from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict, Iterable
import json
import os
import re
import time
from functools import lru_cache
from planner.prompts import SKELETON_PROMPT
import requests

# Pull defaults from your existing prover/config.py
from prover.config import (
    MODEL as DEFAULT_MODEL,
    OLLAMA_HOST,
    TIMEOUT_S as OLLAMA_TIMEOUT_S,
    OLLAMA_NUM_PREDICT,
    TEMP as OLLAMA_TEMP,
    TOP_P as OLLAMA_TOP_P,
)

# Isabelle helpers (for quick sketch check)
from prover.isabelle_api import build_theory, finished_ok, run_theory, last_print_state_block
from prover.utils import parse_subgoals

# Reuse local-context miner from repair (defs/facts list)
from planner.repair import _facts_from_state as _facts_from_state

# One HTTP session (keep-alive)
_SESSION = requests.Session()

@dataclass(slots=True)
class Skeleton:
    text: str
    holes: List[Tuple[int, int]]  # (start_idx, end_idx) spans where 'sorry' occurs

SORRY_RE = re.compile(r"\bsorry\b")
PROOF_RE = re.compile(r"(?m)^\s*proof(?:\b|\s|\()", re.UNICODE)
QED_RE   = re.compile(r"(?m)^\s*qed\b", re.UNICODE)
# New regex helpers for sanitization
CASE_START_RE   = re.compile(r"(?m)^\s*case\b")
NEXT_OR_QED_RE  = re.compile(r"(?m)^\s*(next|qed)\b")
SHOW_THESIS_RE = re.compile(r"(?m)^\s*(?:then\s+)?show\s+\?thesis\b")
# Match the meta-variable immediately after 'show', capturing the 'show ' prefix so we can preserve any
# leading tokens like 'then', 'from ...', 'with ...', 'finally', etc. We only rewrite the '?thesis|?case' token.
SHOW_META_AT_SHOW = re.compile(r"(?m)(?P<prefix>\bshow\s+)\?(?P<meta>thesis|case)\b")
BARE_PROOF_RE  = re.compile(r"(?m)^\s*proof\s*$")
# General proof-mode detectors (for ?case/?thesis normalization)
_PROOF_OPEN_RE   = re.compile(r"(?m)^\s*proof(?:\s*\(([^)]+)\))?\s*$")
_QED_LINE_RE     = re.compile(r"(?m)^\s*qed\b")
_MODE_CASES_RE   = re.compile(r"^\s*cases\b")
_MODE_CASES_RULE = re.compile(r"^\s*cases\s+rule:")
_MODE_INDUCT_RE  = re.compile(r"^\s*(?:induction|induct|coinduction|coinduct)\b")
_HAS_TACTIC_NEXT = re.compile(r"(?m)^\s*(?:by\b|apply\b|proof\b|sorry\b|done\b)")
_HAVE_OR_SHOW    = re.compile(r"(?m)^\s*(have|show)\b")
_INLINE_BY       = re.compile(r"\s+by\s+.+$")
_CONTINUATION_HEAD = re.compile(r"(?m)^\s*(?:using|from|with|then|ultimately|finally|also|moreover)\b")
_STMT_OR_BOUNDARY = re.compile(r"(?m)^\s*(?:have|show|assume|case|next|qed)\b")

# =============================================================================
# Provider shims: Ollama (default), Hugging Face ("hf:"), Gemini ("gemini:")
# =============================================================================

def _ollama_generate_simple(
    prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    num_predict: Optional[int] = None,
    timeout_s: Optional[int] = None,
) -> str:
    url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
    payload = {
        "model": model or DEFAULT_MODEL,
        "prompt": prompt,
        "options": {
            "temperature": OLLAMA_TEMP if temperature is None else temperature,
            "top_p": OLLAMA_TOP_P if top_p is None else top_p,
            "num_predict": OLLAMA_NUM_PREDICT if num_predict is None else num_predict,
        },
        "stream": False,
    }
    resp = _SESSION.post(url, json=payload, timeout=timeout_s or OLLAMA_TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("response") or "").strip()

def _hf_generate_simple(
    prompt: str,
    model_id: str,
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    timeout_s: Optional[int] = None,
) -> str:
    token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACE_API_TOKEN is not set")
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}"}
    payload: Dict[str, Any] = {
        "inputs": prompt,
        "parameters": {
            "temperature": OLLAMA_TEMP if temperature is None else temperature,
            "top_p": OLLAMA_TOP_P if top_p is None else top_p,
            "max_new_tokens": OLLAMA_NUM_PREDICT if max_new_tokens is None else max_new_tokens,
            "return_full_text": False,
        },
        "options": {"wait_for_model": True},
    }
    resp = _SESSION.post(url, headers=headers, json=payload, timeout=timeout_s or OLLAMA_TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and data:
        return (data[0].get("generated_text") or "").strip()
    if isinstance(data, dict):
        if "generated_text" in data:
            return (data["generated_text"] or "").strip()
        choices = data.get("choices") or []
        if choices:
            t = choices[0].get("text") or choices[0].get("generated_text") or ""
            return str(t).strip()
    return str(data).strip()

@lru_cache(maxsize=1)
def _gemini_list_models_cached(api_key: str) -> List[str]:
    if not api_key:
        return []
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        resp = _SESSION.get(url, timeout=OLLAMA_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
        out = []
        for m in data.get("models", []):
            name = m.get("name", "")
            short = name.split("/")[-1] if name else ""
            if short:
                out.append(short)
        return out
    except Exception:
        return []

def _gemini_resolve_model_id(model_id: str, *, timeout_s: Optional[int] = None) -> str:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return model_id
    models = _gemini_list_models_cached(api_key)
    if model_id in models:
        return model_id
    cands = [m for m in models if m.startswith(model_id)]
    if cands:
        stable = [m for m in cands if ("preview" not in m and "exp" not in m)]
        return (stable or cands)[0]
    return model_id

def _gemini_cli_available() -> bool:
    from shutil import which
    return which("gemini") is not None

def _gemini_cli_generate_simple(prompt: str, model_id: str, *, timeout_s: Optional[int] = None) -> str:
    import subprocess
    cmd = ["gemini", "-m", model_id]
    proc = subprocess.run(
        cmd, input=prompt, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=timeout_s or OLLAMA_TIMEOUT_S, env=os.environ.copy()
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gemini CLI failed ({proc.returncode}): {(proc.stderr or proc.stdout).strip()}")
    return proc.stdout.strip()

def _gemini_rest_generate_simple(prompt: str, model_id: str, *, timeout_s: Optional[int] = None) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set (needed for Gemini REST)")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": OLLAMA_NUM_PREDICT}}
    resp = _SESSION.post(url, json=body, timeout=timeout_s or OLLAMA_TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()
    try:
        cands = data.get("candidates") or []
        if cands:
            parts = ((cands[0].get("content") or {}).get("parts")) or []
            if parts:
                return (parts[0].get("text") or "").strip()
    except Exception:
        pass
    return str(data).strip()

def _gemini_generate_simple(prompt: str, model_id: str, *, timeout_s: Optional[int] = None) -> str:
    # Share one deadline across CLI, REST and fallback attempts.  Previously
    # every attempt received the complete timeout, so one outline request
    # could silently consume three or four times its planner allocation.
    budget = max(1.0, float(timeout_s or OLLAMA_TIMEOUT_S))
    deadline = time.monotonic() + budget

    def remaining() -> int:
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("Gemini outline generation deadline exhausted")
        return max(1, int(left))

    resolved = _gemini_resolve_model_id(model_id, timeout_s=remaining())
    if _gemini_cli_available():
        try:
            return _gemini_cli_generate_simple(prompt, resolved, timeout_s=remaining())
        except Exception:
            pass
    try:
        return _gemini_rest_generate_simple(prompt, resolved, timeout_s=remaining())
    except Exception:
        fallback = "gemini-2.5-pro"
        if _gemini_cli_available():
            try:
                return _gemini_cli_generate_simple(prompt, fallback, timeout_s=remaining())
            except Exception:
                pass
        try:
            return _gemini_rest_generate_simple(prompt, fallback, timeout_s=remaining())
        except Exception as exc:
            return f"__ERROR__ {exc}"

def _generate_simple(
    prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    num_predict: Optional[int] = None,
    timeout_s: Optional[int] = None,
) -> str:
    if model:
        if model.startswith("hf:"):
            return _hf_generate_simple(
                prompt, model_id=model[len("hf:"):],
                temperature=temperature, top_p=top_p,
                max_new_tokens=num_predict, timeout_s=timeout_s
            )
        if model.startswith("gemini:"):
            return _gemini_generate_simple(
                prompt, model_id=model[len("gemini:"):],
                timeout_s=timeout_s
            )
        if model.startswith("ollama:"):
            model = model[len("ollama:"):]
    return _ollama_generate_simple(
        prompt, model=model, temperature=temperature, top_p=top_p,
        num_predict=num_predict, timeout_s=timeout_s
    )

# -----------------------------------------------------------------------------
# Utilities: sorry spans, sanitize, state block, facts, scoring
# -----------------------------------------------------------------------------

def find_sorry_spans(isar: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in SORRY_RE.finditer(isar)]


def _generation_failed(text: str) -> bool:
    """Recognise provider/transport diagnostics that are not Isar output."""
    stripped = (text or "").strip()
    return not stripped or stripped.startswith("__ERROR__")

def _ensure_lemma_header(text: str, goal: str) -> str:
    body = text.lstrip()
    if not body.startswith("lemma"):
        return f'lemma "{goal}"\n{body}'
    return text

def _normalize_calculation_ellipsis(text: str) -> str:
    # Replace Unicode ellipsis and spaced PDF (. . .) with Isar token "..."
    text = text.replace("…", "...")
    text = re.sub(r"\.\s*\.\s*\.", "...", text)
    return text

def _crop_to_first_proof_block(text: str) -> str:
    """
    Keep only the first lemma..(proof..qed)* block; be nesting-aware so the cropped
    text ends at the *matching* 'qed' for the first 'proof', not at an inner one.
    """
    # Find the first lemma line
    m_lemma = re.search(r'(?m)^\s*lemma\s+"[^"]*"', text)
    if not m_lemma:
        return text
    tail = text[m_lemma.start():]

    # Find the first 'proof' after that lemma
    m_proof = PROOF_RE.search(tail)
    if not m_proof:
        # No proof — still return from lemma onward (sanitizers may add a skeleton later)
        return tail

    # Walk forward and balance nested 'proof'/'qed'
    depth = 1
    end_idx = None
    pos = m_proof.end()

    # We want earliest upcoming PROOF or QED at each step
    while True:
        m_next_proof = PROOF_RE.search(tail, pos)
        m_next_qed   = QED_RE.search(tail, pos)

        if not m_next_proof and not m_next_qed:
            # No closing 'qed' — return as-is; later passes may append a final 'qed'
            break

        # Choose whichever comes first
        choose_qed = False
        if m_next_qed and m_next_proof:
            choose_qed = (m_next_qed.start() <= m_next_proof.start())
        elif m_next_qed and not m_next_proof:
            choose_qed = True
        else:
            choose_qed = False

        if choose_qed:
            depth -= 1
            pos = m_next_qed.end()
            if depth == 0:
                end_idx = pos
                break
        else:
            depth += 1
            pos = m_next_proof.end()

    if end_idx is None:
        # Could not find the matching outer 'qed'; return tail as-is (other sanitizers add one)
        return tail

    return tail[:end_idx] + "\n"

def _normalize_show_kinds(text: str) -> str:
    """
    Flip ONLY the '?thesis'/'?case' meta after 'show', preserving any prefix (e.g., 'then', 'from', 'with', 'using', 'finally').
    Policy (line-local, nesting-aware):
      • Inside branches of (co)induction → 'show ?case'
      • Inside branches of 'proof (cases …)' (incl. 'cases rule: …') → 'show ?thesis'
      • Outside any explicit case branch → default to 'show ?thesis'
    """
    lines = text.splitlines()
    # Track current proof mode for the *innermost* open proof
    stack: list[str] = []  # values: "induct" | "cases" | "plain"
    in_case_branch = False
    for i, L in enumerate(lines):
        m_open = _PROOF_OPEN_RE.match(L)
        if m_open:
            mode = (m_open.group(1) or "").strip()
            if not mode:
                stack.append("plain")
            elif _MODE_INDUCT_RE.match(mode):
                stack.append("induct")
            elif _MODE_CASES_RULE.match(mode) or _MODE_CASES_RE.match(mode):
                stack.append("cases")
            else:
                stack.append("plain")
            continue
        if _QED_LINE_RE.match(L):
            if stack:
                stack.pop()
            continue
        if CASE_START_RE.match(L):
            in_case_branch = True
            continue
        if NEXT_OR_QED_RE.match(L):
            in_case_branch = False
            continue

        def _repl(m: "re.Match[str]") -> str:
            current = stack[-1] if stack else "plain"
            want = "case" if (in_case_branch and current == "induct") else "thesis"
            # If already correct, return unchanged
            if m.group("meta") == want:
                return m.group(0)
            return f'{m.group("prefix")}?{want}'

        # Rewrite at most once per line; this keeps everything except the meta token intact.
        lines[i] = SHOW_META_AT_SHOW.sub(_repl, L, count=1)
    return "\n".join(lines)

def _ensure_have_show_bodies(text: str) -> str:
    """
    Ensure every 'have …' / 'show …' has a body, but *preserve* local continuations:
    scan forward across lines starting with using/from/with/then/also/moreover/finally,
    and only insert 'sorry' at the *end* of that block if no tactic/proof body occurs
    before the next statement/boundary (have/show/assume/case/next/qed).
    """
    lines = text.splitlines()
    i, n = 0, len(lines)
    out: List[str] = []
    while i < n:
        L = lines[i]
        out.append(L)
        if _HAVE_OR_SHOW.match(L) and not _INLINE_BY.search(L):
            j = i + 1
            # skip blank lines
            while j < n and lines[j].strip() == "":
                out.append(lines[j]); j += 1
            # consume a local continuation block
            saw_body = False
            k = j
            while k < n:
                Nk = lines[k]
                if _HAS_TACTIC_NEXT.match(Nk):
                    saw_body = True
                    break
                if _STMT_OR_BOUNDARY.match(Nk):
                    break
                if _CONTINUATION_HEAD.match(Nk) or Nk.strip() == "":
                    out.append(Nk); k += 1
                    continue
                # unknown line: stop the block here
                break
            if not saw_body:
                indent = L[:len(L) - len(L.lstrip(" "))]
                out.append(f"{indent}  sorry")
            i = k
            continue
        i += 1
    return "\n".join(out)

def _maybe_proof_dash(text: str) -> str:
    """
    If there is a bare 'proof' at top-level and calculational cues present, prefer 'proof -'.
    """
    if not BARE_PROOF_RE.search(text):
        return text
    if re.search(r"(?m)^\s*(have|also|moreover|ultimately|finally|hence|thus)\b", text):
        return BARE_PROOF_RE.sub("proof -", text, count=1)
    return text

def _sanitize_outline(text: str, goal: str, *, force_outline: bool) -> str:
    # Provider errors must never cross the boundary into Isabelle source.
    # Converting an error string into ``lemma ... __ERROR__ ...`` obscures the
    # real failure and makes every subsequent fill and repair unverifiable.
    if _generation_failed(text):
        text = ""
    text = _ensure_lemma_header(text, goal)
    # Normalize ellipsis first (avoid Unicode / spaced form)
    text = _normalize_calculation_ellipsis(text)

    # Keep content from *this* lemma onwards
    goal_header = f'lemma "{goal}"'
    idx = text.find(goal_header)
    if idx >= 0:
        text = text[idx:]
    else:
        first_lemma = text.find("lemma ")
        if first_lemma >= 0:
            text = text[first_lemma:]

    # Preserve a valid one-line proof in auto mode.  Previously a second
    # ``proof ... qed`` block was appended after ``by``, making it invalid.
    if not PROOF_RE.search(text):
        one_line = text.splitlines()
        by_idx = next(
            (i for i, line in enumerate(one_line[1:], start=1) if re.match(r"^\s*by\b", line)),
            -1,
        )
        if by_idx >= 0:
            # A complete one-line proof is a valid plan in every mode.  Opening
            # it into an artificial hole discarded the model's best answer.
            text = "\n".join(one_line[: by_idx + 1]).rstrip() + "\n"
            return text

    # Ensure proof/qed skeleton exists
    if not PROOF_RE.search(text):
        text = text.rstrip() + "\nproof\n  sorry\nqed\n"
    if not QED_RE.search(text):
        text = text.rstrip() + "\nqed\n"

    # Preserve solved leaves in a structured outline.  Deleting every
    # ``by ...`` manufactured extra holes and discarded useful LLM work.  A
    # complete structured proof is already a successful planner result; only
    # a top-level one-line proof is deliberately opened above.

    # Light Isar fixups (order matters)
    #  1) Flip only the meta after 'show', preserving 'then/using/from/with/finally' etc.
    #  2) Ensure every 'have/show' has a body; insert 'sorry' if missing to trigger fill/repair.
    #  3) Prefer 'proof -' when calculational cues are present.
    text = _normalize_show_kinds(text)
    text = _ensure_have_show_bodies(text)
    text = _maybe_proof_dash(text)

    # Trim to the first complete lemma..qed block to avoid trailing splices
    text = _crop_to_first_proof_block(text)

    # ``proof -; show ?thesis; sorry; qed`` adds no planning structure, but it
    # refines a theorem ``A ⟹ B`` to a leaf ``B`` and can leave ``A`` only as
    # an unnamed chained fact.  Canonicalise this vacuous wrapper to a direct
    # theorem-level hole.  Real outlines with assumes, cases, calculations or
    # intermediate facts are deliberately untouched.
    vacuous = re.fullmatch(
        r'(lemma\s+"[^\n]*"[^\n]*\n)\s*proof\s+-\s*\n'
        r'\s*show\s+\?thesis\s*\n\s*sorry\s*\n\s*qed\s*',
        text.strip() + "\n",
    )
    if vacuous:
        text = vacuous.group(1).rstrip() + "\n  sorry\n"

    if not text.endswith("\n"):
        text += "\n"
    return text

def _quick_sketch_score(
    isabelle,
    session_id: str,
    outline_text: str,
    *,
    budget_s: Optional[float] = None,
) -> int:
    """Estimate real outline cost from its reachable hole states.

    The old check appended ``print_state`` after ``qed``, so it did not inspect
    any hole and commonly gave every candidate the same sentinel score.
    """
    started = time.monotonic()
    budget = max(1.0, float(budget_s)) if budget_s is not None else 30.0

    def left() -> float:
        return max(0.0, budget - (time.monotonic() - started))

    try:
        thy = build_theory(outline_text.splitlines(), add_print_state=False, end_with=None)
        resps = run_theory(
            isabelle, session_id, thy,
            timeout_s=max(1, min(8, int(left()) or 1)),
        )
        ok, _ = finished_ok(resps)
        if not ok:
            return 10**6
        holes = find_sorry_spans(outline_text)
        if not holes:
            return 0
        from planner.goals import _print_state_before_hole

        score = 20 * len(holes)
        inspected = 0
        for span in holes[:2]:
            if left() <= 1.0:
                break
            block = _print_state_before_hole(
                isabelle, session_id, outline_text, span, trace=False,
                timeout_s=max(1, min(6, int(left()) or 1)),
            ) or ""
            inspected += 1
            n = parse_subgoals(block)
            if not isinstance(n, int):
                score += 500
            else:
                score += 10 * n + min(200, len(" ".join(block.split())) // 20)
        # Exhausting a deliberately small scoring budget should not classify
        # an otherwise valid outline as invalid.  Uninspected holes simply
        # retain a structural cost.
        score += 80 * max(0, len(holes) - inspected)
        return score
    except Exception:
        return 10**6

def _state_block_for_goal(
    isabelle, session_id: str, goal: str, theory_preamble: str = ""
) -> str:
    prefix = theory_preamble.strip()
    if prefix:
        prefix += "\n\n"
    mini = prefix + f'lemma "{goal}"\nproof\n  print_state\n  sorry\nqed\n'
    try:
        thy = build_theory(mini.splitlines(), add_print_state=False, end_with=None)
        resps = run_theory(isabelle, session_id, thy, timeout_s=12)
        return last_print_state_block(resps) or ""
    except Exception:
        return ""

# --- Pattern detection / simple priors ---

_PAT_INDUCTION = re.compile(r"(?m)^\s*proof\s*\(induction\b", re.UNICODE)
_PAT_CASES     = re.compile(r"(?m)^\s*proof\s*\(cases\b", re.UNICODE)
_PAT_CASES_RULE= re.compile(r"(?m)^\s*proof\s*\(cases\s+rule:\s*([A-Za-z0-9_\.]+)", re.UNICODE)
_ID = r"[A-Za-z_][A-Za-z0-9_']*"

def _detect_pattern_key(outline: str) -> str:
    if _PAT_INDUCTION.search(outline):
        return "induction"
    m = _PAT_CASES_RULE.search(outline)
    if m:
        return f"cases_rule:{m.group(1)}"
    if _PAT_CASES.search(outline):
        return "cases"
    return "plain"

def _tokenize_goal(goal: str) -> set:
    toks = set(re.findall(_ID, goal))
    if "@" in goal: toks.add("@")
    if "⟹" in goal: toks.add("implies")
    return toks

def _load_priors(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Expect either {"rules":[...]} or a bare list of rules.
        if isinstance(data, dict) and isinstance(data.get("rules"), list):
            return data["rules"]
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []

def _pattern_penalty(goal: str, outline: str, rules: List[Dict[str, Any]]) -> float:
    """
    Lower is better. Simple heuristic + optional JSON rules.
    """
    key = _detect_pattern_key(outline)
    toks = _tokenize_goal(goal)
    pen = 0.0
    # Built-in gentle priors
    if ("@" in toks or "map" in toks) and key != "induction":
        pen += 0.4
    if ({"Suc", "0"} & toks) and key != "induction":
        pen += 0.3
    if ({"True", "False"} & toks) and not key.startswith("cases"):
        pen += 0.25
    if ({"Some", "None", "option"} & toks) and "cases_rule:option.exhaust" != key and key != "cases":
        pen += 0.25
    if ({"Inl", "Inr"} & toks) and "cases_rule:sum.exhaust" != key and key != "cases":
        pen += 0.25
    # Optional external rules
    for r in rules:
        cond = set(map(str, r.get("if_any_tokens", [])))
        prefer = set(map(str, r.get("prefer_patterns", [])))
        weight = float(r.get("weight", 0.3))
        if cond and (cond & toks) and key and prefer and key not in prefer:
            pen += weight
    return pen

def _hint_bonus_from_outline(outline: str, recommended: List[str]) -> int:
    if not recommended:
        return 0
    # Count how many recommended tokens appear in the outline text (rough proxy)
    text = outline
    c = 0
    for h in recommended[:10]:
        if h in text:
            c += 1
    return c

# -----------------------------------------------------------------------------
# NEW: Hint lexicon (micro-RAG) utilities
# -----------------------------------------------------------------------------

def _load_hintlex(path: Optional[str]) -> Dict[str, List[str]]:
    """
    Returns token -> [hint,...]. Accepts either {"token":[["hint",count],...]} or {"token":["hint",...]}.
    """
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}
    out: Dict[str, List[str]] = {}
    for tok, val in raw.items():
        if isinstance(val, list):
            if val and isinstance(val[0], list):
                out[tok] = [h for h, _c in val]
            else:
                out[tok] = [str(h) for h in val]
    return out

def _hints_from_hintlex(goal: str, hintlex: Dict[str, List[str]], top: int = 8) -> List[str]:
    toks = _tokenize_goal(goal)
    got: List[str] = []
    for t in toks:
        hs = hintlex.get(t)
        if not hs:
            continue
        for h in hs[:top]:
            got.append(h)
    # stable de-dup
    return list(dict.fromkeys(got))

# -----------------------------------------------------------------------------
# Outline generators
# -----------------------------------------------------------------------------

def propose_isar_skeleton(
    goal: str,
    model: Optional[str] = None,
    temp: float = 0.35,
    *,
    force_outline: bool = False,
    hints: Optional[List[str]] = None,
    timeout_s: Optional[int] = None,
    theory_preamble: str = "",
) -> Skeleton:
    # Inject tiny hint list when available (keeps default behavior if None/empty)
    prompt = SKELETON_PROMPT.format(goal=goal)
    if hints:
        prompt += "\nHINTS: Prefer using " + ", ".join(sorted(set(hints))) + " if applicable.\n"
    if theory_preamble.strip():
        prompt += (
            "\nTHEORY CONTEXT (already declared before the lemma; do not repeat it):\n"
            + theory_preamble.strip()
            + "\nEND THEORY CONTEXT\n"
        )
    raw = _generate_simple(
        prompt=prompt,
        model=model or DEFAULT_MODEL,
        temperature=temp,
        timeout_s=timeout_s or OLLAMA_TIMEOUT_S,
    )
    cleaned = _sanitize_outline(raw, goal=goal, force_outline=force_outline)
    if theory_preamble.strip():
        cleaned = theory_preamble.strip() + "\n\n" + cleaned.lstrip()
    return Skeleton(text=cleaned, holes=find_sorry_spans(cleaned))

def propose_isar_skeletons(
    goal: str,
    *,
    model: Optional[str] = None,
    temps: Iterable[float] = (0.3, 0.5, 0.8),
    k: Optional[int] = None,
    force_outline: bool = False,
    hints: Optional[List[str]] = None,
    total_timeout_s: Optional[float] = None,
    theory_preamble: str = "",
) -> List[Skeleton]:
    seen, out = set(), []
    temp_list = list(temps)
    deadline = (
        time.monotonic() + max(1.0, float(total_timeout_s))
        if total_timeout_s is not None else None
    )
    for index, t in enumerate(temp_list):
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 1.0:
                break
            calls_left = max(1, len(temp_list) - index)
            if index == 0 and calls_left > 1:
                # A usable first outline is more valuable than several calls
                # that all time out just before the model responds.
                call_timeout = max(1, int(min(remaining, remaining * 0.60)))
            else:
                call_timeout = max(1, int(remaining / calls_left))
        else:
            call_timeout = OLLAMA_TIMEOUT_S
        prompt = SKELETON_PROMPT.format(goal=goal)
        if hints:
            prompt += "\nHINTS: Prefer using " + ", ".join(sorted(set(hints))) + " if applicable.\n"
        if theory_preamble.strip():
            prompt += (
                "\nTHEORY CONTEXT (already declared before the lemma; do not repeat it):\n"
                + theory_preamble.strip()
                + "\nEND THEORY CONTEXT\n"
            )
        try:
            raw = _generate_simple(
                prompt=prompt,
                model=model or DEFAULT_MODEL,
                temperature=float(t),
                timeout_s=min(OLLAMA_TIMEOUT_S, call_timeout),
            )
        except Exception:
            raw = ""
        if _generation_failed(raw):
            continue
        cleaned = _sanitize_outline(raw, goal=goal, force_outline=force_outline)
        if theory_preamble.strip():
            cleaned = theory_preamble.strip() + "\n\n" + cleaned.lstrip()
        sk = Skeleton(text=cleaned, holes=[])
        sk.holes = find_sorry_spans(sk.text)
        key = sk.text.strip()
        if key not in seen:
            seen.add(key)
            out.append(sk)
        if k is not None and len(out) >= int(k):
            break
    if not out:
        # Do not spend the final second retrying the same unavailable backend.
        # Use a known-valid structural template so exact-context automation can
        # still solve the theorem locally.
        fallback = _lib_templates_for_goal(goal)[0]
        if theory_preamble.strip():
            fallback.text = theory_preamble.strip() + "\n\n" + fallback.text.lstrip()
            fallback.holes = find_sorry_spans(fallback.text)
        return [fallback]
    return out

def _lib_templates_for_goal(goal: str) -> List[Skeleton]:
    toks = _tokenize_goal(goal)
    used_names = set(re.findall(r"[A-Za-z][A-Za-z0-9_']*", goal))
    head_name = next(
        (name for name in ("a", "x1", "elem", "hd1") if name not in used_names),
        "fresh_head",
    )
    lib: List[str] = []
    if ("@" in toks or "map" in toks) and "xs" in toks:
        lib.append(
f'''lemma "{goal}"
proof (induction xs)
  case Nil
  show ?case by simp
next
  case (Cons {head_name} xs)
  show ?case
    using Cons.IH
    sorry
qed
''')
    if ({"Suc","0"} & toks) and "n" in toks:
        lib.append(
f'''lemma "{goal}"
proof (induction n)
  case 0
  show ?case by simp
next
  case (Suc n)
  show ?case
    using Suc.IH
    sorry
qed
''')
    if ({"True","False"} & toks) and "b" in toks:
        lib.append(
f'''lemma "{goal}"
proof (cases b)
  case True
  show ?thesis
    sorry
next
  case False
  show ?thesis
    sorry
qed
''')
    # A safe neutral candidate.  Never invent placeholder propositions: a
    # quoted comment is not a valid HOL term and cannot be repaired by tactics.
    lib.append(f'''lemma "{goal}"
  sorry
''')
    return [Skeleton(text=s if s.endswith("\n") else s+"\n", holes=find_sorry_spans(s)) for s in lib]

def propose_isar_skeleton_diverse_best(
    goal: str,
    *,
    isabelle,             # required for sketch check
    session_id: str,
    model: Optional[str] = None,
    temps: Iterable[float] = (0.35, 0.55, 0.85),
    k: int = 3,
    force_outline: bool = False,
    # knobs
    priors_path: Optional[str] = None,
    context_hints: bool = False,
    lib_templates: bool = False,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.2,
    # NEW: hint lexicon
    hintlex_path: Optional[str] = None,
    hintlex_top: int = 8,
    total_budget_s: Optional[float] = None,
    theory_preamble: str = "",
) -> Tuple[Skeleton, Dict[str, Any]]:
    """
    Generate K outlines, optionally inject context & hintlex hints, run one-shot sketch checks,
    and return the best using composite score:
      score = alpha * subgoals + beta * pattern_penalty - gamma * hint_bonus
    """
    # Optional context hints from Isabelle state + hint lexicon
    rec_hints: List[str] = []
    if context_hints:
        state_block = _state_block_for_goal(
            isabelle, session_id, goal, theory_preamble
        )
        rec_hints += _facts_from_state(state_block)[:8]
    hintlex = _load_hintlex(hintlex_path)
    if hintlex:
        rec_hints += _hints_from_hintlex(goal, hintlex, top=hintlex_top)
    rec_hints = list(dict.fromkeys(rec_hints))[:12]  # stable de-dup + cap

    # Outline generation and scoring must leave time for the actual filler.
    # The old code allowed each of K LLM calls and every state probe to consume
    # its full independent timeout.
    started = time.monotonic()
    total_budget = max(4.0, float(total_budget_s)) if total_budget_s is not None else None
    generation_budget = total_budget * 0.60 if total_budget is not None else None
    cands = propose_isar_skeletons(
        goal, model=model, temps=temps, k=k,
        force_outline=force_outline, hints=rec_hints,
        total_timeout_s=generation_budget,
        theory_preamble=theory_preamble,
    )
    if lib_templates:
        templates = _lib_templates_for_goal(goal)
        if theory_preamble.strip():
            for template in templates:
                template.text = theory_preamble.strip() + "\n\n" + template.text.lstrip()
                template.holes = find_sorry_spans(template.text)
        cands = templates + cands

    # Load optional priors/rules
    rules = _load_priors(priors_path)

    scored: List[Tuple[float, int, int]] = []  # (score, n_subgoals, idx)
    for i, sk in enumerate(cands):
        if total_budget is None:
            score_budget = None
        else:
            remaining = max(1.0, total_budget - (time.monotonic() - started))
            score_budget = remaining / max(1, len(cands) - i)
        n = _quick_sketch_score(
            isabelle, session_id, sk.text, budget_s=score_budget
        )
        pat_pen = _pattern_penalty(goal, sk.text, rules)
        hint_b = _hint_bonus_from_outline(sk.text, rec_hints)
        score = alpha * float(n) + beta * float(pat_pen) - gamma * float(hint_b)
        scored.append((score, n, i))

    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    if scored and scored[0][1] < 10**6:
        best = cands[scored[0][2]]
        rejected_all = False
    else:
        # Do not feed a known-unreachable outline into fill/repair merely
        # because every generated candidate failed sketch checking.  A neutral
        # theorem-shaped hole is reachable and lets the verified prover search
        # the actual goal instead of repairing a false intermediate claim.
        best = _lib_templates_for_goal(goal)[-1]
        if theory_preamble.strip():
            best.text = theory_preamble.strip() + "\n\n" + best.text.lstrip()
            best.holes = find_sorry_spans(best.text)
        rejected_all = True
    diag = {
        "scores": scored,
        # Internal planner consumers use this ordered archive for bounded
        # backtracking when the best-looking outline reaches a dead frontier.
        "ranked_candidates": [
            cands[index].text for _score, n_subgoals, index in scored
            if n_subgoals < 10**6
        ],
        "num_candidates": len(cands),
        "used_hints": rec_hints[:12],
        "priors_rules": len(rules),
        "alpha_beta_gamma": (alpha, beta, gamma),
        "all_generated_outlines_rejected": rejected_all,
    }
    return best, diag
