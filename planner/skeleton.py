from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict, Iterable
import json
import os
import re
from functools import lru_cache

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
from prover.isabelle_api import build_theory, run_theory, last_print_state_block
from prover.utils import parse_subgoals

# Reuse local-context miner from repair (defs/facts list)
from planner.repair import _facts_from_state as _facts_from_state

# One HTTP session (keep-alive)
_SESSION = requests.Session()

# -----------------------------------------------------------------------------
# Prompt for OUTLINES
# -----------------------------------------------------------------------------
SKELETON_PROMPT = """You are an Isabelle/HOL proof expert.

TASK
Given a lemma statement, produce a CLEAN Isar proof OUTLINE that exposes the decomposition strategy
(e.g., `proof (induction …)`, `proof (cases …)`, or a short `proof` with intermediate `have`/`show` steps).
Think in English first, then translate the outline into Isar. Leave nontrivial reasoning steps as `sorry`.

OUTPUT REQUIREMENTS
- Output ONLY Isabelle text (no explanations, no code fences).
- Start at or after a line exactly of the form:
  lemma "{goal}"
- Ensure there is a well-formed block:
  proof
    … (case/induction structure, small `have`/`show` stubs, `sorry` where reasoning is omitted) …
  qed
- Prefer structured outlines:
  • For lists/nats: `proof (induction xs)` / `proof (induction n)` with `case Nil`/`case Cons` or `case 0`/`case (Suc n)`.
  • For booleans/sum-types: `proof (cases b)` / `proof (cases rule: <type>.exhaust)`.
  • For set/algebraic goals: include helpful rewrites as hints (e.g., `using`/`simp add:`) before a `sorry`.

STYLE EXAMPLES
lemma "{goal}"
proof (induction xs)
  case Nil
  then show ?thesis by simp
next
  case (Cons x xs)
  have H1: "…"
    sorry
  then show ?thesis
    sorry
qed

lemma "{goal}"
proof (cases b)
  case True
  then show ?thesis
    sorry
next
  case False
  then show ?thesis
    sorry
qed

lemma "{goal}"
proof
  show "P"
    sorry
next
  show "Q"
    sorry
qed

lemma "{goal}"
proof -
  have "A ⟹ B"
    sorry
  moreover have "B ⟹ C"
    sorry
  ultimately show ?thesis
    sorry
qed
"""

@dataclass(slots=True)
class Skeleton:
    text: str
    holes: List[Tuple[int, int]]  # (start_idx, end_idx) spans where 'sorry' occurs

SORRY_RE = re.compile(r"\bsorry\b")
PROOF_RE = re.compile(r"(?m)^\s*proof(?:\b|\s|\()", re.UNICODE)
QED_RE   = re.compile(r"(?m)^\s*qed\b", re.UNICODE)
BY_INLINE_RE = re.compile(r"\s+by\s+.*$")

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
    resolved = _gemini_resolve_model_id(model_id, timeout_s=timeout_s)
    if _gemini_cli_available():
        try:
            return _gemini_cli_generate_simple(prompt, resolved, timeout_s=timeout_s)
        except Exception:
            pass
    try:
        return _gemini_rest_generate_simple(prompt, resolved, timeout_s=timeout_s)
    except Exception:
        fallback = "gemini-2.5-pro"
        if _gemini_cli_available():
            try:
                return _gemini_cli_generate_simple(prompt, fallback, timeout_s=timeout_s)
            except Exception:
                pass
        return _gemini_rest_generate_simple(prompt, fallback, timeout_s=timeout_s)

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

def _ensure_lemma_header(text: str, goal: str) -> str:
    body = text.lstrip()
    if not body.startswith("lemma"):
        return f'lemma "{goal}"\n{body}'
    return text

def _sanitize_outline(text: str, goal: str, *, force_outline: bool) -> str:
    text = _ensure_lemma_header(text, goal)
    goal_header = f'lemma "{goal}"'
    idx = text.find(goal_header)
    if idx >= 0:
        text = text[idx:]
    else:
        first_lemma = text.find("lemma ")
        if first_lemma >= 0:
            text = text[first_lemma:]
    if not PROOF_RE.search(text):
        text = text.rstrip() + "\nproof\n  sorry\nqed\n"
    if not QED_RE.search(text):
        text = text.rstrip() + "\nqed\n"
    if force_outline:
        lines = text.splitlines()
        for i, L in enumerate(lines):
            if " by " in L:
                lines[i] = BY_INLINE_RE.sub(" sorry", L)
        text = "\n".join(lines)
        if "sorry" not in text:
            m_qed = QED_RE.search(text)
            if m_qed:
                insert_at = m_qed.start()
                text = text[:insert_at] + "  sorry\n" + text[insert_at:]
    if not text.endswith("\n"):
        text += "\n"
    return text

def _quick_sketch_score(isabelle, session_id: str, outline_text: str) -> int:
    try:
        thy = build_theory(outline_text.splitlines(), add_print_state=True, end_with="sorry")
        resps = run_theory(isabelle, session_id, thy)
        block = last_print_state_block(resps) or ""
        n = parse_subgoals(block)
        return int(n) if isinstance(n, int) else 9999
    except Exception:
        return 9999

def _state_block_for_goal(isabelle, session_id: str, goal: str) -> str:
    mini = f'lemma "{goal}"\nproof\n  sorry\nqed\n'
    try:
        thy = build_theory(mini.splitlines(), add_print_state=True, end_with="sorry")
        resps = run_theory(isabelle, session_id, thy)
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
) -> Skeleton:
    # Inject tiny hint list when available (keeps default behavior if None/empty)
    prompt = SKELETON_PROMPT.format(goal=goal)
    if hints:
        prompt += "\nHINTS: Prefer using " + ", ".join(sorted(set(hints))) + " if applicable.\n"
    raw = _generate_simple(
        prompt=prompt,
        model=model or DEFAULT_MODEL,
        temperature=temp,
        timeout_s=OLLAMA_TIMEOUT_S,
    )
    cleaned = _sanitize_outline(raw, goal=goal, force_outline=force_outline)
    return Skeleton(text=cleaned, holes=find_sorry_spans(cleaned))

def propose_isar_skeletons(
    goal: str,
    *,
    model: Optional[str] = None,
    temps: Iterable[float] = (0.3, 0.5, 0.8),
    k: Optional[int] = None,
    force_outline: bool = False,
    hints: Optional[List[str]] = None,
) -> List[Skeleton]:
    seen, out = set(), []
    for t in temps:
        prompt = SKELETON_PROMPT.format(goal=goal)
        if hints:
            prompt += "\nHINTS: Prefer using " + ", ".join(sorted(set(hints))) + " if applicable.\n"
        raw = _generate_simple(
            prompt=prompt,
            model=model or DEFAULT_MODEL,
            temperature=float(t),
            timeout_s=OLLAMA_TIMEOUT_S,
        )
        sk = Skeleton(text=_sanitize_outline(raw, goal=goal, force_outline=force_outline),
                      holes=[])
        sk.holes = find_sorry_spans(sk.text)
        key = sk.text.strip()
        if key not in seen:
            seen.add(key)
            out.append(sk)
        if k is not None and len(out) >= int(k):
            break
    if not out:
        return [propose_isar_skeleton(goal, model=model, temp=0.3, force_outline=force_outline, hints=hints)]
    return out

def _lib_templates_for_goal(goal: str) -> List[Skeleton]:
    toks = _tokenize_goal(goal)
    lib: List[str] = []
    if ("@" in toks or "map" in toks) and "xs" in toks:
        lib.append(
f'''lemma "{goal}"
proof (induction xs)
  case Nil
  then show ?thesis by simp
next
  case (Cons x xs)
  then show ?thesis
    sorry
qed
''')
    if ({"Suc","0"} & toks) and "n" in toks:
        lib.append(
f'''lemma "{goal}"
proof (induction n)
  case 0
  then show ?thesis by simp
next
  case (Suc n)
  then show ?thesis
    sorry
qed
''')
    if ({"True","False"} & toks) and "b" in toks:
        lib.append(
f'''lemma "{goal}"
proof (cases b)
  case True
  then show ?thesis
    sorry
next
  case False
  then show ?thesis
    sorry
qed
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
) -> Tuple[Skeleton, Dict[str, Any]]:
    """
    Generate K outlines, optionally inject context & hintlex hints, run one-shot sketch checks,
    and return the best using composite score:
      score = alpha * subgoals + beta * pattern_penalty - gamma * hint_bonus
    """
    # Optional context hints from Isabelle state + hint lexicon
    rec_hints: List[str] = []
    if context_hints:
        state_block = _state_block_for_goal(isabelle, session_id, goal)
        rec_hints += _facts_from_state(state_block)[:8]
    hintlex = _load_hintlex(hintlex_path)
    if hintlex:
        rec_hints += _hints_from_hintlex(goal, hintlex, top=hintlex_top)
    rec_hints = list(dict.fromkeys(rec_hints))[:12]  # stable de-dup + cap

    # Outline candidates (LLM) + optional library templates
    cands = propose_isar_skeletons(goal, model=model, temps=temps, k=k,
                                   force_outline=force_outline, hints=rec_hints)
    if lib_templates:
        cands = _lib_templates_for_goal(goal) + cands

    # Load optional priors/rules
    rules = _load_priors(priors_path)

    scored: List[Tuple[float, int, int]] = []  # (score, n_subgoals, idx)
    for i, sk in enumerate(cands):
        n = _quick_sketch_score(isabelle, session_id, sk.text)
        pat_pen = _pattern_penalty(goal, sk.text, rules)
        hint_b = _hint_bonus_from_outline(sk.text, rec_hints)
        score = alpha * float(n) + beta * float(pat_pen) - gamma * float(hint_b)
        scored.append((score, n, i))

    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    best = cands[scored[0][2]]
    diag = {
        "scores": scored,
        "num_candidates": len(cands),
        "used_hints": rec_hints[:12],
        "priors_rules": len(rules),
        "alpha_beta_gamma": (alpha, beta, gamma),
    }
    return best, diag
