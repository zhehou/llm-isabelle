from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import requests

from prover.config import (
    MODEL as DEFAULT_MODEL,
    OLLAMA_HOST,
    TIMEOUT_S as OLLAMA_TIMEOUT_S,
    OLLAMA_NUM_PREDICT,
    TEMP as OLLAMA_TEMP,
    TOP_P as OLLAMA_TOP_P,
)
from prover.isabelle_api import build_theory, run_theory, last_print_state_block
from prover.utils import parse_subgoals

_SESSION = requests.Session()

# =========================
# Backends: Ollama / HF / Gemini
# =========================

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
    payload: Dict[str, object] = {
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

def _gemini_resolve_model_id(model_id: str) -> str:
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
    resolved = _gemini_resolve_model_id(model_id)
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
    *,
    timeout_s: Optional[int] = None,
) -> str:
    m = (model or DEFAULT_MODEL) or ""
    if m.startswith("hf:"):
        return _hf_generate_simple(prompt, model_id=m[len("hf:"):], timeout_s=timeout_s)
    if m.startswith("gemini:"):
        return _gemini_generate_simple(prompt, model_id=m[len("gemini:"):], timeout_s=timeout_s)
    if m.startswith("ollama:"):
        m = m[len("ollama:"):]
    return _ollama_generate_simple(prompt, model=m, timeout_s=timeout_s)

# =========================
# Repair op schema
# =========================

@dataclass(frozen=True)
class InsertBeforeHole:
    line: str

@dataclass(frozen=True)
class ReplaceInSnippet:
    find: str
    replace: str

@dataclass(frozen=True)
class InsertHaveBlock:
    label: str
    statement: str
    after_line_matching: str
    body_hint: str

RepairOp = Tuple[str, object]

def _extract_json_array(text: str) -> Optional[list]:
    """Try to salvage the first JSON array from free-form text."""
    try:
        return json.loads(text)
    except Exception:
        pass
    i = text.find("["); j = text.rfind("]")
    if i != -1 and j != -1 and j > i:
        frag = text[i:j+1]
        try:
            return json.loads(frag)
        except Exception:
            return None
    return None

def _parse_repair_ops(text: str) -> List[RepairOp]:
    data = _extract_json_array(text.strip())
    if not isinstance(data, list):
        return []
    out: List[RepairOp] = []
    for item in data:
        if not isinstance(item, dict) or len(item) != 1:
            continue
        k = next(iter(item.keys()))
        v = item[k]
        if k == "insert_before_hole" and isinstance(v, str) and v.strip():
            out.append(("insert_before_hole", InsertBeforeHole(v.strip())))
        elif k == "replace_in_snippet" and isinstance(v, dict):
            f = v.get("find", ""); r = v.get("replace", "")
            if isinstance(f, str) and isinstance(r, str) and f.strip() and r.strip():
                out.append(("replace_in_snippet", ReplaceInSnippet(f.strip(), r.strip())))
        elif k == "insert_have_block" and isinstance(v, dict):
            lab = v.get("label", "H"); stmt = v.get("statement", "")
            after = v.get("after_line_matching", "then show ?thesis"); hint = v.get("body_hint", "apply simp")
            if all(isinstance(x, str) for x in (lab, stmt, after, hint)) and stmt.strip() and after.strip():
                out.append(("insert_have_block", InsertHaveBlock(lab.strip(), stmt.strip(), after.strip(), hint.strip())))
    return out[:3]

# =========================
# Local context mining
# =========================

_ID = r"[A-Za-z_][A-Za-z0-9_']*"
_DEF_RE = re.compile(rf"\b({_ID})_def\b")
_FACT_RE = re.compile(rf"\b({_ID})\b")
_APPLY_OR_BY = re.compile(r"^\s*(apply\b|by\b)")
_HEADER_RE = re.compile(r"^\s*(proof\s*\(|proof\b|case\s+|then\s+show\b)")

def _hole_line_bounds(full_text: str, hole_span: Tuple[int, int]) -> Tuple[int, int, List[str]]:
    lines = full_text.splitlines()
    upto = full_text[:hole_span[0]]
    hole_line = upto.count("\n")
    line_text = lines[hole_line] if 0 <= hole_line < len(lines) else ""
    indent = len(line_text) - len(line_text.lstrip(" "))
    return hole_line, indent, lines

def _snippet_window(lines: List[str], hole_line: int, radius: int = 12) -> Tuple[int, int]:
    s = max(0, hole_line - radius)
    e = min(len(lines), hole_line + radius + 1)
    return s, e

def _facts_from_state(state_block: str, limit: int = 16) -> List[str]:
    defs = _DEF_RE.findall(state_block or "")
    tokens = _FACT_RE.findall(state_block or "")
    priority = ["algebra_simps", "field_simps", "append_assoc", "map_append", "length_append", "rev_append", "rev_rev_ident"]
    out: List[str] = []
    seen = set()
    for x in priority + defs + tokens:
        if not x or x in seen:
            continue
        seen.add(x); out.append(x)
        if len(out) >= limit:
            break
    return out

def _nearest_header(lines: List[str], hole_line: int) -> str:
    for i in range(hole_line, -1, -1):
        L = lines[i].strip()
        if _HEADER_RE.match(L):
            return L
    return ""

def _recent_steps(lines: List[str], hole_line: int, max_lines: int = 5) -> List[str]:
    out: List[str] = []
    for i in range(hole_line - 1, -1, -1):
        L = lines[i]
        if _APPLY_OR_BY.match(L):
            out.append(L.strip())
            if len(out) >= max_lines:
                break
        if L.strip().startswith(("case ", "proof", "qed", "lemma ")):
            break
    return list(reversed(out))

# =========================
# Patch application helpers
# =========================

def _apply_insert_before_hole(full_text: str, hole_span: Tuple[int, int], line: str) -> str:
    hole_line, indent, lines = _hole_line_bounds(full_text, hole_span)
    insertion = " " * max(2, indent) + line
    lines.insert(hole_line, insertion)
    return "\n".join(lines) + ("\n" if full_text.endswith("\n") else "")

def _apply_replace_in_snippet(full_text: str, hole_span: Tuple[int, int], find: str, replace: str) -> str:
    hole_line, _indent, lines = _hole_line_bounds(full_text, hole_span)
    s, e = _snippet_window(lines, hole_line)
    snippet = lines[s:e]
    try:
        idx = snippet.index(find)
        snippet[idx] = replace
    except ValueError:
        stripped = [L.strip() for L in snippet]
        try:
            idx = stripped.index(find.strip())
            orig = snippet[idx]
            leading = orig[: len(orig) - len(orig.lstrip(" "))]
            snippet[idx] = leading + replace.lstrip(" ")
        except ValueError:
            return full_text
    new_lines = lines[:s] + snippet + lines[e:]
    return "\n".join(new_lines) + ("\n" if full_text.endswith("\n") else "")

def _apply_insert_have_block(full_text: str, hole_span: Tuple[int, int], label: str, statement: str, after_line_matching: str, body_hint: str) -> str:
    hole_line, indent, lines = _hole_line_bounds(full_text, hole_span)
    s, e = _snippet_window(lines, hole_line)
    anchor_idx = None
    for i in range(s, e):
        if lines[i].strip() == after_line_matching.strip():
            anchor_idx = i; break
    if anchor_idx is None:
        anchor_idx = hole_line
    pad = " " * max(2, indent)
    block = [
        f'{pad}have {label}: "{statement}"',
        f"{pad}  sorry",
    ]
    new_lines = lines[:anchor_idx] + block + lines[anchor_idx:]
    return "\n".join(new_lines) + ("\n" if full_text.endswith("\n") else "")

# =========================
# LLM prompt for local repair
# =========================

_REPAIR_SYSTEM = """You patch an Isabelle/Isar proof LOCALLY. Do not regenerate the whole proof.
Return ONLY a JSON array of at most 3 patch operations from the allowed schema.

ALLOWED OPS (choose from):
1) {"insert_before_hole": "<ONE LINE, e.g., apply (simp add: …)>"}
2) {"replace_in_snippet": {"find": "<EXACT LINE>", "replace": "<NEW LINE>"}}
3) {"insert_have_block": {"label":"H", "statement":"<FORMULA>", "after_line_matching":"<LINE IN SNIPPET>", "body_hint":"<ONE LINE>"}}

Rules:
- Edit only inside the provided SNIPPET. Never change the lemma line or text outside the snippet.
- Prefer single-line hints like `apply (simp add: …)`, `apply (unfolding …_def)`, `apply (simp only: …_def)`, `using …`.
- If variables escape induction, adjust header via replace_in_snippet (e.g., add `arbitrary: ys`).
- If necessary, insert a small `have` block just above the failing `show`. Keep its body a single `sorry`.
- The output MUST be valid JSON with no comments.
"""

_REPAIR_USER = """GOAL:
{goal}

STATE_BEFORE_HOLE:
{state_block}

FACTS/DEFS CANDIDATES:
{facts_list}

NEAREST_HEADER:
{nearest_header}

RECENT_STEPS:
{recent_steps}

SNIPPET (edit only inside this region):
<<<SNIPPET
{block_snippet}
SNIPPET

Output a JSON array of at most 3 patch ops according to the allowed schema.
"""

def propose_local_repairs(
    *,
    goal: str,
    state_block: str,
    block_snippet: str,
    nearest_header: str,
    recent_steps: List[str],
    facts: List[str],
    model: Optional[str],
    timeout_s: int,
) -> List[RepairOp]:
    prompt = _REPAIR_SYSTEM + "\n\n" + _REPAIR_USER.format(
        goal=goal,
        state_block=state_block.strip(),
        block_snippet=block_snippet.rstrip(),
        nearest_header=nearest_header.strip(),
        recent_steps="\n".join(recent_steps),
        facts_list=", ".join(facts),
    )
    raw = _generate_simple(prompt, model=model, timeout_s=timeout_s)
    return _parse_repair_ops(raw)

# =========================
# Heuristic fallback ops (when LLM gives nothing usable)
# =========================

def _heuristic_fallback_ops(goal_text: str, state_block: str, header: str, facts: List[str]) -> List[RepairOp]:
    ops: List[RepairOp] = []

    # Unfolding (Let)
    if "Let " in state_block or "Let_def" in facts:
        ops.append(("insert_before_hole", InsertBeforeHole("apply (unfolding Let_def)")))

    # Classic list lemmas
    g = goal_text
    if ("map" in g and "@" in g) or ("map_append" in facts):
        ops.append(("insert_before_hole", InsertBeforeHole("apply (simp add: map_append)")))
    if ("length" in g and "@" in g) or ("length_append" in facts):
        ops.append(("insert_before_hole", InsertBeforeHole("apply (simp add: length_append)")))
    if ("@" in g) or ("append_assoc" in facts):
        ops.append(("insert_before_hole", InsertBeforeHole("apply (simp add: append_assoc)")))

    # Light induction header tweak: add arbitrary: ys if likely
    if header.startswith("proof (induction") and ("arbitrary:" not in header):
        for v in ("ys", "zs"):
            if v in g:
                ops.append(("replace_in_snippet",
                            ReplaceInSnippet(find=header, replace=header.rstrip(")") + f" arbitrary: {v})")))
                break

    return ops[:3]

# =========================
# Quick check & orchestrator
# =========================

def _quick_state_subgoals(isabelle, session: str, text: str) -> int:
    """
    Run Isabelle on the CURRENT text (no offset surgery) and parse the last print_state.
    Returns a large sentinel on failure.
    """
    try:
        thy = build_theory(text.splitlines(), add_print_state=True, end_with="sorry")
        resps = run_theory(isabelle, session, thy)
        block = last_print_state_block(resps) or ""
        n = parse_subgoals(block)
        return int(n) if isinstance(n, int) else 9999
    except Exception:
        return 9999

def _quick_state_text(isabelle, session: str, full_text: str) -> str:
    try:
        thy = build_theory(full_text.splitlines(), add_print_state=True, end_with="sorry")
        resps = run_theory(isabelle, session, thy)
        return last_print_state_block(resps) or ""
    except Exception:
        return ""

def try_local_repairs(
    *,
    full_text: str,
    hole_span: Tuple[int, int],
    goal_text: str,
    model: Optional[str],
    isabelle,
    session: str,
    repair_budget_s: float = 8.0,
    max_ops_to_try: int = 2,
    trace: bool = False,
) -> Tuple[str, bool, str]:
    start = time.monotonic()
    def _left() -> float:
        return max(0.0, repair_budget_s - (time.monotonic() - start))

    # Split budget: reserve ~half for applying/quick-checking
    propose_timeout = int(min(6, max(3, repair_budget_s * 0.5)))

    # Baseline state on current text
    s0 = _quick_state_subgoals(isabelle, session, full_text)

    # Local context (computed once for the prompt)
    hole_line, indent, lines = _hole_line_bounds(full_text, hole_span)
    s, e = _snippet_window(lines, hole_line, radius=12)
    snippet = "\n".join(lines[s:e])
    state_block = _quick_state_text(isabelle, session, full_text)
    facts = _facts_from_state(state_block)
    header = _nearest_header(lines, hole_line)
    rsteps = _recent_steps(lines, hole_line)

    # Ask LLM for repair ops with bounded time
    ops = propose_local_repairs(
        goal=goal_text,
        state_block=state_block,
        block_snippet=snippet,
        nearest_header=header,
        recent_steps=rsteps,
        facts=facts,
        model=model,
        timeout_s=propose_timeout,
    )
    if not ops:
        ops = _heuristic_fallback_ops(goal_text, state_block, header, facts)

    if trace:
        printable = []
        for kind, payload in ops:
            if kind == "insert_before_hole":
                printable.append(f'insert_before_hole("{payload.line}")')
            elif kind == "replace_in_snippet":
                printable.append(f'replace_in_snippet(find="{payload.find}", replace="{payload.replace}")')
            else:
                printable.append(f'insert_have_block("{payload.label}", "...")')
        if printable:
            print("[repair] proposed:", "; ".join(printable))

    # If almost no time left, blind-apply the first simple op so we at least land a visible hint.
    if _left() < 0.75:
        if trace:
            print("[repair] budget exhausted before trying ops → blind-apply first simple patch")
        for kind, payload in ops:
            if kind == "insert_before_hole":
                return _apply_insert_before_hole(full_text, hole_span, payload.line), True, "blind-insert"
            if kind == "replace_in_snippet":
                return _apply_replace_in_snippet(full_text, hole_span, payload.find, payload.replace), True, "blind-replace"
        # otherwise, fall through and just give up
        return full_text, False, "no-time"

    tried = 0
    current = full_text
    for (kind, payload) in ops:
        if _left() <= 0 or tried >= max_ops_to_try:
            break
        tried += 1

        if kind == "insert_before_hole":
            current2 = _apply_insert_before_hole(current, hole_span, payload.line)
        elif kind == "replace_in_snippet":
            current2 = _apply_replace_in_snippet(current, hole_span, payload.find, payload.replace)
        elif kind == "insert_have_block":
            current2 = _apply_insert_have_block(current, hole_span, payload.label, payload.statement, payload.after_line_matching, payload.body_hint)
        else:
            continue

        if current2 == current:
            continue

        s1 = _quick_state_subgoals(isabelle, session, current2)
        accept = (s1 != 9999) and (s1 <= s0 or kind in ("insert_before_hole", "replace_in_snippet"))
        if trace:
            print(f"[repair] trying {kind} → subgoals {s0} → {s1} | {'ACCEPT' if accept else 'reject'}")

        if accept:
            return current2, True, kind

    return full_text, False, "repairs-did-not-help"
