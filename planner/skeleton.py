from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict, Iterable
import os
import re
import time
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

# One HTTP session (keep-alive)
_SESSION = requests.Session()

# -----------------------------------------------------------------------------
# Prompt for OUTLINES (not full proofs unless mode=auto and LLM decides so)
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
BY_INLINE_RE = re.compile(r"\s+by\s+.*$")  # replace inline 'by ...' with 'sorry' when forcing outline

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

# ---------- Gemini helpers ----------
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
    # Try to resolve against ListModels when API key is present; else return as-is
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return model_id
    models = _gemini_list_models_cached(api_key)
    if model_id in models:
        return model_id
    # heuristic startswith fallback
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
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": OLLAMA_NUM_PREDICT},
    }
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
        # final fallback to a stable public model id if possible
        fallback = "gemini-2.5-pro"
        if _gemini_cli_available():
            try:
                return _gemini_cli_generate_simple(prompt, fallback, timeout_s=timeout_s)
            except Exception:
                pass
        return _gemini_rest_generate_simple(prompt, fallback, timeout_s=timeout_s)

# Unified dispatch
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
    # default: Ollama
    return _ollama_generate_simple(
        prompt, model=model, temperature=temperature, top_p=top_p,
        num_predict=num_predict, timeout_s=timeout_s
    )

# -----------------------------------------------------------------------------
# Utilities and main entry (single outline, diverse outlines, quick sketch check)
# -----------------------------------------------------------------------------

def find_sorry_spans(isar: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in SORRY_RE.finditer(isar)]

def _ensure_lemma_header(text: str, goal: str) -> str:
    body = text.lstrip()
    if not body.startswith("lemma"):
        return f'lemma "{goal}"\n{body}'
    return text

def _sanitize_outline(text: str, goal: str, *, force_outline: bool) -> str:
    """
    Ensure: starts from lemma, has proof...qed block.
    If force_outline=True: must contain ≥1 `sorry`, and inline `by ...` lines are replaced with `sorry`.
    If force_outline=False: leave content intact (allow complete proofs), just normalize header/proof/qed.
    """
    text = _ensure_lemma_header(text, goal)

    # Keep only content starting from the lemma matching our goal if possible
    goal_header = f'lemma "{goal}"'
    idx = text.find(goal_header)
    if idx >= 0:
        text = text[idx:]
    else:
        first_lemma = text.find("lemma ")
        if first_lemma >= 0:
            text = text[first_lemma:]

    # Ensure proof/qed block exists
    if not PROOF_RE.search(text):
        text = text.rstrip() + "\nproof\n  sorry\nqed\n"
    if not QED_RE.search(text):
        text = text.rstrip() + "\nqed\n"

    if force_outline:
        # Replace inline '... by ...' finishers with '... sorry'
        lines = text.splitlines()
        for i, L in enumerate(lines):
            if " by " in L:
                lines[i] = BY_INLINE_RE.sub(" sorry", L)
        text = "\n".join(lines)
        # Ensure at least one sorry before qed
        if "sorry" not in text:
            m_qed = QED_RE.search(text)
            if m_qed:
                insert_at = m_qed.start()
                text = text[:insert_at] + "  sorry\n" + text[insert_at:]

    if not text.endswith("\n"):
        text += "\n"
    return text

def propose_isar_skeleton(
    goal: str,
    model: Optional[str] = None,
    temp: float = 0.35,
    *,
    force_outline: bool = False,
) -> Skeleton:
    """
    Backward-compatible single-outline generator (what older code calls).
    """
    raw = _generate_simple(
        prompt=SKELETON_PROMPT.format(goal=goal),
        model=model or DEFAULT_MODEL,
        temperature=temp,
        timeout_s=OLLAMA_TIMEOUT_S,
    )
    cleaned = _sanitize_outline(raw, goal=goal, force_outline=force_outline)
    return Skeleton(text=cleaned, holes=find_sorry_spans(cleaned))

# ---- NEW: produce several outlines by varying temperature ----
def propose_isar_skeletons(
    goal: str,
    *,
    model: Optional[str] = None,
    temps: Iterable[float] = (0.3, 0.5, 0.8),
    k: Optional[int] = None,
    force_outline: bool = False,
) -> List[Skeleton]:
    seen, out = set(), []
    for t in temps:
        raw = _generate_simple(
            prompt=SKELETON_PROMPT.format(goal=goal),
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
        # Fallback: one low-temp outline
        return [propose_isar_skeleton(goal, model=model, temp=0.3, force_outline=force_outline)]
    return out

# ---- NEW: quick sketch scoring (fewer subgoals is better) ----
def _quick_sketch_score(isabelle, session_id: str, outline_text: str) -> int:
    try:
        thy = build_theory(outline_text.splitlines(), add_print_state=True, end_with="sorry")
        resps = run_theory(isabelle, session_id, thy)
        block = last_print_state_block(resps) or ""
        n = parse_subgoals(block)
        return int(n) if isinstance(n, int) else 9999
    except Exception:
        return 9999

# ---- NEW: select the best outline after quick check ----
def propose_isar_skeleton_diverse_best(
    goal: str,
    *,
    isabelle,             # required for sketch check
    session_id: str,
    model: Optional[str] = None,
    temps: Iterable[float] = (0.35, 0.55, 0.85),
    k: int = 3,
    force_outline: bool = False,
) -> Tuple[Skeleton, Dict[str, Any]]:
    """
    Generate K outlines at diverse temps, run a one-shot sketch check, and return the best (fewest subgoals).
    """
    cands = propose_isar_skeletons(goal, model=model, temps=temps, k=k, force_outline=force_outline)
    scored: List[Tuple[int, int]] = []  # (n_subgoals, idx)
    for i, sk in enumerate(cands):
        n = _quick_sketch_score(isabelle, session_id, sk.text)
        scored.append((n, i))
    scored.sort(key=lambda x: (x[0], x[1]))
    best = cands[scored[0][1]]
    diag = {"scores": scored, "num_candidates": len(cands)}
    return best, diag
