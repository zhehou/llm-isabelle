# planner/skeleton.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import re
import requests

# Pull defaults from your existing config
from prover.config import (
    MODEL as DEFAULT_MODEL,
    OLLAMA_HOST,
    TIMEOUT_S as OLLAMA_TIMEOUT_S,   # alias in your config
    OLLAMA_NUM_PREDICT,
    TEMP as OLLAMA_TEMP,              # alias
    TOP_P as OLLAMA_TOP_P,            # alias
)

# --- Prompt that prefers an OUTLINE, but we won't force it unless requested ---
SKELETON_PROMPT = """You are an Isabelle/HOL assistant.

TASK: Given a lemma statement, produce a CLEAN Isar PROOF OUTLINE that shows the structure
(e.g., proof method, cases/induction skeleton). Prefer to leave nontrivial steps as `sorry`.
Output ONLY Isabelle text (no explanations). Start at or after a line:
  lemma "{goal}"
and conclude with:
  qed
"""

@dataclass
class Skeleton:
    text: str
    holes: List[Tuple[int, int]]  # (start_idx, end_idx) spans where 'sorry' occurs

SORRY_RE = re.compile(r"\bsorry\b")
PROOF_RE = re.compile(r"(?m)^\s*proof(?:\b|\s|\()", re.UNICODE)
QED_RE   = re.compile(r"(?m)^\s*qed\b", re.UNICODE)

def _ollama_generate_simple(
    prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    num_predict: Optional[int] = None,
    timeout_s: Optional[int] = None,
) -> str:
    """
    Minimal synchronous call to Ollama's /api/generate endpoint.
    Returns the full 'response' string (not streaming).
    """
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
    resp = requests.post(url, json=payload, timeout=timeout_s or OLLAMA_TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("response") or "").strip()

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
    If force_outline=False (auto): leave content intact (allow complete proofs), just normalize header/proof/qed.
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
        # Replace inline ' by ...' with ' sorry'
        lines = text.splitlines()
        for i, L in enumerate(lines):
            j = L.find(" by ")
            if j != -1:
                lines[i] = L[:j].rstrip() + " sorry"
        text = "\n".join(lines)

        # Ensure at least one sorry before qed
        if "sorry" not in text:
            m_qed = QED_RE.search(text)
            if m_qed:
                insert_at = m_qed.start()
                # simple indent
                text = text[:insert_at] + "  sorry\n" + text[insert_at:]

    # Final newline
    if not text.endswith("\n"):
        text += "\n"
    return text

def propose_isar_skeleton(
    goal: str,
    model: Optional[str] = None,
    temp: float = 0.2,
    *,
    force_outline: bool = False,
) -> Skeleton:
    prompt = SKELETON_PROMPT.format(goal=goal)
    raw = _ollama_generate_simple(
        prompt=prompt,
        model=model or DEFAULT_MODEL,
        temperature=temp,
    )
    cleaned = _sanitize_outline(raw, goal=goal, force_outline=force_outline)
    return Skeleton(text=cleaned, holes=find_sorry_spans(cleaned))
