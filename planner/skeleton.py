# planner/skeleton.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict
import os
import re
import json
import shutil
import subprocess
import time

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
SKELETON_PROMPT = """You are an Isabelle/HOL proof expert.

TASK
Given a lemma statement, produce a CLEAN Isar proof OUTLINE that exposes the decomposition strategy
(e.g., `proof (induction …)`, `proof (cases …)`, or a short `proof` with intermediate `have`/`show` steps).
That is, your aim is to analyse the problem (proof goal) and break it into smaller problems (sub-goals) that are easier to solve. Think about how you can prove it in English, and then translate the proof outline into Isar syntax. Leave nontrivial reasoning steps as `sorry` so that a lower-level prover can fill them later.

OUTPUT REQUIREMENTS
- Output ONLY Isabelle text (no explanations, no code fences).
- Start at or after a line exactly of the form:
  lemma "{goal}"
- Ensure there is a well-formed block:
  proof
    … (case/induction structure, `have` / `show` stubs, etc., with `sorry` where reasoning is omitted) …
  qed
- Prefer *structured* outlines:
  • For lists or natural numbers: `proof (induction xs)` / `proof (induction n)` with `case Nil`/`case Cons` or `case 0`/`case (Suc n)`.
  • For booleans or sum-types: `proof (cases b)` / `proof (cases rule: <type>.exhaust)`.
  • For set/algebraic goals: include helpful rewrites as hints (e.g., `using` lines or `simp add: <facts>` before a `sorry`).
- Keep each subcase minimal: introduce the case with `case …`, then `then show ?thesis` followed by `sorry`.

EXAMPLES OF STYLE
lemma "{goal}"
proof (induction xs)
  case Nil
  then show ?thesis by simp
next
  case (Cons x xs)
  (* outline only; details left to the micro prover *)
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

@dataclass
class Skeleton:
    text: str
    holes: List[Tuple[int, int]]  # (start_idx, end_idx) spans where 'sorry' occurs

SORRY_RE = re.compile(r"\bsorry\b")
PROOF_RE = re.compile(r"(?m)^\s*proof(?:\b|\s|\()", re.UNICODE)
QED_RE   = re.compile(r"(?m)^\s*qed\b", re.UNICODE)

# ----------------------------
# Provider shims (Ollama / HF / Gemini)
# ----------------------------

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

def _hf_generate_simple(
    prompt: str,
    model_id: str,
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    timeout_s: Optional[int] = None,
) -> str:
    """
    Minimal call to Hugging Face Inference API (text-generation).
    Requires env HUGGINGFACE_API_TOKEN.
    """
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
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s or OLLAMA_TIMEOUT_S)
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

# ---------- Gemini helpers (model resolution + CLI/REST) ----------

# Alias map for common-but-wrong or shortened IDs → valid IDs seen in docs.
_GEMINI_MODEL_ALIASES: Dict[str, str] = {
    # Experimental 2.0 Pro is versioned; unversioned often 404:
    "gemini-2.0-pro-exp": "gemini-2.0-pro-exp-02-05",
}

def _gemini_list_models(timeout_s: Optional[int]) -> List[str]:
    """
    Return a list of simple model codes (like 'gemini-2.5-pro') if GEMINI_API_KEY is set.
    Otherwise return [].
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return []
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        resp = requests.get(url, timeout=timeout_s or OLLAMA_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
        out = []
        for m in data.get("models", []):
            name = m.get("name", "")
            # API returns names like 'models/gemini-2.5-pro' → we want the short code
            short = name.split("/")[-1] if name else ""
            if short:
                out.append(short)
        return out
    except Exception:
        return []

def _pick_best_model_id(requested: str, available: List[str]) -> Optional[str]:
    """
    Choose the best available model ID given a requested token and a list of available IDs.
    Strategy:
      1) Exact match
      2) Startswith match preferring non-preview/non-exp if multiple
      3) Fallback: None
    """
    if not available:
        return None
    if requested in available:
        return requested
    # prefer stable over preview/exp
    candidates = [m for m in available if m.startswith(requested)]
    if candidates:
        stable = [m for m in candidates if ("preview" not in m and "exp" not in m)]
        return (stable or candidates)[0]
    return None

def _gemini_resolve_model_id(model_id: str, *, timeout_s: Optional[int] = None, force_list: bool = False) -> str:
    """
    Resolve a user-supplied model ID to a valid Gemini ID:
      - Apply aliases
      - Optionally consult ListModels to find an exact/best match
      - Otherwise return as-is
    """
    model_id = _GEMINI_MODEL_ALIASES.get(model_id, model_id)
    models = _gemini_list_models(timeout_s) if (force_list or os.getenv("GEMINI_API_KEY")) else []
    if models:
        chosen = _pick_best_model_id(model_id, models)
        if chosen:
            return chosen
    return model_id

def _gemini_cli_available() -> bool:
    """Return True if the 'gemini' CLI is on PATH."""
    return shutil.which("gemini") is not None

def _gemini_cli_generate_simple(
    prompt: str,
    model_id: str,
    *,
    timeout_s: Optional[int] = None,
) -> str:
    """
    Use the 'gemini' CLI in non-interactive mode.
    We pass prompt via stdin (no --json, no subcommands).
    """
    cmd = ["gemini", "-m", model_id]
    env = os.environ.copy()
    proc = subprocess.run(
        cmd,
        input=prompt,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s or OLLAMA_TIMEOUT_S,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"gemini CLI failed (code {proc.returncode}): "
            f"{(proc.stderr or proc.stdout).strip()}"
        )
    return proc.stdout.strip()

def _gemini_rest_generate_simple(
    prompt: str,
    model_id: str,
    *,
    max_output_tokens: Optional[int] = None,
    timeout_s: Optional[int] = None,
    retries: int = 2,
    backoff_s: float = 1.5,
) -> str:
    """
    REST fallback using the Generative Language API.
    Requires GEMINI_API_KEY.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set (needed for REST fallback)")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": OLLAMA_NUM_PREDICT if max_output_tokens is None else max_output_tokens,
        },
    }
    attempt = 0
    while True:
        attempt += 1
        resp = requests.post(url, json=body, timeout=timeout_s or OLLAMA_TIMEOUT_S)
        if resp.status_code == 429 and attempt <= retries:
            time.sleep(backoff_s * attempt)
            continue
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

def _gemini_generate_simple(
    prompt: str,
    model_id: str,
    *,
    timeout_s: Optional[int] = None,
) -> str:
    """
    Gemini entry: resolve → CLI (if present) → REST fallback.
    Also retries resolution on NOT_FOUND errors and finally falls back to gemini-2.5-pro.
    """
    # Step 1: best-effort resolution
    resolved = _gemini_resolve_model_id(model_id, timeout_s=timeout_s)

    # Step 2: CLI first (if installed)
    if _gemini_cli_available():
        try:
            return _gemini_cli_generate_simple(prompt, resolved, timeout_s=timeout_s)
        except RuntimeError as e:
            msg = str(e)
            # If the CLI says NOT_FOUND, try resolving via ListModels again (force) then fallback
            if "NOT_FOUND" in msg or "notFound" in msg or "Requested entity was not found" in msg:
                reresolved = _gemini_resolve_model_id(model_id, timeout_s=timeout_s, force_list=True)
                if reresolved != resolved:
                    try:
                        return _gemini_cli_generate_simple(prompt, reresolved, timeout_s=timeout_s)
                    except RuntimeError:
                        pass
            # Fall through to REST
        except Exception:
            pass

    # Step 3: REST fallback (if key available)
    try:
        return _gemini_rest_generate_simple(prompt, resolved, timeout_s=timeout_s)
    except Exception:
        # Last resort: try a safe, stable model
        safe = "gemini-2.5-pro"
        try:
            return _gemini_rest_generate_simple(prompt, safe, timeout_s=timeout_s)
        except Exception:
            # If REST isn't configured, try CLI with the safe model
            if _gemini_cli_available():
                return _gemini_cli_generate_simple(prompt, safe, timeout_s=timeout_s)
            raise

# ---------- Unified dispatch ----------

def _generate_simple(
    prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    num_predict: Optional[int] = None,
    timeout_s: Optional[int] = None,
) -> str:
    """
    Dispatch to Ollama (default), Hugging Face, or Gemini based on model prefix:
      - "hf:<repo_id>"        → Hugging Face Inference API
      - "gemini:<model_id>"   → Gemini CLI (preferred), REST fallback
      - "ollama:<model>" or no prefix → Ollama /api/generate
    """
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
    # Default: Ollama
    return _ollama_generate_simple(
        prompt, model=model,
        temperature=temperature, top_p=top_p,
        num_predict=num_predict, timeout_s=timeout_s
    )

# ----------------------------
# Utilities and main entry
# ----------------------------

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
                text = text[:insert_at] + "  sorry\n" + text[insert_at:]

    # Final newline
    if not text.endswith("\n"):
        text += "\n"
    return text

def propose_isar_skeleton(
    goal: str,
    model: Optional[str] = None,
    temp: float = 0.35,  # kept for ollama/hf; ignored by gemini CLI path
    *,
    force_outline: bool = False,
) -> Skeleton:
    prompt = SKELETON_PROMPT.format(goal=goal)
    raw = _generate_simple(
        prompt=prompt,
        model=model or DEFAULT_MODEL,
        temperature=temp,
    )
    cleaned = _sanitize_outline(raw, goal=goal, force_outline=force_outline)
    return Skeleton(text=cleaned, holes=find_sorry_spans(cleaned))
