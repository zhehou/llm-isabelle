# prover/config.py 
"""
Central config for the LLM-guided Isabelle/HOL prover.

This version keeps full backward compatibility with existing imports and
naming (MODEL, TEMP/TOP_P/TIMEOUT_S aliases, etc.), while simplifying env
parsing and adding tiny safety checks.

Key small improvements:
- Central _get() helper to remove repetitive getenv/parse/strip code
- Light validation/clamping for OLLAMA_TEMP and OLLAMA_TOP_P
- Snapshot dataclass uses slots/frozen for smaller/faster instances
- refresh_from_env() to re-read the environment into module globals
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# ----------------- helpers -----------------
def _get(name: str, default: Any, conv: Optional[Callable[[str], Any]] = None) -> Any:
    """Read env var and convert; empty/whitespace means 'unset' -> default."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    if raw == "":
        return default
    if conv is None:
        return raw
    try:
        return conv(raw)
    except Exception:
        return default

def _to_int(s: str) -> int:
    return int(s)

def _to_float(s: str) -> float:
    return float(s)

def _to_bool(s: str) -> bool:
    return s.strip().lower() in {"1", "true", "yes", "y", "on"}

def _split_ws(s: str) -> list[str]:
    return [t for t in s.split() if t]


# ----------------- load (with minimal validation) -----------------
def _clamp(x: float, lo: float, hi: float) -> float:
    return hi if x > hi else lo if x < lo else x

def _load_from_env() -> Dict[str, Any]:
    d: Dict[str, Any] = {}

    # ---------- Ollama / LLM ----------
    d["OLLAMA_HOST"] = _get("OLLAMA_HOST", "http://127.0.0.1:11434")
    d["MODEL"] = _get("OLLAMA_MODEL", "qwen3-coder:30b")

    # Modern names
    d["OLLAMA_TEMP"] = _clamp(_get("OLLAMA_TEMP", 0.2, _to_float), 0.0, 2.0)
    # Most engines assume (0,1]; allow 0 for deterministic; clamp >1 to 1.0
    d["OLLAMA_TOP_P"] = _clamp(_get("OLLAMA_TOP_P", 0.95, _to_float), 0.0, 1.0)
    d["OLLAMA_TIMEOUT_S"] = _get("OLLAMA_TIMEOUT_S", 120, _to_int)
    d["OLLAMA_NUM_PREDICT"] = _get("OLLAMA_NUM_PREDICT", 256, _to_int)

    # Legacy aliases (kept for compatibility with llm.py/logging_utils.py)
    d["TEMP"] = d["OLLAMA_TEMP"]
    d["TOP_P"] = d["OLLAMA_TOP_P"]
    d["TIMEOUT_S"] = d["OLLAMA_TIMEOUT_S"]

    # How many raw candidates we ask the LLM to produce per model call
    d["NUM_CANDIDATES"] = _get("NUM_CANDIDATES", 6, _to_int)

    # ---------- Search / beam ----------
    d["BEAM_WIDTH"] = _get("BEAM_WIDTH", 3, _to_int)
    d["MAX_DEPTH"] = _get("MAX_DEPTH", 8, _to_int)
    d["HINT_LEMMAS"] = _get("HINT_LEMMAS", 6, _to_int)
    d["FACTS_LIMIT"] = _get("FACTS_LIMIT", 6, _to_int)

    # ---------- Minimization / Variants ----------
    d["MINIMIZE_DEFAULT"] = _get("MINIMIZE_DEFAULT", True, _to_bool)
    d["MINIMIZE_TIMEOUT"] = _get("MINIMIZE_TIMEOUT", 8, _to_int)
    d["MINIMIZE_MAX_FACT_TRIES"] = _get("MINIMIZE_MAX_FACT_TRIES", 6, _to_int)

    d["VARIANTS_DEFAULT"] = _get("VARIANTS_DEFAULT", False, _to_bool)
    d["VARIANT_TIMEOUT"] = _get("VARIANT_TIMEOUT", 6, _to_int)
    d["VARIANT_TRIES"] = _get("VARIANT_TRIES", 24, _to_int)

    # ---------- Reranker ----------
    d["RERANKER_DIR"] = Path(_get("RERANKER_DIR", "models")).resolve()
    d["RERANKER_OFF"] = _get("RERANKER_OFF", False, _to_bool)

    # ---------- Logging ----------
    d["LOG_DIR"] = Path(_get("LOG_DIR", "logs")).resolve()
    d["ATTEMPTS_LOG"] = str(d["LOG_DIR"] / _get("ATTEMPTS_LOG", "attempts.log.jsonl"))
    d["RUNS_LOG"] = str(d["LOG_DIR"] / _get("RUNS_LOG", "runs.log.jsonl"))

    # ---------- Isabelle ----------
    d["ISABELLE_SESSION"] = _get("ISABELLE_SESSION", "HOL")

    # extra imports for the Scratch theory (space-separated)
    d["EXTRA_IMPORTS"] = _split_ws(os.environ.get("EXTRA_IMPORTS", ""))

    # ---------- Premises / Context (optional; default off) ----------
    # Two-stage premise selection and file-aware context boosts.
    # Kept disabled by default for full backward compatibility.
    d["PREMISES_ENABLE"]      = _get("PREMISES_ENABLE", False, _to_bool)
    d["PREMISES_K_SELECT"]    = _get("PREMISES_K_SELECT", 512, _to_int)
    d["PREMISES_K_RERANK"]    = _get("PREMISES_K_RERANK", 64, _to_int)
    d["PROVER_CONTEXT_ENABLE"]= _get("PROVER_CONTEXT_ENABLE", False, _to_bool)
    d["PROVER_CONTEXT_WINDOW"]= _get("PROVER_CONTEXT_WINDOW", 400, _to_int)
    # Space-separated list of .thy files to pre-ingest for retrieval/context boosts
    d["PROVER_CONTEXT_FILES"] = _split_ws(os.environ.get("PROVER_CONTEXT_FILES", ""))

    return d


# ----------------- module globals -----------------
# Create LOG_DIR eagerly as before (import-time side effect preserved)
_env = _load_from_env()
globals().update(_env)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Snapshot ----------
@dataclass(slots=True, frozen=True)
class Snapshot:
    model: str
    beam_width: int
    max_depth: int
    hint_lemmas: int
    facts_limit: int
    minimize_default: bool
    minimize_timeout: int
    minimize_max_fact_tries: int
    variants_default: bool
    variant_timeout: int
    variant_tries: int
    reranker_dir: str
    reranker_off: bool
    ollama_host: str
    ollama_temp: float
    ollama_top_p: float
    ollama_timeout_s: int
    ollama_num_predict: int
    num_candidates: int
    isabelle_session: str
    # new (premises/context)
    premises_enable: bool
    premises_k_select: int
    premises_k_rerank: int
    prover_context_enable: bool
    prover_context_window: int
    prover_context_files: str

def snapshot_dict() -> Dict[str, Any]:
    return asdict(
        Snapshot(
            model=MODEL,
            beam_width=BEAM_WIDTH,
            max_depth=MAX_DEPTH,
            hint_lemmas=HINT_LEMMAS,
            facts_limit=FACTS_LIMIT,
            minimize_default=MINIMIZE_DEFAULT,
            minimize_timeout=MINIMIZE_TIMEOUT,
            minimize_max_fact_tries=MINIMIZE_MAX_FACT_TRIES,
            variants_default=VARIANTS_DEFAULT,
            variant_timeout=VARIANT_TIMEOUT,
            variant_tries=VARIANT_TRIES,
            reranker_dir=str(RERANKER_DIR),
            reranker_off=RERANKER_OFF,
            ollama_host=OLLAMA_HOST,
            ollama_temp=OLLAMA_TEMP,
            ollama_top_p=OLLAMA_TOP_P,
            ollama_timeout_s=OLLAMA_TIMEOUT_S,
            ollama_num_predict=OLLAMA_NUM_PREDICT,
            num_candidates=NUM_CANDIDATES,            
            isabelle_session=ISABELLE_SESSION,
            # new (premises/context)
            premises_enable=PREMISES_ENABLE,
            premises_k_select=PREMISES_K_SELECT,
            premises_k_rerank=PREMISES_K_RERANK,
            prover_context_enable=PROVER_CONTEXT_ENABLE,
            prover_context_window=PROVER_CONTEXT_WINDOW,
            prover_context_files=" ".join(PROVER_CONTEXT_FILES),
        )
    )

def refresh_from_env() -> None:
    """Reload module-level config from the current environment.

    Useful in tests or when scripts set os.environ at runtime and want
    to reflect updates without re-importing this module.
    """
    new = _load_from_env()
    globals().update(new)
    LOG_DIR.mkdir(parents=True, exist_ok=True)  # preserve side-effect


__all__ = [
    # LLM / Ollama
    "OLLAMA_HOST", "MODEL",
    "OLLAMA_TEMP", "OLLAMA_TOP_P", "OLLAMA_TIMEOUT_S", "OLLAMA_NUM_PREDICT",
    # Legacy aliases
    "TEMP", "TOP_P", "TIMEOUT_S", "NUM_CANDIDATES",
    # Search
    "BEAM_WIDTH", "MAX_DEPTH", "HINT_LEMMAS", "FACTS_LIMIT",
    # Minimization / Variants
    "MINIMIZE_DEFAULT", "MINIMIZE_TIMEOUT", "MINIMIZE_MAX_FACT_TRIES",
    "VARIANTS_DEFAULT", "VARIANT_TIMEOUT", "VARIANT_TRIES",
    # Reranker
    "RERANKER_DIR", "RERANKER_OFF",
    # Logging
    "LOG_DIR", "ATTEMPTS_LOG", "RUNS_LOG",
    # Isabelle
    "ISABELLE_SESSION", "EXTRA_IMPORTS",
    # Premises / Context
    "PREMISES_ENABLE", "PREMISES_K_SELECT", "PREMISES_K_RERANK",
    "PROVER_CONTEXT_ENABLE", "PROVER_CONTEXT_WINDOW", "PROVER_CONTEXT_FILES",    
    # Snapshot / utils
    "snapshot_dict", "refresh_from_env",
]
