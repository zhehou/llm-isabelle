# prover/config.py
"""
Central config for the LLM-guided Isabelle/HOL prover.

This file intentionally exposes BOTH the modern names (OLLAMA_TEMP, etc.)
and the legacy names used across existing modules (TEMP, TOP_P, TIMEOUT_S,
NUM_CANDIDATES) to avoid ImportErrors.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict

# ---------- helpers ----------
def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if (v is not None and v.strip() != "") else default

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except Exception:
        return default

def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
    return v in {"1", "true", "yes", "y", "on"}

# ---------- Ollama / LLM ----------
OLLAMA_HOST: str = _env_str("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL: str = _env_str("OLLAMA_MODEL", "qwen3-coder:30b")

# Modern names
OLLAMA_TEMP: float = _env_float("OLLAMA_TEMP", 0.2)
OLLAMA_TOP_P: float = _env_float("OLLAMA_TOP_P", 0.95)
OLLAMA_TIMEOUT_S: int = _env_int("OLLAMA_TIMEOUT_S", 60)
OLLAMA_NUM_PREDICT: int = _env_int("OLLAMA_NUM_PREDICT", 256)

# Legacy aliases (kept for compatibility with llm.py/logging_utils.py)
TEMP: float = OLLAMA_TEMP
TOP_P: float = OLLAMA_TOP_P
TIMEOUT_S: int = OLLAMA_TIMEOUT_S

# How many raw candidates we ask the LLM to produce per model call
NUM_CANDIDATES: int = _env_int("NUM_CANDIDATES", 6)

# ---------- Search / beam ----------
BEAM_WIDTH: int = _env_int("BEAM_WIDTH", 3)
MAX_DEPTH: int = _env_int("MAX_DEPTH", 8)
HINT_LEMMAS: int = _env_int("HINT_LEMMAS", 6)
FACTS_LIMIT: int = _env_int("FACTS_LIMIT", 6)

# ---------- Minimization / Variants ----------
MINIMIZE_DEFAULT: bool = _env_bool("MINIMIZE_DEFAULT", True)
MINIMIZE_TIMEOUT: int = _env_int("MINIMIZE_TIMEOUT", 8)
MINIMIZE_MAX_FACT_TRIES: int = _env_int("MINIMIZE_MAX_FACT_TRIES", 6)

VARIANTS_DEFAULT: bool = _env_bool("VARIANTS_DEFAULT", False)
VARIANT_TIMEOUT: int = _env_int("VARIANT_TIMEOUT", 6)
VARIANT_TRIES: int = _env_int("VARIANT_TRIES", 24)

# ---------- Reranker ----------
RERANKER_DIR: Path = Path(_env_str("RERANKER_DIR", "models")).resolve()
RERANKER_OFF: bool = _env_bool("RERANKER_OFF", False)

# ---------- Logging ----------
LOG_DIR: Path = Path(_env_str("LOG_DIR", "logs")).resolve()
LOG_DIR.mkdir(parents=True, exist_ok=True)
ATTEMPTS_LOG: str = str(LOG_DIR / _env_str("ATTEMPTS_LOG", "attempts.log.jsonl"))
RUNS_LOG: str = str(LOG_DIR / _env_str("RUNS_LOG", "runs.log.jsonl"))

# ---------- Isabelle ----------
ISABELLE_SESSION: str = _env_str("ISABELLE_SESSION", "HOL")

# extra imports for the Scratch theory (space-separated), default empty
EXTRA_IMPORTS = os.getenv("EXTRA_IMPORTS", "").split()

# ---------- Snapshot ----------
@dataclass
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

def snapshot_dict() -> Dict[str, Any]:
    return asdict(Snapshot(
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
    ))

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
    "ISABELLE_SESSION",
    # Snapshot
    "snapshot_dict",
]
