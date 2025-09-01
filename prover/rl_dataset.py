
# prover/rl_dataset.py
"""
Build offline RL-style datasets from existing logs, reusing the current feature
shape (prover/features.py). We intentionally avoid changing any existing files.

We provide two dataset builders:
1) make_bandit_dataset(...)  -> (X, y) where y ~ P(success|s,a) from run-level success
2) make_q_targets(...)       -> (X, q) bootstrapped FQI targets using run-level success

Both functions are robust to existing logs and missing fields.
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple, Optional

# We DO NOT import internal helpers of features.py to avoid tight coupling.
# Instead, we re-implement the tiny pieces we need, mirroring features.py.
STEP_TYPES = [
    "apply (induction", "apply (cases", "apply simp", "apply simp_all",
    "apply auto", "apply (simp", "apply (auto", "apply (rule",
    "apply (erule", "apply (intro", "apply (elim", "apply (subst", "apply (metis",
]

def _step_prefix(cmd: str) -> str:
    s = (cmd or "").strip()
    for t in STEP_TYPES:
        if s.startswith(t): return t
    if s.startswith("by "): return "by"
    return s.split(" ", 1)[0][:32]

def _flags_from_goal(goal: str, state_hint: str) -> Dict[str, int]:
    g = ((goal or "") + " " + (state_hint or "")).lower()
    return {
        "is_listy": int(any(k in g for k in ["@", "append", "rev", "map", "take", "drop"])),
        "is_natty": int(any(k in g for k in ["suc", "nat", "≤", "<", "0", "add", "mult", "+", "-", "*"])),
        "is_sety":  int(any(k in g for k in ["∈", "subset", "⋂", "⋃"])),
        "has_q":    int(any(k in g for k in ["∀", "∃"])),
        "is_bool":  int(any(k in g for k in ["true", "false", "¬", "not", "∧", "∨"])),
    }

def _iter_jsonl(paths: Iterable[Path]) -> Iterator[dict]:
    for p in paths:
        if not p.exists(): continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: 
                    continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict):
                        yield rec
                except Exception:
                    continue

def _feature_row_from_attempt(rec: dict) -> List[float]:
    goal = rec.get("goal", "")
    prefix = rec.get("prefix", []) or []
    cand = rec.get("candidate", "") or ""
    depth = int(rec.get("depth", 0) or 0)
    n_sub = rec.get("n_subgoals")
    n_sub = int(n_sub) if n_sub is not None else 9999
    elapsed = float(rec.get("elapsed_ms", 0.0) or 0.0)
    cache_hit = 1 if rec.get("cache_hit") else 0

    state_proxy = " ".join(prefix[-3:])  # cheap context
    flags = _flags_from_goal(goal, state_proxy)

    step_t = _step_prefix(cand)
    step_one_hot = [1 if step_t == t else 0 for t in STEP_TYPES]

    row = [
        depth, n_sub, elapsed, cache_hit,
        flags["is_listy"], flags["is_natty"], flags["is_sety"], flags["has_q"], flags["is_bool"],
    ] + step_one_hot + [len(cand)]
    return row

def _runs_success_map(runs_paths: Iterable[Path]) -> Dict[str, int]:
    """Map run_id -> 1 if success True else 0"""
    succ = {}
    for rec in _iter_jsonl(runs_paths):
        rid = rec.get("run_id")
        if not rid:
            continue
        succ[rid] = 1 if rec.get("success") else 0
    return succ

def make_bandit_dataset(attempts_paths: List[str], runs_paths: Optional[List[str]] = None) -> Tuple[List[List[float]], List[int]]:
    """
    Contextual bandit view: label each expand (s,a) with the run-level success.
    This is a strong baseline and requires no changes to the logger.
    """
    X: List[List[float]] = []
    y: List[int] = []
    succ = _runs_success_map([Path(p) for p in (runs_paths or [])]) if runs_paths else {}
    for rec in _iter_jsonl([Path(p) for p in attempts_paths]):
        if rec.get("type") != "expand":
            continue
        rid = rec.get("run_id")
        label = succ.get(rid, 1 if rec.get("ok") else 0)  # fallback to ok flag if no run record
        X.append(_feature_row_from_attempt(rec))
        y.append(int(label))
    return X, y

def make_q_targets(attempts_paths: List[str], runs_paths: Optional[List[str]] = None, 
                   discount: float = 0.8) -> Tuple[List[List[float]], List[float]]:
    """
    FQI-style target with a single bootstrap step using run-level success:
      target = r + gamma * V', where r=0 per step and +1 if run succeeds;
    This collapses to gamma * success, serving as a smooth target in [0,1].
    """
    X: List[List[float]] = []
    q: List[float] = []
    succ = _runs_success_map([Path(p) for p in (runs_paths or [])]) if runs_paths else {}
    for rec in _iter_jsonl([Path(p) for p in attempts_paths]):
        if rec.get("type") != "expand":
            continue
        rid = rec.get("run_id")
        success = succ.get(rid, 1 if rec.get("ok") else 0)
        X.append(_feature_row_from_attempt(rec))
        q.append(float(discount) * float(success))
    return X, q

def feature_names() -> List[str]:
    base = ["depth","n_sub","elapsed_ms","cache_hit","is_listy","is_natty","is_sety","has_q","is_bool"]
    return base + [f"step::{t}" for t in STEP_TYPES] + ["len_cmd"]
