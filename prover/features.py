# prover/features.py
import json
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Any

# Keep in sync with heuristics/live features
STEP_TYPES = [
    "apply (induction", "apply (cases", "apply simp", "apply simp_all",
    "apply auto", "apply (simp", "apply (auto", "apply (rule",
    "apply (erule", "apply (intro", "apply (elim", "apply (subst", "apply (metis",
]

def _step_prefix(cmd: str) -> str:
    s = cmd.strip()
    for t in STEP_TYPES:
        if s.startswith(t): return t
    if s.startswith("by "): return "by"
    return s.split(" ", 1)[0][:32]

def _flags_from_goal(goal: str, state_hint: str) -> Dict[str, int]:
    g = (goal + " " + state_hint).lower()
    return {
        "is_listy": int(any(k in g for k in ["@", "append", "rev", "map", "take", "drop"])),
        "is_natty": int(any(k in g for k in ["suc", "nat", "≤", "<", "0", "add", "mult", "+", "-", "*"])),
        "is_sety":  int(any(k in g for k in ["∈", "subset", "⋂", "⋃"])),
        "has_q":    int(any(k in g for k in ["∀", "∃"])),
        "is_bool":  int(any(k in g for k in ["true", "false", "¬", "not", "∧", "∨"])),
    }

def iter_attempt_rows(paths: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    for p in paths:
        if not p.exists(): continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if isinstance(rec, dict):
                    yield rec

def make_dataset(attempts_paths: List[str]) -> Tuple[List[List[float]], List[int]]:
    """Build X,y for binary classification on expand attempts (ok == True)."""
    X, y = [], []
    for rec in iter_attempt_rows([Path(p) for p in attempts_paths]):
        if rec.get("type") != "expand":
            continue
        ok = bool(rec.get("ok", False))
        goal = rec.get("goal", "")
        prefix = rec.get("prefix", [])
        cand = rec.get("candidate", "")
        depth = int(rec.get("depth", 0))
        n_sub = rec.get("n_subgoals")
        n_sub = int(n_sub) if n_sub is not None else 9999
        elapsed = float(rec.get("elapsed_ms", 0.0))
        cache_hit = 1 if rec.get("cache_hit") else 0

        # Crude state proxy: last few accepted steps (we don't log state text here)
        state_proxy = " ".join(prefix[-3:])
        flags = _flags_from_goal(goal, state_proxy)

        step_t = _step_prefix(cand)
        step_one_hot = [1 if step_t == t else 0 for t in STEP_TYPES]

        row = [
            depth, n_sub, elapsed, cache_hit,
            flags["is_listy"], flags["is_natty"], flags["is_sety"], flags["has_q"], flags["is_bool"],
        ] + step_one_hot + [len(cand)]
        X.append(row); y.append(1 if ok else 0)
    return X, y

def feature_names() -> List[str]:
    base = ["depth","n_sub","elapsed_ms","cache_hit","is_listy","is_natty","is_sety","has_q","is_bool"]
    return base + [f"step::{t}" for t in STEP_TYPES] + ["len_cmd"]
