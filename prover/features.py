# prover/features.py
import json
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Any

# Keep in sync with heuristics/live features and proposal generators.
STEP_TYPES: List[str] = [
    "apply (induction", "apply (cases", "apply simp", "apply simp_all",
    "apply auto", "apply (simp", "apply (auto", "apply (rule",
    "apply (erule", "apply (intro", "apply (elim", "apply (subst", "apply (metis",
]

# Small precomputations for speed (no behavior change)
_STEP_INDEX: Dict[str, int] = {t: i for i, t in enumerate(STEP_TYPES)}

_LISTY_TOKENS: Tuple[str, ...] = ("@", "append", "rev", "map", "take", "drop")
_NATTY_TOKENS: Tuple[str, ...] = ("suc", "nat", "≤", "<", "0", "add", "mult", "+", "-", "*")
_SETY_TOKENS:  Tuple[str, ...] = ("∈", "subset", "⋂", "⋃")
_Q_TOKENS:     Tuple[str, ...] = ("∀", "∃")
_BOOL_TOKENS:  Tuple[str, ...] = ("true", "false", "¬", "not", "∧", "∨")


def _any_in(s: str, tokens: Tuple[str, ...]) -> bool:
    # micro-helper avoids repeated generator allocations at call sites
    for k in tokens:
        if k in s:
            return True
    return False


def step_prefix(cmd: str) -> str:
    s = (cmd or "").strip()
    for t in STEP_TYPES:
        if s.startswith(t):
            return t
    if s.startswith("by "):
        return "by"
    # Fallback: first word capped to 32 chars (unchanged behavior)
    return s.split(" ", 1)[0][:32]


def flags_from_goal(goal: str, state_hint: str) -> Dict[str, int]:
    # Lowercase once (original used lower(), keep for identical semantics)
    g = ((goal or "") + " " + (state_hint or "")).lower()
    return {
        "is_listy": int(_any_in(g, _LISTY_TOKENS)),
        "is_natty": int(_any_in(g, _NATTY_TOKENS)),
        "is_sety":  int(_any_in(g, _SETY_TOKENS)),
        "has_q":    int(_any_in(g, _Q_TOKENS)),
        "is_bool":  int(_any_in(g, _BOOL_TOKENS)),
    }


def iter_attempt_rows(paths: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    for p in paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if isinstance(rec, dict):
                    yield rec


def make_dataset(attempts_paths: List[str]) -> Tuple[List[List[float]], List[int]]:
    """Build X,y for binary classification on expand attempts (ok == True)."""
    X: List[List[float]] = []
    y: List[int] = []

    paths = [Path(p) for p in attempts_paths]
    for rec in iter_attempt_rows(paths):
        if rec.get("type") != "expand":
            continue

        ok = 1 if rec.get("ok", False) else 0
        goal = rec.get("goal", "")
        prefix = rec.get("prefix", []) or []
        cand = rec.get("candidate", "") or ""
        depth = int(rec.get("depth", 0) or 0)
        n_sub = rec.get("n_subgoals")
        n_sub = int(n_sub) if n_sub is not None else 9999
        elapsed = float(rec.get("elapsed_ms", 0.0) or 0.0)
        cache_hit = 1 if rec.get("cache_hit") else 0

        # Crude state proxy: last few accepted steps (we don't log full print_state here)
        state_proxy = " ".join(prefix[-3:]) if prefix else ""
        flags = flags_from_goal(goal, state_proxy)

        # Fast one-hot: set a single index if we recognize the prefix
        step_t = step_prefix(cand)
        step_one_hot = [0] * len(STEP_TYPES)
        idx = _STEP_INDEX.get(step_t)
        if idx is not None:
            step_one_hot[idx] = 1

        row = [
            depth, n_sub, elapsed, cache_hit,
            flags["is_listy"], flags["is_natty"], flags["is_sety"], flags["has_q"], flags["is_bool"],
        ] + step_one_hot + [len(cand)]
        X.append(row)
        y.append(ok)

    return X, y


def feature_names() -> List[str]:
    base = ["depth","n_sub","elapsed_ms","cache_hit","is_listy","is_natty","is_sety","has_q","is_bool"]
    return base + [f"step::{t}" for t in STEP_TYPES] + ["len_cmd"]