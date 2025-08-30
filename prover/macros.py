# prover/macros.py
"""
Two-step macro miner.

Reads successful proofs from RUNS_LOG and extracts frequent continuations:
  head_step -> [(next_step, count), ...]

Runtime knobs (env or params):
  MACRO_MIN_COUNT   : keep only pairs seen this many times (default 3)
  MACRO_MAX_PER_HEAD: cap continuations per head (default 5)
"""
import os, json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Iterable, Optional
from .config import RUNS_LOG

# ---- Env-configurable defaults ----
MACRO_MIN_COUNT: int = int(os.environ.get("MACRO_MIN_COUNT", "3"))
MACRO_MAX_PER_HEAD: int = int(os.environ.get("MACRO_MAX_PER_HEAD", "5"))

ContinuationMap = Dict[str, List[Tuple[str, int]]]

def _iter_success_runs(paths: Iterable[str]) -> Iterable[List[str]]:
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    if rec.get("success") and isinstance(rec.get("final_steps"), list):
                        yield rec["final_steps"]
                    elif rec.get("success") and isinstance(rec.get("steps"), list):
                        yield rec["steps"]
        except FileNotFoundError:
            continue

def mine_two_step_macros(
    paths: Optional[List[str]] = None,
    min_count: Optional[int] = None,
    max_per_head: Optional[int] = None,
) -> ContinuationMap:
    """
    Build a mapping: first_step -> list of (second_step, count), sorted by freq.

    Args:
      paths         : runs.jsonl paths (defaults to [RUNS_LOG])
      min_count     : minimum frequency to keep a pair (env MACRO_MIN_COUNT if None)
      max_per_head  : cap continuations per head (env MACRO_MAX_PER_HEAD if None)

    Returns:
      Dict[str, List[Tuple[str, int]]]
    """
    paths = paths or [RUNS_LOG]
    min_count = MACRO_MIN_COUNT if min_count is None else int(min_count)
    max_per_head = MACRO_MAX_PER_HEAD if max_per_head is None else int(max_per_head)

    pair_freq: Dict[Tuple[str, str], int] = Counter()
    for steps in _iter_success_runs(paths):
        # Ignore lemma line (index 0) and finisher (last)
        if not steps or len(steps) < 3:
            continue
        mid = [s for s in steps[1:-1] if s.strip().startswith("apply")]
        for a, b in zip(mid, mid[1:]):
            pair_freq[(a, b)] += 1

    conts: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for (a, b), c in pair_freq.items():
        if c >= min_count:
            conts[a].append((b, c))

    # sort and truncate per head
    for a in list(conts.keys()):
        conts[a].sort(key=lambda t: (-t[1], len(t[0]), t[0]))
        conts[a] = conts[a][:max_per_head]
    return dict(conts)

def suggest_continuations(head: str, cont_map: ContinuationMap, k: int = 2) -> List[str]:
    lst = cont_map.get(head, [])
    return [b for (b, _) in lst[:k]]
