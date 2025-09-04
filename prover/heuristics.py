# prover/heuristics.py

# add at top of prover/heuristics.py
from .features import STEP_TYPES

def live_features_for(cmd: str, goal: str, state_hint: str,
                      depth: int) -> list[float]:
    g = (goal + " " + (state_hint or "")).lower()

    def flag_any(keys): return int(any(k in g for k in keys))
    base = [
        depth,
        9999,          # n_sub unknown at proposal time
        0.0,           # elapsed_ms unknown
        0,             # cache_hit unknown
        flag_any(["@", "append", "rev", "map", "take", "drop"]),
        flag_any(["suc", "nat", "≤", "<", "0", "add", "mult", "+", "-", "*"]),
        flag_any(["∈", "subset", "⋂", "⋃"]),
        flag_any(["∀", "∃"]),
        flag_any(["true", "false", "¬", "not", "∧", "∨"]),
    ]
    step_t = next((t for t in STEP_TYPES if cmd.startswith(t)), None)
    one_hot = [1 if step_t == t else 0 for t in STEP_TYPES]
    return base + one_hot + [len(cmd)]

from typing import List, Optional, Tuple

def suggest_common_lemmas(state_hint: str) -> List[str]:
    txt = state_hint.lower()
    lemmas = []
    if "rev" in txt: lemmas.append("rev_rev_ident")
    if "append" in txt or " @ " in txt: lemmas.append("append_assoc")
    if "length" in txt and ("append" in txt or " @ " in txt): lemmas.append("length_append")
    if "map" in txt and ("append" in txt or " @ " in txt): lemmas.append("map_append")
    out, seen = [], set()
    for l in lemmas:
        if l not in seen:
            seen.add(l); out.append(l)
    return out

def mk_finisher_variants(lemmas: List[str]) -> List[str]:
    return [f"by (simp add: {l})" for l in lemmas] + [f"by (metis {l})" for l in lemmas]

def _heuristic_score(cmd: str, goal: str, state: str, facts: Optional[List[str]] = None) -> float:
    g = (goal + " " + state).lower()
    score = 0.0
    # Prefer unfolding defs matching symbols in the goal/state (small nudge)
    import re
    m = re.search(r"([A-Za-z0-9_']+)_def\b", cmd)
    if m:
        stem = m.group(1).lower()
        if stem and stem in g:
            score -= 0.25
    if any(tok in g for tok in ["rev", "@", "append", "map", "take", "drop"]):
        if "induction" in cmd: score -= 1.2
        if "cases" in cmd: score -= 0.6
        if cmd.startswith("apply simp"): score -= 0.4
        if "metis" in cmd: score -= 0.2
    if any(tok in g for tok in ["suc", "nat", "≤", "<", "+", "-", "*", "dvd"]):
        if "induction" in cmd: score -= 0.8
        if "linarith" in cmd: score -= 0.7
        if "arith" in cmd: score -= 0.5
        if "auto" in cmd: score -= 0.3
    if any(tok in g for tok in ["∈", "subset", "⋂", "⋃", "∀", "∃"]):
        if "blast" in cmd: score -= 0.7
        if "auto" in cmd: score -= 0.4
    if cmd.startswith("apply simp"): score -= 0.2
    if cmd.startswith("apply auto"): score -= 0.15
    if "metis" in cmd: score += 0.1
    if facts:
        for f in facts[:6]:
            if f in cmd:
                score -= 0.35
                break
    return score

def rank_candidates(cands: List[str], goal: str, state_hint: str,
                    facts: Optional[List[str]] = None,
                    reranker=None, depth:int=0) -> List[str]:
    scored = []
    for i, c in enumerate(cands):
        s = _heuristic_score(c, goal, state_hint, facts)
        # gentle length penalty to prefer shorter, cheaper steps
        s += 0.003 * max(0, len(c) - 24)
        # blend reranker score (higher prob = better; our sort is ascending)
        if reranker and getattr(reranker, "available", lambda: False)():
            feats = live_features_for(c, goal, state_hint, depth)
            try:
                p = reranker.score(feats)  # 0..1
                s += -0.5 * p              # subtract to prefer higher p
            except Exception:
                pass
        scored.append((s, len(c), i, c))
    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    return [c for *_, c in scored]

def augment_with_facts_for_steps(cands: List[str], facts: List[str]) -> List[str]:
    if not facts: return cands
    # Prefer *_def early
    defs = [f for f in facts if f.endswith("_def")]
    aug = []
    for c in cands:
        aug.append(c)
        if c.startswith("apply simp"):
            for f in (defs[:3] or facts[:3]):
                aug.append(f"apply (simp add: {f})")
                aug.append(f"apply (simp only: {f})")
        elif c.startswith("apply auto"):
            for f in (defs[:3] or facts[:3]):
                aug.append(f"apply (auto simp: {f})")
        elif c.startswith("apply (simp") and "add:" not in c:
            pref = (defs[0] if defs else facts[0])
            aug.append(c.replace("(simp", f"(simp add: {pref}", 1))
        elif c.startswith("apply (auto") and "simp:" not in c:
            pref = (defs[0] if defs else facts[0])
            aug.append(c.replace("(auto", f"(auto simp: {pref}", 1))
        elif "metis" in c and "(" in c and ")" in c and facts:
            aug.append(c[:-1] + f" {facts[0]})")
    seen, dedup = set(), []
    for x in aug:
        if x not in seen:
            seen.add(x); dedup.append(x)
    return dedup[:max(len(cands), 10)]

def augment_with_facts_for_finishers(base_finishers: List[str], facts: List[str], cap: int = 8) -> List[str]:
    if not facts: return base_finishers
    defs = [f for f in facts if f.endswith("_def")]
    pri = (defs[:6] or facts[:6])  # prefer defs when present
    extras = []
    for f in pri:
        extras.append(f"by (simp add: {f})")
        extras.append(f"by (simp only: {f})")
        extras.append(f"by (auto simp add: {f})")
        extras.append(f"by (metis {f})")
    seen, out = set(), []
    for x in (extras + base_finishers):
        if x not in seen:
            seen.add(x); out.append(x)
        if len(out) >= cap: break
    return out
