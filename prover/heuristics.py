# prover/heuristics.py
from __future__ import annotations

import os
import re
from typing import List, Optional, Dict, Tuple

from .features import STEP_TYPES
from .premises import cand_features

# Reranker controls (kept identical)
_RR_W = float(os.environ.get("RERANKER_WEIGHT", "0.5"))
_SAFE_TOPM = int(os.environ.get("RERANKER_SAFE_TOPM", "0"))  # 0 = no gating

# Token sets used by live features / heuristics (same strings as before)
_LISTY = ("@", "append", "rev", "map", "take", "drop")
_NATTY = ("suc", "nat", "≤", "<", "0", "add", "mult", "+", "-", "*")
_SETY  = ("∈", "subset", "⋂", "⋃")
_QTOK  = ("∀", "∃")
_BOOLY = ("true", "false", "¬", "not", "∧", "∨")

# Precompiled once for tiny speed bump
_DEF_RE = re.compile(r"([A-Za-z0-9_']+)_def\b")

def _any_in(s: str, toks: tuple[str, ...]) -> bool:
    for t in toks:
        if t in s:
            return True
    return False

def _step_index(cmd: str) -> Optional[int]:
    s = (cmd or "").strip()
    for i, t in enumerate(STEP_TYPES):
        if s.startswith(t):
            return i
    return None

def live_features_for(cmd: str, goal: str, state_hint: str, depth: int) -> list[float]:
    g = (goal + " " + (state_hint or "")).lower()
    base = [
        depth,
        9999,   # n_sub unknown at proposal time
        0.0,    # elapsed_ms unknown
        0,      # cache_hit unknown
        int(_any_in(g, _LISTY)),
        int(_any_in(g, _NATTY)),
        int(_any_in(g, _SETY)),
        int(_any_in(g, _QTOK)),
        int(_any_in(g, _BOOLY)),
    ]
    idx = _step_index(cmd)
    one_hot = [0] * len(STEP_TYPES)
    if idx is not None:
        one_hot[idx] = 1
    return base + one_hot + [len(cmd or "")]

from typing import Optional  # re-affirm for older type checkers

def suggest_common_lemmas(state_hint: str) -> List[str]:
    txt = (state_hint or "").lower()
    lemmas: List[str] = []
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

# NEW: conservative extractor for lemma names in a single command
# include dots so tokens like List.append_assoc remain intact
_TOK = re.compile(r"[A-Za-z0-9_'.]+")
def extract_candidate_facts(cmd: str) -> List[str]:
    """
    Pull likely lemma names after markers: add:, simp:, only:, metis, rule, intro, elim, erule, subst.
    Returns unique order-preserving list.
    """
    s = cmd or ""
    spans = []
    for key in ("add:", "simp:", "only:", "metis", "rule", "intro", "elim", "erule", "subst"):
        for m in re.finditer(rf"{re.escape(key)}\s+", s):
            spans.append(m.end())
    names: List[str] = []
    seen = set()
    for st in spans:
        chunk = s[st:]
        # stop at closing paren or end-of-string
        stop = chunk.find(")")
        if stop >= 0:
            chunk = chunk[:stop]
        toks = _TOK.findall(chunk)
        for t in toks:
            if t not in seen:
                seen.add(t); names.append(t)
    return names[:24]

def _heuristic_score(cmd: str, goal: str, state: str, facts: Optional[List[str]] = None) -> float:
    g = (goal + " " + state).lower()
    score = 0.0
    m = _DEF_RE.search(cmd)
    if m:
        stem = m.group(1).lower()
        if stem and stem in g:
            score -= 0.25
    if _any_in(g, _LISTY):
        if "induction" in cmd: score -= 1.2
        if "cases" in cmd: score -= 0.6
        if cmd.startswith("apply simp"): score -= 0.4
        if "metis" in cmd: score -= 0.2
    if _any_in(g, ("suc", "nat", "≤", "<", "+", "-", "*", "dvd")):
        if "induction" in cmd: score -= 0.8
        if "linarith" in cmd: score -= 0.7
        if "arith" in cmd: score -= 0.5
        if "auto" in cmd: score -= 0.3
    if _any_in(g, _SETY + _QTOK):
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

def rank_candidates(
    cands: List[str],
    goal: str,
    state_hint: str,
    facts: Optional[List[str]] = None,
    reranker=None,
    depth: int = 0,
    extra_tail: Optional[List[float]] = None,
    premise_scores: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[str]:
    # 1) Heuristic-only base (keep gentle length penalty)
    base = []
    for i, c in enumerate(cands):
        s = _heuristic_score(c, goal, state_hint, facts)
        s += 0.003 * max(0, len(c) - 24)
        base.append((s, len(c), i, c))

    # 2) No ML → return heuristic order
    if not (reranker and getattr(reranker, "available", lambda: False)()):
        base.sort(key=lambda t: (t[0], t[1], t[2]))
        return [c for *_, c in base]

    # 3) Safe top-M gating: ML rescoring only for top M by heuristic score
    base.sort(key=lambda t: (t[0], t[1], t[2]))
    M = _SAFE_TOPM if _SAFE_TOPM > 0 else len(base)
    rescored = []
    for rank, (s, L, i, c) in enumerate(base):
        s_adj = s
        if rank < M:
            try:
                feats = live_features_for(c, goal, state_hint, depth)
                # If the model expects a longer vector, append tail (pool + per-candidate).
                # Robustly obtain expected dimension (method or attr).
                exp = None
                try:
                    ed = getattr(reranker, "expected_dim", None)
                    exp = ed() if callable(ed) else ed
                except Exception:
                    exp = None
                cand_tail_vals = []
                if premise_scores:
                    cf = cand_features(extract_candidate_facts(c), premise_scores)
                    cand_tail_vals = [
                        cf.get("cand_cos_mean", 0.0),
                        cf.get("cand_cos_max", 0.0),
                        cf.get("cand_rerank_mean", 0.0),
                        cf.get("cand_hit_topk", 0.0),
                        cf.get("cand_n_facts", 0.0),
                    ]
                if isinstance(exp, int) and exp > 0:
                    full_tail = (extra_tail or []) + cand_tail_vals
                    if len(feats) < exp:
                        need = exp - len(feats)
                        tail = full_tail[:need] + [0.0] * max(0, need - len(full_tail))
                        feats = list(feats) + tail
                    elif len(feats) > exp:
                        feats = list(feats[:exp])                
                p = float(reranker.score(feats))  # 0..1
                s_adj += -_RR_W * p               # subtract so higher p ranks earlier
            except Exception:
                pass
        rescored.append((s_adj, L, i, c))

    rescored.sort(key=lambda t: (t[0], t[1], t[2]))
    return [c for *_, c in rescored]

def augment_with_facts_for_steps(cands: List[str], facts: List[str]) -> List[str]:
    if not facts: return cands
    defs = [f for f in facts if f.endswith("_def")]
    aug: List[str] = []
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
    extras: List[str] = []
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