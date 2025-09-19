# =============================================
# File: prover/premises.py
# Purpose: Lightweight two-stage premise retrieval with optional re-ranking.
#          No mandatory heavy deps; uses scikit-learn if present, else falls
#          back to a token-overlap scorer. Designed to feed prover candidates
#          and produce extra reranker features without changing public APIs.
# =============================================
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math
import threading
import re

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    _SK_OK = True
except Exception:
    _SK_OK = False

_TOKEN = re.compile(r"[A-Za-z0-9_'.]+")  # include dots to keep Theory.lemma intact


def _tokenize(s: str) -> List[str]:
    return _TOKEN.findall((s or "").lower())


@dataclass
class _Item:
    text: str
    meta: Dict


class PremisesIndex:
    """In-memory text index for premise selection.

    SELECT: TF-IDF cosine (if sklearn available) or token Jaccard overlap.
    RE-RANK: optional shallow rescoring (here: same as SELECT unless a
             custom scorer is provided via constructor hook).
    Thread-safe for `select` by read-only access after `finalize()`.
    """
    def __init__(self, select_model: Optional[object] = None,
                 rerank_model: Optional[object] = None,
                 store: Optional[Dict[str, Dict]] = None):
        self._items: Dict[str, _Item] = {}
        self._vec = None
        self._mat = None
        self._ids: List[str] = []
        self._lock = threading.RLock()
        self._store = store  # optional external KV for texts/meta
        self._select_model = select_model  # reserved for custom encoders
        self._rerank_model = rerank_model  # reserved for cross-encoders

    # ---- building ----
    def add(self, fact_id: str, text: str, meta: Dict | None = None) -> None:
        with self._lock:
            self._items[str(fact_id)] = _Item(text=text or "", meta=meta or {})

    def add_many(self, pairs: List[Tuple[str, str]]) -> None:
        for fid, txt in pairs:
            self.add(fid, txt, None)

    def finalize(self) -> None:
        with self._lock:
            self._ids = list(self._items.keys())
            corpus = [self._items[i].text for i in self._ids]
            if _SK_OK and len(corpus) > 0:
                self._vec = TfidfVectorizer(min_df=1, max_features=200000)
                self._mat = self._vec.fit_transform(corpus)
            else:
                self._vec = None
                self._mat = None

    # ---- selection ----
    def _select_scores(self, goal_text: str) -> List[Tuple[str, float]]:
        if self._vec is not None and self._mat is not None and _SK_OK:
            q = self._vec.transform([goal_text])
            sim = cosine_similarity(q, self._mat).ravel()
            return list(zip(self._ids, sim.tolist()))
        # fallback: Jaccard over tokens
        gtok = set(_tokenize(goal_text))
        out: List[Tuple[str, float]] = []
        for i in self._ids:
            itok = set(_tokenize(self._items[i].text))
            inter = len(gtok & itok)
            union = max(1, len(gtok | itok))
            out.append((i, inter / union))
        return out

    def select(self, goal_text: str, *, k_select: int = 512, k_rerank: int = 64,
               boost_ids: Optional[List[str]] = None) -> List[Tuple[str, float, float]]:
        """Return [(fact_id, select_score, rerank_score)].
        `boost_ids` are given a mild multiplicative boost to prefer local context.
        """
        with self._lock:
            scores = self._select_scores(goal_text)
        if not scores:
            return []
        boost = set(boost_ids or [])
        boosted = []
        for fid, s in scores:
            if fid in boost:
                s *= 1.15  # mild, safe boost
            boosted.append((fid, s))
        boosted.sort(key=lambda x: x[1], reverse=True)
        top = boosted[:max(1, k_select)]

        # Optional re-rank: apply a stronger model if provided; else mirror select score
        if self._rerank_model is None:
            reranked = [(fid, s, s) for fid, s in top[:k_rerank]]
            return reranked

        # Placeholder hook: custom cross-encoder scorer(goal, fact_text) -> float
        def _score_pair(g: str, fid: str) -> float:
            try:
                return float(self._rerank_model.score_pair(g, self._items[fid].text))
            except Exception:
                return 0.0

        reranked = [(fid, s, _score_pair(goal_text, fid)) for fid, s in top[:k_rerank]]
        reranked.sort(key=lambda x: x[2], reverse=True)
        return reranked

    # ---- convenience for prover integration ----
    def texts_for(self, ids: List[str]) -> List[str]:
        return [self._items[i].text for i in ids if i in self._items]


# --------- Tiny utility for turning selected premises into features ---------

def selection_features(picks: List[Tuple[str, float, float]], k: int = 16) -> Dict[str, float]:
    """Compute cheap numeric features to append to the reranker feature row.
    Safe defaults when `picks` is empty.
    """
    if not picks:
        return {
            "premise_cosine_top1": 0.0,
            "premise_cosine_topk_mean": 0.0,
            "premise_rerank_top1": 0.0,
            "premise_rerank_topk_mean": 0.0,
            "n_premises": 0.0,
        }
    k = max(1, min(k, len(picks)))
    sel = [p[1] for p in picks[:k]]
    rer = [p[2] for p in picks[:k]]
    return {
        "premise_cosine_top1": float(sel[0]),
        "premise_cosine_topk_mean": float(sum(sel) / k),
        "premise_rerank_top1": float(rer[0]),
        "premise_rerank_topk_mean": float(sum(rer) / k),
        "n_premises": float(len(picks)),
    }

# --------- NEW: per-candidate aggregation helpers ----------
def build_score_map(picks: List[Tuple[str, float, float]]) -> Dict[str, Tuple[float, float]]:
    """
    Map lemma-name -> (select_score, rerank_score).
    Input fact_id format expected like 'File.thy:lemma_name:i' (fallback: whole id).
    """
    out: Dict[str, Tuple[float, float]] = {}
    for fid, s, r in picks:
        parts = str(fid).split(":")
        file = parts[0] if len(parts) >= 1 else ""
        name = parts[1] if len(parts) >= 2 else str(fid)
        sel, rer = float(s), float(r)
        # base name
        out[name] = (sel, rer)
        # alias: Theory.lemma from File.thy
        if file.endswith(".thy"):
            theory = file[:-4]
            out[f"{theory}.{name}"] = (sel, rer)
        # alias: full fid key as a last resort
        out[str(fid)] = (sel, rer)
    return out

def cand_features(fact_names: List[str], score_map: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """
    Aggregate scores for the facts *referenced by a single candidate*.
    Returns zeros if names are empty or not in the map.
    """
    f = [n for n in (fact_names or []) if n in score_map]
    if not f:
        return {
            "cand_cos_mean": 0.0,
            "cand_cos_max": 0.0,
            "cand_rerank_mean": 0.0,
            "cand_hit_topk": 0.0,
            "cand_n_facts": float(len(fact_names or [])),
        }
    sels = [score_map[n][0] for n in f]    # select_score
    rers = [score_map[n][1]   for n in f]  # rerank_score at index 1
    hit = float(len(f)) / float(max(1, len(fact_names)))
    return {
        "cand_cos_mean": float(sum(sels) / len(sels)),
        "cand_cos_max": float(max(sels)),
        "cand_rerank_mean": float(sum(rers) / len(rers)),
        "cand_hit_topk": hit,
        "cand_n_facts": float(len(fact_names)),
    }