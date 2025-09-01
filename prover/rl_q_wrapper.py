
# prover/rl_q_wrapper.py
"""
RLQXGBWrapper: wraps an XGBoost regressor that predicts a scalar in [0,1],
but exposes a sklearn-style `predict_proba(X)` returning [[1-p, p], ...],
so it plugs into prover/ranker.py without any code changes.
"""
from __future__ import annotations
from typing import List
import numpy as np

class RLQXGBWrapper:
    def __init__(self, xgb_regressor, clip: bool = True):
        self.reg = xgb_regressor
        self.clip = clip

    def _to_prob(self, arr):
        p = np.asarray(arr, dtype=float)
        if self.clip:
            p = np.clip(p, 0.0, 1.0)
        return p

    def predict_proba(self, X: List[List[float]]):
        p = self._to_prob(self.reg.predict(X))
        # Return 2-class probabilities shape (n,2)
        return np.vstack([1.0 - p, p]).T
