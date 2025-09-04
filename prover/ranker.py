# prover/ranker.py
import os
from pathlib import Path
from typing import List, Optional, Any
from joblib import load

try:
    from .config import RERANKER_DIR as CFG_RERANKER_DIR, RERANKER_OFF as CFG_RERANKER_OFF
except Exception:
    CFG_RERANKER_DIR = Path(os.environ.get("RERANKER_DIR", ".models"))
    CFG_RERANKER_OFF = os.environ.get("RERANKER_OFF", "0").lower() in ("1", "true", "yes", "on")

MODEL_BASENAME = "sk_reranker.joblib"

class SklearnReranker:
    """
    Joblib-saved model wrapper (classifier or regressor-wrapper).

    - Honors RERANKER_OFF (env or config)
    - Supports predict_proba or decision_function
    - Returns 0.5 on any error/unavailable
    """
    def __init__(self):
        self._disabled = CFG_RERANKER_OFF or (os.environ.get("RERANKER_OFF", "0") in ("1","true","True"))
        self.model: Optional[Any] = None
        if not self._disabled:
            self._try_load()

    def _try_load(self):
        p = Path(CFG_RERANKER_DIR) / MODEL_BASENAME
        try:
            if p.exists():
                self.model = load(p)
            else:
                self.model = None
        except Exception:
            self.model = None

    def available(self) -> bool:
        return (not self._disabled) and (self.model is not None)

    def score(self, feat_row: List[float]) -> float:
        if not self.available():
            return 0.5
        try:
            m = self.model
            if hasattr(m, "predict_proba"):
                proba = m.predict_proba([feat_row])
                return float(proba[0][1])
            if hasattr(m, "decision_function"):
                import math
                d = float(m.decision_function([feat_row])[0])
                return 1.0 / (1.0 + math.exp(-d))
        except Exception:
            pass
        return 0.5
