# prover/ranker.py
import os
from pathlib import Path
from typing import List, Optional, Any
from joblib import load

RERANKER_DIR = Path(os.environ.get("RERANKER_DIR", ".models"))
MODEL_PATH = RERANKER_DIR / "sk_reranker.joblib"

class SklearnReranker:
    """
    Thin wrapper around a joblib-saved classifier.

    Behavior:
      - If env RERANKER_OFF=1, acts as unavailable (no reranking).
      - Supports models with either predict_proba or decision_function.
      - Returns neutral score 0.5 on any error.
    """
    def __init__(self):
        self._disabled = os.environ.get("RERANKER_OFF", "0") in ("1", "true", "True")
        self.model: Optional[Any] = None
        if not self._disabled:
            try:
                if MODEL_PATH.exists():
                    self.model = load(MODEL_PATH)
            except Exception:
                self.model = None

    def available(self) -> bool:
        return (not self._disabled) and (self.model is not None)

    def score(self, feat_row: List[float]) -> float:
        """Return P(success) in [0,1]. If unavailable, return neutral 0.5."""
        if not self.available():
            return 0.5
        try:
            m = self.model
            # predict_proba path
            if hasattr(m, "predict_proba"):
                proba = m.predict_proba([feat_row])
                return float(proba[0][1])
            # decision_function path -> squash to 0..1
            if hasattr(m, "decision_function"):
                import math
                d = float(m.decision_function([feat_row])[0])
                return 1.0 / (1.0 + math.exp(-d))
        except Exception:
            pass
        return 0.5
