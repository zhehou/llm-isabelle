# prover/ranker.py
import os, glob
from pathlib import Path
from typing import List, Optional, Any
from joblib import load

try:
    from .config import RERANKER_DIR as CFG_RERANKER_DIR, RERANKER_OFF as CFG_RERANKER_OFF
except Exception:
    CFG_RERANKER_DIR = Path(os.environ.get("RERANKER_DIR", ".models"))
    CFG_RERANKER_OFF = os.environ.get("RERANKER_OFF", "0").lower() in ("1", "true", "yes", "on")

MODEL_BASENAME = "reranker.joblib"
LATEST_POINTER = "latest.joblib"

class Reranker:
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
        dirp = Path(CFG_RERANKER_DIR)
        candidates = []
        # 1) explicit "latest" pointer (file/symlink/copy)
        p_latest = dirp / LATEST_POINTER
        if p_latest.exists():
            candidates.append(p_latest)
        # 2) newest *.joblib artifact in directory
        try:
            joblibs = sorted((Path(p) for p in glob.glob(str(dirp / "*.joblib"))),
                             key=lambda p: p.stat().st_mtime, reverse=True)
            candidates.extend(joblibs)
        except Exception:
            pass
        # 3) stable basenames for back-compat (new then legacy)
        candidates.append(dirp / MODEL_BASENAME)          # "reranker.joblib"
        candidates.append(dirp / "sk_reranker.joblib")    # legacy name

        self.model = None
        for p in candidates:
            try:
                if p.exists():
                    self.model = load(p)
                    break
            except Exception:
                continue

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
