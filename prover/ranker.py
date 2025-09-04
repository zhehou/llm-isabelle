# prover/ranker.py
import os, glob, math
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
TORCH_LATEST = "latest.pt"

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
        self.torch_model: Optional[Any] = None    # TorchScript .pt/.pth
        if not self._disabled:
            self._try_load()

    def _try_load(self):
        dirp = Path(CFG_RERANKER_DIR)
        # Prefer TorchScript (.pt/.pth) if present
        pt_candidates = []
        p_latest_pt = dirp / TORCH_LATEST
        if p_latest_pt.exists():
            pt_candidates.append(p_latest_pt)
        try:
            pts = sorted(
                [Path(p) for pat in ("*.pt","*.pth") for p in glob.glob(str(dirp / pat))],
                key=lambda p: p.stat().st_mtime, reverse=True
            )
            pt_candidates.extend(pts)
        except Exception:
            pass
        pt_candidates += [dirp / "reranker.pt", dirp / "reranker.pth"]

        for p in pt_candidates:
            try:
                import torch  # lazy import
                if p.exists():
                    m = torch.jit.load(str(p), map_location="cpu")
                    m.eval()
                    self.torch_model = m
                    return
            except Exception:
                continue

        # Fallback to joblib models
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
        return (not self._disabled) and (self.torch_model is not None or self.model is not None)

    def score(self, feat_row: List[float]) -> float:
        if not self.available():
            return 0.5
        try:
            if self.torch_model is not None:
                import torch
                with torch.no_grad():
                    x = torch.tensor([feat_row], dtype=torch.float32)
                    p = self.torch_model(x)  # expected shape [N,1] or [N]
                    p = p.view(-1)[0].item()
                    if not (p == p):  # NaN guard
                        return 0.5
                    return max(0.0, min(1.0, float(p)))            
            m = self.model
            if hasattr(m, "predict_proba"):
                proba = m.predict_proba([feat_row])
                return float(proba[0][1])
            if hasattr(m, "decision_function"):                
                d = float(m.decision_function([feat_row])[0])
                return 1.0 / (1.0 + math.exp(-d))
        except Exception:
            pass
        return 0.5
