# prover/train_reranker_sklearn.py
import os, json, sys
from pathlib import Path
from typing import Optional
from .features import make_dataset, feature_names
from .config import ATTEMPTS_LOG

# Where to save the model
OUT_DIR = Path(os.environ.get("RERANKER_DIR", ".models"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT_DIR / "sk_reranker.joblib"
META_PATH  = OUT_DIR / "sk_reranker.meta.json"

def _save_meta(names, extra: Optional[dict] = None):
    meta = {"features": names}
    if extra:
        meta.update(extra)
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

def train(min_rows: Optional[int] = None) -> int:
    # allow env override
    min_rows = int(os.environ.get("RERANKER_MIN_ROWS", "200")) if min_rows is None else int(min_rows)

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from joblib import dump

    X, y = make_dataset([ATTEMPTS_LOG])
    n = len(X)
    if n < min_rows:
        print(f"Not enough rows ({n}) to train. Run more proofs first.")
        return 1

    names = feature_names()

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  # robust for sparse-ish numeric features
        ("clf", LogisticRegression(
            max_iter=400,
            class_weight="balanced",   # handle imbalance
            solver="lbfgs",
            n_jobs=None
        )),
    ])
    pipe.fit(X, y)

    try:
        import numpy as np
        pred = pipe.predict_proba(np.array(X))[:, 1]
        auc = roc_auc_score(y, pred)
        print(f"Train AUC: {auc:.3f} on {n} rows")
    except Exception:
        pass

    dump(pipe, MODEL_PATH)
    _save_meta(names, extra={"trainer": "sklearn/logreg", "rows": n})
    print(f"Saved sklearn reranker → {MODEL_PATH}")
    return 0

if __name__ == "__main__":
    sys.exit(train())
