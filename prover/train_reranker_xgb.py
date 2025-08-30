# prover/train_reranker_xgb.py
import os, json, sys
from pathlib import Path
from typing import Optional, Tuple
from .features import make_dataset, feature_names
from .config import ATTEMPTS_LOG

# Where to save the model
OUT_DIR = Path(os.environ.get("RERANKER_DIR", ".models"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT_DIR / "sk_reranker.joblib"   # keep same path so runtime loader doesn't change
META_PATH  = OUT_DIR / "sk_reranker.meta.json"

def _save_meta(names, extra: Optional[dict] = None):
    meta = {"features": names}
    if extra:
        meta.update(extra)
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

def _pos_neg_counts(y) -> Tuple[int, int]:
    pos = sum(1 for v in y if v == 1)
    neg = len(y) - pos
    return pos, neg

def train(min_rows: Optional[int] = None) -> int:
    # allow env override
    min_rows = int(os.environ.get("RERANKER_MIN_ROWS", "200")) if min_rows is None else int(min_rows)

    try:
        from xgboost import XGBClassifier
    except Exception as e:
        print("xgboost is not installed. Try: pip install xgboost")
        return 2

    from sklearn.metrics import roc_auc_score
    from joblib import dump

    X, y = make_dataset([ATTEMPTS_LOG])
    n = len(X)
    if n < min_rows:
        print(f"Not enough rows ({n}) to train. Run more proofs first.")
        return 1

    names = feature_names()
    pos, neg = _pos_neg_counts(y)
    spw = (neg / max(pos, 1)) if pos > 0 else 1.0  # scale_pos_weight

    _save_meta(
        names,
        extra={"trainer": "xgboost", "rows": n, "pos": pos, "neg": neg, "scale_pos_weight": spw},
    )

    # Strong, CPU-friendly baseline
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        learning_rate=0.05,
        reg_lambda=1.0,
        n_jobs=0,             # 0 = use all cores
        tree_method="hist",
        objective="binary:logistic",
        scale_pos_weight=spw, # handle imbalance
        eval_metric="logloss",
    )
    clf.fit(X, y)

    # quick sanity metric on train (we’ll add holdout later)
    try:
        from numpy import array
        pred = clf.predict_proba(array(X))[:, 1]
        auc = roc_auc_score(y, pred)
        print(f"Train AUC: {auc:.3f} on {n} rows (pos={pos}, neg={neg}, spw={spw:.2f})")
    except Exception:
        pass

    dump(clf, MODEL_PATH)
    print(f"Saved XGBoost reranker → {MODEL_PATH}")
    return 0

if __name__ == "__main__":
    sys.exit(train())
