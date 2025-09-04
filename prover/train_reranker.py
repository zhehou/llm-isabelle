# prover/train_reranker.py
"""
Unified reranker trainer.

Targets
-------
  bandit : contextual bandit labels from expand attempts (ok ∨ run-success)
  q      : smoothed Q-like targets in [0,1] using run-level success with discount

Algorithms
----------
  xgb-classifier  : XGBoost binary classifier (predict_proba)
  sklearn-logreg  : LogisticRegression (+StandardScaler with_mean=False)
  xgb-regressor   : XGBoost regressor wrapped to expose predict_proba

Output
------
  Saves to {RERANKER_DIR}/sk_reranker.joblib (what ranker.py expects).

Usage
-----
  python -m prover.train_reranker --algo xgb-classifier --target bandit
  python -m prover.train_reranker --algo sklearn-logreg --target bandit
  python -m prover.train_reranker --algo xgb-regressor --target q
"""
from __future__ import annotations
import argparse, os, sys, json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple, Dict, Any, Optional

from joblib import dump

from . import config
from .features import STEP_TYPES, feature_names  # schema only

# ---------- local dataset builders (merged former rl_dataset.py) ----------
def _iter_jsonl(paths: Iterable[Path]) -> Iterator[dict]:
    for p in paths:
        if not p.exists(): continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict): yield rec
                except Exception:
                    continue

def _flags_from(goal: str, state_proxy: str) -> Dict[str, int]:
    g = (goal + " " + state_proxy).lower()
    return {
        "is_listy": int(any(k in g for k in ["@", "append", "rev", "map", "take", "drop"])),
        "is_natty": int(any(k in g for k in ["suc", "nat", "≤", "<", "0", "add", "mult", "+", "-", "*"])),
        "is_sety":  int(any(k in g for k in ["∈", "subset", "⋂", "⋃"])),
        "has_q":    int(any(k in g for k in ["∀", "∃"])),
        "is_bool":  int(any(k in g for k in ["true", "false", "¬", "not", "∧", "∨"])),
    }

def _step_prefix(cmd: str) -> str:
    s = (cmd or "").strip()
    for t in STEP_TYPES:
        if s.startswith(t): return t
    if s.startswith("by "): return "by"
    return s.split(" ", 1)[0][:32]

def _row_from_attempt(rec: dict) -> List[float]:
    goal = rec.get("goal","") or ""
    prefix = rec.get("prefix", []) or []
    cand = rec.get("candidate","") or ""
    depth = int(rec.get("depth", 0) or 0)
    n_sub = rec.get("n_subgoals"); n_sub = int(n_sub) if n_sub is not None else 9999
    elapsed = float(rec.get("elapsed_ms", 0.0) or 0.0)
    cache_hit = 1 if rec.get("cache_hit") else 0
    state_proxy = " ".join(prefix[-3:])
    flags = _flags_from(goal, state_proxy)
    step_t = _step_prefix(cand)
    step_one_hot = [1 if step_t == t else 0 for t in STEP_TYPES]
    return [depth, n_sub, elapsed, cache_hit,
            flags["is_listy"], flags["is_natty"], flags["is_sety"], flags["has_q"], flags["is_bool"]] \
           + step_one_hot + [len(cand)]

def _runs_success(paths: Iterable[Path]) -> Dict[str, int]:
    succ = {}
    for rec in _iter_jsonl(paths):
        rid = rec.get("run_id")
        if not rid: continue
        succ[rid] = 1 if rec.get("success") else 0
    return succ

def _make_bandit(attempts: List[str], runs: Optional[List[str]]) -> Tuple[List[List[float]], List[int]]:
    X: List[List[float]] = []; y: List[int] = []
    succ = _runs_success([Path(p) for p in (runs or [])]) if runs else {}
    for rec in _iter_jsonl([Path(p) for p in attempts]):
        if rec.get("type") != "expand": continue
        rid = rec.get("run_id")
        label = succ.get(rid, 1 if rec.get("ok") else 0)
        X.append(_row_from_attempt(rec)); y.append(int(label))
    return X, y

def _make_q(attempts: List[str], runs: Optional[List[str]], discount: float = 0.8) -> Tuple[List[List[float]], List[float]]:
    X: List[List[float]] = []; q: List[float] = []
    succ = _runs_success([Path(p) for p in (runs or [])]) if runs else {}
    for rec in _iter_jsonl([Path(p) for p in attempts]):
        if rec.get("type") != "expand": continue
        rid = rec.get("run_id")
        success = succ.get(rid, 1 if rec.get("ok") else 0)
        X.append(_row_from_attempt(rec)); q.append(float(discount) * float(success))
    return X, q

# ---------- wrapper for regressor so ranker can call predict_proba ----------
class _RLQXGBWrapper:
    def __init__(self, reg):
        self.reg = reg
    def predict_proba(self, X):
        import numpy as np
        y = self.reg.predict(X)
        # clip into [0,1], provide as [:,1]
        y = np.clip(y, 0.0, 1.0)
        return np.stack([1.0 - y, y], axis=1)

# ---------- main ----------
def _autopaths(paths: List[str] | None, fallback: List[str]) -> List[str]:
    out: List[str] = []
    for p in (paths or []):
        if p and Path(p).exists():
            out.append(p)
    if out: return out
    for p in fallback:
        if Path(p).exists(): out.append(p)
    return out

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Unified reranker trainer")
    ap.add_argument("--attempts", nargs="*", default=None, help="attempts log paths")
    ap.add_argument("--runs", nargs="*", default=None, help="runs log paths")
    ap.add_argument("--out", default=None, help="output model path (default: timestamped under RERANKER_DIR)")
    ap.add_argument("--target", choices=["bandit","q"], default="bandit")
    ap.add_argument("--algo", choices=["xgb-classifier","sklearn-logreg","xgb-regressor"], default="xgb-classifier")
    ap.add_argument("--min_rows", type=int, default=200)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--n_estimators", type=int, default=300)
    args = ap.parse_args(argv)

    attempts = _autopaths(args.attempts or [config.ATTEMPTS_LOG], ["log/attempts.log.jsonl", "logs/attempts.log.jsonl"])
    runs     = _autopaths(args.runs or [config.RUNS_LOG],       ["log/runs.log.jsonl",     "logs/runs.log.jsonl"])

    if not attempts:
        print("No attempts logs found. Use --attempts or set config.ATTEMPTS_LOG", file=sys.stderr)
        return 2

    if args.target == "bandit":
        X, y = _make_bandit(attempts, runs)
    else:
        X, y = _make_q(attempts, runs)

    n = len(X)
    if n < args.min_rows:
        pos = sum(1 for v in y if (v if args.target == "q" else v >= 1))
        print(f"Not enough rows ({n}) to train. Run more proofs first. (pos~{pos})", file=sys.stderr)
        return 1

    out_dir = Path(os.environ.get("RERANKER_DIR", str(config.RERANKER_DIR))); out_dir.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_path = Path(args.out)
    else:
        import time
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_path = out_dir / f"reranker-{stamp}-{args.algo}-{args.target}.joblib"

    if args.algo == "xgb-classifier":
        try:
            from xgboost import XGBClassifier
        except Exception:
            print("xgboost not installed. pip install xgboost", file=sys.stderr); return 2
        pos = sum(1 for v in y if v >= 1); spw = ((n - pos) / max(pos, 1)) if pos > 0 else 1.0
        clf = XGBClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.eta,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, n_jobs=0, tree_method="hist",
            objective="binary:logistic", scale_pos_weight=spw, eval_metric="logloss", random_state=42,
        )
        clf.fit(X, y); dump(clf, out_path); print(f"[ok] XGBClassifier on {n} rows → {out_path}")

    elif args.algo == "sklearn-logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        pipe = Pipeline([("scaler", StandardScaler(with_mean=False)),
                         ("clf", LogisticRegression(max_iter=400, class_weight="balanced", solver="lbfgs"))])
        pipe.fit(X, y); dump(pipe, out_path); print(f"[ok] sklearn-LogReg on {n} rows → {out_path}")

    else:  # xgb-regressor
        try:
            from xgboost import XGBRegressor
        except Exception:
            print("xgboost not installed. pip install xgboost", file=sys.stderr); return 2
        reg = XGBRegressor(
            n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.eta,
            subsample=0.9, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0,
            objective="reg:squarederror", n_jobs=4, random_state=42,
        )
        reg.fit(X, y); dump(_RLQXGBWrapper(reg), out_path); print(f"[ok] RL-XGB regressor on {n} rows → {out_path}")
    
    # --- Update 'latest.joblib' pointer in out_dir ---
    latest = out_dir / "latest.joblib"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        # Prefer a relative symlink (same dir) on POSIX
        try:
            latest.symlink_to(out_path.name)
        except Exception:
            # Fall back to a plain file copy (Windows/no symlink perms)
            import shutil
            shutil.copyfile(out_path, latest)
    except Exception:
        pass
    # Optional manifest for auditing
    try:
        (out_dir / "latest.json").write_text(
            json.dumps(
                {"path": out_path.name, "algo": args.algo, "target": args.target, "rows": n},
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
