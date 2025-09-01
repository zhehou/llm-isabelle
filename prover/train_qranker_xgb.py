
# prover/train_qranker_xgb.py
"""
Train an offline RL-flavored reranker that drops in with zero code changes:
- Reads attempts/runs logs (auto-detects common paths).
- Builds a contextual-bandit target from run success.
- Fits XGBRegressor and saves an RLQXGBWrapper to .models/sk_reranker.joblib
  so prover/ranker.py will load it seamlessly.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
from typing import List, Tuple

from joblib import dump
import xgboost as xgb

# Local imports
from . import config
from .rl_dataset import make_bandit_dataset, make_q_targets, feature_names
from .rl_q_wrapper import RLQXGBWrapper

DEFAULT_ATTEMPTS = [config.ATTEMPTS_LOG]
DEFAULT_RUNS = [config.RUNS_LOG]

def _autopaths(paths: List[str], fallback: List[str]) -> List[str]:
    out = []
    for p in (paths or []):
        if p and Path(p).exists():
            out.append(p)
    if out:
        return out
    # Try fallbacks commonly used by earlier configs
    for p in fallback:
        if Path(p).exists():
            out.append(p)
    return out

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts", nargs="*", default=None, help="attempts log paths")
    ap.add_argument("--runs", nargs="*", default=None, help="runs log paths")
    ap.add_argument("--out", default=None, help="output model path (defaults to .models/sk_reranker.joblib)")
    ap.add_argument("--mode", choices=["bandit","q"], default="bandit", help="target type")
    ap.add_argument("--min_rows", type=int, default=200, help="minimum rows to train")
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--n_estimators", type=int, default=300)
    args = ap.parse_args(argv)

    # Resolve input paths
    attempts = _autopaths(args.attempts or DEFAULT_ATTEMPTS, ["log/attempts.log.jsonl", "logs/attempts.log.jsonl"])
    runs = _autopaths(args.runs or DEFAULT_RUNS, ["log/runs.log.jsonl", "logs/runs.log.jsonl"])

    if not attempts:
        print("No attempts logs found. Use --attempts or set config.ATTEMPTS_LOG", file=sys.stderr)
        return 2

    # Build dataset
    if args.mode == "bandit":
        X, y = make_bandit_dataset(attempts, runs)
        target_name = "success(bandit)"
    else:
        X, y = make_q_targets(attempts, runs)
        target_name = "q_target"

    n = len(X)
    pos = sum(y_i > 0.5 for y_i in y)
    neg = n - pos
    if n < args.min_rows:
        print(f"Not enough rows ({n}) to train. (expand rows only; pos={pos}, neg={neg}). Run more proofs.", file=sys.stderr)
        return 1

    # Fit XGB regressor on scalar target in [0,1]
    reg = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.eta,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=4,
        random_state=42,
    )
    reg.fit(X, y)

    # Wrap and save to the SAME filename that ranker.py expects
    out_dir = Path(os.environ.get("RERANKER_DIR", ".models"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (out_dir / "sk_reranker.joblib")
    model = RLQXGBWrapper(reg)
    dump(model, out_path)
    print(f"[ok] Trained RL-XGB reranker on {n} rows (pos={pos}, neg={neg}) [{target_name}]")
    print(f"[ok] Saved to {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
