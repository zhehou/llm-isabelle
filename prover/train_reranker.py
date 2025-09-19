# prover/train_reranker.py
"""
Unified reranker trainer (classic + DeepRL) with safe teacher sandbox.

Targets
-------
  bandit : contextual bandit labels from expand attempts (ok ∨ run-success)
  q      : smoothed Q-like targets in [0,1] using run-level success with discount

Algorithms
----------
  xgb-classifier  : XGBoost binary classifier (predict_proba)   -> .joblib
  sklearn-logreg  : LogisticRegression (+StandardScaler)        -> .joblib
  xgb-regressor   : XGBoost regressor (wrapped as proba)        -> .joblib
  awr             : Offline Advantage-Weighted Regression       -> TorchScript .pt
  dqn             : Offline fitted-Q learning (DQN-style)       -> TorchScript .pt

Output locations (compatible with ranker.Reranker)
--------------------------------------------------
  Joblib models → {RERANKER_DIR}/*.joblib  + pointer {RERANKER_DIR}/latest.joblib
  TorchScript   → {RERANKER_DIR}/*.pt      + pointer {RERANKER_DIR}/latest.pt

Usage
-----
  # classic
  python -m prover.train_reranker --algo xgb-classifier --target bandit
  python -m prover.train_reranker --algo sklearn-logreg --target bandit
  python -m prover.train_reranker --algo xgb-regressor --target q

  # DeepRL (AWR with safe teacher; 'auto' is sandboxed if XGBoost)
  python -m prover.train_reranker --algo awr --epochs 8 --batch 1024 --tau 0.6 --teacher auto
  python -m prover.train_reranker --algo dqn --epochs 12 --batch 2048 --gamma 0.92 --target_update 500
"""
from __future__ import annotations
import argparse, os, sys, json, time, math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple, Dict, Any, Optional, DefaultDict
from collections import defaultdict

from joblib import dump

from . import config
from .features import STEP_TYPES, feature_names  # schema only


# -----------------------
# Small path/pointer helpers
# -----------------------
def _reranker_dir() -> Path:
    return Path(os.environ.get("RERANKER_DIR", str(config.RERANKER_DIR)))


def _write_pointer(out_path: Path, pointer_name: str, meta: Optional[Dict[str, Any]] = None) -> None:
    """Create/refresh {pointer_name} alongside the model; fall back to copy if symlink fails."""
    out_dir = out_path.parent
    ptr = out_dir / pointer_name
    try:
        if ptr.exists() or ptr.is_symlink():
            ptr.unlink()
        try:
            ptr.symlink_to(out_path.name)
        except Exception:
            import shutil
            shutil.copyfile(out_path, ptr)
    except Exception:
        pass
    if meta:
        try:
            (out_dir / "latest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            pass


# -----------------------
# Shared IO & features
# -----------------------
def _iter_jsonl(paths: Iterable[Path]) -> Iterator[dict]:
    for p in paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                    if isinstance(rec, dict):
                        yield rec
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
        if s.startswith(t):
            return t
    if s.startswith("by "):
        return "by"
    return s.split(" ", 1)[0][:32]


def _row_from_attempt(rec: dict) -> List[float]:
    goal = rec.get("goal", "") or ""
    prefix = rec.get("prefix", []) or []
    cand = rec.get("candidate", "") or ""
    depth = int(rec.get("depth", 0) or 0)
    n_sub = rec.get("n_subgoals"); n_sub = int(n_sub) if n_sub is not None else 9999
    elapsed = float(rec.get("elapsed_ms", 0.0) or 0.0)
    cache_hit = 1 if rec.get("cache_hit") else 0
    state_proxy = " ".join(prefix[-3:])
    flags = _flags_from(goal, state_proxy)
    step_t = _step_prefix(cand)
    step_one_hot = [1 if step_t == t else 0 for t in STEP_TYPES]
    feats = [depth, n_sub, elapsed, cache_hit,
             flags["is_listy"], flags["is_natty"], flags["is_sety"], flags["has_q"], flags["is_bool"]] \
            + step_one_hot + [len(cand)]
    # Premise-selection tail (appended at end; defaults to zeros if absent in logs)
    feats += [
        float(rec.get("premise_cosine_top1", 0.0) or 0.0),
        float(rec.get("premise_cosine_topk_mean", 0.0) or 0.0),
        float(rec.get("premise_rerank_top1", 0.0) or 0.0),
        float(rec.get("premise_rerank_topk_mean", 0.0) or 0.0),
        float(rec.get("n_premises", 0.0) or 0.0),
        float(rec.get("cand_cos_mean", 0.0) or 0.0),
        float(rec.get("cand_cos_max", 0.0) or 0.0),
        float(rec.get("cand_rerank_mean", 0.0) or 0.0),
        float(rec.get("cand_hit_topk", 0.0) or 0.0),
        float(rec.get("cand_n_facts", 0.0) or 0.0),        
    ]
    return feats


def _runs_success(paths: Iterable[Path]) -> Dict[str, int]:
    succ = {}
    for rec in _iter_jsonl(paths):
        rid = rec.get("run_id")
        if rid:
            succ[rid] = 1 if rec.get("success") else 0
    return succ


def _autopaths(paths: List[str] | None, fallback: List[str]) -> List[str]:
    out: List[str] = []
    for p in (paths or []):
        if p and Path(p).exists():
            out.append(p)
    if out:
        return out
    for p in fallback:
        if Path(p).exists():
            out.append(p)
    return out


# --------------------------
# Classic datasets (bandit/q)
# --------------------------
def _make_bandit(attempts: List[str], runs: Optional[List[str]]) -> Tuple[List[List[float]], List[int]]:
    X: List[List[float]] = []; y: List[int] = []
    succ = _runs_success([Path(p) for p in (runs or [])]) if runs else {}
    for rec in _iter_jsonl([Path(p) for p in attempts]):
        if rec.get("type") not in ("expand", "expand_macro", "finish"):
            continue
        rid = rec.get("run_id")
        label = succ.get(rid, 1 if rec.get("ok") else 0)
        X.append(_row_from_attempt(rec)); y.append(int(label))
    return X, y


def _make_q(attempts: List[str], runs: Optional[List[str]], discount: float = 0.8) -> Tuple[List[List[float]], List[float]]:
    X: List[List[float]] = []; q: List[float] = []
    succ = _runs_success([Path(p) for p in (runs or [])]) if runs else {}
    for rec in _iter_jsonl([Path(p) for p in attempts]):
        if rec.get("type") not in ("expand", "expand_macro"):
            continue
        rid = rec.get("run_id")
        success = float(succ.get(rid, 1 if rec.get("ok") else 0))
        sb = rec.get("subgoals_before")
        sa = rec.get("n_subgoals")
        el = float(rec.get("elapsed_ms", 0.0) or 0.0)
        dsub = 0.0
        if isinstance(sb, int) and isinstance(sa, int):
            dsub = max(0.0, float(sb - sa))
        rate = dsub / max(1.0, el)  # progress per millisecond
        target = 0.5 * discount * success + 0.5 * min(1.0, rate)
        X.append(_row_from_attempt(rec)); q.append(target)
    return X, q


# --------------------------
# Wrapper for regressors
# --------------------------
class _RLQXGBWrapper:
    def __init__(self, reg):
        self.reg = reg
    def predict_proba(self, X):
        import numpy as np
        y = self.reg.predict(X)
        y = np.clip(y, 0.0, 1.0)
        return np.stack([1.0 - y, y], axis=1)


# --------------------------
# DeepRL datasets & models
# --------------------------
@dataclass
class Row:
    x: List[float]
    y: float  # soft label (teacher blend) or 0/1
    w: float
    group: str


def _group_key(rec: dict) -> str:
    pid = rec.get("proposal_id")
    if pid:
        return f"pid:{pid}"
    rid = rec.get("run_id") or "?"
    d = int(rec.get("depth", 0) or 0)
    prefix = rec.get("prefix", []) or []
    tail = " | ".join(prefix[-2:])
    return f"rdp:{rid}|{d}|{tail}"


# --- safe teacher (subprocess) for risky joblibs like xgboost ---
class _TeacherProc:
    """Run a risky joblib model (e.g., XGBoost on macOS) in a spawned subprocess.
       API: .score_one(x: List[float]) -> float in [0,1]; .close() when done."""
    def __init__(self, model_path: str):
        from multiprocessing import get_context, Pipe
        self._ctx = get_context("spawn")
        parent, child = Pipe()
        self._parent = parent
        self._proc = self._ctx.Process(target=self._worker, args=(child, model_path), daemon=True)
        self._proc.start()

    @staticmethod
    def _worker(conn, model_path: str):
        import traceback, math
        try:
            from joblib import load
            m = load(model_path)
            while True:
                msg = conn.recv()
                if msg is None:
                    break
                x = msg  # list[float]
                try:
                    if hasattr(m, "predict_proba"):
                        p = float(m.predict_proba([x])[0][1])
                    elif hasattr(m, "decision_function"):
                        d = float(m.decision_function([x])[0])
                        p = 1.0 / (1.0 + math.exp(-d))
                    else:
                        p = 0.5
                except Exception:
                    p = 0.5
                conn.send(p)
        except Exception:
            # If import/unpickle crashes, parent will detect failure.
            traceback.print_exc()
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def score_one(self, x: list) -> Optional[float]:
        try:
            self._parent.send(x)
            return float(self._parent.recv())
        except Exception:
            return None

    def close(self):
        try:
            self._parent.send(None)
        except Exception:
            pass
        try:
            self._parent.close()
        except Exception:
            pass
        try:
            if self._proc.is_alive():
                self._proc.join(timeout=1.0)
        except Exception:
            pass


def _maybe_make_safe_teacher(teacher_arg: str | None):
    """
    Always sandbox joblib teachers to avoid interpreter segfaults from xgboost/libomp.
    Returns None or a _TeacherProc instance.
    """
    if not teacher_arg:
        return None
    t = teacher_arg.strip().lower()
    if t in ("none", "off", "no"):
        return None

    # Resolve 'auto' to the newest joblib in RERANKER_DIR
    if t == "auto":
        d = _reranker_dir()
        cands = sorted(d.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            return None
        teacher_path = str(cands[0])
    else:
        teacher_path = teacher_arg

    try:
        return _TeacherProc(teacher_path)
    except Exception:
        return None


def build_offline_awr(
    attempts_paths: List[str],
    runs_paths: Optional[List[str]],
    tau: float,
    listwise_norm: bool,
    teacher=None,
    teacher_w: float = 0.3,
) -> List[Row]:
    succ = _runs_success([Path(p) for p in (runs_paths or [])]) if runs_paths else {}
    groups: DefaultDict[str, List[dict]] = defaultdict(list)
    for rec in _iter_jsonl([Path(p) for p in attempts_paths]):
        if rec.get("type") not in ("expand", "expand_macro"):
            continue
        groups[_group_key(rec)].append(rec)

    rows: List[Row] = []
    for gkey, batch in groups.items():
        scores, t_scores = [], []
        for r in batch:
            ok = 1.0 if r.get("ok") else 0.0
            sb = r.get("subgoals_before")
            sa = r.get("n_subgoals")
            delta = 0.0
            if isinstance(sb, int) and isinstance(sa, int):
                delta = max(0.0, float(sb - sa))
            scores.append(ok + 0.25 * delta)

            # optional teacher score in [0,1]
            if teacher is not None:
                x = _row_from_attempt(r)
                try:
                    if hasattr(teacher, "predict_proba") or hasattr(teacher, "decision_function"):
                        if hasattr(teacher, "predict_proba"):
                            p = float(teacher.predict_proba([x])[0][1])
                        else:
                            d = float(teacher.decision_function([x])[0])
                            p = 1.0 / (1.0 + math.exp(-d))
                    elif isinstance(teacher, _TeacherProc):
                        pv = teacher.score_one(x)
                        p = 0.5 if pv is None else float(pv)
                    else:
                        p = 0.5
                except Exception:
                    p = 0.5
            else:
                p = 0.5
            t_scores.append(p)

        if not scores:
            continue

        # listwise z-score before softmax
        s_arr = scores
        if listwise_norm:
            m = sum(s_arr) / len(s_arr)
            v = sum((s - m) * (s - m) for s in s_arr) / max(1, len(s_arr) - 1)
            std = math.sqrt(max(1e-12, v))
            s_arr = [(s - m) / std for s in s_arr]

        # AWR weights via softmax(scores / tau)
        mx = max(s_arr)
        exps = [math.exp((s - mx) / max(1e-6, tau)) for s in s_arr]
        Z = sum(exps) or 1.0
        aw = [e / Z for e in exps]

        for r, w, pt in zip(batch, aw, t_scores):
            rid = r.get("run_id")
            run_ok = succ.get(rid, 1 if r.get("ok") else 0)
            x = _row_from_attempt(r)
            y = 1.0 if r.get("ok") else 0.0
            y_soft = (1.0 - teacher_w) * float(y) + teacher_w * float(pt)
            rows.append(Row(x=x, y=y_soft, w=float(w) * (1.0 + 0.5 * float(run_ok)), group=gkey))
    return rows


# Torch models (lazy import in main)
class _MLP_Torch:
    def __init__(self, d_in: int, hidden: int = 128, layers: int = 2):
        import torch.nn as nn
        blocks = []
        h = d_in
        for _ in range(layers):
            blocks += [nn.Linear(h, hidden), nn.ReLU()]
            h = hidden
        blocks += [nn.Linear(h, 1)]
        self.net = nn.Sequential(*blocks)
        self.nn = nn
    def __call__(self, x):
        import torch
        logits = self.net(x)
        return torch.sigmoid(logits).squeeze(-1)


@dataclass
class QRow:
    x: List[float]
    r: float
    sp: Optional[str]
    done: bool


def _reward_from(rec: dict) -> float:
    sb = rec.get("subgoals_before")
    sa = rec.get("n_subgoals")
    dsub = 0.0
    if isinstance(sb, int) and isinstance(sa, int):
        dsub = max(0.0, float(sb - sa))
    base = 0.05 * dsub - 0.005
    if rec.get("type") == "finish" and rec.get("ok"):
        base += 1.0
    elif rec.get("ok"):
        base += 0.2
    return float(base)


def build_offline_q(attempts_paths: List[str]) -> Tuple[List[QRow], Dict[str, List[int]]]:
    rows: List[QRow] = []
    state_index: Dict[str, List[int]] = defaultdict(list)
    for rec in _iter_jsonl([Path(p) for p in attempts_paths]):
        if rec.get("type") not in ("expand", "expand_macro", "finish"):
            continue
        x = _row_from_attempt(rec)
        r = _reward_from(rec)
        sp = rec.get("state_fp_after") if rec.get("ok") else rec.get("state_fp_before")
        done = bool(rec.get("type") == "finish" and rec.get("ok"))
        fpb = rec.get("state_fp_before")
        idx = len(rows)
        rows.append(QRow(x=x, r=r, sp=sp if isinstance(sp, str) else None, done=done))
        if isinstance(fpb, str):
            state_index[fpb].append(idx)
    return rows, state_index


class _QMLP_Torch:
    def __init__(self, d_in: int, hidden: int = 192, layers: int = 3):
        import torch.nn as nn
        blocks = []
        h = d_in
        for _ in range(layers):
            blocks += [nn.Linear(h, hidden), nn.ReLU()]
            h = hidden
        blocks += [nn.Linear(h, 1)]
        self.net = nn.Sequential(*blocks)
    def __call__(self, x):
        return self.net(x).squeeze(-1)


# --------------------------
# Main
# --------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Unified reranker trainer (classic + DeepRL)")
    ap.add_argument("--attempts", nargs="*", default=None, help="attempts log paths")
    ap.add_argument("--runs", nargs="*", default=None, help="runs log paths")
    ap.add_argument("--out", default=None, help="output model path (default: timestamped under RERANKER_DIR)")

    # classic targets & algos
    ap.add_argument("--target", choices=["bandit","q"], default="bandit")
    ap.add_argument("--algo",
        choices=["xgb-classifier","sklearn-logreg","xgb-regressor","awr","dqn"],
        default="xgb-classifier")
    ap.add_argument("--min_rows", type=int, default=200)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--n_estimators", type=int, default=300)

    # DeepRL hyperparams
    ap.add_argument("--tau", type=float, default=0.5, help="AWR temperature")
    ap.add_argument("--listwise_norm", dest="listwise_norm", action="store_true", default=True,
                    help="Apply per-batch z-score before softmax (AWR).")
    ap.add_argument("--listwise-norm", dest="listwise_norm", action="store_true")
    ap.add_argument("--no-listwise-norm", dest="listwise_norm", action="store_false")
    ap.add_argument("--teacher", default="",
                    help="Path to joblib teacher or 'auto' (pick latest in RERANKER_DIR). Omit/none to disable.")
    ap.add_argument("--teacher_w", type=float, default=0.3, help="Teacher blend weight into soft-label (0..1).")
    ap.add_argument("--teacher-w", dest="teacher_w", type=float)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_split", type=float, default=0.1)
    # DQN specifics
    ap.add_argument("--gamma", type=float, default=0.92)
    ap.add_argument("--target_update", type=int, default=500)

    args = ap.parse_args(argv)

    # macOS safety: prefer 'spawn' for all multiprocessing before any heavy imports
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    # resolve inputs
    attempts = _autopaths(args.attempts or [config.ATTEMPTS_LOG], ["log/attempts.log.jsonl", "logs/attempts.log.jsonl"])
    runs     = _autopaths(args.runs     or [config.RUNS_LOG],     ["log/runs.log.jsonl",     "logs/runs.log.jsonl"])

    if not attempts:
        print("No attempts logs found. Use --attempts or set config.ATTEMPTS_LOG", file=sys.stderr)
        return 2

    # branch: classic joblib learners
    if args.algo in ("xgb-classifier","sklearn-logreg","xgb-regressor"):
        if args.target == "bandit":
            X, y = _make_bandit(attempts, runs)
        else:
            X, y = _make_q(attempts, runs)

        n = len(X)
        if n < args.min_rows:
            pos = sum(1 for v in y if (v if args.target == "q" else v >= 1))
            print(f"Not enough rows ({n}) to train. Run more proofs first. (pos~{pos})", file=sys.stderr)
            return 1

        out_dir = _reranker_dir(); out_dir.mkdir(parents=True, exist_ok=True)
        out_path = Path(args.out) if args.out else out_dir / f"reranker-{time.strftime('%Y%m%d-%H%M%S')}-{args.algo}-{args.target}.joblib"

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

        # also expose feature schema in latest.json for downstream consumers
        try:
            feat_schema = feature_names()
        except Exception:
            feat_schema = []
        _write_pointer(out_path, "latest.joblib",
                       meta={"path": out_path.name, "algo": args.algo, "target": args.target, "rows": n, "features": feat_schema})
        return 0

    # branch: DeepRL (TorchScript)
    try:
        import torch
        import torch.nn as nn
        import torch.utils.data as td
    except Exception:
        print("PyTorch not installed. pip install torch", file=sys.stderr)
        return 2

    out_dir = _reranker_dir(); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / f"reranker-{time.strftime('%Y%m%d-%H%M%S')}-{args.algo}.pt"

    if args.algo == "awr":
        # teacher is opt-in; sandbox XGBoost joblibs in a spawned subprocess
        teacher = _maybe_make_safe_teacher(args.teacher if isinstance(args.teacher, str) else "")
        rows = build_offline_awr(
            attempts, runs, tau=args.tau, listwise_norm=args.listwise_norm,
            teacher=teacher, teacher_w=args.teacher_w
        )
        n = len(rows)
        if n < args.min_rows:
            pos = sum(1 for r in rows if r.y >= 0.5)
            print(f"Not enough rows ({n}) to train. Run more proofs first. (soft-pos~{pos})", file=sys.stderr)
            try:
                if isinstance(teacher, _TeacherProc):
                    teacher.close()
            except Exception:
                pass
            return 1

        # tensors
        X = torch.tensor([r.x for r in rows], dtype=torch.float32)
        Y = torch.tensor([r.y for r in rows], dtype=torch.float32)
        W = torch.tensor([max(1e-6, r.w) for r in rows], dtype=torch.float32)

        nv = int(n * args.val_split)
        idx_tr = torch.arange(nv, n)
        idx_va = torch.arange(0, nv) if nv > 0 else torch.arange(0, 0)

        model = _MLP_Torch(d_in=X.shape[1]).net.train()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        bce = nn.BCELoss(reduction="none")

        def _run_epoch(I: torch.Tensor, train=True):
            model.train() if train else model.eval()
            if train:
                total = 0.0
                loader = td.DataLoader(I.tolist(), batch_size=args.batch, shuffle=True)
                for idx in loader:
                    xb, yb, wb = X[idx], Y[idx], W[idx]
                    opt.zero_grad(set_to_none=True)
                    pb = torch.sigmoid(model(xb).squeeze(-1))
                    loss = (bce(pb, yb) * wb).mean()
                    loss.backward(); opt.step()
                    total += float(loss.detach()) * len(idx)
                return total / max(1, len(I))
            else:
                with torch.no_grad():
                    if len(I) == 0: return 0.0
                    p = torch.sigmoid(model(X[I]).squeeze(-1))
                    return float((bce(p, Y[I]) * W[I]).mean())

        for ep in range(1, args.epochs + 1):
            tr = _run_epoch(idx_tr, train=True)
            va = _run_epoch(idx_va, train=False)
            print(f"[epoch {ep}] train_loss={tr:.4f}" + (f"  val_loss={va:.4f}" if len(idx_va) > 0 else ""))

        model.eval()
        scripted = torch.jit.script(torch.nn.Sequential(model, torch.nn.Sigmoid()))
        scripted.save(str(out_path))
        print(f"[ok] TorchScript saved → {out_path}")

        try:
            if isinstance(teacher, _TeacherProc):
                teacher.close()
        except Exception:
            pass

    else:  # dqn
        qrows, sidx = build_offline_q(attempts)
        n = len(qrows)
        if n < args.min_rows:
            print(f"Not enough rows ({n}) for DQN. Run more proofs first.", file=sys.stderr)
            return 1

        X = torch.tensor([r.x for r in qrows], dtype=torch.float32)
        R = torch.tensor([r.r for r in qrows], dtype=torch.float32)
        done = torch.tensor([1.0 if r.done else 0.0 for r in qrows], dtype=torch.float32)

        next_lists: List[List[int]] = []
        for r in qrows:
            if r.sp is None or r.sp not in sidx:
                next_lists.append([])
            else:
                next_lists.append(sidx[r.sp])

        model = _QMLP_Torch(d_in=X.shape[1]).net.train()
        target = _QMLP_Torch(d_in=X.shape[1]).net
        target.load_state_dict(model.state_dict()); target.eval()

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        huber = nn.SmoothL1Loss()
        step = 0

        for ep in range(args.epochs):
            idx = torch.randperm(X.shape[0])
            for i in range(0, X.shape[0], args.batch):
                b = idx[i:i+args.batch]
                xb, rb, db = X[b], R[b], done[b]
                with torch.no_grad():
                    q_next = torch.zeros_like(rb)
                    for j, qi in enumerate(b.tolist()):
                        nxt = next_lists[qi]
                        if nxt:
                            Xn = X[nxt]
                            qn = target(Xn)
                            q_next[j] = qn.max()
                y = rb + (1.0 - db) * args.gamma * q_next
                q = model(xb).squeeze(-1)
                loss = huber(q, y)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                step += 1
                if step % max(1, args.target_update) == 0:
                    target.load_state_dict(model.state_dict())
            print(f"[epoch {ep+1}] dqn_loss={float(loss):.4f}")

        # wrap with sigmoid prob head for runtime compatibility
        class QSig(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
            def forward(self, x):
                return torch.sigmoid(self.base(x))

        scripted = torch.jit.script(QSig(model).eval())
        scripted.save(str(out_path))
        print(f"[ok] TorchScript saved → {out_path}")

    # pointer: latest.pt (for DeepRL branches)
    if args.algo in ("awr","dqn"):
        try:
            feat_schema = feature_names()
        except Exception:
            feat_schema = []
        _write_pointer(out_path, "latest.pt",
                       meta={"path": out_path.name, "algo": args.algo, "rows": n, "features": feat_schema})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())