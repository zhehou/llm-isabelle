# prover/train_deeprl.py
from __future__ import annotations
import argparse, os, sys, json, time, math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Any, DefaultDict
from collections import defaultdict

import torch
import torch.nn as nn
import torch.utils.data as td

from . import config
from .features import STEP_TYPES, feature_names  # for schema parity


# ---------- IO ----------
def _iter_jsonl(paths: Iterable[Path]) -> Iterator[dict]:
    for p in paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict):
                        yield rec
                except Exception:
                    continue


def _runs_success(paths: Iterable[Path]) -> Dict[str, int]:
    succ = {}
    for rec in _iter_jsonl(paths):
        rid = rec.get("run_id")
        if not rid:
            continue
        succ[rid] = 1 if rec.get("success") else 0
    return succ


# ---------- features (mirror classic trainer) ----------
def _flags_from(goal: str, state_proxy: str) -> Dict[str, int]:
    g = (goal + " " + state_proxy).lower()
    return {
        "is_listy": int(any(k in g for k in ["@", "append", "rev", "map", "take", "drop"])),
        "is_natty": int(any(k in g for k in ["suc", "nat", "≤", "<", "0", "add", "mult", "+", "-", "*"])),
        "is_sety": int(any(k in g for k in ["∈", "subset", "⋂", "⋃"])),
        "has_q": int(any(k in g for k in ["∀", "∃"])),
        "is_bool": int(any(k in g for k in ["true", "false", "¬", "not", "∧", "∨"])),
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
    n_sub = rec.get("n_subgoals")
    n_sub = int(n_sub) if n_sub is not None else 9999
    elapsed = float(rec.get("elapsed_ms", 0.0) or 0.0)
    cache_hit = 1 if rec.get("cache_hit") else 0
    state_proxy = " ".join(prefix[-3:])
    flags = _flags_from(goal, state_proxy)
    step_t = _step_prefix(cand)
    step_one_hot = [1 if step_t == t else 0 for t in STEP_TYPES]
    return (
        [depth, n_sub, elapsed, cache_hit,
         flags["is_listy"], flags["is_natty"], flags["is_sety"], flags["has_q"], flags["is_bool"]]
        + step_one_hot
        + [len(cand)]
    )


# ---------- dataset ----------
@dataclass
class Row:
    x: List[float]
    y: float  # may be soft when teacher blending is used
    w: float
    group: str


def _group_key(rec: dict) -> str:
    # Prefer logger-provided batch id; otherwise fallback to (run, depth, last-2 prefix)
    pid = rec.get("proposal_id")
    if pid:
        return f"pid:{pid}"
    rid = rec.get("run_id") or "?"
    d = int(rec.get("depth", 0) or 0)
    prefix = rec.get("prefix", []) or []
    tail = " | ".join(prefix[-2:])
    return f"rdp:{rid}|{d}|{tail}"


def _load_teacher(path: Optional[str]) -> Optional[Any]:
    try:
        from joblib import load
        if path:
            p = Path(path)
            return load(p) if p.exists() else None
        # auto-pick latest *.joblib in RERANKER_DIR
        d = Path(os.environ.get("RERANKER_DIR", str(config.RERANKER_DIR)))
        cands = sorted(d.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
        return load(cands[0]) if cands else None
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
        # raw utility: ok + positive delta subgoals
        scores, t_scores = [], []
        for r in batch:
            ok = 1.0 if r.get("ok") else 0.0
            sb = r.get("subgoals_before")
            sa = r.get("n_subgoals")
            delta = 0.0
            if isinstance(sb, int) and isinstance(sa, int):
                delta = max(0.0, float(sb - sa))
            scores.append(ok + 0.25 * delta)

            # optional teacher score (0..1)
            if teacher is not None:
                try:
                    x = _row_from_attempt(r)
                    if hasattr(teacher, "predict_proba"):
                        p = float(teacher.predict_proba([x])[0][1])
                    elif hasattr(teacher, "decision_function"):
                        d = float(teacher.decision_function([x])[0])
                        p = 1.0 / (1.0 + math.exp(-d))
                    else:
                        p = 0.5
                except Exception:
                    p = 0.5
            else:
                p = 0.5
            t_scores.append(p)

        if not scores:
            continue

        # listwise normalization (z-score per batch) before softmax
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
            run_ok = succ.get(rid, 1 if r.get("ok") else 0)  # fallback helps bootstrap
            x = _row_from_attempt(r)
            y = 1.0 if r.get("ok") else 0.0
            # teacher-blended soft target
            y_soft = (1.0 - teacher_w) * float(y) + teacher_w * float(pt)
            # combine: awr weight * (1 + 0.5*run_success)
            rows.append(Row(x=x, y=y_soft, w=float(w) * (1.0 + 0.5 * float(run_ok)), group=gkey))
    return rows


# ---------- models ----------
class MLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128, layers: int = 2):
        super().__init__()
        blocks = []
        h = d_in
        for _ in range(layers):
            blocks += [nn.Linear(h, hidden), nn.ReLU()]
            h = hidden
        blocks += [nn.Linear(h, 1)]  # logits
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return probability in [0,1]
        logits = self.net(x)
        return torch.sigmoid(logits).squeeze(-1)


# ---------- training (AWR) ----------
def train_awr(
    rows: List[Row], *, epochs=6, batch=512, lr=1e-3, val_split=0.1, seed=42
) -> MLP:
    import random
    random.Random(seed).shuffle(rows)

    X = torch.tensor([r.x for r in rows], dtype=torch.float32)
    Y = torch.tensor([r.y for r in rows], dtype=torch.float32)  # may be soft
    W = torch.tensor([max(1e-6, r.w) for r in rows], dtype=torch.float32)

    n = len(rows)
    nv = int(n * val_split)
    idx_tr = torch.arange(nv, n)
    idx_va = torch.arange(0, nv) if nv > 0 else torch.arange(0, 0)

    model = MLP(d_in=X.shape[1]).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    bce = nn.BCELoss(reduction="none")

    def _run_epoch(I: torch.Tensor, train=True):
        model.train() if train else model.eval()
        total = 0.0
        if train:
            loader = td.DataLoader(I.tolist(), batch_size=batch, shuffle=True)
            for idx in loader:
                xb, yb, wb = X[idx], Y[idx], W[idx]
                opt.zero_grad(set_to_none=True)
                pb = model(xb)
                loss_vec = bce(pb, yb) * wb
                loss = loss_vec.mean()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(idx)
            return total / max(1, len(I))
        else:
            with torch.no_grad():
                if len(I) == 0:
                    return 0.0
                p = model(X[I])
                loss = (bce(p, Y[I]) * W[I]).mean()
                return float(loss)

    for ep in range(1, epochs + 1):
        tr = _run_epoch(idx_tr, train=True)
        va = _run_epoch(idx_va, train=False)
        print(
            f"[epoch {ep}] train_loss={tr:.4f}"
            + (f"  val_loss={va:.4f}" if len(idx_va) > 0 else "")
        )

    return model.eval()


# ------------------ DQN (offline fitted Q) ------------------
@dataclass
class QRow:
    x: List[float]
    r: float
    sp: Optional[str]  # next state's fingerprint key
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
    # index of rows sharing the same "state_fp_before"
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


class QMLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 192, layers: int = 3):
        super().__init__()
        blocks = []
        h = d_in
        for _ in range(layers):
            blocks += [nn.Linear(h, hidden), nn.ReLU()]
            h = hidden
        blocks += [nn.Linear(h, 1)]
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # Q(s,a)


def train_dqn(
    qrows: List[QRow],
    state_index: Dict[str, List[int]],
    *,
    epochs=8,
    batch=1024,
    lr=1e-3,
    gamma=0.92,
    target_update=500,
    seed=42,
) -> QMLP:
    import random
    random.Random(seed).shuffle(qrows)

    X = torch.tensor([r.x for r in qrows], dtype=torch.float32)
    R = torch.tensor([r.r for r in qrows], dtype=torch.float32)
    done = torch.tensor([1.0 if r.done else 0.0 for r in qrows], dtype=torch.float32)

    # map next-state indices list
    next_lists: List[List[int]] = []
    for r in qrows:
        if r.sp is None or r.sp not in state_index:
            next_lists.append([])
        else:
            next_lists.append(state_index[r.sp])

    model = QMLP(d_in=X.shape[1]).train()
    target = QMLP(d_in=X.shape[1])
    target.load_state_dict(model.state_dict())
    target.eval()

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    huber = nn.SmoothL1Loss()
    step = 0

    for ep in range(epochs):
        # simple full-batch sampler (shuffle indices)
        idx = torch.randperm(X.shape[0])
        for i in range(0, X.shape[0], batch):
            b = idx[i : i + batch]
            xb, rb, db = X[b], R[b], done[b]
            with torch.no_grad():
                # bootstrap from observed next-state action sets
                q_next = torch.zeros_like(rb)
                for j, qi in enumerate(b.tolist()):
                    nxt = next_lists[qi]
                    if nxt:
                        Xn = X[nxt]
                        qn = target(Xn)  # [kn]
                        q_next[j] = qn.max()
            y = rb + (1.0 - db) * gamma * q_next
            q = model(xb)
            loss = huber(q, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % max(1, target_update) == 0:
                target.load_state_dict(model.state_dict())
        print(f"[epoch {ep + 1}] dqn_loss={float(loss):.4f}")

    return model.eval()


# ---------- main ----------
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Deep RL / AWR reranker trainer (.pt export)")
    ap.add_argument("--attempts", nargs="*", default=None, help="attempts log paths")
    ap.add_argument("--runs", nargs="*", default=None, help="runs log paths")
    ap.add_argument("--algo", choices=["awr", "dqn"], default="awr")
    ap.add_argument("--tau", type=float, default=0.5, help="AWR temperature")
    # accept both underscore and hyphen aliases
    ap.add_argument(
        "--listwise_norm",
        dest="listwise_norm",
        action="store_true",
        default=True,
        help="Apply per-batch z-score before softmax for AWR (on by default).",
    )
    ap.add_argument("--listwise-norm", dest="listwise_norm", action="store_true")
    ap.add_argument("--no-listwise-norm", dest="listwise_norm", action="store_false")
    ap.add_argument(
        "--teacher",
        default=None,
        help="Optional path to joblib teacher (XGBRanker). If omitted, auto-pick latest in RERANKER_DIR.",
    )
    ap.add_argument("--teacher_w", type=float, default=0.3, help="Teacher blend weight into soft-label (0..1).")
    ap.add_argument("--teacher-w", dest="teacher_w", type=float)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--min_rows", type=int, default=200)
    # DQN specifics
    ap.add_argument("--gamma", type=float, default=0.92)
    ap.add_argument("--target_update", type=int, default=500)
    ap.add_argument("--out", default=None, help="output model path (default: RERANKER_DIR/reranker-*.pt)")
    args = ap.parse_args(argv)

    attempts = _autopaths(
        args.attempts or [config.ATTEMPTS_LOG], ["log/attempts.log.jsonl", "logs/attempts.log.jsonl"]
    )
    runs = _autopaths(args.runs or [config.RUNS_LOG], ["log/runs.log.jsonl", "logs/runs.log.jsonl"])

    if not attempts:
        print("No attempts logs found. Use --attempts or set config.ATTEMPTS_LOG", file=sys.stderr)
        return 2

    # ---- train (AWR or DQN) ----
    if args.algo == "awr":
        teacher = _load_teacher(args.teacher) if (args.teacher is not None) else _load_teacher(None)
        rows = build_offline_awr(
            attempts, runs, tau=args.tau, listwise_norm=args.listwise_norm, teacher=teacher, teacher_w=args.teacher_w
        )
        n = len(rows)
        if n < args.min_rows:
            pos = sum(1 for r in rows if r.y >= 0.5)
            print(f"Not enough rows ({n}) to train. Run more proofs first. (soft-pos~{pos})", file=sys.stderr)
            return 1
        print(f"Training AWR on {n} rows …")
        model = train_awr(rows, epochs=args.epochs, batch=args.batch, lr=args.lr, val_split=args.val_split)
        out_suffix = "awr"
    else:
        qrows, sidx = build_offline_q(attempts)
        n = len(qrows)
        if n < args.min_rows:
            print(f"Not enough rows ({n}) for DQN. Run more proofs first.", file=sys.stderr)
            return 1
        print(f"Training DQN on {n} rows …")
        qnet = train_dqn(
            qrows, sidx, epochs=args.epochs, batch=args.batch, lr=args.lr, gamma=args.gamma, target_update=args.target_update
        )

        # wrap into a sigmoid probability head for compatibility with reranker.score()
        class QSig(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base

            def forward(self, x):
                return torch.sigmoid(self.base(x))

        model = QSig(qnet).eval()
        out_suffix = "dqn"

    # ---- save artifact ----
    out_dir = Path(os.environ.get("RERANKER_DIR", str(config.RERANKER_DIR)))
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_path = Path(args.out)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_path = out_dir / f"reranker-{stamp}-{out_suffix}.pt"

    scripted = torch.jit.script(model)
    scripted.save(str(out_path))
    print(f"[ok] TorchScript saved → {out_path}")

    # Update latest.pt pointer
    latest = out_dir / "latest.pt"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        try:
            latest.symlink_to(out_path.name)
        except Exception:
            import shutil
            shutil.copyfile(out_path, latest)
    except Exception:
        pass

    # Small manifest (optional)
    try:
        (out_dir / "latest.json").write_text(
            json.dumps({"path": out_path.name, "algo": out_suffix, "rows": n}, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
