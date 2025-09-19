# =============================================
# File: prover/train_premises.py
# Mine (goal, premise) pairs from attempts logs and
# train (a) a bi-encoder retriever and (b) an optional cross-encoder re-ranker.
# Models are saved under: <out>/premises/{encoder,rerank}, with metadata:
#   <out>/premises/premises.json
#   <out>/premises/premises_reranker.json
# Default <out> = "models"
# =============================================
from __future__ import annotations
import argparse, json, os, random, sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

def _iter_attempts(paths: List[str]):
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                yield rec

def build_pairs(log_paths: List[str], min_pos_per_goal: int = 1, max_negs_per_pos: int = 8):
    """Return a list of dict records: {'goal', 'pos', 'negs'} with hard negatives from retrieval pool."""
    bag: Dict[str, Dict[str, Set[str]]] = {}
    for rec in _iter_attempts(log_paths):
        t = rec.get("type")
        if t not in ("expand", "finish"):
            continue
        g = rec.get("goal")
        if not g:
            continue
        pool = rec.get("retrieval_picks") or []
        facts = rec.get("cand_facts") or []
        ok = bool(rec.get("ok", False))

        info = bag.setdefault(g, {"pos": set(), "pool": set()})
        for fid in pool:
            info["pool"].add(str(fid))
        if ok:
            for name in facts:
                if name:
                    info["pos"].add(str(name))

    out = []
    for g, info in bag.items():
        pos = list(info["pos"])
        pool = list(info["pool"])
        if len(pos) < min_pos_per_goal:
            continue
        # negatives: from pool, excluding positive lemma names
        pool_names = []
        for fid in pool:
            parts = str(fid).split(":")
            pool_names.append(parts[1] if len(parts) >= 2 else str(fid))
        neg = [nm for nm in set(pool_names) if nm not in set(pos)]
        for p in pos:
            nns = random.sample(neg, k=min(max_negs_per_pos, len(neg))) if neg else []
            out.append({"goal": g, "pos": p, "negs": nns})
    return out

# ---------- Training: Bi-encoder (SELECT) ----------
def train_bi_encoder(pairs, base_model: str, epochs: int, batch_size: int, out_dir: Path):
    try:
        from sentence_transformers import SentenceTransformer, losses, InputExample
        from torch.utils.data import DataLoader
    except Exception as e:
        print("[train_premises] ERROR: sentence-transformers (and torch) are required to train a bi-encoder.", file=sys.stderr)
        print("Install: pip install 'sentence-transformers>=2.6' torch --extra-index-url https://download.pytorch.org/whl/cpu", file=sys.stderr)
        sys.exit(2)

    examples: List["InputExample"] = []
    for rec in pairs:
        g = rec["goal"]; p = rec["pos"]
        examples.append(InputExample(texts=[g, p]))

    if not examples:
        print("[train_premises] No bi-encoder training pairs mined; skipping.", file=sys.stderr)
        return None

    model = SentenceTransformer(base_model)
    train_loader = DataLoader(examples, shuffle=True, batch_size=batch_size, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = max(10, int(0.1 * epochs * len(train_loader)))
    model.fit(
        train_objectives=[(train_loader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        output_path=None,
    )

    enc_dir = out_dir / "premises" / "encoder"
    enc_dir.mkdir(parents=True, exist_ok=True)
    model.save(enc_dir)
    meta = {
        "type": "sbert",
        "model_relpath": "encoder",
        "normalize": True,
        "base_model": base_model,
        "epochs": epochs,
        "batch_size": batch_size,
        "pairs": len(examples),
    }
    (out_dir / "premises" / "premises.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[train_premises] Bi-encoder saved to: {enc_dir.resolve()}")
    return enc_dir

# ---------- Training: Cross-encoder (RE-RANK) ----------
def build_cross_examples(pairs, max_negs_per_pos: int = 4):
    """Expand pairs into labeled (goal, premise, y) tuples."""
    pos = [(r["goal"], r["pos"], 1.0) for r in pairs]
    neg = []
    for r in pairs:
        g = r["goal"]
        for n in (r.get("negs") or [])[:max_negs_per_pos]:
            neg.append((g, n, 0.0))
    return pos + neg

def train_cross_encoder(pairs, base_model: str, epochs: int, batch_size: int, out_dir: Path,
                        max_negs_per_pos: int = 4):
    try:
        from sentence_transformers import CrossEncoder, InputExample
        from torch.utils.data import DataLoader
    except Exception:
        print("[train_premises] ERROR: sentence-transformers (and torch) are required to train a cross-encoder.", file=sys.stderr)
        sys.exit(2)

    triples = build_cross_examples(pairs, max_negs_per_pos=max_negs_per_pos)
    if not triples:
        print("[train_premises] No cross-encoder examples mined; skipping.", file=sys.stderr)
        return None

    random.shuffle(triples)
    examples = [InputExample(texts=[g, p], label=float(y)) for (g, p, y) in triples]
    train_loader = DataLoader(examples, shuffle=True, batch_size=batch_size, drop_last=True)

    model = CrossEncoder(base_model, num_labels=1)  # regression w/ BCEWithLogits
    warmup_steps = max(10, int(0.1 * epochs * len(train_loader)))
    model.fit(
        train_dataloader=train_loader,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=None,
        show_progress_bar=True,
    )

    rr_dir = out_dir / "premises" / "rerank"
    rr_dir.mkdir(parents=True, exist_ok=True)
    model.save(rr_dir)
    meta = {
        "type": "sbert-cross",
        "model_relpath": "rerank",
        "base_model": base_model,
        "epochs": epochs,
        "batch_size": batch_size,
        "pairs": len(triples),
    }
    (out_dir / "premises" / "premises_reranker.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[train_premises] Cross-encoder saved to: {rr_dir.resolve()}")
    return rr_dir

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(
        description="Mine (goal, premise) pairs from attempts logs and train premise SELECT/RE-RANK models.",
    )
    ap.add_argument("--logs", type=str, nargs="+", default=["logs/attempts.log.jsonl"],
                    help="Paths to attempts log files (JSONL).")
    # model dir default is 'models/' (the script will save under models/premises/…)
    ap.add_argument("--out", type=str, default="models", help="Model root directory (default: models/).")
    ap.add_argument("--dump-pairs", action="store_true", help="Write mined pairs to data/premise_pairs.jsonl for inspection.")
    ap.add_argument("--min-pos-per-goal", type=int, default=1)
    ap.add_argument("--max-negs-per-pos", type=int, default=8)

    # Bi-encoder (SELECT)
    ap.add_argument("--train-bi", action="store_true", default=True, help="Train bi-encoder (SELECT).")
    ap.add_argument("--base-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Sentence-Transformers base model for the bi-encoder.")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)

    # Cross-encoder (RE-RANK)
    ap.add_argument("--train-cross", action="store_true", help="Also train cross-encoder (RE-RANK).")
    ap.add_argument("--cross-base-model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                    help="Sentence-Transformers CrossEncoder base model.")
    ap.add_argument("--epochs-cross", type=int, default=2)
    ap.add_argument("--batch-size-cross", type=int, default=32)
    ap.add_argument("--negs-per-pos-cross", type=int, default=4)

    args = ap.parse_args()

    pairs = build_pairs(args.logs, min_pos_per_goal=args.min_pos_per_goal, max_negs_per_pos=args.max_negs_per_pos)
    if args.dump_pairs:
        outp = Path("data"); outp.mkdir(parents=True, exist_ok=True)
        with (outp / "premise_pairs.jsonl").open("w", encoding="utf-8") as f:
            for r in pairs:
                f.write(json.dumps(r) + "\n")
        print(f"[train_premises] Wrote mined pairs: {outp/'premise_pairs.jsonl'}  (n={len(pairs)})")

    out_dir = Path(args.out)

    if args.train_bi if hasattr(args, "train_bi") else True:
        train_bi_encoder(
            pairs,
            base_model=args.base_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            out_dir=out_dir,
        )

    if args.train_cross:
        train_cross_encoder(
            pairs,
            base_model=args.cross_base_model,
            epochs=args.epochs_cross,
            batch_size=args.batch_size_cross,
            out_dir=out_dir,
            max_negs_per_pos=args.negs_per_pos_cross,
        )

    print("[train_premises] Done.")

if __name__ == "__main__":
    main()
