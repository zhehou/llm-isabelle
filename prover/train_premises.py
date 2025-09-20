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
import glob, random

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
    ap.add_argument("--logs-glob", type=str, default="",
                    help="Glob for many shards, e.g., 'logs/magnus_shards/shard_*'.")
    ap.add_argument("--max-shards", type=int, default=0,
                    help="Limit number of shards processed from --logs / --logs-glob (0 = no limit).")
    ap.add_argument("--shuffle-shards", action="store_true",
                    help="Shuffle shard order before training.")    
    # model dir default is 'models/' (the script will save under models/premises/…)
    ap.add_argument("--out", type=str, default="models", help="Model root directory (default: models/).")
    ap.add_argument("--dump-pairs", action="store_true", help="Write mined pairs to logs/premise_pairs.jsonl for inspection.")
    ap.add_argument("--min-pos-per-goal", type=int, default=1)
    ap.add_argument("--max-negs-per-pos", type=int, default=8)

    # Bi-encoder (SELECT)
    ap.add_argument("--train-bi", action="store_true", default=True, help="Train bi-encoder (SELECT).")
    ap.add_argument("--base-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Sentence-Transformers base model for the bi-encoder.")
    ap.add_argument("--resume-bi", type=str, default="",
                    help="Resume bi-encoder from this model dir (overrides base-model for first shard if set).")    
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)

    # Cross-encoder (RE-RANK)
    ap.add_argument("--train-cross", action="store_true", help="Also train cross-encoder (RE-RANK).")
    ap.add_argument("--cross-base-model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                    help="Sentence-Transformers CrossEncoder base model.")
    ap.add_argument("--resume-cross", type=str, default="",
                    help="Resume cross-encoder from this model dir (overrides cross-base-model for first shard if set).")    
    ap.add_argument("--epochs-cross", type=int, default=2)
    ap.add_argument("--batch-size-cross", type=int, default=32)
    ap.add_argument("--negs-per-pos-cross", type=int, default=4)

    args = ap.parse_args()

    # Expand logs: explicit list + glob
    shard_paths = []
    if args.logs:
        shard_paths.extend(args.logs)
    if args.logs_glob:
        shard_paths.extend(sorted(glob.glob(args.logs_glob)))
    # Dedup while preserving order
    seen = set(); shards = []
    for p in shard_paths:
        if p not in seen:
            seen.add(p); shards.append(p)
    if args.shuffle_shards:
        random.shuffle(shards)
    if args.max_shards and args.max_shards > 0:
        shards = shards[:args.max_shards]
    if not shards:
        shards = ["logs/attempts.log.jsonl"]

    out_dir = Path(args.out)
    enc_ckpt = args.resume_bi.strip()
    rr_ckpt  = args.resume_cross.strip()

    def _train_bi_on(shard: str):
        nonlocal enc_ckpt
        if not (args.train_bi if hasattr(args, "train_bi") else True):
            return
        if args.epochs <= 0:
            return
        base = enc_ckpt or args.base_model
        print(f"[train_premises] BI on {shard} (base={base})")
        pairs = build_pairs([shard], min_pos_per_goal=args.min_pos_per_goal, max_negs_per_pos=args.max_negs_per_pos)
        res = train_bi_encoder(pairs, base_model=base, epochs=args.epochs, batch_size=args.batch_size, out_dir=out_dir)
        if res is not None:
            enc_ckpt = str(out_dir / "premises" / "encoder")

    def _train_cross_on(shard: str):
        nonlocal rr_ckpt
        if not args.train_cross:
            return
        if args.epochs_cross <= 0:
            return
        base = rr_ckpt or args.cross_base_model
        print(f"[train_premises] CROSS on {shard} (base={base})")
        pairs = build_pairs([shard], min_pos_per_goal=args.min_pos_per_goal, max_negs_per_pos=args.max_negs_per_pos)
        res = train_cross_encoder(pairs, base_model=base, epochs=args.epochs_cross,
                                  batch_size=args.batch_size_cross, out_dir=out_dir,
                                  max_negs_per_pos=args.negs_per_pos_cross)
        if res is not None:
            rr_ckpt = str(out_dir / "premises" / "rerank")

    # Optional one-shot dump from the first shard
    if args.dump_pairs:
        first = shards[0]
        pairs0 = build_pairs([first], min_pos_per_goal=args.min_pos_per_goal, max_negs_per_pos=args.max_negs_per_pos)
        outp = Path("data"); outp.mkdir(parents=True, exist_ok=True)
        with (outp / "premise_pairs.jsonl").open("w", encoding="utf-8") as f:
            for r in pairs0:
                f.write(json.dumps(r) + "\n")
        print(f"[train_premises] Wrote mined pairs from {first}: {outp/'premise_pairs.jsonl'}  (n={len(pairs0)})")

    # Shard loop: resume from last saved checkpoint automatically
    for i, shard in enumerate(shards):
        print(f"[train_premises] ===== Shard {i+1}/{len(shards)}: {shard} =====")
        _train_bi_on(shard)
        _train_cross_on(shard)

    print("[train_premises] Done.")

if __name__ == "__main__":
    main()
