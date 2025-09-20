import argparse, json, random, sys
from pathlib import Path

def _load_streaming(dataset_path: str, split: str = "train"):
    try:
        from datasets import load_dataset
    except Exception as e:
        print("Please `pip install -U datasets` to stream full_dataset.json", file=sys.stderr)
        raise

    if dataset_path.endswith(".jsonl"):
        ds = load_dataset("json", data_files=dataset_path, split=split, streaming=True)
    elif dataset_path.endswith(".json"):
        ds = load_dataset("json", data_files=dataset_path, split=split, streaming=True)
    else:
        # Allow loading by HF dataset name as well (e.g., Simontwice/premise_selection_in_isabelle)
        ds = load_dataset(dataset_path, split=split, streaming=True)
    return ds

def _collect_all_names(dataset_path: str, limit: int | None = None, seed: int = 7):
    ds = _load_streaming(dataset_path)
    names = []
    n = 0
    for ex in ds:
        n += 1
        nm = ex.get("premise_name")
        if nm:
            names.append(str(nm))
        if limit and n >= limit:
            break
    # de-dup while preserving order (large list OK for sampling)
    seen = set()
    uniq = []
    for x in names:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq

def convert(dataset_path: str, out_path: str,
            pool_size: int = 64, max_rows: int | None = None, seed: int = 7,
            names_cache: str | None = None):
    random.seed(seed)
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Names cache (so we don't do two passes every time)
    if names_cache and Path(names_cache).exists():
        names = json.loads(Path(names_cache).read_text(encoding="utf-8"))
    else:
        # First pass: collect all premise names (streaming)
        names = _collect_all_names(dataset_path)
        if names_cache:
            Path(names_cache).write_text(json.dumps(names), encoding="utf-8")

    # Second pass: stream examples and write pseudo-logs
    ds = _load_streaming(dataset_path)
    out = open(out_path, "w", encoding="utf-8")
    m = 0
    for ex in ds:
        pos = ex.get("premise_name")
        if not pos:
            continue
        goal = ex.get("state") or ex.get("statement") or ""
        # sample a small pool for retrieval_picks (include the positive)
        if names:
            pool = set(random.sample(names, k=min(pool_size, len(names))))
        else:
            pool = set()
        pool.add(pos)
        rec = {
            "type": "finish",
            "ok": True,
            "goal": goal,
            "cand_facts": [pos],
            "retrieval_picks": sorted(pool),
        }
        out.write(json.dumps(rec) + "\n")
        m += 1
        if max_rows and m >= max_rows:
            break
    out.close()
    print(f"Wrote {m} rows to {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Convert MagnusData (full_dataset.json) to pseudo attempts logs.")
    ap.add_argument("--input", type=str, required=True, help="Path to full_dataset.json (or HF dataset name).")
    ap.add_argument("--out", type=str, default="logs/attempts.magnus.jsonl", help="Output JSONL path.")
    ap.add_argument("--pool-size", type=int, default=64, help="Size of retrieval_picks pool per row.")
    ap.add_argument("--max-rows", type=int, default=None, help="Limit rows (for quick dry runs).")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--names-cache", type=str, default="data/magnus_names.json",
                    help="Cache file for all premise names (speeds up re-runs).")
    args = ap.parse_args()

    convert(args.input, args.out,
            pool_size=args.pool_size, max_rows=args.max_rows, seed=args.seed,
            names_cache=args.names_cache)

if __name__ == "__main__":
    main()