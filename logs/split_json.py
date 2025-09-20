#!/usr/bin/env python3
import argparse, os, json
from pathlib import Path

def human(n):
    for unit in ['B','KB','MB','GB','TB']:
        if n < 1024.0:
            return f"{n:3.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"

def split_jsonl(input_path: str, out_dir: str, target_size_mb: int = 200, lines_per_shard: int | None = None, pad: int = 3):
    src = Path(input_path)
    dst = Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    total_bytes = src.stat().st_size
    print(f"[split] Source: {src} ({human(total_bytes)})")
    print(f"[split] Output dir: {dst}")

    shard_idx = 0
    lines = 0
    bytes_in_shard = 0
    target_bytes = target_size_mb * 1024 * 1024 if target_size_mb else None

    def shard_name(i):
        return dst / f"shard_{i:0{pad}d}"

    out = open(shard_name(shard_idx), "w", encoding="utf-8")
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            out.write(line)
            lines += 1
            bytes_in_shard += len(line.encode("utf-8"))
            cond_size = target_bytes and bytes_in_shard >= target_bytes
            cond_lines = lines_per_shard and (lines % lines_per_shard == 0)
            if cond_size or cond_lines:
                out.close()
                shard_idx += 1
                bytes_in_shard = 0
                out = open(shard_name(shard_idx), "w", encoding="utf-8")
    out.close()
    created = shard_idx + 1
    print(f"[split] Done. Created {created} shard(s) under {dst}")
    # Write a simple manifest
    man = {
        "source": str(src),
        "target_size_mb": target_size_mb,
        "lines_per_shard": lines_per_shard,
        "shards": []
    }
    total_lines = 0
    total_size = 0
    for i in range(created):
        p = shard_name(i)
        size = p.stat().st_size
        # count lines quickly (may be slow for huge shards; acceptable once)
        lc = sum(1 for _ in open(p, "r", encoding="utf-8"))
        man["shards"].append({"path": str(p), "bytes": size, "lines": lc})
        total_lines += lc
        total_size += size
    man["total_bytes"] = total_size
    man["total_lines"] = total_lines
    (dst / "MANIFEST.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"[split] Wrote manifest: {dst / 'MANIFEST.json'}")

def main():
    ap = argparse.ArgumentParser(description="Split a large JSONL attempts log into shards by size or line count.")
    ap.add_argument("--input", required=True, help="Path to logs/attempts.magnus.jsonl")
    ap.add_argument("--outdir", default="logs/magnus_shards", help="Output directory for shards")
    ap.add_argument("--target-size-mb", type=int, default=200, help="Target approx size per shard in MB (ignored if --lines-per-shard is set)")
    ap.add_argument("--lines-per-shard", type=int, default=None, help="Exact number of lines per shard (overrides --target-size-mb)")
    ap.add_argument("--pad", type=int, default=3, help="Zero padding width in shard filenames")
    args = ap.parse_args()
    split_jsonl(args.input, args.outdir, args.target_size_mb, args.lines_per_shard, args.pad)

if __name__ == "__main__":
    main()