#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path

def _bool(x):
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    return str(x).strip().lower() in ("1","true","yes","y")

def main():
    ap = argparse.ArgumentParser(description="Filter planner logs to positives only and convert to AFP-rich format.")
    ap.add_argument("log", help="planner.log*.jsonl")
    ap.add_argument("--positives-jsonl", help="Output JSONL of positive planner records (keeps original fields)")
    ap.add_argument("--isar-pairs-jsonl", help="Output JSONL of AFP-rich pairs: {goal, outline, source, model}")
    ap.add_argument("--require-verified", action="store_true",
                    help="Require (success && !had_sorry && verified_ok). If not set, uses (success && !had_sorry).")
    args = ap.parse_args()

    src = Path(args.log)
    if not src.exists():
        raise SystemExit(f"File not found: {src}")

    pos_path = Path(args.positives_jsonl) if args.positives_jsonl else None
    rich_path = Path(args.isar_pairs_jsonl) if args.isar_pairs_jsonl else None
    if pos_path:
        pos_path.parent.mkdir(parents=True, exist_ok=True)
    if rich_path:
        rich_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    total = 0
    with src.open("r", encoding="utf-8") as f, \
         (pos_path.open("w", encoding="utf-8") if pos_path else open("/dev/null","w")) as fp, \
         (rich_path.open("w", encoding="utf-8") if rich_path else open("/dev/null","w")) as fr:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue
            success = _bool(rec.get("success"))
            had_sorry = _bool(rec.get("had_sorry"))
            verified_ok = _bool(rec.get("verified_ok"))

            ok = (success and not had_sorry and verified_ok) if args.require_verified \
                 else (success and not had_sorry)
            if not ok:
                continue

            kept += 1

            # Write positives-only original record
            if pos_path:
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # Convert to "rich AFP-like" pair that priors.aggregate can consume.
            goal = rec.get("goal") or ""
            outline = rec.get("outline") or rec.get("final") or ""
            # Provide minimal fields used downstream; extra fields harmless.
            pair = {
                "goal": goal,
                "outline": outline,
                "source": "planner_log",
                "model": rec.get("model"),
                "suite": rec.get("suite"),
            }
            fr.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Kept {kept} / {total} records")
    if pos_path:
        print(f"Wrote positives to: {pos_path}")
    if rich_path:
        print(f"Wrote rich pairs to: {rich_path}")

if __name__ == "__main__":
    main()
