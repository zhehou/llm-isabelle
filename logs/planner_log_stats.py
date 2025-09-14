#!/usr/bin/env python3
import json, math, statistics as stats
from pathlib import Path
import argparse

def _bool(x):
    return bool(x) if isinstance(x, bool) else (str(x).strip().lower() in ("1","true","yes","y"))

def compute_stats(path: Path):
    total = 0
    n_success = 0
    n_no_sorry = 0
    n_success_no_sorry = 0
    n_verified = 0
    n_transport = 0
    n_fail_verify = 0
    n_outline_mode = 0
    n_auto_mode = 0

    times_all = []
    times_succ = []

    by_mode = {"auto": {"total":0,"succ":0,"succ_no_sorry":0,"verified":0},
               "outline": {"total":0,"succ":0,"succ_no_sorry":0,"verified":0}}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not rec.get("goal"):
                continue

            mode = str(rec.get("mode") or "").lower()
            if mode not in by_mode:
                mode = "auto" if mode == "" else mode
            by_mode.setdefault(mode, {"total":0,"succ":0,"succ_no_sorry":0,"verified":0})
            by_mode[mode]["total"] += 1

            success = _bool(rec.get("success"))
            had_sorry = _bool(rec.get("had_sorry"))
            verified_ok = _bool(rec.get("verified_ok"))
            verify_details = str(rec.get("verify_details") or "")
            elapsed_s = rec.get("elapsed_s")

            if mode == "outline":
                n_outline_mode += 1
            elif mode == "auto":
                n_auto_mode += 1

            if isinstance(elapsed_s, (int, float)):
                times_all.append(float(elapsed_s))

            total += 1
            if success:
                n_success += 1
                by_mode[mode]["succ"] += 1
                if isinstance(elapsed_s, (int, float)):
                    times_succ.append(float(elapsed_s))

            if not had_sorry:
                n_no_sorry += 1

            if success and not had_sorry:
                n_success_no_sorry += 1
                by_mode[mode]["succ_no_sorry"] += 1

            if success and not had_sorry and verified_ok:
                n_verified += 1
                by_mode[mode]["verified"] += 1

            if "[transport_error]" in verify_details:
                n_transport += 1
            elif verify_details.startswith("[fail]") or "[fail]" in verify_details:
                n_fail_verify += 1

    success_rate = (n_success / total) if total else 0.0
    success_no_sorry_rate = (n_success_no_sorry / total) if total else 0.0
    verified_rate = (n_verified / total) if total else 0.0

    med_time_all = stats.median(times_all) if times_all else float("nan")
    med_time_succ = stats.median(times_succ) if times_succ else float("nan")

    return {
        "total_records": total,
        "mode_counts": {"auto": n_auto_mode, "outline": n_outline_mode},
        "success": {"count": n_success, "rate": round(success_rate*100, 2)},
        "success_no_sorry": {"count": n_success_no_sorry, "rate": round(success_no_sorry_rate*100, 2)},
        "verified_success": {"count": n_verified, "rate": round(verified_rate*100, 2)},
        "verify_transport_errors": n_transport,
        "verify_failures": n_fail_verify,
        "median_time_all_s": None if math.isnan(med_time_all) else round(med_time_all, 3),
        "median_time_success_s": None if math.isnan(med_time_succ) else round(med_time_succ, 3),
        "by_mode": by_mode,
    }

def main():
    ap = argparse.ArgumentParser(description="Summarize planner.log*.jsonl proof quality stats.")
    ap.add_argument("log", help="Path to planner log JSONL")
    ap.add_argument("--json", action="store_true", help="Print JSON only")
    args = ap.parse_args()

    path = Path(args.log)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    s = compute_stats(path)
    if args.json:
        print(json.dumps(s, indent=2))
    else:
        print("=== Planner Log Stats ===")
        print(f"File: {path}")
        print(f"Total records: {s['total_records']}  |  modes: auto={s['mode_counts']['auto']}, outline={s['mode_counts']['outline']}")
        print(f"Success:            {s['success']['count']} / {s['total_records']}  ({s['success']['rate']}%)")
        print(f"Hole-free success:  {s['success_no_sorry']['count']} / {s['total_records']}  ({s['success_no_sorry']['rate']}%)")
        print(f"Verified success:   {s['verified_success']['count']} / {s['total_records']}  ({s['verified_success']['rate']}%)")
        print(f"Verify transport errors: {s['verify_transport_errors']}  |  verify failures: {s['verify_failures']}")
        mta = s['median_time_all_s']; mts = s['median_time_success_s']
        print(f"Median time (all): {mta if mta is not None else '-'} s  |  Median time (success): {mts if mts is not None else '-'} s")
        print("\nBy mode:")
        for mode, d in s["by_mode"].items():
            if d["total"] == 0: 
                continue
            rate = (d["succ"] / d["total"]) * 100 if d["total"] else 0.0
            rate_strict = (d["succ_no_sorry"] / d["total"]) * 100 if d["total"] else 0.0
            rate_ver = (d["verified"] / d["total"]) * 100 if d["total"] else 0.0
            print(f"  {mode:7s}  total={d['total']:4d}  succ={d['succ']:4d} ({rate:5.1f}%)  "
                  f"hole-free={d['succ_no_sorry']:4d} ({rate_strict:5.1f}%)  verified={d['verified']:4d} ({rate_ver:5.1f}%)")

if __name__ == "__main__":
    main()
