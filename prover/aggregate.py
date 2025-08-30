# prover/aggregate.py
from __future__ import annotations

import argparse
import csv
import statistics as stats
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Default results root (matches bench.py)
DEFAULT_RESULTS_DIR = Path("benchmarks") / "results"


@dataclass
class Row:
    suite: str          # inferred from filename
    config: str         # inferred from filename (cfg.name)
    goal: str
    success: bool
    elapsed_s: float
    depth: int
    model: str
    timeout: bool
    use_calls: int
    steps_len: int


def _parse_filename(p: Path) -> Tuple[str, str]:
    """
    Extract (suite, config) from bench.py filename scheme:
      <ts>-<suite>-<cfg_name>.csv
    Example:
      20250829-103522-lists-beam2_d5_t6_rron_sdgoff__model_qwen.csv
        -> suite = 'lists'
        -> config = 'beam2_d5_t6_rron_sdgoff__model_qwen'
    """
    name = p.stem
    parts = name.split("-")
    if len(parts) < 3:
        return ("unknown", name)
    # parts[0] = timestamp like 20250829-103522
    suite = parts[1]
    config = "-".join(parts[2:])
    return suite, config


def _coerce_bool(x: str) -> bool:
    s = str(x).strip().lower()
    return s in ("1", "true", "yes", "y")


def load_results(results_dir: Path) -> List[Row]:
    rows: List[Row] = []
    for p in sorted(results_dir.glob("*.csv")):
        suite, cfg = _parse_filename(p)
        try:
            with p.open("r", encoding="utf-8") as f:
                rd = csv.DictReader(f)
                # Expect fields written by bench.py:
                # goal, success, depth, elapsed_s, model, timeout, use_calls, steps_len
                for r in rd:
                    try:
                        rows.append(
                            Row(
                                suite=suite,
                                config=cfg,
                                goal=r.get("goal", ""),
                                success=_coerce_bool(r.get("success", "false")),
                                elapsed_s=float(r.get("elapsed_s", "0") or 0.0),
                                depth=int(r.get("depth", "-1") or -1),
                                model=r.get("model", ""),
                                timeout=_coerce_bool(r.get("timeout", "false")),
                                use_calls=int(r.get("use_calls", "0") or 0),
                                steps_len=int(r.get("steps_len", "0") or 0),
                            )
                        )
                    except Exception:
                        # Skip malformed lines but keep going
                        continue
        except FileNotFoundError:
            continue
    return rows


@dataclass
class Summary:
    suite: str
    config: str
    n: int
    succ: int
    rate: float
    med_all: float
    med_succ: float
    avg_depth: float


def summarize(rows: List[Row]) -> Summary:
    if not rows:
        return Summary("", "", 0, 0, 0.0, 0.0, 0.0, 0.0)
    suite = rows[0].suite
    config = rows[0].config
    n = len(rows)
    succ_rows = [r for r in rows if r.success]
    succ = len(succ_rows)
    rate = (succ / n) if n else 0.0
    med_all = stats.median([r.elapsed_s for r in rows]) if rows else 0.0
    med_succ = stats.median([r.elapsed_s for r in succ_rows]) if succ_rows else 0.0
    avg_depth = sum(r.depth for r in rows) / n if n else 0.0
    return Summary(suite, config, n, succ, rate, med_all, med_succ, avg_depth)


def group_by_suite_config(rows: List[Row]) -> Dict[Tuple[str, str], List[Row]]:
    d: Dict[Tuple[str, str], List[Row]] = {}
    for r in rows:
        d.setdefault((r.suite, r.config), []).append(r)
    return d


def print_tables(
    by_suite_cfg: Dict[Tuple[str, str], List[Row]],
    min_rows: int = 1,
    best_only: bool = False,
    top_k: int = 3,
) -> None:
    # Organize summaries by suite
    suites = sorted(set(s for (s, _) in by_suite_cfg.keys()))
    try:
        from tabulate import tabulate
        use_tabulate = True
    except Exception:
        use_tabulate = False

    print("\n=== Benchmark summary (grouped by suite, sorted by success rate ↓ then median time ↑) ===")
    for suite in suites:
        # Collect summaries for this suite
        summaries = []
        for (s, cfg), rows in by_suite_cfg.items():
            if s != suite:
                continue
            sm = summarize(rows)
            if sm.n >= min_rows:
                summaries.append(sm)
        if not summaries:
            continue

        # Sort: success rate (desc), median time (asc), then config name
        summaries.sort(key=lambda x: (-x.rate, x.med_all, x.config))

        # Optionally keep only top-k
        display = summaries[:top_k] if (best_only and top_k > 0) else summaries

        print(f"\n[{suite}]")
        table = [
            [
                sm.config,
                f"{sm.rate*100:.1f}%",
                f"{sm.succ}/{sm.n}",
                f"{sm.med_all:.2f}",
                f"{sm.med_succ:.2f}" if sm.succ else "-",
                f"{sm.avg_depth:.2f}",
            ]
            for sm in display
        ]
        headers = ["config", "succ_rate", "succ/total", "median_time_all(s)", "median_time_succ(s)", "avg_depth"]

        if use_tabulate:
            print(tabulate(table, headers=headers, tablefmt="github"))
        else:
            # simple fallback
            colw = [max(len(str(x)) for x in col) for col in zip(*([headers] + table))]

            def fmt_row(row: List[str]) -> str:
                return "  ".join(str(cell).ljust(w) for cell, w in zip(row, colw))

            print(fmt_row(headers))
            print("  ".join("-" * w for w in colw))
            for r in table:
                print(fmt_row(r))


def main():
    ap = argparse.ArgumentParser(description="Aggregate benchmark CSVs into readable tables")
    ap.add_argument("--dir", type=str, default=str(DEFAULT_RESULTS_DIR),
                    help="Directory with benchmark CSV files (default: benchmarks/results)")
    ap.add_argument("--min-rows", type=int, default=1,
                    help="Minimum number of goals for a (suite,config) to be shown")
    ap.add_argument("--best-only", action="store_true",
                    help="Show only the top-k configs per suite")
    ap.add_argument("--top-k", type=int, default=3,
                    help="How many top configs to show when --best-only is used")
    args = ap.parse_args()

    results_dir = Path(args.dir)
    if not results_dir.exists():
        print(f"No such directory: {results_dir}")
        return

    rows = load_results(results_dir)
    if not rows:
        print(f"No CSV rows found in {results_dir}")
        return

    by_suite_cfg = group_by_suite_config(rows)
    print_tables(by_suite_cfg, min_rows=args.min_rows, best_only=args.best_only, top_k=args.top_k)


if __name__ == "__main__":
    main()
