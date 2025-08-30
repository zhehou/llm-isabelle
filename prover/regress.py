# prover/regress.py
"""
Deterministic-ish regression harness for the prover.

Usage examples:

# 1) Record a baseline on the lists suite
python -m prover.regress --suite lists --save-baseline benchmarks/baselines/lists.json

# 2) Re-run and compare to baseline (nonzero exit if regression)
python -m prover.regress --suite lists --baseline benchmarks/baselines/lists.json

# 3) Use a custom file of goals
python -m prover.regress --file benchmarks/lists.txt --baseline benchmarks/baselines/lists.json

# 4) Choose an explicit model / ensemble (same as bench/cli)
python -m prover.regress --suite nat --model 'qwen3-coder:30b'
python -m prover.regress --suite sets --models 'qwen3-coder:30b,llama3.1:8b-instruct'

Notes:
- We keep a single-process run (no parallelism).
- We fix random seeding for shuffle to improve repeatability.
- We keep minimization ON by default; disable with --no-minimize to match cli semantics.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as stats
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project imports
from .isabelle_api import start_isabelle_server, get_isabelle_client
from .prover import prove_goal

# macOS / *nix child watcher (pattern known to work for you)
if sys.platform != "win32":
    import asyncio
    try:
        asyncio.get_event_loop_policy().set_child_watcher(asyncio.SafeChildWatcher())
    except Exception:
        pass  # best-effort; ignore if policy doesn't support this

# ---- Paths shared with bench.py conventions ----
BENCH_DIR = Path("benchmarks")
RESULTS_DIR = BENCH_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUITE_MAP = {
    "lists": BENCH_DIR / "lists.txt",
    "nat":   BENCH_DIR / "nat.txt",
    "sets":  BENCH_DIR / "sets.txt",
    "logic": BENCH_DIR / "logic.txt",
}

# ---- IO helpers ----
def read_goals_file(path: Path) -> List[str]:
    goals: List[str] = []
    if not path.exists():
        return goals
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.lower().startswith("lemma "):
                import re
                m = re.search(r'lemma\s+"(.+)"', s, re.IGNORECASE)
                goals.append(m.group(1) if m else s[len("lemma "):].strip().strip('"'))
            else:
                goals.append(s.strip('"'))
    return goals

# ---- Data structures ----
@dataclass
class OneGoal:
    goal: str
    success: bool
    elapsed_s: float
    depth: int
    timeout: bool
    model: str
    use_calls: int
    steps_len: int

@dataclass
class Summary:
    suite: str
    config: str
    n_goals: int
    n_success: int
    success_rate: float
    median_time_all: float
    median_time_success: float
    avg_depth: float
    stamp: str

@dataclass
class Report:
    suite: str
    config: str
    params: Dict[str, Any]
    goals: List[OneGoal]
    summary: Summary

# ---- Summaries ----
def summarize_rows(suite: str, config: str, rows: List[OneGoal]) -> Summary:
    n = len(rows)
    succ = [r for r in rows if r.success]
    times_all = [r.elapsed_s for r in rows]
    times_succ = [r.elapsed_s for r in succ]
    depths = [r.depth for r in rows]
    return Summary(
        suite=suite,
        config=config,
        n_goals=n,
        n_success=len(succ),
        success_rate=(len(succ)/n) if n else 0.0,
        median_time_all=stats.median(times_all) if times_all else 0.0,
        median_time_success=stats.median(times_succ) if times_succ else 0.0,
        avg_depth=(sum(depths)/n) if n else 0.0,
        stamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

# ---- Core run ----
def run_suite(
    isabelle,
    session_id: str,
    suite_name: str,
    goals_path: Path,
    *,
    beam: int,
    max_depth: int,
    budget_s: int,
    reranker: bool,
    sledge: bool,
    quickcheck: bool,
    nitpick: bool,
    facts_limit: int,
    minimize: bool,
    variants: bool,
    model: Optional[str],
    models: Optional[List[str]],
    shuffle_seed: int,
) -> Report:

    goals = read_goals_file(goals_path)
    if not goals:
        raise SystemExit(f"No goals found in {goals_path}")

    # stable order with seed (0 => time-based)
    if shuffle_seed != -1:
        import random
        rnd = random.Random(shuffle_seed or int(time.time()))
        rnd.shuffle(goals)

    cfg = {
        "beam": beam, "max_depth": max_depth, "budget_s": budget_s,
        "reranker": reranker, "sledge": sledge,
        "quickcheck": quickcheck, "nitpick": nitpick,
        "facts_limit": facts_limit, "minimize": minimize, "variants": variants,
        "model": model, "models": models, "shuffle_seed": shuffle_seed,
        "env_MODEL": os.environ.get("OLLAMA_MODEL", ""),
        "env_TEMP": os.environ.get("OLLAMA_TEMP", ""),
        "env_TOP_P": os.environ.get("OLLAMA_TOP_P", ""),
        "env_TIMEOUT_S": os.environ.get("OLLAMA_TIMEOUT_S", ""),
    }

    # config tag similar to bench.py naming
    if models:
        model_tag = "ensemble_" + "_".join(models)
    elif model:
        model_tag = model
    else:
        model_tag = os.environ.get("OLLAMA_MODEL", "env_default")

    config_name = f"beam{beam}_d{max_depth}_t{budget_s}_rr{'on' if reranker else 'off'}_sdg{'on' if sledge else 'off'}__model_{model_tag}"

    rows: List[OneGoal] = []
    for i, g in enumerate(goals, 1):
        print(f"[{suite_name}] [{i}/{len(goals)}] {g}")
        res = prove_goal(
            isabelle, session_id, g,
            model_name_or_ensemble=(model or os.environ.get("OLLAMA_MODEL", "qwen3-coder:30b")),
            beam_w=beam, max_depth=max_depth, hint_lemmas=6, budget_s=budget_s,
            models=models, save_dir=None,
            use_sledge=sledge, sledge_timeout=5, sledge_every=2,
            trace=False, use_color=False,
            use_qc=quickcheck, qc_timeout=2, qc_every=1,
            use_np=nitpick, np_timeout=5, np_every=2,
            facts_limit=facts_limit,
            do_minimize=minimize, minimize_timeout=8,
            do_variants=variants, variant_timeout=6, variant_tries=24,
            enable_reranker=reranker,
        )
        rows.append(OneGoal(
            goal=g,
            success=bool(res.get("success", False)),
            elapsed_s=float(res.get("elapsed_s", 0.0)),
            depth=int(res.get("depth", -1)),
            timeout=bool(res.get("timeout", False)),
            model=str(res.get("model", "")),
            use_calls=int(res.get("use_calls", 0)),
            steps_len=len(res.get("steps", [])),
        ))

    summ = summarize_rows(suite_name, config_name, rows)
    return Report(
        suite=suite_name,
        config=config_name,
        params=cfg,
        goals=rows,
        summary=summ,
    )

# ---- Baseline compare ----
def load_baseline(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def save_report(path: Path, rep: Report) -> None:
    data = {
        "suite": rep.suite,
        "config": rep.config,
        "params": rep.params,
        "summary": asdict(rep.summary),
        "goals": [asdict(g) for g in rep.goals],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def format_pct(x: float) -> str:
    return f"{x*100:.1f}%"

def compare_and_print(current: Report, baseline_data: Dict[str, Any], *, tol_rate: float, tol_time: float) -> bool:
    cur = current.summary
    base = baseline_data.get("summary", {})
    # Pull safely
    b_rate = float(base.get("success_rate", 0.0))
    b_med_all = float(base.get("median_time_all", 0.0))
    b_succ = int(base.get("n_success", 0))
    b_n = int(base.get("n_goals", 0))

    print("\n=== Regression comparison ===")
    print(f"Suite:   {current.suite}")
    print(f"Config:  {current.config}")
    print(f"Goals:   baseline {b_n}, current {cur.n_goals}")
    print(f"Success: baseline {b_succ} / {b_n} ({format_pct(b_rate)}), current {cur.n_success} / {cur.n_goals} ({format_pct(cur.success_rate)})")
    print(f"Median time (all): baseline {b_med_all:.2f}s, current {cur.median_time_all:.2f}s")

    regressed = False
    # Success rate drop beyond tolerance?
    if cur.success_rate + tol_rate < b_rate:
        print(f"⚠️  Success rate drop exceeds tolerance ({format_pct(b_rate)} → {format_pct(cur.success_rate)}, tol={format_pct(tol_rate)})")
        regressed = True
    # Time increase beyond tolerance? (only compare if both non-zero)
    if b_med_all > 0 and (cur.median_time_all - b_med_all) > tol_time:
        print(f"⚠️  Median time increased by {cur.median_time_all - b_med_all:.2f}s (tol={tol_time:.2f}s)")
        regressed = True

    if not regressed:
        print("✅ No regression detected within tolerances.")
    return regressed

# ---- CLI ----
def main():
    ap = argparse.ArgumentParser(description="Regression test harness for the LLM-guided Isabelle prover")
    grp_target = ap.add_mutually_exclusive_group(required=True)
    grp_target.add_argument("--suite", type=str, choices=sorted(list(SUITE_MAP.keys())), help="Built-in suite (benchmarks/*.txt)")
    grp_target.add_argument("--file", type=str, help="Custom goals file (one goal per line)")

    # proving knobs (keep aligned with bench/cli)
    ap.add_argument("--beam", type=int, default=2)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--budget-s", type=int, default=8)
    ap.add_argument("--facts-limit", type=int, default=6)
    ap.add_argument("--no-minimize", action="store_true", help="Disable minimization (enabled by default)")
    ap.add_argument("--variants", action="store_true", help="Enable structured proof variants")
    ap.add_argument("--quickcheck", action="store_true")
    ap.add_argument("--nitpick", action="store_true")
    ap.add_argument("--sledge", action="store_true")

    # model selection
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--models", type=str, default=None, help="Comma-separated list for ensemble")

    # reproducibility
    ap.add_argument("--shuffle-seed", type=int, default=0, help="0 = time-based, -1 = keep file order")

    # regression I/O
    ap.add_argument("--baseline", type=str, help="Compare against this baseline JSON; exit 1 on regression")
    ap.add_argument("--save-baseline", type=str, help="Write current run as baseline JSON to this path")
    ap.add_argument("--out", type=str, default=None, help="Also save current run JSON report here")

    # tolerances
    ap.add_argument("--tol-rate", type=float, default=0.00, help="Allowed absolute drop in success rate (e.g., 0.02 = 2%)")
    ap.add_argument("--tol-time", type=float, default=2.0, help="Allowed increase in median time (seconds)")

    args = ap.parse_args()

    # resolve goals source
    if args.file:
        suite_name = Path(args.file).stem
        goals_path = Path(args.file)
    else:
        suite_name = args.suite
        goals_path = SUITE_MAP[suite_name]

    # parse models
    models_list = [m.strip() for m in args.models.split(",")] if args.models else None

    # Isabelle lifecycle (single process)
    server_info, proc = start_isabelle_server(name="isabelle", log_file="regress_server.log")
    print(server_info.strip())
    isabelle = get_isabelle_client(server_info)
    session_id = isabelle.session_start(session="HOL")
    print("session_id:", session_id)

    try:
        rep = run_suite(
            isabelle, session_id, suite_name, goals_path,
            beam=args.beam, max_depth=args.max_depth, budget_s=args.budget_s,
            reranker=True,  # keep reranker ON by default; adjust here if you want
            sledge=args.sledge,
            quickcheck=args.quickcheck, nitpick=args.nitpick,
            facts_limit=args.facts_limit,
            minimize=(not args.no_minimize),
            variants=args.variants,
            model=args.model,
            models=models_list,
            shuffle_seed=args.shuffle_seed,
        )

        # Emit current report if requested
        if args.out:
            save_report(Path(args.out), rep)
            print(f"Wrote report → {args.out}")

        # Save as baseline
        if args.save_baseline:
            save_report(Path(args.save_baseline), rep)
            print(f"Saved baseline → {args.save_baseline}")
            return

        # Compare to baseline
        if args.baseline:
            base = load_baseline(Path(args.baseline))
            if not base:
                print(f"Baseline not found or unreadable: {args.baseline}")
                sys.exit(2)
            regressed = compare_and_print(rep, base, tol_rate=args.tol_rate, tol_time=args.tol_time)
            sys.exit(1 if regressed else 0)

        # No baseline I/O: just print summary
        s = rep.summary
        print("\n=== Regression run summary (no baseline) ===")
        print(f"Suite:        {s.suite}")
        print(f"Config:       {s.config}")
        print(f"Goals:        {s.n_goals}")
        print(f"Success:      {s.n_success} / {s.n_goals}  ({s.success_rate*100:.1f}%)")
        print(f"Median time:  all={s.median_time_all:.2f}s, succ={s.median_time_success:.2f}s")
        print(f"Avg depth:    {s.avg_depth:.2f}")

    finally:
        # shutdown order known to be stable on macOS
        try:
            isabelle.shutdown()
        except Exception:
            pass
        try:
            proc.terminate(); proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill(); proc.wait(timeout=3)
            except Exception:
                pass

        # optional watcher close (best-effort)
        if sys.platform != "win32":
            try:
                import asyncio
                w = asyncio.get_event_loop_policy().get_child_watcher()
                if w is not None and hasattr(w, "close"):
                    w.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
