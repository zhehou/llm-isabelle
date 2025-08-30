# prover/bench.py
"""
Benchmark harness for the prover.

Examples:
  # Single suite, default model from env, reranker on, sledge off
  python -m prover.bench --suite lists

  # Compare reranker on/off, and sledge on/off
  python -m prover.bench --suite nat --reranker both --sledge both --budget-s 6

  # Run all built-in suites
  python -m prover.bench --suite all --reranker both --sledge off

  # Use a specific single model
  python -m prover.bench --suite sets --model 'qwen3-coder:30b'

  # Use an ensemble
  python -m prover.bench --suite lists --models 'qwen3-coder:30b,llama3.1:8b-instruct'

  # Custom goals file
  python -m prover.bench --file benchmarks/lists.txt --reranker both
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics as stats
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---- Project imports (align with your refactor) ----
from .isabelle_api import start_isabelle_server, get_isabelle_client
from .prover import prove_goal

# place near imports
import sys
if sys.platform != "win32":
    import asyncio
    try:
        asyncio.get_event_loop_policy().set_child_watcher(asyncio.SafeChildWatcher())
    except Exception:
        pass  # older Pythons or already set; ignore


# ---- File paths ----
BENCH_DIR = Path("benchmarks")
RESULTS_DIR = BENCH_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUITE_MAP = {
    "lists": BENCH_DIR / "lists.txt",
    "nat":   BENCH_DIR / "nat.txt",
    "sets":  BENCH_DIR / "sets.txt",
    "logic": BENCH_DIR / "logic.txt",
}

# ---- Helpers ----
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


@dataclass
class BenchConfig:
    name: str
    beam: int
    max_depth: int
    budget_s: int
    reranker: bool
    sledge: bool
    sledge_timeout: int
    sledge_every: int
    quickcheck: bool
    quickcheck_timeout: int
    quickcheck_every: int
    nitpick: bool
    nitpick_timeout: int
    nitpick_every: int
    facts_limit: int
    minimize: bool
    variants: bool


def run_one_goal(
    isabelle,
    session_id: str,
    goal: str,
    cfg: BenchConfig,
    model_name: Optional[str],
    models_list: Optional[List[str]],
) -> Dict[str, Any]:
    res = prove_goal(
        isabelle,
        session_id,
        goal,
        model_name_or_ensemble=(model_name or os.environ.get("OLLAMA_MODEL", "qwen3-coder:30b")),
        beam_w=cfg.beam,
        max_depth=cfg.max_depth,
        hint_lemmas=6,
        budget_s=cfg.budget_s,
        models=models_list,
        save_dir=None,
        use_sledge=cfg.sledge,
        sledge_timeout=cfg.sledge_timeout,
        sledge_every=cfg.sledge_every,
        trace=False,
        use_color=False,
        use_qc=cfg.quickcheck,
        qc_timeout=cfg.quickcheck_timeout,
        qc_every=cfg.quickcheck_every,
        use_np=cfg.nitpick,
        np_timeout=cfg.nitpick_timeout,
        np_every=cfg.nitpick_every,
        facts_limit=cfg.facts_limit,
        do_minimize=cfg.minimize,
        minimize_timeout=8,
        do_variants=cfg.variants,
        variant_timeout=6,
        variant_tries=24,
        enable_reranker=cfg.reranker,
    )
    # normalize into flat row
    return {
        "goal": goal,
        "success": bool(res.get("success")),
        "depth": int(res.get("depth", -1)),
        "elapsed_s": float(res.get("elapsed_s", 0.0)),
        "model": str(res.get("model", "")),
        "timeout": bool(res.get("timeout", False)),
        "use_calls": int(res.get("use_calls", 0)),
        "steps_len": len(res.get("steps", [])),
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    succ = [r for r in rows if r["success"]]
    times = [r["elapsed_s"] for r in rows]
    succ_times = [r["elapsed_s"] for r in succ]
    depths = [r["depth"] for r in rows]
    return {
        "n_goals": n,
        "n_success": len(succ),
        "success_rate": (len(succ) / n) if n else 0.0,
        "median_time_all": stats.median(times) if times else 0.0,
        "median_time_success": stats.median(succ_times) if succ_times else 0.0,
        "avg_depth": (sum(depths) / n) if n else 0.0,
    }


def pretty_report(suite_name: str, cfg: BenchConfig, rows: List[Dict[str, Any]]):
    s = summarize(rows)
    def pct(x: float) -> str:
        return f"{x*100:.1f}%"
    print("\n=== Benchmark Report ===")
    print(f"Suite:  {suite_name}")
    print(f"Config: {cfg.name}")
    print(f"  Beam={cfg.beam}  MaxDepth={cfg.max_depth}  Budget={cfg.budget_s}s")
    print(f"  Reranker={'ON' if cfg.reranker else 'OFF'}  Sledge={'ON' if cfg.sledge else 'OFF'}  "
          f"QC={'ON' if cfg.quickcheck else 'OFF'}  NP={'ON' if cfg.nitpick else 'OFF'}")
    print(f"Results on {s['n_goals']} goals:")
    print(f"  Success: {s['n_success']} / {s['n_goals']}  ({pct(s['success_rate'])})")
    print(f"  Median time (all):     {s['median_time_all']:.2f}s")
    print(f"  Median time (success): {s['median_time_success']:.2f}s")
    print(f"  Avg depth:             {s['avg_depth']:.2f}")


def write_csv(suite_name: str, cfg: BenchConfig, rows: List[Dict[str, Any]]) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    safe_model_tag = cfg.name.replace(" ", "_")
    out = RESULTS_DIR / f"{ts}-{suite_name}-{safe_model_tag}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out


def main():
    ap = argparse.ArgumentParser(description="Benchmark harness for the prover")
    ap.add_argument(
        "--suite",
        type=str,
        choices=sorted(list(SUITE_MAP.keys()) + ["all"]),
        help="Built-in suite name (benchmarks/*.txt) or 'all'",
    )
    ap.add_argument("--file", type=str, help="Custom goals file (one goal per line)")
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--budget-s", type=int, default=6)
    ap.add_argument("--reranker", choices=["on", "off", "both"], default="on")
    ap.add_argument("--sledge", choices=["on", "off", "both"], default="off")
    ap.add_argument("--quickcheck", action="store_true")
    ap.add_argument("--nitpick", action="store_true")
    ap.add_argument("--repeats", type=int, default=1, help="Repeat each suite R times")
    ap.add_argument("--facts-limit", type=int, default=6)
    ap.add_argument("--no-minimize", action="store_true", help="Disable proof minimization (enabled by default)")
    ap.add_argument("--variants", action="store_true")
    # Model selection
    ap.add_argument("--model", type=str, default=None, help="Single Ollama model (overrides env OLLAMA_MODEL)")
    ap.add_argument("--models", type=str, default=None, help="Comma-separated Ollama models for ensemble")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle goal order per repeat (single-process)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for shuffling (0 = time-based)")

    args = ap.parse_args()

    # Resolve suites
    if args.file:
        suites: List[Tuple[str, Path]] = [(Path(args.file).stem, Path(args.file))]
    elif args.suite == "all":
        suites = [(name, path) for name, path in SUITE_MAP.items()]
    elif args.suite:
        suites = [(args.suite, SUITE_MAP[args.suite])]
    else:
        ap.error("Provide --suite (or 'all') or --file")

    # Parse models
    single_model: Optional[str] = args.model
    models_ensemble: Optional[List[str]] = None
    if args.models:
        models_ensemble = [m.strip() for m in args.models.split(",") if m.strip()]

    # Start Isabelle once
    server_info, proc = start_isabelle_server(name="isabelle", log_file="bench_server.log")
    print(server_info.strip())
    isabelle = get_isabelle_client(server_info)
    session_id = isabelle.session_start(session="HOL")
    print("session_id:", session_id)

    def build_cfg(rerank_on: bool, sledge_on: bool) -> BenchConfig:
        return BenchConfig(
            name=f"rerank_{'on' if rerank_on else 'off'}__sledge_{'on' if sledge_on else 'off'}",
            beam=args.beam,
            max_depth=args.max_depth,
            budget_s=args.budget_s,
            reranker=rerank_on,
            sledge=sledge_on,
            sledge_timeout=5,
            sledge_every=2,
            quickcheck=args.quickcheck,
            quickcheck_timeout=2,
            quickcheck_every=1,
            nitpick=args.nitpick,
            nitpick_timeout=5,
            nitpick_every=2,
            facts_limit=args.facts_limit,
            minimize = not args.no_minimize,
            variants=args.variants,
        )

    def run_suite(suite_name: str, goals_path: Path):
        goals = read_goals_file(goals_path)
        if not goals:
            print(f"[SKIP] No goals in {goals_path}")
            return

        print(f"\n=== Running suite: {suite_name} ({len(goals)} goals) ===")

        rerank_opts = {"on": [True], "off": [False], "both": [False, True]}[args.reranker]
        sledge_opts = {"on": [True], "off": [False], "both": [False, True]}[args.sledge]

        for rr in rerank_opts:
            for sh in sledge_opts:
                cfg = build_cfg(rr, sh)

                # Tag config name with model info
                if models_ensemble:
                    model_tag = "ensemble_" + "_".join(models_ensemble)
                elif single_model:
                    model_tag = single_model
                else:
                    model_tag = os.environ.get("OLLAMA_MODEL", "env_default")
                cfg.name = f"{cfg.name}__model_{model_tag}"

                import random, time
                base_seed = args.seed or int(time.time())

                rows: List[Dict[str, Any]] = []
                for r in range(args.repeats):
                    goals_run = list(goals)
                    if args.shuffle:
                        rnd = random.Random(base_seed + r)  # different order per repeat but reproducible
                        rnd.shuffle(goals_run)
                    for i, g in enumerate(goals_run, 1):
                        print(f"[{cfg.name}] (run {r+1}/{args.repeats}) [{i}/{len(goals_run)}] {g}")
                        rows.append(run_one_goal(isabelle, session_id, g, cfg, single_model, models_ensemble))

                pretty_report(suite_name, cfg, rows)
                out = write_csv(suite_name, cfg, rows)
                print(f"CSV → {out}")

    # Run all suites requested
    for suite_name, path in suites:
        run_suite(suite_name, path)

    # after you're done using Isabelle
    try:
        isabelle.shutdown()
    except Exception:
        pass

    # ensure the server process is really gone
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=3)
        except Exception:
            pass

    # (optional) close any asyncio child watcher to avoid late logging
    import sys
    if sys.platform != "win32":
        try:
            import asyncio
            w = asyncio.get_event_loop_policy().get_child_watcher()
            if w is not None:
                w.close()
        except Exception:
            pass



if __name__ == "__main__":
    main()
