# prover/experiments.py
"""
Unified experiments tool:
  • Bench:    run suites/files and write CSVs
  • Regress:  compare a fresh run against a saved baseline JSON
  • Aggregate: summarize CSVs into readable tables

Examples
--------
Bench:
  python -m prover.experiments bench --suite lists --beam 3 --timeout 6 --reranker both
  python -m prover.experiments bench --file datasets/lists.txt --model 'qwen3-coder:30b'

Regress:
  python -m prover.experiments regress --suite lists --baseline baselines/lists.json
  python -m prover.experiments regress --file datasets/lists.txt --save-baseline baselines/lists.json

Aggregate:
  python -m prover.experiments aggregate --dir datasets/results --best-only --top-k 3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as stats
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .isabelle_api import start_isabelle_server, get_isabelle_client
from .prover import prove_goal

# ---------- Common paths ----------
BENCH_DIR = Path("datasets")
RESULTS_DIR = BENCH_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUITE_MAP = {
    "lists": BENCH_DIR / "lists.txt",
    "nat":   BENCH_DIR / "nat.txt",
    "sets":  BENCH_DIR / "sets.txt",
    "logic": BENCH_DIR / "logic.txt",
}

# Precompile once for small speedup on large files
import re
_LEMMA_RE = re.compile(r'lemma\\s+"(.+)"', re.IGNORECASE)

# ---------- Shared goal IO ----------
def _read_goals_file(path: Path) -> List[str]:
    goals: List[str] = []
    if not path.exists():
        return goals
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.lower().startswith("lemma "):
                m = _LEMMA_RE.search(s)
                if m:
                    goals.append(m.group(1))
                else:
                    payload = s[len("lemma "):].strip().strip('"')
                    goals.append(payload)
            else:
                goals.append(s.strip('"'))
    return goals

# =============================================================================
# BENCH
# =============================================================================
@dataclass(slots=True)
class BenchConfig:
    name: str
    beam: int
    max_depth: int
    timeout: int
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

def _bench_run_one(
    isabelle,
    session_id: str,
    goal: str,
    cfg: BenchConfig,
    single_model: Optional[str],
    models_list: Optional[List[str]],
) -> Dict[str, Any]:
    model_default = os.environ.get("OLLAMA_MODEL", "qwen3-coder:30b")
    res = prove_goal(
        isabelle, session_id, goal,
        model_name_or_ensemble=(single_model or model_default),
        beam_w=cfg.beam, max_depth=cfg.max_depth, hint_lemmas=6, timeout=cfg.timeout,
        models=models_list, save_dir=None,
        use_sledge=cfg.sledge, sledge_timeout=cfg.sledge_timeout, sledge_every=cfg.sledge_every,
        trace=False, use_color=False,
        use_qc=cfg.quickcheck, qc_timeout=cfg.quickcheck_timeout, qc_every=cfg.quickcheck_every,
        use_np=cfg.nitpick,    np_timeout=cfg.nitpick_timeout,    np_every=cfg.nitpick_every,
        facts_limit=cfg.facts_limit,
        do_minimize=cfg.minimize, minimize_timeout=8,
        do_variants=cfg.variants, variant_timeout=6, variant_tries=24,
        enable_reranker=cfg.reranker,
    )
    return {
        "goal": goal,
        "success": bool(res.get("success", False)),
        "depth": int(res.get("depth", -1)),
        "elapsed_s": float(res.get("elapsed_s", 0.0)),
        "model": str(res.get("model", "")),
        "timeout": bool(res.get("timeout", False)),
        "use_calls": int(res.get("use_calls", 0)),
        "steps_len": len(res.get("steps", []) or []),
    }

def _bench_summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n_goals": 0, "n_success": 0, "success_rate": 0.0, "median_time_all": 0.0, "median_time_success": 0.0, "avg_depth": 0.0}
    succ = [r for r in rows if r["success"]]
    times = [r["elapsed_s"] for r in rows]
    succ_times = [r["elapsed_s"] for r in succ]
    depths = [r["depth"] for r in rows]
    return {
        "n_goals": n,
        "n_success": len(succ),
        "success_rate": (len(succ)/n),
        "median_time_all": stats.median(times) if times else 0.0,
        "median_time_success": stats.median(succ_times) if succ_times else 0.0,
        "avg_depth": (sum(depths)/n),
    }

def _bench_write_csv(suite_name: str, cfg_name: str, rows: List[Dict[str, Any]]) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    safe_tag = cfg_name.replace(" ", "_")
    out = RESULTS_DIR / f"{ts}-{suite_name}-{safe_tag}.csv"
    if not rows:
        # Create header-only file for traceability
        with out.open("w", newline="", encoding="utf-8") as f:
            f.write("goal,success,depth,elapsed_s,model,timeout,use_calls,steps_len\\n")
        return out
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return out

def cmd_bench(args: argparse.Namespace) -> None:
    # Resolve suites
    if args.file:
        suites: List[Tuple[str, Path]] = [(Path(args.file).stem, Path(args.file))]
    elif args.suite == "all":
        suites = list(SUITE_MAP.items())
    else:
        suites = [(args.suite, SUITE_MAP[args.suite])]

    # Models
    single_model = args.model
    models_ensemble = [m.strip() for m in args.models.split(",")] if args.models else None

    # Start Isabelle once
    server_info, proc = start_isabelle_server(name="isabelle", log_file="bench_server.log")
    print(server_info.strip())
    isabelle = get_isabelle_client(server_info)
    session_id = isabelle.session_start(session="HOL")
    print("session_id:", session_id)

    try:
        def build_cfg(rerank_on: bool, sledge_on: bool) -> BenchConfig:
            return BenchConfig(
                name=f"rerank_{'on' if rerank_on else 'off'}__sledge_{'on' if sledge_on else 'off'}",
                beam=args.beam, max_depth=args.max_depth, timeout=args.timeout,
                reranker=rerank_on, sledge=sledge_on, sledge_timeout=5, sledge_every=2,
                quickcheck=args.quickcheck, quickcheck_timeout=2, quickcheck_every=1,
                nitpick=args.nitpick, nitpick_timeout=5, nitpick_every=2,
                facts_limit=args.facts_limit, minimize=(not args.no_minimize),
                variants=args.variants,
            )

        import random
        base_seed = args.seed or int(time.time())

        for suite_name, goals_path in suites:
            goals = _read_goals_file(goals_path)
            if not goals:
                print(f"[SKIP] No goals in {goals_path}")
                continue

            print(f"\\n=== Running suite: {suite_name} ({len(goals)} goals) ===")
            rr_opts = {"on": [True], "off": [False], "both": [False, True]}[args.reranker]
            # NEW: --sledge is a simple boolean flag
            sh_opts = [bool(args.sledge)]

            for rr in rr_opts:
                for sh in sh_opts:
                    cfg = build_cfg(rr, sh)
                    # Model tag
                    if models_ensemble:
                        model_tag = "ensemble_" + "_".join(models_ensemble)
                    elif single_model:
                        model_tag = single_model
                    else:
                        model_tag = os.environ.get("OLLAMA_MODEL", "env_default")
                    cfg_name = f"{cfg.name}__model_{model_tag}"

                    rows: List[Dict[str, Any]] = []
                    for r in range(args.repeats):
                        goals_run = list(goals)
                        if args.shuffle:
                            rnd = random.Random(base_seed + r)
                            rnd.shuffle(goals_run)
                        for i, g in enumerate(goals_run, 1):
                            print(f"[{cfg_name}] (run {r+1}/{args.repeats}) [{i}/{len(goals_run)}] {g}")
                            rows.append(_bench_run_one(isabelle, session_id, g, cfg, single_model, models_ensemble))

                    s = _bench_summarize(rows)
                    def pct(x: float) -> str: return f"{x*100:.1f}%"
                    print("\\n=== Benchmark Report ===")
                    print(f"Suite:  {suite_name}")
                    print(f"Config: {cfg_name}")
                    print(f"  Success: {s['n_success']} / {s['n_goals']}  ({pct(s['success_rate'])})")
                    print(f"  Median time (all): {s['median_time_all']:.2f}s | Avg depth: {s['avg_depth']:.2f}")
                    out = _bench_write_csv(suite_name, cfg_name, rows)
                    print(f"CSV → {out}")
    finally:
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

# =============================================================================
# REGRESS
# =============================================================================
@dataclass(slots=True)
class OneGoal:
    goal: str; success: bool; elapsed_s: float; depth: int; timeout: bool; model: str; use_calls: int; steps_len: int

@dataclass(slots=True)
class Summary:
    suite: str; config: str; n_goals: int; n_success: int; success_rate: float
    median_time_all: float; median_time_success: float; avg_depth: float; stamp: str

@dataclass(slots=True)
class Report:
    suite: str; config: str; params: Dict[str, Any]; goals: List[OneGoal]; summary: Summary

def _reg_summarize(suite: str, config: str, rows: List[OneGoal]) -> Summary:
    if not rows:
        return Summary(suite=suite, config=config, n_goals=0, n_success=0, success_rate=0.0,
                       median_time_all=0.0, median_time_success=0.0, avg_depth=0.0,
                       stamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    succ = [r for r in rows if r.success]
    times_all = [r.elapsed_s for r in rows]
    times_succ = [r.elapsed_s for r in succ]
    depths = [r.depth for r in rows]
    return Summary(
        suite=suite, config=config, n_goals=len(rows), n_success=len(succ),
        success_rate=(len(succ)/len(rows)),
        median_time_all=stats.median(times_all) if times_all else 0.0,
        median_time_success=stats.median(times_succ) if times_succ else 0.0,
        avg_depth=(sum(depths)/len(rows)) if rows else 0.0,
        stamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

def _reg_read_goals(path: Path) -> List[str]:
    return _read_goals_file(path)

def _reg_save_report(path: Path, rep: Report) -> None:
    data = {
        "suite": rep.suite, "config": rep.config, "params": rep.params,
        "summary": asdict(rep.summary), "goals": [asdict(g) for g in rep.goals],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _reg_load_baseline(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def _reg_compare(current: Report, baseline_data: Dict[str, Any], *, tol_rate: float, tol_time: float) -> bool:
    cur = current.summary
    base = baseline_data.get("summary", {})
    b_rate = float(base.get("success_rate", 0.0))
    b_med_all = float(base.get("median_time_all", 0.0))
    b_succ = int(base.get("n_success", 0))
    b_n = int(base.get("n_goals", 0))

    print("\\n=== Regression comparison ===")
    print(f"Suite:   {current.suite}")
    print(f"Config:  {current.config}")
    print(f"Goals:   baseline {b_n}, current {cur.n_goals}")
    print(f"Success: baseline {b_succ} / {b_n} ({b_rate*100:.1f}%), current {cur.n_success} / {cur.n_goals} ({cur.success_rate*100:.1f}%)")
    print(f"Median time (all): baseline {b_med_all:.2f}s, current {cur.median_time_all:.2f}s")

    regressed = False
    if cur.success_rate + tol_rate < b_rate:
        print(f"⚠️  Success rate drop exceeds tolerance ({b_rate*100:.1f}% → {cur.success_rate*100:.1f}%, tol={tol_rate*100:.1f}%)")
        regressed = True
    if b_med_all > 0 and (cur.median_time_all - b_med_all) > tol_time:
        print(f"⚠️  Median time increased by {cur.median_time_all - b_med_all:.2f}s (tol={tol_time:.2f}s)")
        regressed = True
    if not regressed:
        print("✅ No regression detected within tolerances.")
    return regressed

def cmd_regress(args: argparse.Namespace) -> None:
    if args.file:
        suite_name = Path(args.file).stem
        goals_path = Path(args.file)
    else:
        suite_name = args.suite
        goals_path = SUITE_MAP[suite_name]

    models_list = [m.strip() for m in args.models.split(",")] if args.models else None

    server_info, proc = start_isabelle_server(name="isabelle", log_file="regress_server.log")
    print(server_info.strip())
    isabelle = get_isabelle_client(server_info)
    session_id = isabelle.session_start(session="HOL")
    print("session_id:", session_id)

    try:
        goals = _reg_read_goals(goals_path)
        if not goals:
            raise SystemExit(f"No goals found in {goals_path}")

        cfg = {
            "beam": args.beam, "max_depth": args.max_depth, "timeout": args.timeout,
            "reranker": True, "sledge": args.sledge,
            "quickcheck": args.quickcheck, "nitpick": args.nitpick,
            "facts_limit": args.facts_limit, "minimize": (not args.no_minimize),
            "variants": args.variants, "model": args.model, "models": models_list,
            "shuffle_seed": (args.shuffle_seed if args.shuffle_seed is not None else (args.seed if args.shuffle else -1)),
        }
        model_tag = ("ensemble_" + "_".join(models_list)) if models_list else (args.model or os.environ.get("OLLAMA_MODEL", "env_default"))
        config_name = f"beam{args.beam}_d{args.max_depth}_t{args.timeout}_rron_sdg{'on' if args.sledge else 'off'}__model_{model_tag}"

        rows: List[OneGoal] = []
        import random
        gs = list(goals)
        # Back-compat path: if legacy --shuffle-seed was given, use it
        if args.shuffle_seed is not None:
            if args.shuffle_seed != -1:
                rnd = random.Random(args.shuffle_seed or int(time.time()))
                rnd.shuffle(gs)
        else:
            # New path: --shuffle / --seed
            if args.shuffle:
                rnd = random.Random(args.seed or int(time.time()))
                rnd.shuffle(gs)

        for i, g in enumerate(gs, 1):
            print(f"[{suite_name}] [{i}/{len(gs)}] {g}")
            res = prove_goal(
                isabelle, session_id, g,
                model_name_or_ensemble=(args.model or os.environ.get("OLLAMA_MODEL", "qwen3-coder:30b")),
                beam_w=args.beam, max_depth=args.max_depth, hint_lemmas=6, timeout=args.timeout,
                models=models_list, save_dir=None,
                use_sledge=args.sledge, sledge_timeout=5, sledge_every=2,
                trace=False, use_color=False,
                use_qc=args.quickcheck, qc_timeout=2, qc_every=1,
                use_np=args.nitpick, np_timeout=5, np_every=2,
                facts_limit=args.facts_limit,
                do_minimize=(not args.no_minimize), minimize_timeout=8,
                do_variants=args.variants, variant_timeout=6, variant_tries=24,
                enable_reranker=True,
            )
            rows.append(OneGoal(
                goal=g, success=bool(res.get("success", False)), elapsed_s=float(res.get("elapsed_s", 0.0)),
                depth=int(res.get("depth", -1)), timeout=bool(res.get("timeout", False)),
                model=str(res.get("model","")), use_calls=int(res.get("use_calls",0)),
                steps_len=len(res.get("steps", []) or []),
            ))

        summ = _reg_summarize(suite_name, config_name, rows)
        rep = {
            "suite": suite_name, "config": config_name, "params": cfg,
            "summary": asdict(summ), "goals": [asdict(r) for r in rows],
        }

        if args.out:
            out_p = Path(args.out); out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(rep, indent=2), encoding="utf-8")
            print(f"Wrote report → {out_p}")

        if args.save_baseline:
            Path(args.save_baseline).write_text(json.dumps(rep, indent=2), encoding="utf-8")
            print(f"Saved baseline → {args.save_baseline}")
            return

        if args.baseline:
            base = _reg_load_baseline(Path(args.baseline))
            if not base:
                print(f"Baseline not found or unreadable: {args.baseline}")
                sys.exit(2)
            regressed = _reg_compare(
                current=Report(
                    suite=suite_name,
                    config=config_name,
                    params=cfg,
                    goals=[OneGoal(**g) for g in rep["goals"]],
                    summary=summ,
                ),
                baseline_data=base,
                tol_rate=args.tol_rate,
                tol_time=args.tol_time,
            )
            sys.exit(1 if regressed else 0)

        # No baseline compare: print summary
        print("\\n=== Regression run summary (no baseline) ===")
        print(f"Suite:        {summ.suite}")
        print(f"Config:       {summ.config}")
        print(f"Goals:        {summ.n_goals}")
        print(f"Success:      {summ.n_success} / {summ.n_goals}  ({summ.success_rate*100:.1f}%)")
        print(f"Median time:  all={summ.median_time_all:.2f}s, succ={summ.median_time_success:.2f}s")
        print(f"Avg depth:    {summ.avg_depth:.2f}")
    finally:
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

# =============================================================================
# AGGREGATE
# =============================================================================
@dataclass(slots=True)
class AggRow:
    suite: str; config: str; goal: str; success: bool; elapsed_s: float; depth: int; model: str; timeout: bool; use_calls: int; steps_len: int

def _agg_parse_filename(p: Path) -> Tuple[str, str]:
    name = p.stem; parts = name.split("-")
    if len(parts) < 3: return ("unknown", name)
    return parts[1], "-".join(parts[2:])

def _agg_load(results_dir: Path) -> List[AggRow]:
    rows: List[AggRow] = []
    for p in sorted(results_dir.glob("*.csv")):
        suite, cfg = _agg_parse_filename(p)
        try:
            with p.open("r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    try:
                        rows.append(AggRow(
                            suite=suite, config=cfg, goal=r.get("goal",""),
                            success=str(r.get("success","")).strip().lower() in ("1","true","yes","y"),
                            elapsed_s=float(r.get("elapsed_s","0") or 0.0),
                            depth=int(r.get("depth","-1") or -1),
                            model=r.get("model",""),
                            timeout=str(r.get("timeout","")).strip().lower() in ("1","true","yes","y"),
                            use_calls=int(r.get("use_calls","0") or 0),
                            steps_len=int(r.get("steps_len","0") or 0),
                        ))
                    except Exception:
                        continue
        except FileNotFoundError:
            continue
    return rows

@dataclass(slots=True)
class AggSummary:
    suite: str; config: str; n: int; succ: int; rate: float; med_all: float; med_succ: float; avg_depth: float

def _agg_summarize(rows: List[AggRow]) -> AggSummary:
    if not rows:
        return AggSummary("", "", 0, 0, 0.0, 0.0, 0.0, 0.0)
    suite = rows[0].suite; cfg = rows[0].config
    n = len(rows); succ_rows = [r for r in rows if r.success]; succ = len(succ_rows)
    rate = (succ / n) if n else 0.0
    med_all = stats.median([r.elapsed_s for r in rows]) if rows else 0.0
    med_succ = stats.median([r.elapsed_s for r in succ_rows]) if succ_rows else 0.0
    avg_depth = sum(r.depth for r in rows) / n if n else 0.0
    return AggSummary(suite, cfg, n, succ, rate, med_all, med_succ, avg_depth)

def cmd_aggregate(args: argparse.Namespace) -> None:
    results_dir = Path(args.dir)
    if not results_dir.exists():
        print(f"No such directory: {results_dir}")
        return
    rows = _agg_load(results_dir)
    if not rows:
        print(f"No CSV rows found in {results_dir}")
        return
    # group by (suite, config)
    by: Dict[Tuple[str,str], List[AggRow]] = {}
    for r in rows:
        by.setdefault((r.suite, r.config), []).append(r)

    try:
        from tabulate import tabulate
        use_tab = True
    except Exception:
        use_tab = False

    print("\\n=== Benchmark summary (by suite, success ↓ then time ↑) ===")
    suites = sorted(set(s for (s, _) in by.keys()))
    for suite in suites:
        summaries = []
        for (s, cfg), rs in by.items():
            if s != suite: continue
            sm = _agg_summarize(rs)
            if sm.n >= args.min_rows:
                summaries.append(sm)
        if not summaries: continue
        summaries.sort(key=lambda x: (-x.rate, x.med_all, x.config))
        display = summaries[:args.top_k] if (args.best_only and args.top_k > 0) else summaries
        print(f"\\n[{suite}]")
        table = [
            [sm.config, f"{sm.rate*100:.1f}%", f"{sm.succ}/{sm.n}", f"{sm.med_all:.2f}", f"{sm.med_succ:.2f}" if sm.succ else "-", f"{sm.avg_depth:.2f}"]
            for sm in display
        ]
        headers = ["config", "succ_rate", "succ/total", "median_time_all(s)", "median_time_succ(s)", "avg_depth"]
        if use_tab: print(tabulate(table, headers=headers, tablefmt="github"))
        else:
            colw = [max(len(str(x)) for x in col) for x in zip(*([headers] + table))]
            def fmt_row(row: List[str]) -> str: return "  ".join(str(cell).ljust(w) for cell, w in zip(row, colw))
            print(fmt_row(headers)); print("  ".join("-"*w for w in colw))
            for r in table: print(fmt_row(r))

# =============================================================================
# CLI
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="Prover experiments (bench | regress | aggregate)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("bench", help="Run suites/files and write CSVs")
    pb.add_argument("--suite", type=str, choices=sorted(list(SUITE_MAP.keys()) + ["all"]))
    pb.add_argument("--file", type=str)
    pb.add_argument("--beam", type=int, default=3)
    pb.add_argument("--max-depth", type=int, default=8)
    pb.add_argument("--timeout", type=int, default=6)
    pb.add_argument("--reranker", choices=["on", "off", "both"], default="on")
    pb.add_argument("--sledge", action="store_true", help="Enable Sledgehammer during bench runs")
    pb.add_argument("--quickcheck", action="store_true")
    pb.add_argument("--nitpick", action="store_true")
    pb.add_argument("--repeats", type=int, default=1)
    pb.add_argument("--facts-limit", type=int, default=6)
    pb.add_argument("--no-minimize", action="store_true")
    pb.add_argument("--variants", action="store_true")
    pb.add_argument("--model", type=str, default=None)
    pb.add_argument("--models", type=str, default=None)
    pb.add_argument("--shuffle", action="store_true")
    pb.add_argument("--seed", type=int, default=0)
    pb.set_defaults(func=cmd_bench)

    pr = sub.add_parser("regress", help="Run and compare to baseline")
    pr.add_argument("--suite", type=str, choices=sorted(list(SUITE_MAP.keys())))
    pr.add_argument("--file", type=str, default=None)
    pr.add_argument("--beam", type=int, default=2)
    pr.add_argument("--max-depth", type=int, default=6)
    pr.add_argument("--timeout", type=int, default=8)
    pr.add_argument("--facts-limit", type=int, default=6)
    pr.add_argument("--reranker", choices=["on", "off", "both"], default="on")
    pr.add_argument("--no-minimize", action="store_true")
    pr.add_argument("--variants", action="store_true")
    pr.add_argument("--quickcheck", action="store_true")
    pr.add_argument("--nitpick", action="store_true")
    pr.add_argument("--sledge", action="store_true", help="Enable Sledgehammer during regress runs")
    pr.add_argument("--model", type=str, default=None)
    pr.add_argument("--models", type=str, default=None)
    pr.add_argument("--shuffle", action="store_true")
    pr.add_argument("--seed", type=int, default=0)
    # Back-compat: old flag (if provided, it overrides --shuffle/--seed)
    pr.add_argument("--shuffle-seed", type=int, default=None)
    pr.add_argument("--baseline", type=str)
    pr.add_argument("--save-baseline", type=str)
    pr.add_argument("--out", type=str, default=None)
    pr.add_argument("--tol-rate", type=float, default=0.00)
    pr.add_argument("--tol-time", type=float, default=2.0)
    pr.set_defaults(func=cmd_regress)

    pa = sub.add_parser("aggregate", help="Summarize CSVs")
    pa.add_argument("--dir", type=str, default=str(RESULTS_DIR))
    pa.add_argument("--min-rows", type=int, default=1)
    pa.add_argument("--best-only", action="store_true")
    pa.add_argument("--top-k", type=int, default=3)
    pa.set_defaults(func=cmd_aggregate)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
