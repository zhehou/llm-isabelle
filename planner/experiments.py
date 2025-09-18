# planner/experiments.py
"""
Planner experiments tool:
  • Bench:    run a file (or named suite) of goals and write CSVs
  • Regress:  compare a fresh run against a saved baseline JSON
  • Aggregate:summarize CSVs into readable tables

Success criteria
---------------
- mode="auto": success == (planner filled all holes; i.e., the returned Isar has NO 'sorry').
               This matches planner.driver.PlanAndFillResult.success semantics.
- mode="outline": success == True if an outline is produced (holes are expected).
                  Use --strict-no-sorry to force success only when outline is actually hole-free.

Optionally add --verify to compile the final Isar with Isabelle to double-check (best-effort).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics as stats
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import asyncio, atexit, gc
from asyncio.base_subprocess import BaseSubprocessTransport as _BST

# Planner + Isabelle
from planner.driver import plan_and_fill
from planner.skeleton import find_sorry_spans
from prover.isabelle_api import (
    start_isabelle_server,
    get_isabelle_client,
    build_theory,
    run_theory,
    finished_ok,
)

def _drain_and_close_loop(loop: asyncio.AbstractEventLoop | None) -> None:
    if not loop or loop.is_closed():
        return
    try:
        # cancel pending tasks so transports can close while loop is alive
        tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for t in tasks: t.cancel()
        if tasks:
            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        # drain async generators
        loop.run_until_complete(loop.shutdown_asyncgens())
    except Exception:
        pass
    finally:
        try:
            loop.close()
        except Exception:
            pass

def _close_client_loop_safely(client) -> None:
    loop = getattr(client, "loop", None) or getattr(client, "_loop", None)
    # Force GC first so transports’ __del__ run while loop is still alive
    gc.collect()
    _drain_and_close_loop(loop)

# last-ditch guard for interpreter teardown (silences only the closed-loop case)
_orig_del = _BST.__del__
def _quiet_bst_del(self, *a, **kw):
    try:
        _orig_del(self, *a, **kw)
    except RuntimeError as e:
        if "Event loop is closed" in str(e):
            return
        raise
_BST.__del__ = _quiet_bst_del

@atexit.register
def _planner_atexit_drain():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None
    # Drain/close any still-open default loop at process exit
    _drain_and_close_loop(loop)

# ---------- Common paths ----------
BENCH_DIR = Path("datasets")
RESULTS_DIR = BENCH_DIR / "planner_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# default structured log path (append-only JSONL)
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_LOG_PATH = LOGS_DIR / "planner.log.jsonl"

# Optional named suites
SUITE_MAP = {
    "lists": BENCH_DIR / "lists.txt",
    "nat":   BENCH_DIR / "nat.txt",
    "sets":  BENCH_DIR / "sets.txt",
    "logic": BENCH_DIR / "logic.txt",
}

# Precompile once for small speedup on large files
_LEMMA_RE = re.compile(r'lemma\s+"(.+)"', re.IGNORECASE)

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

# ---------- Logging helpers ----------
def _append_proof_log(log_path: Path, rec: Dict[str, Any]) -> None:
    """Append a single JSON record to the planner proof log; never crash the run."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

# ---------- Optional Isabelle verification ----------

_BLANKS = re.compile(r"\n[ \t]*\n[ \t]*\n+", re.MULTILINE)
_TRAIL_WS = re.compile(r"[ \t]+$", re.MULTILINE)

def _normalize_isar_for_verify(s: str) -> str:
    """
    Harmless whitespace/token tweaks for batch mode:
      - normalize newlines, strip trailing spaces
      - collapse 3+ blank lines to one
      - ensure 'have/show …' is immediately followed by its 'by …'
      - ensure 'using …' (for the same step) is adjacent to 'by …'
      - if there is a lone 'finally' without any 'also', drop the 'finally'
    We do NOT change any other tokens.
    """
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _TRAIL_WS.sub("", s)
    s = _BLANKS.sub("\n\n", s)

    # have/show ... <blank> by ...
    s = re.sub(r"(?m)^(?P<i>\s*)(have\b[^\n]*?)\n\s*\n(?P<j>\s*)by\b", r"\g<i>\g<2>\n\g<j>by", s)
    s = re.sub(r"(?m)^(?P<i>\s*)(show\b[^\n]*?)\n\s*\n(?P<j>\s*)by\b", r"\g<i>\g<2>\n\g<j>by", s)
    # using ... <blank> by ...
    s = re.sub(r"(?m)^(?P<i>\s*)(using\b[^\n]*?)\n\s*\n(?P<j>\s*)by\b", r"\g<i>\g<2>\n\g<j>by", s)

    # Orphan 'finally' -> drop keyword (keep 'show …')
    if re.search(r"(?m)^\s*finally\b", s) and not re.search(r"(?m)^\s*also\b", s):
        s = re.sub(r"(?m)^\s*finally\s+(?=show\b)", "", s)

    return s.strip() + "\n"

# ---- Session helpers (robust start with fallback) ----
def _pick_session_name() -> str:
    """Preferred session name from environment, defaulting to HOL."""
    return (os.environ.get("ISABELLE_LOGIC")
            or os.environ.get("ISABELLE_SESSION")
            or "HOL")

def _session_start_with_fallback(client):
    """
    Try the preferred session; on FAILED fall back to HOL.
    Returns (session_id, used_session_name).
    """
    wanted = _pick_session_name()
    try:
        sid = client.session_start(session=wanted)
        return sid, wanted
    except Exception as e:
        msg = str(e)
        # "FAILED" is what isabelle_client raises when the image isn't available.
        if "FAILED" in msg and wanted != "HOL":
            try:
                sid = client.session_start(session="HOL")
                print(f"[experiments] session '{wanted}' unavailable → falling back to 'HOL'")
                return sid, "HOL"
            except Exception:
                pass
        raise

def _responses_to_text(resps) -> str:
    chunks = []
    for r in (resps or []):
        body = getattr(r, "response_body", None)
        if isinstance(body, bytes):
            chunks.append(body.decode(errors="replace"))
        elif isinstance(body, str):
            chunks.append(body)
        else:
            chunks.append(str(r))
    return "\n".join(chunks)

def _verify_full_isar(isabelle, session_id: str, isar_text: str) -> Tuple[bool, str]:
    """
    Compile a full Isar theory and return (ok, brief_diag).
    This is a completely new implementation that properly parses Isabelle's JSON responses
    to avoid the false positives of previous versions.
    """
    try:
        # Step 1: Prepare and run the theory, assuming isar_text is a complete file.
        theory_lines = _normalize_isar_for_verify(isar_text).splitlines()
        if not theory_lines:
            return False, "Empty proof provided."
            
        thy = build_theory(theory_lines, add_print_state=False, end_with=None)
        resps = run_theory(isabelle, session_id, thy)

        # Step 2: Intelligently parse responses instead of naive string matching.
        # We look for the final summary message from Isabelle.
        final_summary = None
        all_errors = []
        for r in reversed(resps or []):
            body = getattr(r, "response_body", None)
            if not isinstance(body, (str, bytes)):
                continue
            
            text_body = body.decode(errors="replace") if isinstance(body, bytes) else body
            try:
                # A summary will be a JSON object with "ok" and "errors" keys.
                data = json.loads(text_body)
                if isinstance(data, dict):
                    if "ok" in data and "errors" in data:
                        final_summary = data
                        break # We found the main summary.
                    # Also collect individual error messages.
                    if data.get("kind") == "error" and "message" in data:
                        all_errors.append(data["message"])

            except json.JSONDecodeError:
                continue # This response was not a valid JSON object.

        # Step 3: Make a final decision based on the parsed summary.
        if final_summary:
            is_ok = final_summary.get("ok", False)
            errors_list = final_summary.get("errors", [])
            if is_ok and not errors_list:
                return True, ""  # Definitive success.
            else:
                # Definitive failure. Format the error message.
                error_msgs = [e.get("message", "Unknown error") for e in errors_list]
                diag = "Isabelle reported failure:\n" + "\n".join(error_msgs)
                return False, diag

        # Step 4: Fallback if no JSON summary was found (e.g., older Isabelle version).
        # Check for legacy error markers.
        all_txt = _responses_to_text(resps)
        if any(e in all_txt for e in ("*** Error:", "*** Outer syntax error", "*** Failed")):
             return False, f"[Legacy error detected]\n{all_txt[-1000:]}"
        
        # If no errors were found, and we saw signs of completion, assume success.
        if "100%" in all_txt or "theory processed" in all_txt:
            return True, ""

        # If all else fails, we have to assume failure.
        diag = "Verification inconclusive. No summary found."
        if all_errors:
            diag += "\nDetected errors:\n" + "\n".join(all_errors)
        return False, diag

    except Exception as e:
        return False, f"verify_error: {type(e).__name__}: {e}"


# Classify errors that are transport-level (server died/socket closed) vs. “real” Isabelle failures
_TRANSPORT_HINTS = (
    "ConnectionRefusedError",
    "ConnectionResetError",
    "BrokenPipeError",
    "Connect call failed",
    "Event loop is closed",
    "Server not running",
    "Connection aborted",
    "ConnectionError",
)

def _is_transport_error(msg: str) -> bool:
    if not msg:
        return False
    m = msg if isinstance(msg, str) else str(msg)
    return any(hint in m for hint in _TRANSPORT_HINTS)

def _verify_with_auto_restart(isabelle, session_id: str, isar_text: str) -> Tuple[bool, str]:
    """
    Try verification once on the given (isabelle, session_id).
    If a transport error occurs, spin up a *temporary* Isabelle server/session,
    retry verification once, then tear the temp server down.
    """
    ok, details = _verify_full_isar(isabelle, session_id, isar_text)
    if ok or not _is_transport_error(details):
        return ok, details

    # Transport error: retry with a fresh, temporary server
    try:
        server_info, proc = start_isabelle_server(name="planner-verify-retry", log_file="planner_verify_retry.log")
        client = get_isabelle_client(server_info)
        new_session, _used = _session_start_with_fallback(client)
        print("session_id:", new_session, "| session:", _used)
        try:
            ok2, details2 = _verify_full_isar(client, new_session, isar_text)
        finally:
            try:
                client.shutdown()
                try: _close_client_loop_safely(client)
                except Exception: pass

            except Exception:
                pass
            try:
                proc.terminate(); proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill(); proc.wait(timeout=2)
                except Exception:
                    pass
        # Prefer the second attempt’s diagnostics
        if ok2:
            return True, ""
        return False, (details2 or "verify_failed_after_restart")
    except Exception as e:
        # Still a transport problem (or startup failed)
        return False, f"transport_error_after_restart: {type(e).__name__}: {e}"

# ---------- plan_and_fill safety wrapper ----------
def _safe_plan_and_fill(*, goal: str, model: Optional[str], cfg) -> Tuple[Optional[Any], str]:
    """
    Run planner.plan_and_fill and trap any provider/network exceptions.
    Returns (res, err_text). When res is None, the caller should mark failure but continue the run.
    """
    try:
        res = plan_and_fill(
            goal,
            model=model,
            timeout=cfg.timeout,
            mode=cfg.mode,
            outline_k=(cfg.k),
            outline_temps=cfg.temps,
            legacy_single_outline=(cfg.k == 1),
            repairs=cfg.repairs,
            max_repairs_per_hole=cfg.max_repairs_per_hole,
            repair_trace=cfg.repair_trace,
            priors_path=cfg.priors,
            context_hints=cfg.context_hints,
            lib_templates=cfg.lib_templates,
            alpha=cfg.alpha, beta=cfg.beta, gamma=cfg.gamma,
            hintlex_path=cfg.hintlex, hintlex_top=cfg.hintlex_top,
        )
        return res, ""
    except Exception as e:
        tb = "".join(traceback.format_exception_only(type(e), e)).strip()
        return None, tb

# =============================================================================
# BENCH
# =============================================================================
@dataclass(slots=True)
class BenchConfig:
    mode: str                      # "auto" or "outline"
    timeout: int
    k: int
    temps: Optional[List[float]]
    repairs: bool
    max_repairs_per_hole: int
    repair_trace: bool
    priors: Optional[str]
    context_hints: bool
    lib_templates: bool
    alpha: float
    beta: float
    gamma: float
    hintlex: Optional[str]
    hintlex_top: int
    strict_no_sorry: bool
    verify: bool

@dataclass(slots=True)
class BenchRow:
    goal: str
    success: bool
    elapsed_s: float
    mode: str
    model: str
    outline_chars: int
    fills: int
    failed_holes: int
    had_sorry: bool
    verified_ok: bool

def _bench_run_one(
    isabelle,
    session_id: str,
    goal: str,
    cfg: BenchConfig,
    model: Optional[str],
    diverse: bool,
) -> Tuple[BenchRow, str, str]:
    """
    Returns (metrics_row, outline_text, verify_details) so the caller can persist the proof text.
    """
    t0 = time.time()

    # call through safety wrapper (prevents run-wide crashes on provider timeouts)
    call_cfg = BenchConfig(**{k: getattr(cfg, k) for k in cfg.__dataclass_fields__.keys()})
    call_cfg.k = (cfg.k if diverse else 1)
    res, plan_err = _safe_plan_and_fill(goal=goal, model=model, cfg=call_cfg)

    dt = time.time() - t0

    if res is None:
        # Planner crashed for this goal: mark as failure, log error, skip verify
        outline_text = ""
        had_sorry = False
        success = False
        verified_ok = True
        verify_details = f"[planner_error] {plan_err}"
        row = BenchRow(
            goal=goal,
            success=success,
            elapsed_s=float(dt),
            mode=cfg.mode,
            model=(model or os.environ.get("OLLAMA_MODEL", "env_default")),
            outline_chars=0,
            fills=0,
            failed_holes=0,
            had_sorry=had_sorry,
            verified_ok=verified_ok,
        )
        return row, outline_text, verify_details

    outline_text = res.outline or ""
    had_sorry = bool(find_sorry_spans(outline_text))

    # Decide success from *artifact* first, not from res.success
    if cfg.mode == "auto":
        # Auto expects all holes filled → success iff no 'sorry'
        success = (not had_sorry)
    else:
        # Outline mode: success iff we produced an outline; optionally require hole-free
        success = True
        if cfg.strict_no_sorry:
            success = success and (not had_sorry)

    verified_ok = True
    verify_details = ""
    verify_status = "skipped"  # skipped | ok | fail | transport_error

    if cfg.verify and not had_sorry:
        verified_ok, verify_details = _verify_with_auto_restart(isabelle, session_id, outline_text)        
        if verified_ok:
            verify_status = "ok"
            success = True  # verification is authoritative
        else:
            if _is_transport_error(verify_details):
                # Transport issues = inconclusive; keep current success decision
                verify_status = "transport_error"
            else:
                verify_status = "fail"
                success = False

    row = BenchRow(
        goal=goal,
        success=success,
        elapsed_s=float(dt),
        mode=cfg.mode,
        model=(model or os.environ.get("OLLAMA_MODEL", "env_default")),
        outline_chars=len(outline_text),
        fills=len(getattr(res, "fills", []) or []),
        failed_holes=len(getattr(res, "failed_holes", []) or []),
        had_sorry=had_sorry,
        verified_ok=bool(verified_ok),
    )
    # Return verify_status via the third string so callers can log it
    return row, outline_text, (f"[{verify_status}] {verify_details}" if verify_details or verify_status != "skipped" else "")

def _bench_summarize(rows: List[BenchRow]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "n_goals": 0, "n_success": 0, "success_rate": 0.0,
            "median_time_all": 0.0, "median_time_success": 0.0,
            "median_outline_chars": 0, "median_fills": 0,
        }
    succ = [r for r in rows if r.success]
    times = [r.elapsed_s for r in rows]
    succ_times = [r.elapsed_s for r in succ]
    outlines = [r.outline_chars for r in rows]
    fills = [r.fills for r in rows]
    return {
        "n_goals": n,
        "n_success": len(succ),
        "success_rate": (len(succ) / n) if n else 0.0,
        "median_time_all": stats.median(times) if times else 0.0,
        "median_time_success": stats.median(succ_times) if succ_times else 0.0,
        "median_outline_chars": int(stats.median(outlines)) if outlines else 0,
        "median_fills": int(stats.median(fills)) if fills else 0,
    }

def _bench_write_csv(suite_name: str, cfg_name: str, rows: List[BenchRow]) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    safe_tag = cfg_name.replace(" ", "_")
    out = RESULTS_DIR / f"{ts}-{suite_name}-{safe_tag}.csv"
    headers = ["goal", "success", "elapsed_s", "mode", "model", "outline_chars", "fills", "failed_holes", "had_sorry", "verified_ok"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({
                "goal": r.goal,
                "success": r.success,
                "elapsed_s": f"{r.elapsed_s:.4f}",
                "mode": r.mode,
                "model": r.model,
                "outline_chars": r.outline_chars,
                "fills": r.fills,
                "failed_holes": r.failed_holes,
                "had_sorry": r.had_sorry,
                "verified_ok": r.verified_ok,
            })
    return out

def cmd_bench(args: argparse.Namespace) -> None:
    # Resolve suites
    if args.file:
        suites: List[Tuple[str, Path]] = [(Path(args.file).stem, Path(args.file))]
    elif args.suite == "all":
        suites = list(SUITE_MAP.items())
    else:
        suites = [(args.suite, SUITE_MAP[args.suite])]

    # Start Isabelle once (we also optionally use it to verify proofs)
    server_info, proc = start_isabelle_server(name="planner", log_file="planner_bench.log")
    print(server_info.strip())
    isabelle = get_isabelle_client(server_info)
    session_id, used_session = _session_start_with_fallback(isabelle)
    print("session_id:", session_id, "| session:", used_session)

    try:
        import random
        base_seed = args.seed or int(time.time())

        for suite_name, goals_path in suites:
            goals = _read_goals_file(goals_path)
            if not goals:
                print(f"[SKIP] No goals in {goals_path}")
                continue
            print(f"\n=== Planner suite: {suite_name} ({len(goals)} goals) ===")

            cfg = BenchConfig(
                mode=args.mode, timeout=args.timeout,
                k=args.k, temps=([float(x) for x in args.temps.split(",")] if args.temps else None),
                repairs=(not args.no_repairs),
                max_repairs_per_hole=args.max_repairs_per_hole,
                repair_trace=args.repair_trace,
                priors=args.priors,
                context_hints=args.context_hints,
                lib_templates=args.lib_templates,
                alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                hintlex=args.hintlex, hintlex_top=args.hintlex_top,
                strict_no_sorry=args.strict_no_sorry,
                verify=args.verify,
            )

            model_tag = args.model or os.environ.get("OLLAMA_MODEL", "env_default")
            cfg_name = (
                f"mode_{cfg.mode}"
                f"__k{cfg.k if args.diverse else 1}"
                f"__t{cfg.timeout}"
                f"__repairs_{'on' if cfg.repairs else 'off'}"
                f"__model_{model_tag}"
                f"{'__verify' if cfg.verify else ''}"
                f"{'__strict' if cfg.strict_no_sorry else ''}"
            )

            rows: List[BenchRow] = []
            for r in range(args.repeats):
                goals_run = list(goals)
                if args.shuffle:
                    rnd = random.Random(base_seed + r)
                    rnd.shuffle(goals_run)
                for i, g in enumerate(goals_run, 1):
                    print(f"[{cfg_name}] (run {r+1}/{args.repeats}) [{i}/{len(goals_run)}] {g}")
                    row, outline_text, verify_details = _bench_run_one(
                        isabelle, session_id, g, cfg,
                        model=args.model, diverse=bool(args.diverse)
                    )
                    rows.append(row)

                    # append proof log (always)
                    _append_proof_log(
                        Path(args.log_path or DEFAULT_LOG_PATH),
                        {
                            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "cmd": "bench",
                            "suite": suite_name,
                            "config": cfg_name,
                            "run_index": r,
                            "goal_index": i - 1,
                            "model": model_tag,
                            "mode": cfg.mode,
                            "timeout": cfg.timeout,
                            "k": (cfg.k if args.diverse else 1),
                            "temps": (args.temps or None),
                            "repairs": cfg.repairs,
                            "max_repairs_per_hole": cfg.max_repairs_per_hole,
                            "priors": cfg.priors,
                            "context_hints": cfg.context_hints,
                            "lib_templates": cfg.lib_templates,
                            "hintlex": cfg.hintlex,
                            "hintlex_top": cfg.hintlex_top,
                            "strict_no_sorry": cfg.strict_no_sorry,
                            "verify": cfg.verify,
                            "goal": g,
                            "success": row.success,
                            "had_sorry": row.had_sorry,
                            "verified_ok": row.verified_ok,
                            "verify_details": (verify_details or "")[:1000],
                            "elapsed_s": row.elapsed_s,
                            "outline_chars": row.outline_chars,
                            "fills": row.fills,
                            "failed_holes": row.failed_holes,
                            "outline": outline_text,
                        }
                    )

            s = _bench_summarize(rows)
            def pct(x: float) -> str: return f"{x*100:.1f}%"
            print("\n=== Planner Benchmark Report ===")
            print(f"Suite:  {suite_name}")
            print(f"Config: {cfg_name}")
            print(f"  Success: {s['n_success']} / {s['n_goals']}  ({pct(s['success_rate'])})")
            print(f"  Median time (all): {s['median_time_all']:.2f}s | Median fills: {s['median_fills']}")
            out = _bench_write_csv(suite_name, cfg_name, rows)
            print(f"CSV → {out}")
    finally:
        try:
            isabelle.shutdown()
            try: _close_client_loop_safely(isabelle)
            except Exception: pass
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
    goal: str
    success: bool
    elapsed_s: float
    mode: str
    model: str
    outline_chars: int
    fills: int
    failed_holes: int
    had_sorry: bool
    verified_ok: bool

@dataclass(slots=True)
class Summary:
    suite: str
    config: str
    n_goals: int
    n_success: int
    success_rate: float
    median_time_all: float
    median_time_success: float
    stamp: str

@dataclass(slots=True)
class Report:
    suite: str
    config: str
    params: Dict[str, Any]
    goals: List[OneGoal]
    summary: Summary

def _reg_summarize(suite: str, config: str, rows: List[OneGoal]) -> Summary:
    if not rows:
        return Summary(
            suite=suite, config=config, n_goals=0, n_success=0, success_rate=0.0,
            median_time_all=0.0, median_time_success=0.0, stamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    succ = [r for r in rows if r.success]
    times_all = [r.elapsed_s for r in rows]
    times_succ = [r.elapsed_s for r in succ]
    return Summary(
        suite=suite, config=config,
        n_goals=len(rows), n_success=len(succ), success_rate=(len(succ)/len(rows)),
        median_time_all=(stats.median(times_all) if times_all else 0.0),
        median_time_success=(stats.median(times_succ) if times_succ else 0.0),
        stamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

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

    print("\n=== Planner Regression comparison ===")
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
    # Resolve suite path
    if args.file:
        suite_name = Path(args.file).stem
        goals_path = Path(args.file)
    else:
        suite_name = args.suite
        goals_path = SUITE_MAP[suite_name]

    # Start Isabelle once
    server_info, proc = start_isabelle_server(name="planner", log_file="planner_regress.log")
    print(server_info.strip())
    isabelle = get_isabelle_client(server_info)
    session_id, used_session = _session_start_with_fallback(isabelle)
    print("session_id:", session_id, "| session:", used_session)

    try:
        goals = _read_goals_file(goals_path)
        if not goals:
            raise SystemExit(f"No goals found in {goals_path}")

        diverse = bool(args.diverse)
        cfg = {
            "mode": args.mode, "timeout": args.timeout,
            "k": (args.k if diverse else 1), "temps": args.temps,
            "repairs": (not args.no_repairs),
            "max_repairs_per_hole": args.max_repairs_per_hole,
            "repair_trace": args.repair_trace,
            "priors": args.priors, "context_hints": args.context_hints,
            "lib_templates": args.lib_templates,
            "alpha": args.alpha, "beta": args.beta, "gamma": args.gamma,
            "hintlex": args.hintlex, "hintlex_top": args.hintlex_top,
            "strict_no_sorry": args.strict_no_sorry,
            "verify": args.verify,
            "model": args.model,
            "shuffle": args.shuffle,
            "seed": args.seed,
        }
        model_tag = args.model or os.environ.get("OLLAMA_MODEL", "env_default")
        config_name = (
            f"mode_{args.mode}"
            f"__k{args.k if diverse else 1}"
            f"__t{args.timeout}"
            f"__repairs_{'on' if (not args.no_repairs) else 'off'}"
            f"__model_{model_tag}"
            f"{'__verify' if args.verify else ''}"
            f"{'__strict' if args.strict_no_sorry else ''}"
        )

        rows: List[OneGoal] = []
        import random
        gs = list(goals)
        if args.shuffle:
            rnd = random.Random(args.seed or int(time.time()))
            rnd.shuffle(gs)

        # Run
        for i, g in enumerate(gs, 1):
            print(f"[{suite_name}] [{i}/{len(gs)}] {g}")
            t0 = time.time()

            # safe planner call (prevents run aborts on provider/HTTP timeouts)
            class _CfgShim:
                pass
            _c = _CfgShim()
            _c.mode = args.mode
            _c.timeout = args.timeout
            _c.k = (args.k if diverse else 1)
            _c.temps = ([float(x) for x in args.temps.split(",")] if args.temps else None)
            _c.repairs = (not args.no_repairs)
            _c.max_repairs_per_hole = args.max_repairs_per_hole
            _c.repair_trace = args.repair_trace
            _c.priors = args.priors
            _c.context_hints = args.context_hints
            _c.lib_templates = args.lib_templates
            _c.alpha = args.alpha; _c.beta = args.beta; _c.gamma = args.gamma
            _c.hintlex = args.hintlex; _c.hintlex_top = args.hintlex_top

            res, plan_err = _safe_plan_and_fill(goal=g, model=args.model, cfg=_c)
            dt = time.time() - t0

            if res is None:
                text = ""
                had_sorry = False
                success = False
                verified_ok = True
                verify_details = f"[planner_error] {plan_err}"
            else:
                text = res.outline or ""
                had_sorry = bool(find_sorry_spans(text))

                if args.mode == "auto":
                    success = (not had_sorry)
                else:
                    success = True
                    if args.strict_no_sorry:
                        success = success and (not had_sorry)

                verified_ok = True
                verify_details = ""
                verify_status = "skipped"

                if args.verify and not had_sorry:
                    verified_ok, verify_details = _verify_with_auto_restart(isabelle, session_id, text)
                    if verified_ok:
                        verify_status = "ok"
                        success = True
                    else:
                        if _is_transport_error(verify_details):
                            verify_status = "transport_error"
                            # keep success as-is
                        else:
                            verify_status = "fail"
                            success = False
                    verify_details = (f"[{verify_status}] {verify_details}"
                                      if verify_details or verify_status != "skipped" else "")

            rows.append(OneGoal(
                goal=g, success=success, elapsed_s=float(dt),
                mode=args.mode, model=model_tag,
                outline_chars=len(text), fills=len(getattr(res, "fills", []) or []) if res else 0,
                failed_holes=len(getattr(res, "failed_holes", []) or []) if res else 0,
                had_sorry=had_sorry, verified_ok=bool(verified_ok),
            ))

            # append proof log (always)
            _append_proof_log(
                Path(args.log_path or DEFAULT_LOG_PATH),
                {
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "cmd": "regress",
                    "suite": suite_name,
                    "config": config_name,
                    "goal_index": i - 1,
                    "model": model_tag,
                    "mode": args.mode,
                    "timeout": args.timeout,
                    "k": (args.k if diverse else 1),
                    "temps": (args.temps or None),
                    "repairs": (not args.no_repairs),
                    "max_repairs_per_hole": args.max_repairs_per_hole,
                    "priors": args.priors,
                    "context_hints": args.context_hints,
                    "lib_templates": args.lib_templates,
                    "hintlex": args.hintlex,
                    "hintlex_top": args.hintlex_top,
                    "strict_no_sorry": args.strict_no_sorry,
                    "verify": args.verify,
                    "goal": g,
                    "success": success,
                    "had_sorry": had_sorry,
                    "verified_ok": bool(verified_ok),
                    "verify_details": verify_details,
                    "elapsed_s": float(dt),
                    "outline_chars": len(text),
                    "fills": len(getattr(res, "fills", []) or []) if res else 0,
                    "failed_holes": len(getattr(res, "failed_holes", []) or []) if res else 0,
                    "outline": text,
                }
            )

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
        print("\n=== Planner Regression run summary (no baseline) ===")
        print(f"Suite:        {summ.suite}")
        print(f"Config:       {summ.config}")
        print(f"Goals:        {summ.n_goals}")
        print(f"Success:      {summ.n_success} / {summ.n_goals}  ({summ.success_rate*100:.1f}%)")
        print(f"Median time:  all={summ.median_time_all:.2f}s, succ={summ.median_time_success:.2f}s")
    finally:
        try:
            isabelle.shutdown()
            try: _close_client_loop_safely(isabelle)
            except Exception: pass
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
    suite: str; config: str; goal: str; success: bool; elapsed_s: float; mode: str; model: str
    outline_chars: int; fills: int; failed_holes: int; had_sorry: bool; verified_ok: bool

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
                            mode=r.get("mode",""), model=r.get("model",""),
                            outline_chars=int(r.get("outline_chars","0") or 0),
                            fills=int(r.get("fills","0") or 0),
                            failed_holes=int(r.get("failed_holes","0") or 0),
                            had_sorry=str(r.get("had_sorry","")).strip().lower() in ("1","true","yes","y"),
                            verified_ok=str(r.get("verified_ok","")).strip().lower() in ("1","true","yes","y"),
                        ))
                    except Exception:
                        continue
        except FileNotFoundError:
            continue
    return rows

@dataclass(slots=True)
class AggSummary:
    suite: str; config: str; n: int; succ: int; rate: float; med_all: float; med_succ: float; med_fills: float

def _agg_summarize(rows: List[AggRow]) -> AggSummary:
    if not rows:
        return AggSummary("", "", 0, 0, 0.0, 0.0, 0.0, 0.0)
    suite = rows[0].suite; cfg = rows[0].config
    n = len(rows); succ_rows = [r for r in rows if r.success]; succ = len(succ_rows)
    rate = (succ / n) if n else 0.0
    med_all = stats.median([r.elapsed_s for r in rows]) if rows else 0.0
    med_succ = stats.median([r.elapsed_s for r in succ_rows]) if succ_rows else 0.0
    med_fills = stats.median([r.fills for r in rows]) if rows else 0.0
    return AggSummary(suite, cfg, n, succ, rate, med_all, med_succ, med_fills)

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

    print("\n=== Planner benchmark summary (by suite, success ↓ then time ↑) ===")
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
        print(f"\n[{suite}]")
        table = [
            [sm.config, f"{sm.rate*100:.1f}%", f"{sm.succ}/{sm.n}", f"{sm.med_all:.2f}", f"{sm.med_succ:.2f}" if sm.succ else "-", f"{sm.med_fills:.1f}"]
            for sm in display
        ]
        headers = ["config", "succ_rate", "succ/total", "median_time_all(s)", "median_time_succ(s)", "median_fills"]
        if use_tab: print(tabulate(table, headers=headers, tablefmt="github"))
        else:
            colw = [max(len(str(x)) for x in col) for col in zip(*([headers] + table))]
            def fmt_row(row: List[str]) -> str: return "  ".join(str(cell).ljust(w) for cell, w in zip(row, colw))
            print(fmt_row(headers)); print("  ".join("-"*w for w in colw))
            for r in table: print(fmt_row(r))

# =============================================================================
# CLI
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="Planner experiments (bench | regress | aggregate)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Bench
    pb = sub.add_parser("bench", help="Run file/suite and write CSVs")
    pb.add_argument("--suite", type=str, choices=sorted(list(SUITE_MAP.keys()) + ["all"]))
    pb.add_argument("--file", type=str)
    pb.add_argument("--mode", choices=["auto", "outline"], default="auto")
    pb.add_argument("--timeout", type=int, default=100)
    pb.add_argument("--diverse", action="store_true", help="Use diverse outlines (k, temps); otherwise single outline")
    pb.add_argument("--k", type=int, default=3)
    pb.add_argument("--temps", type=str, default=None, help="Comma-separated temps, e.g., '0.35,0.55,0.85'")
    pb.add_argument("--no-repairs", action="store_true")
    pb.add_argument("--max-repairs-per-hole", type=int, default=2)
    pb.add_argument("--repair-trace", action="store_true")
    pb.add_argument("--context-hints", action="store_true")
    pb.add_argument("--lib-templates", action="store_true")
    pb.add_argument("--priors", type=str, default=None)
    pb.add_argument("--alpha", type=float, default=1.0)
    pb.add_argument("--beta", type=float, default=0.5)
    pb.add_argument("--gamma", type=float, default=0.2)
    pb.add_argument("--hintlex", type=str, default=None)
    pb.add_argument("--hintlex-top", type=int, default=8)
    pb.add_argument("--model", type=str, default=None)
    pb.add_argument("--strict-no-sorry", action="store_true", help="Count success only when no 'sorry' is present")
    pb.add_argument("--verify", action="store_true", help="Compile outline with Isabelle if no 'sorry'")
    pb.add_argument("--repeats", type=int, default=1)
    pb.add_argument("--shuffle", action="store_true")
    pb.add_argument("--seed", type=int, default=0)
    # override planner proof log path
    pb.add_argument("--log-path", type=str, default=str(DEFAULT_LOG_PATH))
    pb.set_defaults(func=cmd_bench)

    # Regress
    pr = sub.add_parser("regress", help="Run and compare to baseline")
    pr.add_argument("--suite", type=str, choices=sorted(list(SUITE_MAP.keys())))
    pr.add_argument("--file", type=str, default=None)
    pr.add_argument("--mode", choices=["auto", "outline"], default="auto")
    pr.add_argument("--timeout", type=int, default=100)
    pr.add_argument("--diverse", action="store_true")
    pr.add_argument("--k", type=int, default=3)
    pr.add_argument("--temps", type=str, default=None)
    pr.add_argument("--no-repairs", action="store_true")
    pr.add_argument("--max-repairs-per-hole", type=int, default=2)
    pr.add_argument("--repair-trace", action="store_true")
    pr.add_argument("--context-hints", action="store_true")
    pr.add_argument("--lib_templates", action="store_true")
    pr.add_argument("--priors", type=str, default=None)
    pr.add_argument("--alpha", type=float, default=1.0)
    pr.add_argument("--beta", type=float, default=0.5)
    pr.add_argument("--gamma", type=float, default=0.2)
    pr.add_argument("--hintlex", type=str, default=None)
    pr.add_argument("--hintlex-top", type=int, default=8)
    pr.add_argument("--model", type=str, default=None)
    pr.add_argument("--strict-no-sorry", action="store_true")
    pr.add_argument("--verify", action="store_true")
    pr.add_argument("--shuffle", action="store_true")
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--baseline", type=str)
    pr.add_argument("--save-baseline", type=str)
    pr.add_argument("--out", type=str, default=None)
    pr.add_argument("--tol-rate", type=float, default=0.00)
    pr.add_argument("--tol-time", type=float, default=2.0)
    # override planner proof log path
    pr.add_argument("--log-path", type=str, default=str(DEFAULT_LOG_PATH))
    pr.set_defaults(func=cmd_regress)

    # Aggregate
    pa = sub.add_parser("aggregate", help="Summarize CSVs in planner_results/")
    pa.add_argument("--dir", type=str, default=str(RESULTS_DIR))
    pa.add_argument("--min-rows", type=int, default=1)
    pa.add_argument("--best-only", action="store_true")
    pa.add_argument("--top-k", type=int, default=3)
    pa.set_defaults(func=cmd_aggregate)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()