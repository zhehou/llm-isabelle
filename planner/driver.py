from __future__ import annotations

import time
import re
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout
import hashlib

from planner.skeleton import (
    Skeleton, find_sorry_spans, propose_isar_skeleton, propose_isar_skeleton_diverse_best,
)
from planner.repair import try_cegis_repairs, regenerate_whole_proof, _APPLY_OR_BY as _TACTIC_LINE_RE
from prover.config import ISABELLE_SESSION
from prover.isabelle_api import (
    build_theory, get_isabelle_client, last_print_state_block, start_isabelle_server,
)
from prover.prover import prove_goal
from planner.goals import _print_state_before_hole, _log_state_block, _effective_goal_from_state, _first_lemma_line, _extract_goal_from_lemma_line, _cleanup_resources, _verify_full_proof, _run_theory_with_timeout

def _hole_fingerprint(full_text: str, span: tuple[int, int], context: int = 80) -> str:
    """Stable key for a hole: hash a small window around the 'sorry'."""
    s, e = span
    lo = max(0, s - context)
    hi = min(len(full_text), e + context)
    snippet = full_text[lo:hi]
    return hashlib.sha1(snippet.encode("utf-8")).hexdigest()[:16]

# Constants
_INLINE_BY_TAIL = re.compile(r"\s+by\s+.+$")
_BARE_DOT = re.compile(r"(?m)^\s*\.\s*$")
_HEAD_CMD_RE = re.compile(r"^\s*(have|show|obtain)\b")  # local copy to avoid new imports
_ISA_VERIFY_TIMEOUT_S = int(os.getenv("ISABELLE_VERIFY_TIMEOUT_S", "30"))

@dataclass(slots=True)
class PlanAndFillResult:
    success: bool
    outline: str
    fills: List[str]
    failed_holes: List[int]


# ============================================================================
# Hole Filling
# ============================================================================

def _fill_one_hole(isabelle, session: str, full_text: str, hole_span: Tuple[int, int], 
                  goal_text: str, model: Optional[str], per_hole_timeout: int, *, trace: bool = False) -> Tuple[str, bool, str]:
    """Fill single hole in proof."""
    
    # Check for stale hole
    try:
        s_line_start = full_text.rfind("\n", 0, hole_span[0]) + 1
        prev_line_end = s_line_start - 1
        prev_prev_nl = full_text.rfind("\n", 0, prev_line_end) + 1
        prev_line = full_text[prev_prev_nl:prev_line_end+1]
    except Exception:
        prev_line = ""
    
    if (_INLINE_BY_TAIL.search(prev_line) or _TACTIC_LINE_RE.match(prev_line) or 
        prev_line.strip() in {"done", "."}):
        s, e = hole_span
        return full_text[:s] + "\n" + full_text[e:], True, "(stale-hole)"
    
    state_block = _print_state_before_hole(isabelle, session, full_text, hole_span, trace)
    _log_state_block("fill", state_block, trace=trace)
    
    # orig_goal = _original_goal_from_state(state_block)
    eff_goal = _effective_goal_from_state(state_block, goal_text, full_text, hole_span, trace)
    
    # if trace:
    #     # if orig_goal:
    #     #     print(f"[fill] Original goal: {orig_goal}")
    #     print(f"[fill] Effective goal: {eff_goal}")
    
    res = prove_goal(
        isabelle, session, eff_goal, model_name_or_ensemble=model,
        beam_w=3, max_depth=6, hint_lemmas=6, timeout=per_hole_timeout,
        models=None, save_dir=None, use_sledge=True, sledge_timeout=10,
        sledge_every=1, trace=trace, use_color=False, use_qc=False,
        qc_timeout=2, qc_every=1, use_np=False, np_timeout=5, np_every=2,
        facts_limit=8, do_minimize=False, minimize_timeout=8,
        do_variants=False, variant_timeout=6, variant_tries=24,
        enable_reranker=True, initial_state_hint=state_block,
    )
    
    steps = [str(s) for s in res.get("steps", [])]

    # Fallbacks: some backends return finishers/applies in separate keys
    fin_candidates = []
    # singular fields
    for k in ("finisher", "finish", "final"):
        v = res.get(k)
        if isinstance(v, str):
            fin_candidates.append(v)
    # list fields
    for k in ("finishers", "sledge_finishers"):
        vs = res.get(k)
        if isinstance(vs, (list, tuple)):
            fin_candidates.extend([str(x) for x in vs if isinstance(x, str)])
    applies_from_keys = []
    for k in ("applies", "apply_steps"):
        vs = res.get(k)
        if isinstance(vs, (list, tuple)):
            applies_from_keys.extend([str(x) for x in vs if isinstance(x, str) and x.startswith("apply")])

    applies = [s for s in steps if s.startswith("apply")]
    if applies_from_keys:
        applies = applies or applies_from_keys  # prefer explicit list if steps were empty

    fin = next((s for s in steps if s.startswith("by ") or s.strip() == "done"), "")
    if not fin:
        fin = next((x for x in fin_candidates if isinstance(x, str) and (x.startswith("by ") or x.strip() == "done")), "")

    # If neither steps nor recognized finishers were returned, report no-steps
    if not (applies or fin):
        return full_text, False, "no-steps"
    
    # Handle finisher
    if fin:
        script_lines = applies + [fin]
        insert = "\n  " + "\n  ".join(script_lines) + "\n"
        s, e = hole_span
        new_text = full_text[:s] + insert + full_text[e:]
        
        if _verify_full_proof(isabelle, session, new_text):
            return new_text, True, "\n".join(script_lines)
        return full_text, False, "finisher-unverified"
    
    # Handle apply-only  (NEVER mark success for apply-only scripts)
    if applies:
        # Decide if the hole sits under a have/show/obtain head; if so, we must NOT
        # leave a bare 'apply' there (illegal in 'prove' mode). Replace the hole with
        # a tiny subproof instead of inserting above the hole.
        s, e = hole_span
        # scan a small window upwards to find the enclosing head line
        head_line_start = full_text.rfind("\n", 0, s) + 1
        scan_start = max(0, full_text.rfind("\n", 0, max(0, head_line_start - 512)) + 1)
        segment = full_text[scan_start:s]
        lines = segment.splitlines()
        head_idx = None
        for i in range(len(lines) - 1, -1, -1):
            if _HEAD_CMD_RE.match(lines[i] or ""):
                head_idx = i
                break

        # Deduplicate against already-present lines in the local window
        dedup_window = segment
        dedup = [a for a in applies if a not in dedup_window]
        if not dedup:
            return full_text, False, "apply-duplicate"

        if head_idx is not None:
            # Apply-only inside have/show is illegal in 'prove' mode unless closed by 'by ...'.
            # Do NOT fabricate 'proof ... qed'. Leave the hole as-is and let the caller escalate to repair.
            if trace:
                print("[fill] apply-only inside have/show; not inserting proof/qed; escalating to repair.")
            return full_text, False, "apply-inside-have/show"
        else:
            # Non have/show context — keep existing behaviour (insert above, keep sorry)
            probe_text = _insert_above_hole_keep_sorry(full_text, hole_span, dedup)
            return probe_text, False, "\n".join(dedup)
    
    return full_text, False, "no-tactics"


def _insert_above_hole_keep_sorry(text: str, hole: Tuple[int, int], lines_to_insert: List[str]) -> str:
    """Insert lines above hole while keeping sorry."""
    s, _ = hole
    ls = text.rfind("\n", 0, s) + 1
    le = text.find("\n", s)
    hole_line = text[ls:(le if le != -1 else len(text))]
    indent = hole_line[:len(hole_line) - len(hole_line.lstrip(" "))]
    payload = "".join(f"{indent}{ln.strip()}\n" for ln in lines_to_insert if ln.strip())
    return text[:s] + payload + text[s:]

# --- helper: pick the sorry-span nearest a target offset (to preserve focus) ---
def _nearest_sorry_span(spans: List[Tuple[int, int]], target_s: int) -> Optional[Tuple[int, int]]:
    if not spans:
        return None
    return min(spans, key=lambda sp: abs(sp[0] - target_s))

# ============================================================================
# Repair
# ============================================================================

def _proof_bounds_top_level(text: str) -> Optional[Tuple[int, int]]:
    """Return (start,end) offsets of last top-level proof..qed block."""
    qed_matches = list(re.finditer(r"(?m)^\s*qed\b", text))
    if not qed_matches:
        return None
    
    end = qed_matches[-1].end()
    proof_matches = list(re.finditer(r"(?m)^\s*proof\b.*$", text[:qed_matches[-1].start()]))
    if not proof_matches:
        return None
    
    return (proof_matches[-1].start(), end)


def _tactic_spans_topdown(text: str) -> List[Tuple[int, int]]:
    """Top-down tactic line spans within last proof..qed block."""
    bounds = _proof_bounds_top_level(text)
    if not bounds:
        return []
    
    b0, b1 = bounds
    seg = text[b0:b1]
    lines = seg.splitlines(True)
    spans, off = [], b0
    
    for line in lines:
        if _TACTIC_LINE_RE.match(line or "") or _INLINE_BY_TAIL.search(line or ""):
            spans.append((off, off + len(line.rstrip("\n"))))
        off += len(line)
    
    return spans

def _repair_failed_proof_topdown(isa, session, full: str, goal_text: str, model: Optional[str],
                                 left_s, max_repairs_per_hole: int, trace: bool) -> Tuple[str, bool]:
    """Walk tactics from top; attempt CEGIS-repair on the first failing one.

    This must never crash the UI route. Timeouts / broken Isabelle responses are treated as
    'repair failed', and the caller may decide to fall back (e.g. open minimal sorries).
    """
    t_spans = _tactic_spans_topdown(full)
    if not t_spans:
        return full, False

    i = 0
    while i < len(t_spans) and left_s() > 3.0:
        span = t_spans[i]
        try:
            st = _print_state_before_hole(isa, session, full, span, trace)
            eff_goal = _effective_goal_from_state(st, goal_text, full, span, trace)
        except Exception as ex:
            if trace:
                print(f"[repair] Could not extract state/goal before tactic (skipping): {ex}")
            i += 1
            continue

        per_budget = min(30.0, max(15.0, left_s() * 0.33))

        try:
            patched, applied, _ = try_cegis_repairs(
                full_text=full, hole_span=span, goal_text=eff_goal, model=model,
                isabelle=isa, session=session, repair_budget_s=per_budget,
                max_ops_to_try=max_repairs_per_hole, beam_k=2,
                allow_whole_fallback=False, trace=trace, resume_stage=0,
            )
        except (TimeoutError, _FuturesTimeout, ValueError) as ex:
            # TimeoutError: verifier timed out; ValueError: isabelle_client returned unexpected/empty response
            if trace:
                print(f"[repair] CEGIS repair aborted (treat as failed): {type(ex).__name__}: {ex}")
            return full, False
        except Exception as ex:
            if trace:
                print(f"[repair] CEGIS repair crashed (treat as failed): {type(ex).__name__}: {ex}")
            return full, False

        if applied and patched != full:
            if _verify_full_proof(isa, session, patched):
                return patched, True

            # Partial progress: keep it, then try to open the failing spot into a 'sorry'
            if trace:
                print("[repair] Partial progress in topdown repair (unverified). Opening sorries...")
            full = patched
            full2, opened = _open_minimal_sorries(isa, session, full)
            if opened:
                full = full2
                t_spans = _tactic_spans_topdown(full)
                i = 0
                continue

        i += 1

    return full, False

def _quick_state_and_errors(isabelle, session: str, text: str, *, timeout_s: Optional[int] = None) -> Tuple[str, List[str]]:
    """Run a theory quickly and return (last_state_block, error_messages).

    Best-effort utility used only to locate an error line for opening with 'sorry'.
    It must be robust: on any exception, return empty state and a single error string.
    """
    try:
        ts = text.splitlines()
        thy = build_theory(ts, add_print_state=True, end_with=None)
        out = _run_theory_with_timeout(
            isabelle, session, thy,
            timeout_s=int(timeout_s) if timeout_s is not None else min(_ISA_VERIFY_TIMEOUT_S, 15),
        )
        state = ""
        try:
            state = last_print_state_block(out)
        except Exception:
            state = ""

        # Normalize messages to strings for simple scanning
        if isinstance(out, (list, tuple)):
            msgs = [str(m) for m in out]
        else:
            msgs = [str(out)]

        errs = [m for m in msgs if any(tok in m.lower() for tok in ("error", "exception", "failed"))]
        return state, errs
    except Exception as ex:
        return "", [str(ex)]
    
def _extract_error_lines(errs: List[str]) -> List[int]:
    """Extract 1-based line numbers from Isabelle error messages (best-effort)."""
    if not errs:
        return []

    patts = [
        re.compile(r"(?i)\bline\s+(\d+)\b"),          # 'line 23'
        re.compile(r"(?i)\bLine\s+(\d+)\b"),          # 'Line 23'
        re.compile(r":(\d+):(\d+)\b"),                # 'Scratch.thy:23:5'
        re.compile(r"\((\d+),(\d+)\)"),               # '(23,5)'
    ]

    found: set[int] = set()
    for raw in errs:
        s = str(raw)
        for p in patts:
            for m in p.finditer(s):
                try:
                    n = int(m.group(1))
                    if n > 0:
                        found.add(n)
                except Exception:
                    pass

    return sorted(found)

def _open_minimal_sorries(isabelle, session: str, text: str) -> Tuple[str, bool]:
    """Localize a failing finisher with minimal opening (replace 1 tactic with 'sorry').

    Returns (new_text, opened). Never raises.
    """
    def _ensure_nl(s: str) -> str:
        return s if s.endswith("\n") else s + "\n"

    # First check if the whole thing passes
    def runs(ts):
        try:
            thy = build_theory(ts, add_print_state=False, end_with=None)
            _run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_VERIFY_TIMEOUT_S)
            return True
        except Exception:
            return False

    try:
        if runs(text.splitlines()):
            return _ensure_nl(text), False
    except Exception:
        # If even 'runs' crashes, do nothing.
        return _ensure_nl(text), False

    # Document fails: find first error line, then open nearest tactic by turning it into 'sorry'
    try:
        _, errs = _quick_state_and_errors(isabelle, session, text)
        err_lines = _extract_error_lines(errs)
    except Exception:
        err_lines = []

    if not err_lines:
        return _ensure_nl(text), False

    failing_line_1based = min(err_lines)
    lines = text.splitlines()
    failing_idx = failing_line_1based - 1

    for i in range(min(failing_idx, len(lines) - 1), -1, -1):
        line = lines[i]

        if _TACTIC_LINE_RE.match(line) or line.strip() == "done" or _BARE_DOT.match(line):
            indent = line[:len(line) - len(line.lstrip(" "))]
            lines[i] = f"{indent}sorry"
            return _ensure_nl("\n".join(lines)), True

        m = _INLINE_BY_TAIL.search(line)
        if m:
            indent = line[:len(line) - len(line.lstrip(" "))]
            header = line[:m.start()].rstrip()
            lines[i] = header
            lines.insert(i + 1, f"{indent}sorry")
            return _ensure_nl("\n".join(lines)), True

    return _ensure_nl(text), False

# ============================================================================
# Public API
# ============================================================================

def plan_outline(goal: str, *, model: Optional[str] = None, outline_k: Optional[int] = None,
                outline_temps: Optional[Iterable[float]] = None, legacy_single_outline: bool = False,
                priors_path: Optional[str] = None, context_hints: bool = False,
                lib_templates: bool = False, alpha: float = 1.0, beta: float = 0.5,
                gamma: float = 0.2, hintlex_path: Optional[str] = None, hintlex_top: int = 8) -> str:
    """Generate Isar outline with 'sorry' placeholders."""
    server_info, proc = start_isabelle_server(name="planner", log_file="logs/planner_ui.log")
    isa = get_isabelle_client(server_info)
    session = isa.session_start(session=ISABELLE_SESSION)
    
    try:
        if legacy_single_outline:
            return propose_isar_skeleton(goal, model=model, temp=0.35, force_outline=True).text
        
        temps = tuple(outline_temps) if outline_temps else (0.35, 0.55, 0.85)
        k = int(outline_k) if outline_k is not None else 3
        
        best, _ = propose_isar_skeleton_diverse_best(
            goal, isabelle=isa, session_id=session, model=model, temps=temps, k=k,
            force_outline=True, priors_path=priors_path, context_hints=context_hints,
            lib_templates=lib_templates, alpha=alpha, beta=beta, gamma=gamma,
            hintlex_path=hintlex_path, hintlex_top=hintlex_top,
        )
        return best.text
    finally:
        _cleanup_resources(isa, proc)

def plan_and_fill(goal: str, model: Optional[str] = None, timeout: int = 100, *, mode: str = "auto",
                 outline_k: Optional[int] = None, outline_temps: Optional[Iterable[float]] = None,
                 legacy_single_outline: bool = False, repairs: bool = True,
                 max_repairs_per_hole: int = 2, trace: bool = False, repair_trace: bool = False,
                 priors_path: Optional[str] = None, context_hints: bool = False,
                 lib_templates: bool = False, alpha: float = 1.0, beta: float = 0.5,
                 gamma: float = 0.2, hintlex_path: Optional[str] = None,
                 hintlex_top: int = 8) -> PlanAndFillResult:
    """Plan and fill holes in Isar proofs.

    Notes:
      - 'repair_trace' is a backwards-compatible alias used by the UI. It enables 'trace'.
      - Repair/verification timeouts or broken Isabelle responses must not crash the caller.
        We treat them as repair failures, and (best-effort) restart Isabelle for subsequent calls.
    """
    if repair_trace and not trace:
        trace = True

    server_info, proc = start_isabelle_server(name="planner", log_file="logs/planner_ui.log")
    isa = get_isabelle_client(server_info)
    session = isa.session_start(session=ISABELLE_SESSION)

    t0 = time.monotonic()
    left_s = lambda: max(0.0, timeout - (time.monotonic() - t0))

    restart_count = 0

    def _restart_isabelle(reason: str, ex: Optional[BaseException] = None) -> None:
        nonlocal isa, session, proc, restart_count
        if restart_count >= 2:
            return
        restart_count += 1
        if trace:
            msg = f"[planner] Restarting Isabelle (#{restart_count}) due to {reason}"
            if ex is not None:
                msg += f": {type(ex).__name__}: {ex}"
            print(msg)
        try:
            _cleanup_resources(isa, proc)
        except Exception:
            pass
        server_info2, proc2 = start_isabelle_server(name="planner", log_file="logs/planner_ui.log")
        isa2 = get_isabelle_client(server_info2)
        session2 = isa2.session_start(session=ISABELLE_SESSION)
        isa, session, proc = isa2, session2, proc2

    try:
        # Generate outline
        if legacy_single_outline:
            full = propose_isar_skeleton(goal, model=model, temp=0.35, force_outline=(mode == "outline")).text
        else:
            temps = tuple(outline_temps) if outline_temps else (0.35, 0.55, 0.85)
            k = int(outline_k) if outline_k is not None else 3
            best, _ = propose_isar_skeleton_diverse_best(
                goal, isabelle=isa, session_id=session, model=model, temps=temps, k=k,
                force_outline=(mode == "outline"), priors_path=priors_path,
                context_hints=context_hints, lib_templates=lib_templates,
                alpha=alpha, beta=beta, gamma=gamma, hintlex_path=hintlex_path,
                hintlex_top=hintlex_top,
            )
            full = best.text

        spans = find_sorry_spans(full)

        if mode == "outline":
            return PlanAndFillResult(True, full, [], [])

        # Handle complete proofs
        if not spans:
            try:
                if _verify_full_proof(isa, session, full):
                    return PlanAndFillResult(True, full, [], [])
            except (TimeoutError, _FuturesTimeout, ValueError) as ex:
                _restart_isabelle("verify_full_proof", ex)

            if repairs and left_s() > 6.0:
                full, ok = _repair_failed_proof_topdown(isa, session, full, goal, model, left_s, max_repairs_per_hole, trace)
                if ok:
                    return PlanAndFillResult(True, full, [], [])

            full2, opened = _open_minimal_sorries(isa, session, full)
            full = full2 if opened else full
            if not opened:
                return PlanAndFillResult(False, full, [], [0])

        # Fill holes
        lemma_line = _first_lemma_line(full)
        if not lemma_line:
            return PlanAndFillResult(False, full, [], [0])

        goal_text = _extract_goal_from_lemma_line(lemma_line)
        fills: List[str] = []
        failed: List[int] = []
        repair_progress: dict[str, int] = {}
        stage_tries: dict[Tuple[str, int], int] = {}
        _skip_fill_logged_once: set[Tuple[str, int]] = set()

        focused_hole_key: Optional[str] = None

        while "sorry" in full and left_s() > 0:
            spans = find_sorry_spans(full)
            if not spans:
                break

            span = None
            if focused_hole_key is not None:
                for s in spans:
                    if _hole_fingerprint(full, s) == focused_hole_key:
                        span = s
                        break
                if span is None:
                    if trace:
                        print(f"[fill] Focused hole @{focused_hole_key} was closed. Moving to first hole.")
                    focused_hole_key = None

            if span is None:
                span = spans[0]

            hole_key = _hole_fingerprint(full, span)
            per_hole_budget = int(max(5, left_s() / max(1, len(spans))))
            start_stage = repair_progress.get(hole_key, 0)

            # Always try fill first unless we're in escalated repair stages
            if start_stage == 0:
                try:
                    full2, ok, script = _fill_one_hole(
                        isa, session, full, span, goal_text, model,
                        per_hole_timeout=per_hole_budget, trace=trace
                    )
                except (TimeoutError, _FuturesTimeout, ValueError) as ex:
                    _restart_isabelle("fill_one_hole", ex)
                    full2, ok, script = full, False, "fill-exception"
                except Exception as ex:
                    if trace:
                        print(f"[fill] _fill_one_hole crashed: {type(ex).__name__}: {ex}")
                    full2, ok, script = full, False, "fill-exception"

                if ok and full2 != full:
                    full = full2
                    fills.append(script)
                    repair_progress.pop(hole_key, None)
                    focused_hole_key = None
                    continue
                elif not ok and full2 != full:
                    if trace:
                        print("[fill] Partial progress from fill (unverified). Opening sorries and staying focused...")
                    old_start = span[0]
                    full = full2
                    full2, opened = _open_minimal_sorries(isa, session, full)
                    if opened:
                        full = full2
                        new_spans = find_sorry_spans(full)
                        near = _nearest_sorry_span(new_spans, old_start)
                        focused_hole_key = _hole_fingerprint(full, near) if near else None
                        continue
                    else:
                        if trace:
                            print("[fill] Could not open sorries. Escalating to repair stage 1...")
                        repair_progress[hole_key] = 1
                        focused_hole_key = hole_key
                        start_stage = 1
                else:
                    if trace:
                        print("[fill] Fill made no progress. Escalating to repair stage 1...")
                    repair_progress[hole_key] = 1
                    focused_hole_key = hole_key
                    start_stage = 1
            else:
                if trace and (hole_key, start_stage) not in _skip_fill_logged_once:
                    print(f"[fill] Skipping fill for hole @{hole_key}; running repairs at stage {start_stage}")
                    _skip_fill_logged_once.add((hole_key, start_stage))

            # Try CEGIS repairs
            current_stage = repair_progress.get(hole_key, 0)
            if current_stage > 0 and repairs and left_s() > 6:
                try:
                    state = _print_state_before_hole(isa, session, full, span, trace)
                    eff_goal = _effective_goal_from_state(state, goal_text, full, span, trace)
                except (TimeoutError, _FuturesTimeout, ValueError) as ex:
                    _restart_isabelle("print_state_before_hole", ex)
                    continue
                except Exception as ex:
                    if trace:
                        print(f"[repair] Could not compute effective goal: {type(ex).__name__}: {ex}")
                    continue

                try:
                    patched, applied, _ = try_cegis_repairs(
                        full_text=full, hole_span=span, goal_text=eff_goal, model=model,
                        isabelle=isa, session=session,
                        repair_budget_s=min(30.0, max(15.0, left_s() * 0.33)),
                        max_ops_to_try=max_repairs_per_hole, beam_k=2,
                        allow_whole_fallback=False, trace=trace, resume_stage=current_stage,
                    )
                except (TimeoutError, _FuturesTimeout, ValueError) as ex:
                    _restart_isabelle("try_cegis_repairs", ex)
                    patched, applied = full, False
                except Exception as ex:
                    if trace:
                        print(f"[repair] try_cegis_repairs crashed: {type(ex).__name__}: {ex}")
                    patched, applied = full, False

                if patched != full:
                    try:
                        if _verify_full_proof(isa, session, patched):
                            if trace:
                                print(f"[repair] Stage {current_stage} repair verified! Clearing progress and moving on.")
                            full = patched
                            repair_progress.clear()
                            stage_tries.clear()
                            focused_hole_key = None
                            continue
                    except (TimeoutError, _FuturesTimeout, ValueError) as ex:
                        _restart_isabelle("verify_full_proof_after_repair", ex)

                    # Unverified change: count attempt and decide escalation
                    key = (hole_key, start_stage)
                    stage_tries[key] = stage_tries.get(key, 0) + 1

                    STAGE1_CAP = 2
                    STAGE2_CAP = 3

                    should_escalate = False
                    if start_stage == 1 and stage_tries[key] >= STAGE1_CAP:
                        should_escalate = True
                        if trace:
                            print(f"[repair] Stage 1 cap ({STAGE1_CAP}) reached. Escalating to stage 2...")
                    elif start_stage == 2 and stage_tries.get((hole_key, 2), 0) >= STAGE2_CAP:
                        should_escalate = True
                        if trace:
                            print(f"[repair] Stage 2 cap ({STAGE2_CAP}) reached. Regenerating whole proof...")

                    if should_escalate:
                        if start_stage < 2:
                            repair_progress[hole_key] = 2
                            focused_hole_key = hole_key
                            continue
                        else:
                            regen_budget = min(40.0, max(8.0, left_s() * 0.8))
                            try:
                                new_full, ok_re, _ = regenerate_whole_proof(
                                    full_text=full, goal_text=goal_text, model=model,
                                    isabelle=isa, session=session, budget_s=regen_budget,
                                    trace=trace, prior_outline_text=full
                                )
                            except (TimeoutError, _FuturesTimeout, ValueError) as ex:
                                _restart_isabelle("regenerate_whole_proof", ex)
                                new_full, ok_re = full, False
                            except Exception as ex:
                                if trace:
                                    print(f"[repair] regenerate_whole_proof crashed: {type(ex).__name__}: {ex}")
                                new_full, ok_re = full, False

                            if ok_re and new_full != full:
                                full = new_full
                                repair_progress.clear()
                                stage_tries.clear()
                                focused_hole_key = None
                                continue

                            if trace:
                                print("[repair] Whole regeneration failed to verify; proposing a fresh outline…")
                            temps = tuple(outline_temps) if outline_temps else (0.35, 0.55, 0.85)
                            k = int(outline_k) if outline_k is not None else 3
                            best, _ = propose_isar_skeleton_diverse_best(
                                goal_text, isabelle=isa, session_id=session, model=model, temps=temps, k=k,
                                force_outline=True, priors_path=priors_path, context_hints=context_hints,
                                lib_templates=lib_templates, alpha=alpha, beta=beta, gamma=gamma,
                                hintlex_path=hintlex_path, hintlex_top=hintlex_top,
                            )
                            full = best.text
                            repair_progress.clear()
                            stage_tries.clear()
                            focused_hole_key = None
                            continue

                    if trace:
                        cap = STAGE1_CAP if start_stage == 1 else STAGE2_CAP
                        print(f"[repair] Stage {start_stage} changed but unverified (attempt {stage_tries[key]}/{cap}). Opening sorries...")
                    full = patched
                    full2, opened = _open_minimal_sorries(isa, session, full)
                    if opened:
                        full = full2
                        focused_hole_key = None
                        continue
                    else:
                        if trace:
                            print("[repair] Could not open sorries; escalating stage...")
                        if start_stage < 2:
                            repair_progress[hole_key] = 2
                            focused_hole_key = hole_key
                        continue

                # No change from repair: count attempt and escalate
                key = (hole_key, start_stage)
                stage_tries[key] = stage_tries.get(key, 0) + 1
                if start_stage < 2:
                    repair_progress[hole_key] = min(start_stage + 1, 2)
                    focused_hole_key = hole_key
                else:
                    repair_progress[hole_key] = 2
                    focused_hole_key = hole_key

        # Final verification
        success = ("sorry" not in full)
        if success:
            try:
                if _verify_full_proof(isa, session, full):
                    return PlanAndFillResult(True, full, fills, failed)
            except (TimeoutError, _FuturesTimeout, ValueError) as ex:
                _restart_isabelle("final_verify_full_proof", ex)

        return PlanAndFillResult(False, full, fills, failed)

    finally:
        _cleanup_resources(isa, proc)