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
    if not steps:
        return full_text, False, "no-steps"
    
    applies = [s for s in steps if s.startswith("apply")]
    fin = next((s for s in steps if s.startswith("by ") or s.strip() == "done"), "")
    
    # Handle finisher
    if fin:
        script_lines = applies + [fin]
        insert = "\n  " + "\n  ".join(script_lines) + "\n"
        s, e = hole_span
        new_text = full_text[:s] + insert + full_text[e:]
        
        if _verify_full_proof(isabelle, session, new_text):
            return new_text, True, "\n".join(script_lines)
        return full_text, False, "finisher-unverified"
    
    # Handle apply-only
    if applies:
        s, _ = hole_span
        win_s = max(0, full_text.rfind("\n", 0, max(0, s-256)) + 1)
        window = full_text[win_s:s]
        dedup = [a for a in applies if a not in window]
        
        if not dedup:
            return full_text, False, "apply-duplicate"
        
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


def _repair_failed_proof_topdown(isa, session, full: str, goal_text: str, model: str,
                                 left_s, max_repairs_per_hole: int, trace: bool) -> Tuple[str, bool]:
    """Walk tactics from top; CEGIS-repair first failing one."""
    t_spans = _tactic_spans_topdown(full)
    if not t_spans:
        return full, False
    
    i = 0
    while i < len(t_spans) and left_s() > 3.0:
        span = t_spans[i]
        st = _print_state_before_hole(isa, session, full, span, trace)
        eff_goal = _effective_goal_from_state(st, goal_text, full, span, trace)
        per_budget = min(30.0, max(15.0, left_s() * 0.33))
        
        patched, applied, _ = try_cegis_repairs(
            full_text=full, hole_span=span, goal_text=eff_goal, model=model,
            isabelle=isa, session=session, repair_budget_s=per_budget,
            max_ops_to_try=max_repairs_per_hole, beam_k=2,
            allow_whole_fallback=False, trace=trace, resume_stage=0,
        )
        
        if applied and patched != full:
            if _verify_full_proof(isa, session, patched):
                full = patched
                return full, True
            # FIX 3: Open sorries on unverified partial progress
            if trace:
                print("[repair] Partial progress in topdown repair (unverified). Opening sorries...")
            full = patched
            full2, opened = _open_minimal_sorries(isa, session, full)
            if opened:
                full = full2
                t_spans = _tactic_spans_topdown(full)
                continue
        i += 1
    
    return full, False


def _open_minimal_sorries(isabelle, session: str, text: str) -> Tuple[str, bool]:
    """Localize failing finisher with minimal opening."""
    def runs(ts):
        try:
            thy = build_theory(ts, add_print_state=True, end_with="sorry")
            _run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_VERIFY_TIMEOUT_S)
            return True
        except Exception:
            return False
    
    lines = text.splitlines()
    
    # Try whole-line tactics
    for i, line in enumerate(lines):
        if not (_TACTIC_LINE_RE.match(line) or line.strip() == "done" or _BARE_DOT.match(line)):
            continue
        
        indent = line[:len(line) - len(line.lstrip(" "))]
        if runs(lines[:i] + [f"{indent}sorry"] + lines[i + 1:]):
            lines[i] = f"{indent}sorry"
            return "\n".join(lines) + ("" if text.endswith("\n") else "\n"), True
    
    # Try inline 'by TACTIC' patterns
    for i, line in enumerate(lines):
        m = _INLINE_BY_TAIL.search(line) if line else None
        if not m:
            continue
        
        indent = line[:len(line) - len(line.lstrip(" "))]
        header = line[:m.start()].rstrip()
        if runs(lines[:i] + [header, f"{indent}sorry"] + lines[i + 1:]):
            lines[i] = header
            lines.insert(i + 1, f"{indent}sorry")
            return "\n".join(lines) + ("" if text.endswith("\n") else "\n"), True
    
    return (text if text.endswith("\n") else text + "\n"), False


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
                 max_repairs_per_hole: int = 2, trace: bool = False,
                 priors_path: Optional[str] = None, context_hints: bool = False,
                 lib_templates: bool = False, alpha: float = 1.0, beta: float = 0.5,
                 gamma: float = 0.2, hintlex_path: Optional[str] = None,
                 hintlex_top: int = 8) -> PlanAndFillResult:
    """Plan and fill holes in Isar proofs."""
    server_info, proc = start_isabelle_server(name="planner", log_file="logs/planner_ui.log")
    isa = get_isabelle_client(server_info)
    session = isa.session_start(session=ISABELLE_SESSION)
    
    t0 = time.monotonic()
    left_s = lambda: max(0.0, timeout - (time.monotonic() - t0))
    
    try:
        # Generate outline
        if legacy_single_outline:
            full = propose_isar_skeleton(goal, model=model, temp=0.35, force_outline=(mode=="outline")).text
        else:
            temps = tuple(outline_temps) if outline_temps else (0.35, 0.55, 0.85)
            k = int(outline_k) if outline_k is not None else 3
            best, _ = propose_isar_skeleton_diverse_best(
                goal, isabelle=isa, session_id=session, model=model, temps=temps, k=k,
                force_outline=(mode=="outline"), priors_path=priors_path,
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
            if _verify_full_proof(isa, session, full):
                return PlanAndFillResult(True, full, [], [])
            
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
        fills, failed = [], []
        repair_progress: dict[int, int] = {}
        stage_tries: dict[Tuple[int, int], int] = {}
        STAGE2_CAP = 3
        _skip_fill_logged_once: set[Tuple[str, int]] = set()
        
        # NEW: Track which hole we're currently focusing on to ensure we don't skip ahead
        focused_hole_key: Optional[int] = None
        
        while "sorry" in full and left_s() > 0:
            spans = find_sorry_spans(full)
            if not spans:
                break
            
            # NEW: If we have a focused hole, find it. Otherwise take the first.
            span = None
            if focused_hole_key is not None:
                for s in spans:
                    if _hole_fingerprint(full, s) == focused_hole_key:
                        span = s
                        break
                if span is None:
                    # Focused hole was closed! Clear focus and take first hole.
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
                full2, ok, script = _fill_one_hole(
                    isa, session, full, span, goal_text, model,
                    per_hole_timeout=per_hole_budget, trace=trace
                )

                if ok and full2 != full:
                    # Fill succeeded and verified!
                    full = full2
                    fills.append(script)
                    repair_progress.pop(hole_key, None)
                    focused_hole_key = None  # Clear focus, move to next hole
                    continue
                elif not ok and full2 != full:
                    # Fill made partial progress but didn't verify
                    if trace:
                        print(f"[fill] Partial progress from fill (unverified). Opening sorries and staying focused on this block...")
                    full = full2
                    full2, opened = _open_minimal_sorries(isa, session, full)
                    if opened:
                        full = full2
                        # CRITICAL: Stay focused on this hole (or the first sorry in this block)
                        # Don't reset progress - keep trying to fill the newly opened sorries
                        focused_hole_key = hole_key  # Keep working on this area
                        continue
                    else:
                        # Couldn't open sorries - escalate to repair
                        if trace:
                            print(f"[fill] Could not open sorries. Escalating to repair stage 1...")
                        repair_progress[hole_key] = 1
                        focused_hole_key = hole_key
                        # Fall through to repair
                else:
                    # Fill made no progress - escalate to repair
                    if trace:
                        print(f"[fill] Fill made no progress. Escalating to repair stage 1...")
                    repair_progress[hole_key] = 1
                    focused_hole_key = hole_key
                    # Fall through to repair
            else:
                if trace and (hole_key, start_stage) not in _skip_fill_logged_once:
                    print(
                        f"[fill] Skipping fill for hole @{hole_key}; running repairs at stage {start_stage}"
                    )
                    _skip_fill_logged_once.add((hole_key, start_stage))

            # Try CEGIS repairs (only if we're in repair mode for this hole)
            if repair_progress.get(hole_key, 0) > 0 and repairs and left_s() > 6:
                state = _print_state_before_hole(isa, session, full, span, trace)
                eff_goal = _effective_goal_from_state(state, goal_text, full, span, trace)
                
                patched, applied, _ = try_cegis_repairs(
                    full_text=full, hole_span=span, goal_text=eff_goal, model=model,
                    isabelle=isa, session=session, 
                    repair_budget_s=min(30.0, max(15.0, left_s() * 0.33)),
                    max_ops_to_try=max_repairs_per_hole, beam_k=2, 
                    allow_whole_fallback=False,
                    trace=trace, resume_stage=start_stage,
                )
                
                if applied and patched != full:
                    if _verify_full_proof(isa, session, patched):
                        # Repair succeeded and verified!
                        if trace:
                            print(f"[repair] Stage {start_stage} repair verified! Clearing progress and moving on.")
                        full = patched
                        repair_progress.clear()
                        stage_tries.clear()
                        focused_hole_key = None  # Clear focus, move to next hole
                        continue
                    else:
                        # Repair made progress but didn't verify - open sorries and retry fill
                        if trace:
                            print(f"[repair] Stage {start_stage} partial progress (unverified). Opening sorries and retrying fill...")
                        full = patched
                        full2, opened = _open_minimal_sorries(isa, session, full)
                        if opened:
                            full = full2
                            # CRITICAL: Reset to stage 0 (fill) and stay focused on this block
                            repair_progress[hole_key] = 0
                            stage_tries.pop((hole_key, start_stage), None)
                            focused_hole_key = hole_key
                            continue
                        else:
                            # Couldn't open sorries - escalate stage
                            if trace:
                                print(f"[repair] Could not open sorries; escalating stage...")
                
                # Repair didn't help or couldn't open sorries → count attempt and escalate
                key = (hole_key, start_stage)
                stage_tries[key] = stage_tries.get(key, 0) + 1
                
                if start_stage < 2:
                    repair_progress[hole_key] = min(start_stage + 1, 2)
                    focused_hole_key = hole_key  # Stay focused
                else:
                    # Stage 2 cap reached → regenerate whole proof
                    if stage_tries.get((hole_key, 2), 0) >= STAGE2_CAP:
                        if trace:
                            print(f"[repair] Stage-2 cap reached for hole @{hole_key}. Regenerating whole proof…")
                        regen_budget = min(40.0, max(8.0, left_s() * 0.8))
                        new_full, ok_re, _ = regenerate_whole_proof(
                            full_text=full, goal_text=goal_text, model=model,
                            isabelle=isa, session=session, budget_s=regen_budget,
                            trace=trace, prior_outline_text=full
                        )
                        if ok_re and new_full != full:
                            full = new_full
                            repair_progress.clear()
                            stage_tries.clear()
                            focused_hole_key = None
                            continue
                        
                        # Regeneration failed → fresh outline
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
                    
                    repair_progress[hole_key] = 2
                    focused_hole_key = hole_key  # Stay focused
        
        # Final verification
        success = ("sorry" not in full)
        if success:
            if _verify_full_proof(isa, session, full):
                return PlanAndFillResult(True, full, fills, failed)
        
        return PlanAndFillResult(False, full, fills, failed)
    
    finally:
        _cleanup_resources(isa, proc)