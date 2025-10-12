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
    """
    Localize failing finisher with minimal opening.
    Finds the first tactic that fails verification and replaces it with sorry.
    """
    # First check if the whole thing passes
    try:
        thy = build_theory(text.splitlines(), add_print_state=False, end_with=None)
        _run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_VERIFY_TIMEOUT_S)
        print("[_open_minimal_sorries] Document already passes, nothing to open")  # DEBUG
        return (text if text.endswith("\n") else text + "\n"), False
    except Exception:
        print("[_open_minimal_sorries] Document fails, looking for failing tactic")  # DEBUG
        pass
    
    # Get error information
    _, errs = _quick_state_and_errors(isabelle, session, text)
    err_lines = _extract_error_lines(errs)
    
    if not err_lines:
        print("[_open_minimal_sorries] No error lines found")  # DEBUG
        return (text if text.endswith("\n") else text + "\n"), False
    
    lines = text.splitlines()
    failing_line_1based = min(err_lines)
    failing_idx = failing_line_1based - 1
    
    print(f"[_open_minimal_sorries] First error at line {failing_line_1based}: {lines[failing_idx] if 0 <= failing_idx < len(lines) else 'N/A'}")  # DEBUG
    
    # Search backwards from error
    for i in range(min(failing_idx, len(lines) - 1), -1, -1):
        line = lines[i]
        
        # Check for tactics
        if _TACTIC_LINE_RE.match(line) or line.strip() == "done" or _BARE_DOT.match(line):
            print(f"[_open_minimal_sorries] Found tactic at line {i+1}: {line}")  # DEBUG
            indent = line[:len(line) - len(line.lstrip(" "))]
            lines[i] = f"{indent}sorry"
            return "\n".join(lines) + ("" if text.endswith("\n") else "\n"), True
        
        m = _INLINE_BY_TAIL.search(line)
        if m:
            print(f"[_open_minimal_sorries] Found inline 'by' at line {i+1}: {line}")  # DEBUG
            indent = line[:len(line) - len(line.lstrip(" "))]
            header = line[:m.start()].rstrip()
            lines[i] = header
            lines.insert(i + 1, f"{indent}sorry")
            return "\n".join(lines) + ("" if text.endswith("\n") else "\n"), True
    
    print("[_open_minimal_sorries] No tactic found to open")  # DEBUG
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
        
        # Track which hole we're currently focusing on
        focused_hole_key: Optional[int] = None
        
        while "sorry" in full and left_s() > 0:
            spans = find_sorry_spans(full)
            if not spans:
                break
            
            # If we have a focused hole, find it. Otherwise take the first.
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
                    # Keep the old start for nearest-hole tracking
                    old_start = span[0]
                    full = full2
                    full2, opened = _open_minimal_sorries(isa, session, full)
                    if opened:
                        full = full2
                        # Recompute focus: choose the sorry span nearest where the old hole began
                        new_spans = find_sorry_spans(full)
                        near = _nearest_sorry_span(new_spans, old_start)
                        focused_hole_key = _hole_fingerprint(full, near) if near else None
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
                    start_stage = 1  # make sure we resume at stage 1 now
            else:
                if trace and (hole_key, start_stage) not in _skip_fill_logged_once:
                    print(
                        f"[fill] Skipping fill for hole @{hole_key}; running repairs at stage {start_stage}"
                    )
                    _skip_fill_logged_once.add((hole_key, start_stage))

            # Try CEGIS repairs (only if we're in repair mode for this hole)
            current_stage = repair_progress.get(hole_key, 0)
            if current_stage > 0 and repairs and left_s() > 6:
                # Save the current state before repair
                full_before_repair = full
                
                state = _print_state_before_hole(isa, session, full, span, trace)
                eff_goal = _effective_goal_from_state(state, goal_text, full, span, trace)
                
                patched, applied, _ = try_cegis_repairs(
                    full_text=full, hole_span=span, goal_text=eff_goal, model=model,
                    isabelle=isa, session=session, 
                    repair_budget_s=min(30.0, max(15.0, left_s() * 0.33)),
                    max_ops_to_try=max_repairs_per_hole, beam_k=2, 
                    allow_whole_fallback=False,
                    trace=trace, resume_stage=current_stage,
                )
                
                # Check if repair made any changes
                if patched != full:
                    if _verify_full_proof(isa, session, patched):
                        # Repair succeeded and verified!
                        if trace:
                            print(f"[repair] Stage {current_stage} repair verified! Clearing progress and moving on.")
                        full = patched
                        repair_progress.clear()
                        stage_tries.clear()
                        focused_hole_key = None
                        continue
                    else:
                        # Repair made changes but didn't verify
                        # Count this attempt FIRST before deciding what to do
                        key = (hole_key, start_stage)
                        stage_tries[key] = stage_tries.get(key, 0) + 1
                        
                        # Define per-stage attempt caps
                        STAGE1_CAP = 2  # Try stage 1 twice before escalating
                        STAGE2_CAP = 3  # Try stage 2 three times before whole regen
                        
                        # Check if we should escalate to next stage
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
                            # Don't open sorries, just escalate
                            if start_stage < 2:
                                repair_progress[hole_key] = 2
                                focused_hole_key = hole_key
                                continue
                            else:
                                # Stage 2 exhausted → whole regeneration
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
                        
                        # Still have attempts left at this stage → open sorries and retry
                        if trace:
                            print(f"[repair] Stage {start_stage} made changes but unverified (attempt {stage_tries[key]}/{STAGE1_CAP if start_stage == 1 else STAGE2_CAP}). Opening sorries and retrying fill...")
                        
                        full = patched
                        full2, opened = _open_minimal_sorries(isa, session, full)
                        
                        if opened:
                            full = full2
                            # Find the new sorry and keep the same stage
                            old_start = span[0]
                            new_spans = find_sorry_spans(full)
                            near = _nearest_sorry_span(new_spans, old_start)
                            if near:
                                new_hole_key = _hole_fingerprint(full, near)
                                # Keep the current stage - we're still repairing the same logical block
                                repair_progress[new_hole_key] = start_stage
                                focused_hole_key = new_hole_key
                            else:
                                focused_hole_key = None
                            continue
                        else:
                            # Could not open sorries - this means repair removed the sorry but didn't fix the proof
                            # Check if there are any sorries left at all
                            remaining_sorries = find_sorry_spans(full)
                            if not remaining_sorries:
                                # Repair removed the sorry without fixing the proof - revert and escalate
                                if trace:
                                    print(f"[repair] Repair removed sorry without fixing proof. Reverting and escalating...")
                                full = full_before_repair
                                
                            # Escalate immediately
                            if trace:
                                print(f"[repair] Could not open sorries; escalating stage...")
                            if start_stage < 2:
                                repair_progress[hole_key] = 2
                                focused_hole_key = hole_key
                            continue
                
                # Repair made no changes → count attempt and escalate
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
            if _verify_full_proof(isa, session, full):
                return PlanAndFillResult(True, full, fills, failed)
        
        return PlanAndFillResult(False, full, fills, failed)
    
    finally:
        _cleanup_resources(isa, proc)