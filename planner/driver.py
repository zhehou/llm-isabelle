from __future__ import annotations

import time
import re
import os
import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from planner.skeleton import (
    Skeleton, find_sorry_spans, propose_isar_skeleton, propose_isar_skeleton_diverse_best,
)
from planner.repair import try_cegis_repairs, regenerate_whole_proof, _APPLY_OR_BY as _TACTIC_LINE_RE
from prover.config import ISABELLE_SESSION
from prover.isabelle_api import build_theory, get_isabelle_client, start_isabelle_server
from planner.goals import (
    _print_state_before_hole, _log_state_block, _effective_goal_from_state,
    _first_lemma_line, _extract_goal_from_lemma_line, _cleanup_resources,
    _verify_full_proof, _run_theory_with_timeout
)
from prover.prover import prove_goal

# Constants
_INLINE_BY_TAIL = re.compile(r"\s+by\s+.+$")
_BARE_DOT = re.compile(r"(?m)^\s*\.\s*$")
_HEAD_CMD_RE = re.compile(r"^\s*(have|show|obtain)\b")
_ISA_VERIFY_TIMEOUT_S = int(os.getenv("ISABELLE_VERIFY_TIMEOUT_S", "30"))

@dataclass(slots=True)
class PlanAndFillResult:
    success: bool
    outline: str
    fills: List[str]
    failed_holes: List[int]

def _hole_fingerprint(full_text: str, span: tuple[int, int], context: int = 80) -> str:
    s, e = span
    snippet = full_text[max(0, s - context):min(len(full_text), e + context)]
    return hashlib.sha1(snippet.encode("utf-8")).hexdigest()[:16]

def _nearest_sorry_span(spans: List[Tuple[int, int]], target_s: int) -> Optional[Tuple[int, int]]:
    return min(spans, key=lambda sp: abs(sp[0] - target_s)) if spans else None

def _get_prev_line(text: str, span: Tuple[int, int]) -> str:
    try:
        line_start = text.rfind("\n", 0, span[0]) + 1
        prev_start = text.rfind("\n", 0, line_start - 1) + 1
        return text[prev_start:line_start]
    except Exception:
        return ""

def _is_stale_hole(text: str, span: Tuple[int, int]) -> bool:
    prev = _get_prev_line(text, span)
    return bool(_INLINE_BY_TAIL.search(prev) or _TACTIC_LINE_RE.match(prev) or prev.strip() in {"done", "."})

def _insert_above_hole(text: str, hole: Tuple[int, int], lines: List[str]) -> str:
    s = hole[0]
    ls = text.rfind("\n", 0, s) + 1
    le = text.find("\n", s)
    indent = (text[ls:(le if le != -1 else len(text))]).split()[0] if ls < len(text) else ""
    indent = text[ls:ls + len(text[ls:]) - len(text[ls:].lstrip(" "))]
    payload = "".join(f"{indent}{ln.strip()}\n" for ln in lines if ln.strip())
    return text[:s] + payload + text[s:]

def _is_inside_have_show(text: str, span: Tuple[int, int]) -> bool:
    scan_start = max(0, text.rfind("\n", 0, max(0, text.rfind("\n", 0, span[0]) - 512)) + 1)
    segment = text[scan_start:span[0]]
    for line in reversed(segment.splitlines()):
        if _HEAD_CMD_RE.match(line or ""):
            return True
    return False

def _extract_steps_and_finisher(res: dict) -> Tuple[List[str], str]:
    steps = [str(s) for s in res.get("steps", [])]
    
    # Finisher candidates
    fin_candidates = []
    for k in ("finisher", "finish", "final"):
        if isinstance(v := res.get(k), str):
            fin_candidates.append(v)
    for k in ("finishers", "sledge_finishers"):
        if isinstance(vs := res.get(k), (list, tuple)):
            fin_candidates.extend([str(x) for x in vs if isinstance(x, str)])
    
    # Applies
    applies = [s for s in steps if s.startswith("apply")]
    if not applies:
        for k in ("applies", "apply_steps"):
            if isinstance(vs := res.get(k), (list, tuple)):
                applies = [str(x) for x in vs if isinstance(x, str) and x.startswith("apply")]
                if applies:
                    break
    
    # Finisher
    finisher = next((s for s in steps if s.startswith("by ") or s.strip() == "done"), "")
    if not finisher:
        finisher = next((x for x in fin_candidates if x.startswith("by ") or x.strip() == "done"), "")
    
    return applies, finisher

def _fill_one_hole(isabelle, session: str, full_text: str, hole_span: Tuple[int, int],
                  goal_text: str, model: Optional[str], per_hole_timeout: int, 
                  *, trace: bool = False) -> Tuple[str, bool, str]:
    if _is_stale_hole(full_text, hole_span):
        s, e = hole_span
        return full_text[:s] + "\n" + full_text[e:], True, "(stale-hole)"
    
    state_block = _print_state_before_hole(isabelle, session, full_text, hole_span, trace)
    _log_state_block("fill", state_block, trace=trace)
    eff_goal = _effective_goal_from_state(state_block, goal_text, full_text, hole_span, trace)
    
    res = prove_goal(
        isabelle, session, eff_goal, model_name_or_ensemble=model, beam_w=3, max_depth=6, 
        hint_lemmas=6, timeout=per_hole_timeout, use_sledge=True, sledge_timeout=10,
        sledge_every=1, trace=trace, use_color=False, enable_reranker=True, 
        initial_state_hint=state_block, use_qc=False, use_np=False, facts_limit=8,
    )
    
    applies, finisher = _extract_steps_and_finisher(res)
    if not (applies or finisher):
        return full_text, False, "no-steps"
    
    if finisher:
        script = applies + [finisher]
        s, e = hole_span
        new_text = full_text[:s] + "\n  " + "\n  ".join(script) + "\n" + full_text[e:]
        return (new_text, True, "\n".join(script)) if _verify_full_proof(isabelle, session, new_text) else (full_text, False, "finisher-unverified")
    
    if applies:
        if _is_inside_have_show(full_text, hole_span):
            if trace:
                print("[fill] apply-only inside have/show; escalating to repair.")
            return full_text, False, "apply-inside-have/show"
        
        segment = full_text[max(0, full_text.rfind("\n", 0, max(0, hole_span[0] - 512)) + 1):hole_span[0]]
        dedup = [a for a in applies if a not in segment]
        if not dedup:
            return full_text, False, "apply-duplicate"
        
        return _insert_above_hole(full_text, hole_span, dedup), False, "\n".join(dedup)
    
    return full_text, False, "no-tactics"

def _proof_bounds_top_level(text: str) -> Optional[Tuple[int, int]]:
    qed_matches = list(re.finditer(r"(?m)^\s*qed\b", text))
    if not qed_matches:
        return None
    end = qed_matches[-1].end()
    proof_matches = list(re.finditer(r"(?m)^\s*proof\b.*$", text[:qed_matches[-1].start()]))
    return (proof_matches[-1].start(), end) if proof_matches else None

def _tactic_spans_topdown(text: str) -> List[Tuple[int, int]]:
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

def _extract_error_lines(error_text: str) -> List[int]:
    return [int(m.group(1)) for m in re.finditer(r"line\s+(\d+)", error_text)]

def _open_minimal_sorries(isabelle, session: str, text: str) -> Tuple[str, bool]:
    try:
        thy = build_theory(text.splitlines(), add_print_state=False, end_with=None)
        _run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_VERIFY_TIMEOUT_S)
        return (text if text.endswith("\n") else text + "\n"), False
    except Exception as e:
        err_lines = _extract_error_lines(str(e))
        if not err_lines:
            return (text if text.endswith("\n") else text + "\n"), False
        
        lines = text.splitlines()
        failing_idx = min(err_lines) - 1
        
        for i in range(min(failing_idx, len(lines) - 1), -1, -1):
            line = lines[i]
            indent = line[:len(line) - len(line.lstrip(" "))]
            
            if _TACTIC_LINE_RE.match(line) or line.strip() in {"done", "."} or _BARE_DOT.match(line):
                lines[i] = f"{indent}sorry"
                return "\n".join(lines) + ("" if text.endswith("\n") else "\n"), True
            
            if m := _INLINE_BY_TAIL.search(line):
                lines[i] = line[:m.start()].rstrip()
                lines.insert(i + 1, f"{indent}sorry")
                return "\n".join(lines) + ("" if text.endswith("\n") else "\n"), True
        
        return (text if text.endswith("\n") else text + "\n"), False

def _repair_failed_proof_topdown(isa, session, full: str, goal_text: str, model: str,
                                 left_s, max_repairs_per_hole: int, trace: bool) -> Tuple[str, bool]:
    t_spans = _tactic_spans_topdown(full)
    if not t_spans:
        return full, False
    
    for i, span in enumerate(t_spans):
        if left_s() <= 3.0:
            break
        st = _print_state_before_hole(isa, session, full, span, trace)
        eff_goal = _effective_goal_from_state(st, goal_text, full, span, trace)
        
        patched, applied, _ = try_cegis_repairs(
            full_text=full, hole_span=span, goal_text=eff_goal, model=model,
            isabelle=isa, session=session, repair_budget_s=min(30.0, max(15.0, left_s() * 0.33)),
            max_ops_to_try=max_repairs_per_hole, beam_k=2, allow_whole_fallback=False, 
            trace=trace, resume_stage=0,
        )
        
        if applied and patched != full:
            if _verify_full_proof(isa, session, patched):
                return patched, True
            if trace:
                print("[repair] Partial progress in topdown repair. Opening sorries...")
            full2, opened = _open_minimal_sorries(isa, session, patched)
            if opened:
                full = full2
                t_spans = _tactic_spans_topdown(full)
    
    return full, False

def _handle_repair_result(full: str, patched: str, span: Tuple[int, int], hole_key: int,
                         start_stage: int, stage_tries: dict, repair_progress: dict,
                         isa, session: str, trace: bool) -> Tuple[str, Optional[int], bool]:
    STAGE1_CAP, STAGE2_CAP = 2, 3
    key = (hole_key, start_stage)
    stage_tries[key] = stage_tries.get(key, 0) + 1
    
    should_escalate = ((start_stage == 1 and stage_tries[key] >= STAGE1_CAP) or 
                      (start_stage == 2 and stage_tries.get((hole_key, 2), 0) >= STAGE2_CAP))
    
    if should_escalate:
        repair_progress[hole_key] = 2 if start_stage < 2 else 2
        return full, hole_key, (start_stage < 2)
    
    if trace:
        cap = STAGE1_CAP if start_stage == 1 else STAGE2_CAP
        print(f"[repair] Stage {start_stage} unverified (attempt {stage_tries[key]}/{cap}). Opening sorries...")
    
    full2, opened = _open_minimal_sorries(isa, session, patched)
    if opened:
        near = _nearest_sorry_span(find_sorry_spans(full2), span[0])
        if near:
            new_hole_key = _hole_fingerprint(full2, near)
            repair_progress[new_hole_key] = start_stage
            return full2, new_hole_key, True
        return full2, None, True
    
    if trace:
        print("[repair] Could not open sorries; escalating...")
    if not find_sorry_spans(patched):
        patched = full
    repair_progress[hole_key] = 2 if start_stage < 2 else 2
    return patched, hole_key, True

def _run_fill_loop(isa, session: str, full: str, goal_text: str, model: Optional[str],
                  left_s, repairs: bool, max_repairs_per_hole: int, trace: bool,
                  outline_params: dict) -> Tuple[str, List[str], List[int]]:
    fills, failed = [], []
    repair_progress: dict[int, int] = {}
    stage_tries: dict[Tuple[int, int], int] = {}
    focused_hole_key: Optional[int] = None
    skip_fill_logged: set[Tuple[str, int]] = set()
    
    while "sorry" in full and left_s() > 0:
        spans = find_sorry_spans(full)
        if not spans:
            break
        
        span = None
        if focused_hole_key:
            span = next((s for s in spans if _hole_fingerprint(full, s) == focused_hole_key), None)
            if not span and trace:
                print(f"[fill] Focused hole closed. Moving to first hole.")
        
        span = span or spans[0]
        hole_key = _hole_fingerprint(full, span)
        per_hole_budget = int(max(5, left_s() / max(1, len(spans))))
        start_stage = repair_progress.get(hole_key, 0)
        
        if start_stage == 0:
            full2, ok, script = _fill_one_hole(isa, session, full, span, goal_text, model, per_hole_budget, trace=trace)
            
            if ok and full2 != full:
                full, fills, focused_hole_key = full2, fills + [script], None
                repair_progress.pop(hole_key, None)
                continue
            
            if full2 != full:
                if trace:
                    print("[fill] Partial progress from fill. Opening sorries...")
                old_start = span[0]
                full2, opened = _open_minimal_sorries(isa, session, full2)
                if opened:
                    full = full2
                    near = _nearest_sorry_span(find_sorry_spans(full), old_start)
                    focused_hole_key = _hole_fingerprint(full, near) if near else None
                    continue
            
            if trace:
                print("[fill] Fill made no progress. Escalating to repair stage 1...")
            repair_progress[hole_key], focused_hole_key, start_stage = 1, hole_key, 1
        else:
            if trace and (hole_key, start_stage) not in skip_fill_logged:
                print(f"[fill] Skipping fill for hole; running repairs at stage {start_stage}")
                skip_fill_logged.add((hole_key, start_stage))
        
        current_stage = repair_progress.get(hole_key, 0)
        if current_stage > 0 and repairs and left_s() > 6:
            full_before = full
            state = _print_state_before_hole(isa, session, full, span, trace)
            eff_goal = _effective_goal_from_state(state, goal_text, full, span, trace)
            
            patched, applied, _ = try_cegis_repairs(
                full_text=full, hole_span=span, goal_text=eff_goal, model=model,
                isabelle=isa, session=session, repair_budget_s=min(30.0, max(15.0, left_s() * 0.33)),
                max_ops_to_try=max_repairs_per_hole, beam_k=2, allow_whole_fallback=False,
                trace=trace, resume_stage=current_stage,
            )
            
            if patched != full:
                if _verify_full_proof(isa, session, patched):
                    if trace:
                        print(f"[repair] Stage {current_stage} repair verified!")
                    full, focused_hole_key = patched, None
                    repair_progress.clear()
                    stage_tries.clear()
                    continue
                
                new_full, new_focus, should_cont = _handle_repair_result(
                    full, patched, span, hole_key, start_stage, stage_tries, repair_progress, isa, session, trace
                )
                full, focused_hole_key = new_full, new_focus
                
                if should_cont:
                    continue
                
                if trace:
                    print("[repair] Stage 2 cap reached. Regenerating whole proof...")
                
                new_full, ok_re, _ = regenerate_whole_proof(
                    full_text=full, goal_text=goal_text, model=model, isabelle=isa, session=session,
                    budget_s=min(40.0, max(8.0, left_s() * 0.8)), trace=trace, prior_outline_text=full
                )
                
                if ok_re and new_full != full:
                    full, focused_hole_key = new_full, None
                    repair_progress.clear()
                    stage_tries.clear()
                    continue
                
                if trace:
                    print("[repair] Whole regeneration failed; proposing fresh outline...")
                
                best, _ = propose_isar_skeleton_diverse_best(
                    goal_text, isabelle=isa, session_id=session, model=model, force_outline=True, **outline_params
                )
                full, focused_hole_key = best.text, None
                repair_progress.clear()
                stage_tries.clear()
                continue
            
            key = (hole_key, start_stage)
            stage_tries[key] = stage_tries.get(key, 0) + 1
            repair_progress[hole_key] = min(start_stage + 1, 2)
            focused_hole_key = hole_key
    
    return full, fills, failed

def plan_outline(goal: str, *, model: Optional[str] = None, outline_k: Optional[int] = None,
                outline_temps: Optional[Iterable[float]] = None, legacy_single_outline: bool = False,
                priors_path: Optional[str] = None, context_hints: bool = False,
                lib_templates: bool = False, alpha: float = 1.0, beta: float = 0.5,
                gamma: float = 0.2, hintlex_path: Optional[str] = None, hintlex_top: int = 8) -> str:
    server_info, proc = start_isabelle_server(name="planner", log_file="logs/planner_ui.log")
    isa = get_isabelle_client(server_info)
    session = isa.session_start(session=ISABELLE_SESSION)
    
    try:
        if legacy_single_outline:
            return propose_isar_skeleton(goal, model=model, temp=0.35, force_outline=True).text
        
        temps = tuple(outline_temps) if outline_temps else (0.35, 0.55, 0.85)
        k = int(outline_k) if outline_k is not None else 3
        best, _ = propose_isar_skeleton_diverse_best(
            goal, isabelle=isa, session_id=session, model=model, temps=temps, k=k, force_outline=True,
            priors_path=priors_path, context_hints=context_hints, lib_templates=lib_templates,
            alpha=alpha, beta=beta, gamma=gamma, hintlex_path=hintlex_path, hintlex_top=hintlex_top,
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
    server_info, proc = start_isabelle_server(name="planner", log_file="logs/planner_ui.log")
    isa = get_isabelle_client(server_info)
    session = isa.session_start(session=ISABELLE_SESSION)
    
    t0 = time.monotonic()
    left_s = lambda: max(0.0, timeout - (time.monotonic() - t0))
    
    try:
        temps = tuple(outline_temps) if outline_temps else (0.35, 0.55, 0.85)
        k = int(outline_k) if outline_k is not None else 3
        outline_params = {
            'temps': temps, 'k': k, 'priors_path': priors_path, 'context_hints': context_hints,
            'lib_templates': lib_templates, 'alpha': alpha, 'beta': beta, 'gamma': gamma,
            'hintlex_path': hintlex_path, 'hintlex_top': hintlex_top,
        }
        
        if legacy_single_outline:
            full = propose_isar_skeleton(goal, model=model, temp=0.35, force_outline=(mode == "outline")).text
        else:
            best, _ = propose_isar_skeleton_diverse_best(
                goal, isabelle=isa, session_id=session, model=model, force_outline=(mode == "outline"), **outline_params
            )
            full = best.text
        
        if mode == "outline":
            return PlanAndFillResult(True, full, [], [])
        
        spans = find_sorry_spans(full)
        if not spans:
            if _verify_full_proof(isa, session, full):
                return PlanAndFillResult(True, full, [], [])
            if repairs and left_s() > 6.0:
                full, ok = _repair_failed_proof_topdown(isa, session, full, goal, model, left_s, max_repairs_per_hole, trace)
                if ok:
                    return PlanAndFillResult(True, full, [], [])
            full2, opened = _open_minimal_sorries(isa, session, full)
            if not opened:
                return PlanAndFillResult(False, full2 if opened else full, [], [0])
            full = full2
        
        lemma_line = _first_lemma_line(full)
        if not lemma_line:
            return PlanAndFillResult(False, full, [], [0])
        
        goal_text = _extract_goal_from_lemma_line(lemma_line)
        full, fills, failed = _run_fill_loop(isa, session, full, goal_text, model, left_s, repairs, max_repairs_per_hole, trace, outline_params)
        
        success = "sorry" not in full
        if success and _verify_full_proof(isa, session, full):
            return PlanAndFillResult(True, full, fills, failed)
        
        return PlanAndFillResult(False, full, fills, failed)
    finally:
        _cleanup_resources(isa, proc)