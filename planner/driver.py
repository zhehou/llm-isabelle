from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from planner.skeleton import (
    Skeleton,
    find_sorry_spans,
    propose_isar_skeleton,               # legacy single-outline path
    propose_isar_skeleton_diverse_best,  # diverse outlines + quick sketch check (+ scoring)
)
from planner.repair import try_local_repairs  # LLM-guided local repair

from prover.config import ISABELLE_SESSION
from prover.isabelle_api import (
    build_theory,
    get_isabelle_client,
    last_print_state_block,
    run_theory,
    start_isabelle_server,
)
from prover.prover import prove_goal


@dataclass(slots=True)
class PlanAndFillResult:
    """
    success:
      - In 'auto' mode: True iff the proof is fully filled (no 'sorry').
      - In 'outline' mode: True (operation succeeded), even though 'sorry' is expected.
    """
    success: bool
    outline: str  # final Isar text (outline-only or with holes filled)
    fills: List[str]  # text of per-hole proof scripts inserted (empty in outline mode)
    failed_holes: List[int]  # indices of holes that failed to fill (empty in outline mode)


# ----------------------------
# Small helpers
# ----------------------------

def _extract_goal_from_lemma_line(lemma_line: str) -> str:
    q1 = lemma_line.find('"')
    q2 = lemma_line.rfind('"')
    if q1 == -1 or q2 == -1 or q2 <= q1:
        raise ValueError(f"Cannot parse lemma line: {lemma_line!r}")
    return lemma_line[q1 + 1 : q2]


def _first_lemma_line(full_text: str) -> str:
    for L in full_text.splitlines():
        if L.strip().startswith("lemma "):
            return L
    return ""


def _print_state_before_hole(isabelle, session: str, full_text: str, hole_span: Tuple[int, int]) -> str:
    s, e = hole_span
    prefix_with_placeholder = full_text[:s] + "sorry\n" + full_text[e:]
    thy = build_theory(prefix_with_placeholder.splitlines(), add_print_state=True, end_with="sorry")
    resps = run_theory(isabelle, session, thy)
    return last_print_state_block(resps) or ""


def _fill_one_hole(
    isabelle,
    session: str,
    full_text: str,
    hole_span: Tuple[int, int],
    goal_text: str,
    model: Optional[str],
    per_hole_timeout: int,
) -> Tuple[str, bool, str]:
    state_block = _print_state_before_hole(isabelle, session, full_text, hole_span)

    res = prove_goal(
        isabelle,
        session,
        goal_text,
        model_name_or_ensemble=model,
        beam_w=2,
        max_depth=8,
        hint_lemmas=6,
        timeout=per_hole_timeout,
        models=None,
        save_dir=None,
        use_sledge=True,
        sledge_timeout=10,  # keep your current default
        sledge_every=2,
        trace=False,
        use_color=False,
        use_qc=True,
        qc_timeout=2,
        qc_every=1,
        use_np=True,
        np_timeout=5,
        np_every=2,
        facts_limit=6,
        do_minimize=True,
        minimize_timeout=8,
        do_variants=True,
        variant_timeout=6,
        variant_tries=24,
        enable_reranker=True,
        initial_state_hint=state_block,
    )

    steps: List[str] = [str(s) for s in res.get("steps", [])]
    if not steps:
        return full_text, False, "no-steps"

    applies = [s for s in steps if s.startswith("apply")]
    fin = next((s for s in steps if s.startswith("by ") or s.strip() == "done"), "")
    script_lines: List[str] = []
    script_lines += applies
    if fin:
        script_lines.append(fin)
    if not script_lines:
        return full_text, False, "no-tactics"

    insert = "\n  " + "\n  ".join(script_lines) + "\n"
    s, e = hole_span
    new_text = full_text[:s] + insert + full_text[e:]
    return new_text, True, "\n".join(script_lines)


# ----------------------------
# Public API
# ----------------------------

def plan_outline(
    goal: str,
    *,
    model: Optional[str] = None,
    outline_k: Optional[int] = None,
    outline_temps: Optional[Iterable[float]] = None,
    legacy_single_outline: bool = False,
    priors_path: Optional[str] = None,
    context_hints: bool = False,
    lib_templates: bool = False,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.2,
    hintlex_path: Optional[str] = None,
    hintlex_top: int = 8,
) -> str:
    """
    Generate an Isar outline with 'sorry' placeholders and return it. No filling.
    """
    server_info, proc = start_isabelle_server(name="planner", log_file="logs/planner_ui.log")
    isa = get_isabelle_client(server_info)
    session = isa.session_start(session=ISABELLE_SESSION)
    try:
        if legacy_single_outline:
            skel = propose_isar_skeleton(goal, model=model, temp=0.35, force_outline=True)
            return skel.text
        temps = tuple(outline_temps) if outline_temps else (0.35, 0.55, 0.85)
        k = int(outline_k) if outline_k is not None else 3
        best, _diag = propose_isar_skeleton_diverse_best(
            goal,
            isabelle=isa,
            session_id=session,
            model=model,
            temps=temps,
            k=k,
            force_outline=True,          # <- enforce outline shape
            priors_path=priors_path,
            context_hints=context_hints,
            lib_templates=lib_templates,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            hintlex_path=hintlex_path,
            hintlex_top=hintlex_top,
        )
        return best.text
    finally:
        try:
            isa.shutdown()
            try:
                from planner.experiments import _close_client_loop_safely  # reuse helper
                _close_client_loop_safely(isa)
            except Exception:
                pass
        except Exception:
            pass
        try:
            if hasattr(proc, "terminate"):
                try: proc.terminate()
                except Exception: pass
            if hasattr(proc, "kill"):
                try: proc.kill()
                except Exception: pass
        except Exception:
            pass


def plan_and_fill(
    goal: str,
    model: Optional[str] = None,
    timeout: int = 100,
    *,
    mode: str = "auto",  # "auto" (plan+fill) or "outline" (plan-only)
    # Diverse-outline controls
    outline_k: Optional[int] = None,
    outline_temps: Optional[Iterable[float]] = None,
    legacy_single_outline: bool = False,
    # Local repair controls
    repairs: bool = True,
    max_repairs_per_hole: int = 2,
    repair_trace: bool = False,
    # planner scoring/context flags
    priors_path: Optional[str] = None,
    context_hints: bool = False,
    lib_templates: bool = False,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.2,
    # micro-RAG hint lexicon
    hintlex_path: Optional[str] = None,
    hintlex_top: int = 8,
) -> PlanAndFillResult:
    """
    'auto' mode: Plan → Sketch → Fill all holes (attempt).
    'outline' mode: Plan → Sketch only (return outline with 'sorry' and do NOT fill).
    """
    force_outline = (mode == "outline")

    server_info, proc = start_isabelle_server(name="planner", log_file="logs/planner_ui.log")
    isa = get_isabelle_client(server_info)
    session = isa.session_start(session=ISABELLE_SESSION)

    t0 = time.monotonic()
    def left_s() -> float:
        return max(0.0, timeout - (time.monotonic() - t0))

    try:
        # 1) Outline
        if legacy_single_outline:
            skel = propose_isar_skeleton(goal, model=model, temp=0.35, force_outline=force_outline)
            full = skel.text
        else:
            temps = tuple(outline_temps) if outline_temps else (0.35, 0.55, 0.85)
            k = int(outline_k) if outline_k is not None else 3
            best, _diag = propose_isar_skeleton_diverse_best(
                goal,
                isabelle=isa,
                session_id=session,
                model=model,
                temps=temps,
                k=k,
                force_outline=force_outline,
                # scoring/context
                priors_path=priors_path,
                context_hints=context_hints,
                lib_templates=lib_templates,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                # hintlex
                hintlex_path=hintlex_path,
                hintlex_top=hintlex_top,
            )
            full = best.text

        # NEW: If the caller asked for 'outline', stop here and return the outline.
        if mode == "outline":
            return PlanAndFillResult(
                success=True,      # operation succeeded; outline intentionally contains 'sorry'
                outline=full,
                fills=[],
                failed_holes=[],
            )

        # 2) Early exit in 'auto' if the model already produced a complete proof
        spans = find_sorry_spans(full)
        if not spans:
            return PlanAndFillResult(True, full, [], [])

        # 3) Fill holes with shared time + local repair
        lemma_line = _first_lemma_line(full)
        if not lemma_line:
            # Not a well-formed lemma; return outline so the caller can inspect
            return PlanAndFillResult(False, full, [], [0])

        goal_text = _extract_goal_from_lemma_line(lemma_line)
        fills: List[str] = []
        failed: List[int] = []

        hole_idx = 0
        while "sorry" in full and left_s() > 0:
            spans = find_sorry_spans(full)
            if not spans:
                break

            span = spans[0]
            remaining = max(1, len(spans))
            per_hole_budget = int(max(5, left_s() / remaining))

            full2, ok, script = _fill_one_hole(isa, session, full, span, goal_text, model, per_hole_timeout=per_hole_budget)
            if ok:
                full = full2
                fills.append(script)
                hole_idx += 1
                continue

            # Local repair on failure
            if repairs and left_s() > 6:
                patched, applied, _reason = try_local_repairs(
                    full_text=full,
                    hole_span=span,
                    goal_text=goal_text,
                    model=model,
                    isabelle=isa,
                    session=session,
                    repair_budget_s=min(8.0, max(4.0, left_s() * 0.25)),
                    max_ops_to_try=max_repairs_per_hole,
                    trace=repair_trace,
                )
                if applied and patched != full:
                    full = patched
                    # retry same hole (spans recomputed)
                    continue

            failed.append(hole_idx)
            hole_idx += 1

        success = ("sorry" not in full)
        return PlanAndFillResult(success, full, fills, failed)

    finally:
        # Robust shutdown across subprocess.Popen vs asyncio.subprocess.Process
        try:
            isa.shutdown()
            try:
                from planner.experiments import _close_client_loop_safely  # reuse helper
                _close_client_loop_safely(isa)
            except Exception:
                pass
        except Exception:
            pass
        try:
            if hasattr(proc, "terminate"):
                try:
                    proc.terminate()
                except Exception:
                    pass
            if hasattr(proc, "kill"):
                try:
                    proc.kill()
                except Exception:
                    pass
            is_popen = all(hasattr(proc, attr) for attr in ("poll", "communicate", "pid"))
            if is_popen:
                try:
                    proc.wait(timeout=2)
                except Exception:
                    pass
        except Exception:
            pass
