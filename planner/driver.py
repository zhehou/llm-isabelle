# planner/driver.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Iterable

from planner.skeleton import (
    Skeleton,
    find_sorry_spans,
    propose_isar_skeleton_diverse_best,  # diverse outlines + quick sketch check
    propose_isar_skeleton,               # legacy single-outline path
)
from prover.config import ISABELLE_SESSION
from prover.isabelle_api import (
    start_isabelle_server,
    get_isabelle_client,
    build_theory,
    run_theory,
    last_print_state_block,
)
from prover.prover import prove_goal


@dataclass(slots=True)
class PlanAndFillResult:
    success: bool
    outline: str           # final Isar text (with holes filled if successful)
    fills: List[str]       # text of per-hole proof scripts inserted
    failed_holes: List[int]  # indices of holes that failed to fill (if any)


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
    """
    Build a theory ending at the hole, run once, and return the latest print_state block.
    We temporarily leave a 'sorry' at the hole so the theory is well-formed.
    """
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
    """
    Best-effort fill for a single 'sorry' hole:
      1) Get local subgoal print_state right before the hole
      2) Call micro-prover seeded with that state (initial_state_hint)
      3) Splice produced steps into the hole
    Returns: (new_full_text, ok, script_or_reason)
    """
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
        sledge_timeout=5,
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
        initial_state_hint=state_block,  # seed first beam with local print_state
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
# Public entry
# ----------------------------

def plan_and_fill(
    goal: str,
    model: Optional[str] = None,
    timeout: int = 100,
    *,
    mode: str = "auto",  # "auto" (allow complete) or "outline" (force placeholders)
    # New: diverse-outline controls (CLI pass-through)
    outline_k: Optional[int] = None,
    outline_temps: Optional[Iterable[float]] = None,
    legacy_single_outline: bool = False,
) -> PlanAndFillResult:
    """
    Plan (diverse outlines + quick sketch check) → Sketch → Fill all holes.
    - mode="auto": if the planner emits a complete proof (no `sorry`), we return it.
    - mode="outline": force placeholders and try to fill them.
    - If legacy_single_outline=True, bypass diversity and use a single low-temp outline.
    - If outline_k/outline_temps are provided, override defaults used in the diverse step.
    """
    force_outline = (mode == "outline")

    server_info, proc = start_isabelle_server(name="planner", log_file="planner_ui.log")
    isa = get_isabelle_client(server_info)
    session = isa.session_start(session=ISABELLE_SESSION)

    t0 = time.monotonic()
    def left_s() -> float:
        return max(0.0, timeout - (time.monotonic() - t0))

    try:
        # 1) Outline selection
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
            )
            full = best.text

        # If auto-mode and no holes ⇒ return the (checked) whole-proof as-is
        spans = find_sorry_spans(full)
        if mode == "auto" and not spans:
            return PlanAndFillResult(True, full, [], [])

        # 2) Fill holes left→right under a shared wall-clock
        lemma_line = _first_lemma_line(full)
        if not lemma_line:
            return PlanAndFillResult(False, full, [], [0])

        goal_text = _extract_goal_from_lemma_line(lemma_line)
        fills: List[str] = []
        failed: List[int] = []

        hole_idx = 0
        while True:
            if "sorry" not in full:
                break
            if left_s() <= 0:
                break
            spans = find_sorry_spans(full)
            if not spans:
                break

            span = spans[0]  # earliest hole
            remaining = max(1, len(spans))
            per_hole_budget = int(max(5, left_s() / remaining))

            full, ok, script = _fill_one_hole(
                isa, session, full, span, goal_text, model, per_hole_budget
            )
            if ok:
                fills.append(script)
            else:
                failed.append(hole_idx)

            hole_idx += 1

        success = ("sorry" not in full)
        return PlanAndFillResult(success, full, fills, failed)

    finally:
        # Robust shutdown
        try:
            isa.shutdown()
        except Exception:
            pass
        try:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except TypeError:
                proc.wait()
        except Exception:
            try:
                proc.kill()
                try:
                    proc.wait(timeout=2)
                except TypeError:
                    proc.wait()
            except Exception:
                pass
