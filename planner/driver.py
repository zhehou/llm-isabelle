from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple

from planner.skeleton import Skeleton, propose_isar_skeleton, find_sorry_spans
from prover.isabelle_api import start_isabelle_server, get_isabelle_client
from prover.prover import prove_goal
from prover.config import ISABELLE_SESSION


@dataclass(slots=True)
class PlanAndFillResult:
    success: bool
    outline: str          # final isar text
    fills: List[str]      # per-hole proof scripts inserted
    failed_holes: List[int]


def _extract_goal_from_lemma_line(lemma_line: str) -> str:
    q1 = lemma_line.find('"')
    q2 = lemma_line.rfind('"')
    if q1 == -1 or q2 == -1 or q2 <= q1:
        raise ValueError(f"Cannot parse lemma line: {lemma_line!r}")
    return lemma_line[q1 + 1 : q2]


def _fill_one_hole(
    isabelle,
    session,
    full_text: str,
    hole_span: Tuple[int, int],
    model: Optional[str],
    timeout: int,
) -> tuple[str, bool, str]:
    # Extract top-level goal from the first 'lemma "…"' line.
    lemma_line = ""
    for L in full_text.splitlines():
        if L.strip().startswith("lemma "):
            lemma_line = L
            break
    if not lemma_line:
        return full_text, False, "no-lemma-head"
    goal = _extract_goal_from_lemma_line(lemma_line)

    # Call micro prover (top-level goal; simple prototype).
    res = prove_goal(
        isabelle,
        session,
        goal,
        model_name_or_ensemble=model,
        beam_w=2,
        max_depth=8,
        hint_lemmas=6,
        timeout=timeout,
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

    # Preserve previous indentation behavior: indent two spaces inside the hole.
    insert = "\n  " + "\n  ".join(script_lines) + "\n"
    s, e = hole_span
    new_text = full_text[:s] + insert + full_text[e:]
    return new_text, True, "\n".join(script_lines)


def plan_and_fill(
    goal: str,
    model: Optional[str] = None,
    timeout: int = 100,
    *,
    mode: str = "auto",  # "auto" (allow complete) or "outline" (force placeholders)
) -> PlanAndFillResult:
    """
    mode="auto": if the LLM produces a complete proof (no `sorry`), return it directly.
    mode="outline": force an outline with `sorry` and try to fill the first hole.
    """
    force_outline = mode == "outline"

    # Start Isabelle once for this call.
    server_info, proc = start_isabelle_server(name="planner", log_file="planner_ui.log")
    isa = get_isabelle_client(server_info)
    session = isa.session_start(session=ISABELLE_SESSION)
    try:
        skel: Skeleton = propose_isar_skeleton(
            goal, model=model, temp=0.4, force_outline=force_outline
        )
        full = skel.text
        spans = find_sorry_spans(full)

        if mode == "auto" and not spans:
            # LLM produced a complete proof; return as-is.
            return PlanAndFillResult(success=True, outline=full, fills=[], failed_holes=[])

        # Otherwise, try to fill the first hole.
        fills: List[str] = []
        failed: List[int] = []
        if spans:
            span = spans[0]
            full, ok, script = _fill_one_hole(
                isa, session, full, span, model, timeout=timeout
            )
            if ok:
                fills.append(script)
            else:
                failed.append(0)

        success = ("sorry" not in full)
        return PlanAndFillResult(
            success=success, outline=full, fills=fills, failed_holes=failed
        )
    finally:
        # Robust shutdown (works across platforms / Python versions).
        try:
            isa.shutdown()
        except Exception:
            pass
        try:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except TypeError:
                # Some Process/older Python variants don't accept timeout kw.
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