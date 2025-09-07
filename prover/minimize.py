# prover/minimize.py
import re, time
from typing import List, Tuple

from .isabelle_api import build_theory, run_theory, finished_ok
from .config import MINIMIZE_MAX_FACT_TRIES

# --- Precompiled patterns (unchanged semantics) ---
_FACTS_IN_FINISH = re.compile(r"\b(add:|simp:)\s+([A-Za-z0-9_ ]+)\)")
_METIS_FACTS     = re.compile(r"by\s*\(\s*metis\s+([^)]+)\)", re.IGNORECASE)

# Common one-liners, in the same order as before
_ONE_LINERS: Tuple[str, ...] = (
    "by simp", "by auto", "by clarsimp", "by arith", "by presburger",
    "by fastforce", "by blast", "by (simp)", "by (auto)"
)


def _check_finish(isabelle, session_id: str, lemma_line: str, mid_steps: List[str], finisher: str) -> bool:
    """Run the tiny theory and check the FINISHED-ok result; exact behavior preserved."""
    thy = build_theory([lemma_line] + mid_steps + [finisher], add_print_state=False, end_with=None)
    ok, _ = finished_ok(run_theory(isabelle, session_id, thy))
    return ok


def _try_one_liners(isabelle, session_id: str, lemma_line: str) -> str | None:
    """Return the first successful one-liner, or None (order preserved)."""
    for fin in _ONE_LINERS:
        if _check_finish(isabelle, session_id, lemma_line, [], fin):
            return fin
    return None


def _parse_fact_list_from_finisher(finisher: str) -> Tuple[str, List[str], str]:
    """
    Extract (kind, facts, template) from a finisher.
    kind in {"", "simp_add", "auto_simp", "metis"}; template uses "{facts}".
    """
    s = finisher.strip()

    m = _FACTS_IN_FINISH.search(s)
    if m:
        tag   = m.group(1)
        facts = [x for x in m.group(2).strip().split() if x]
        if "simp add:" in s:   return "simp_add",  facts, "by (simp add: {facts})"
        if "auto simp:" in s:  return "auto_simp", facts, "by (auto simp: {facts})"
        if tag == "add:":      return "simp_add",  facts, "by (simp add: {facts})"
        if tag == "simp:":     return "auto_simp", facts, "by (auto simp: {facts})"

    m2 = _METIS_FACTS.match(s)
    if m2:
        facts = [x for x in m2.group(1).split() if x]
        return "metis", facts, "by (metis {facts})"

    return "", [], s


def _format_finisher(template: str, facts: List[str]) -> str:
    """Fill the template with a space-joined fact list (no behavior change)."""
    return template.replace("{facts}", " ".join(facts)) if "{facts}" in template else template


def minimize_proof(isabelle, session_id: str, final_steps: List[str],
                   timeout_s: int = 8, trace: bool = False, use_color: bool = True) -> List[str]:
    """
    Best-effort shrink:
      1) Try collapse to a one-liner
      2) Try shrinking fact lists in finisher (simp/auto/metis)
      3) Try deleting intermediate steps from the end
      4) Final attempt at a one-liner
    All behavior preserved; only minor speedups and code cleanup applied.
    """
    t0 = time.monotonic()
    def left() -> float:
        return timeout_s - (time.monotonic() - t0)

    # Trivial case
    if len(final_steps) < 2:
        return final_steps[:]

    lemma_line, mid_steps, finisher = final_steps[0], final_steps[1:-1], final_steps[-1]

    # (1) Try collapsing to one-liner
    if left() > 0:
        ol = _try_one_liners(isabelle, session_id, lemma_line)
        if ol:
            return [lemma_line, ol]

    # (2) Try shrinking fact lists / simplifying finisher
    kind, facts, template = _parse_fact_list_from_finisher(finisher)
    if left() > 0 and facts:
        changed = True
        tries   = 0
        # Greedy backward pass over facts, with max tries cap
        while changed and left() > 0 and tries < MINIMIZE_MAX_FACT_TRIES:
            changed = False
            for i in range(len(facts) - 1, -1, -1):
                if left() <= 0 or tries >= MINIMIZE_MAX_FACT_TRIES:
                    break
                trial = facts[:i] + facts[i+1:]
                if not trial:
                    continue
                fin_try = _format_finisher(template, trial)
                if _check_finish(isabelle, session_id, lemma_line, mid_steps, fin_try):
                    facts    = trial
                    finisher = fin_try
                    changed  = True
                tries += 1

        # Consider simpler forms if possible (logic intentionally identical to original)
        if left() > 0 and kind in ("simp_add", "auto_simp"):
            simpler = "by simp" if "simp" in kind else "by auto"
            if _check_finish(isabelle, session_id, lemma_line, mid_steps, simpler):
                finisher = simpler

        if left() > 0 and kind == "metis":
            # Try metis without facts
            if _check_finish(isabelle, session_id, lemma_line, mid_steps, "by (metis)"):
                finisher = "by (metis)"
                facts = []
            else:
                # Try small prefixes of the fact list
                for target in (1, 2):
                    if len(facts) <= target:
                        break
                    cand = facts[:target]
                    fin_try = _format_finisher(template, cand)
                    if _check_finish(isabelle, session_id, lemma_line, mid_steps, fin_try):
                        finisher = fin_try
                        facts    = cand
                        break

    # (3) Try deleting intermediate steps from the end
    if left() > 0 and mid_steps:
        for idx in range(len(mid_steps) - 1, -1, -1):
            if left() <= 0:
                break
            trial_mid = mid_steps[:idx] + mid_steps[idx+1:]
            if _check_finish(isabelle, session_id, lemma_line, trial_mid, finisher):
                mid_steps = trial_mid

    minimized = [lemma_line] + mid_steps + [finisher]

    # (4) Final attempt to collapse to a one-liner
    if left() > 0 and len(minimized) > 2:
        ol2 = _try_one_liners(isabelle, session_id, lemma_line)
        if ol2:
            return [lemma_line, ol2]

    return minimized