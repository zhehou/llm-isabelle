# prover/variants.py
import re, time
from typing import List, Optional
from .isabelle_api import build_theory, run_theory, finished_ok

_IND_STEP = re.compile(r"^\s*apply\s*\(\s*induct(?:ion)?\s+([^)]+)\)\s*$", re.IGNORECASE)
_CASES_STEP = re.compile(r"^\s*apply\s*\(\s*cases\s+([^)]+)\)\s*$", re.IGNORECASE)

def booleanish(s: str) -> bool:
    s = s.lower()
    return any(tok in s for tok in ["true", "false", "¬", "not", "∧", "∨"])

def _case_finishers(facts: List[str]) -> List[str]:
    base = ["by simp", "by auto", "by blast"]
    extras = []
    for f in facts[:4]:
        extras.append(f"by (simp add: {f})")
        extras.append(f"by (metis {f})")
    seen, out = set(), []
    for x in extras + base:
        if x not in seen:
            seen.add(x); out.append(x)
    return out[:10]

def _build_structured_induction_steps(goal: str, var_raw: str, nil_fin: str, cons_fin: str) -> List[str]:
    return [
        f'lemma "{goal}"',
        f'proof (induction {var_raw.strip().split()[0]})',
        'case Nil',
        f'  then show ?case {nil_fin}',
        'next',
        'case (Cons x xs)',
        f'  then show ?case {cons_fin}',
        'qed',
    ]

def _build_structured_cases_bool_steps(goal: str, var_raw: str, t_fin: str, f_fin: str) -> List[str]:
    return [
        f'lemma "{goal}"',
        f'proof (cases {var_raw.strip().split()[0]})',
        'case True',
        f'  then show ?thesis {t_fin}',
        'next',
        'case False',
        f'  then show ?thesis {f_fin}',
        'qed',
    ]

def try_structured_variants(isabelle, session_id: str, goal: str,
                            final_steps: List[str], facts_seed: List[str],
                            timeout_s: int, max_tries: int,
                            trace: bool, use_color: bool) -> Optional[List[str]]:
    ind_var = None; cases_var = None
    for s in final_steps:
        if ind_var is None:
            m = _IND_STEP.match(s)
            if m: ind_var = m.group(1)
        if cases_var is None:
            c = _CASES_STEP.match(s)
            if c: cases_var = c.group(1)
        if ind_var or cases_var: break
    if not ind_var and not cases_var: return None

    facts = list(facts_seed) if facts_seed else []
    t0 = time.monotonic()
    def left(): return timeout_s - (time.monotonic() - t0)
    tries = 0

    if ind_var:
        finishers = _case_finishers(facts)
        for nil_fin in finishers:
            for cons_fin in finishers:
                if tries >= max_tries or left() <= 0: return None
                steps = _build_structured_induction_steps(goal, ind_var, nil_fin, cons_fin)
                ok, _ = finished_ok(run_theory(isabelle, session_id, build_theory([steps[0]] + steps[1:], False, None)))
                tries += 1
                if ok: return steps

    if cases_var and booleanish(goal):
        finishers = _case_finishers(facts)
        for t_fin in finishers:
            for f_fin in finishers:
                if tries >= max_tries or left() <= 0: return None
                steps = _build_structured_cases_bool_steps(goal, cases_var, t_fin, f_fin)
                ok, _ = finished_ok(run_theory(isabelle, session_id, build_theory([steps[0]] + steps[1:], False, None)))
                tries += 1
                if ok: return steps

    return None
