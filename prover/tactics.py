# prover/tactics.py
"""
Unified Isabelle-facing tactics & helpers:
- Quickcheck/Nitpick prechecks
- Sledgehammer finishers
- Lemma/fact mining from state
- Structured-variant builders (induction/cases)
- Macros: step-continuation suggestions (moved from macros.py)

This collapses: prechecks.py, sledge.py, mining.py, variants.py, macros.py
"""

from __future__ import annotations

import json
import re
import time
import textwrap
from typing import List, Optional, Dict, Tuple

from .isabelle_api import build_theory, run_theory, finished_ok  # reuse shared API

# --- Variant step templates (tiny helper) ---
VAR_TOKEN = re.compile(r"\b([A-Za-z][A-Za-z0-9_']*)\b")

def _likely_vars_from_state_hint(state_hint: str, k: int = 3) -> List[str]:
    if not state_hint:
        return []
    toks = [t for t in VAR_TOKEN.findall(state_hint)]
    if not toks:
        return []
    # Prefer common names; then short identifiers
    prefer = ["n", "m", "k", "xs", "ys", "zs", "x", "y", "z"]
    seen, ordered = set(), []
    for p in prefer:
        if p in toks and p not in seen:
            seen.add(p); ordered.append(p)
    for t in sorted(toks, key=lambda s: (len(s), s)):
        if t not in seen:
            seen.add(t); ordered.append(t)
    return ordered[:k]

def variant_step_templates(state_hint: str, limit: int = 4) -> List[str]:
    vars_ = _likely_vars_from_state_hint(state_hint, k=3)
    out, seen = [], set()
    for v in vars_:
        # Lists/sequences → try induction & cases; naturals → induction; otherwise cases.
        if v in ("xs", "ys", "zs") or v.endswith("s"):
            for c in (f"apply (induction {v})", f"apply (cases {v})"):
                if c not in seen:
                    seen.add(c); out.append(c)
        elif v in ("n", "m", "k"):
            c = f"apply (induction {v})"
            if c not in seen:
                seen.add(c); out.append(c)
        else:
            c = f"apply (cases {v})"
            if c not in seen:
                seen.add(c); out.append(c)
        if len(out) >= limit:
            break
    return out[:limit]

# =============================================================================
# Prechecks (Quickcheck / Nitpick)
# =============================================================================
_QC_HIT = re.compile(r"(?i)(counterexample\s+found|Quickcheck\s+found\s+a\s+counterexample)")
_NP_HIT = re.compile(r"(?i)(Nitpick\s+found\s+a\s+counterexample|genuine\s+counterexample)")

def precheck_quickcheck_refutes(isabelle, session_id: str, steps_with_candidate: List[str], timeout_s: int) -> bool:
    cmd = f"quickcheck[timeout = {int(timeout_s)}]"
    thy = build_theory(steps_with_candidate + [cmd], add_print_state=False, end_with="sorry")
    for r in run_theory(isabelle, session_id, thy):
        if getattr(r, "response_type", "") != "NOTE":
            continue
        try:
            msg = json.loads(r.response_body).get("message", "")
        except Exception:
            continue
        if _QC_HIT.search(str(msg)):
            return True
    return False

def precheck_nitpick_refutes(isabelle, session_id: str, steps_with_candidate: List[str], timeout_s: int) -> bool:
    cmd = f"nitpick[timeout = {int(timeout_s)}]"
    thy = build_theory(steps_with_candidate + [cmd], add_print_state=False, end_with="sorry")
    for r in run_theory(isabelle, session_id, thy):
        if getattr(r, "response_type", "") != "NOTE":
            continue
        try:
            msg = json.loads(r.response_body).get("message", "")
        except Exception:
            continue
        if _NP_HIT.search(str(msg)):
            return True
    return False

# =============================================================================
# Sledgehammer finishers
# =============================================================================
_SLEDGE_BY = re.compile(r"(?i)(?:try this:\s*)?(by\s+\([^)]+\)|by\s+\w+(?:\s+.+)?)")
_SLEDGE_METIS_LINE = re.compile(r"(?i)^metis\b.*?:\s*(.*)$")

def _extract_sledge_by_lines(text: str) -> List[str]:
    out: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        m = _SLEDGE_BY.search(s)
        if m:
            out.append(re.sub(r"\s+", " ", m.group(1).strip()))
            continue
        m2 = _SLEDGE_METIS_LINE.match(s)
        if m2:
            facts = m2.group(1).strip()
            out.append(f"by (metis {facts})" if facts else "by (metis)")
    # de-dup
    seen, dedup = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup

def sledgehammer_finishers(isabelle, session_id: str, steps: List[str], timeout_s: int = 5, limit: int = 5) -> List[str]:
    sh_cmd = f"sledgehammer [timeout = {int(timeout_s)}]"
    thy = build_theory(steps + [sh_cmd], add_print_state=False, end_with="sorry")
    texts: List[str] = []
    for r in run_theory(isabelle, session_id, thy):
        if getattr(r, "response_type", "") != "NOTE":
            continue
        try:
            msg = json.loads(r.response_body).get("message", "")
        except Exception:
            continue
        if any(k in str(msg) for k in ("Try this:", "by ", "metis", "Metis")):
            texts.append(str(msg))
    cands: List[str] = []
    for t in texts:
        cands.extend(_extract_sledge_by_lines(t))
    out, seen = [], set()
    for c in cands:
        if c.startswith("by ") or c == "done":
            if c not in seen:
                seen.add(c)
                out.append(c)
                if len(out) >= limit:
                    break
    return out

# =============================================================================
# Mining lemmas / prioritized facts
# =============================================================================
FIND_NAME_TOKENS = re.compile(r"[A-Za-z][A-Za-z0-9_']{2,}")
_LEMMA_LINE = re.compile(r"^\s*\d+\.\s*([A-Za-z0-9_\.]+):")

def _build_find_theorems_theory(symbols: List[str]) -> str:
    body = "\n".join(f'find_theorems name:{s}' for s in symbols[:8])
    return textwrap.dedent(f"theory FT_Scratch\nimports Main\nbegin\n\n{body}\n\nend").strip()

def _build_find_theorems_filtered(symbols: List[str], filters: List[str]) -> str:
    lines = []
    for s in symbols[:8]:
        for flt in filters:
            lines.append(f"find_theorems {flt} name:{s}" if flt else f"find_theorems name:{s}")
    return textwrap.dedent(
        "theory FT2_Scratch\nimports Main\nbegin\n\n{}\n\nend".format("\n".join(lines))
    ).strip()

def mine_lemmas_from_state(isabelle, session_id: str, state_hint: str, max_lemmas: int = 6) -> List[str]:
    toks = list(dict.fromkeys(FIND_NAME_TOKENS.findall(state_hint)))
    if not toks:
        return []
    theory_text = _build_find_theorems_theory(toks)
    lemmas, seen = [], set()
    for r in run_theory(isabelle, session_id, theory_text):
        if getattr(r, "response_type", "") != "NOTE":
            continue
        try:
            msg = json.loads(r.response_body).get("message", "")
        except Exception:
            continue
        for line in str(msg).splitlines():
            m = _LEMMA_LINE.match(line.strip())
            if m:
                name = m.group(1).split(".")[-1]
                if name not in seen:
                    seen.add(name)
                    lemmas.append(name)
                    if len(lemmas) >= max_lemmas:
                        return lemmas
    return lemmas

def mine_facts_prioritized(isabelle, session_id: str, state_hint: str, limit: int = 6) -> List[str]:
    toks = list(dict.fromkeys(FIND_NAME_TOKENS.findall(state_hint)))
    if not toks:
        return []
    # Pull general rules plus defs; defs are often decisive for user-defined symbols.
    theory_text = _build_find_theorems_filtered(toks, ["simp", "intro", "rule", "elim", "dest"])
    # Then add a pass that prefers *_def by name (keep short and cheap)
    def_name_patterns = [f'name: "{stem}_def"' for stem in toks[:6]]
    if def_name_patterns:
        theory_text += "\n" + "\n".join(f"find_theorems {p} - 10" for p in def_name_patterns)
    freq: Dict[str, int] = {}
    for r in run_theory(isabelle, session_id, theory_text):
        if getattr(r, "response_type", "") != "NOTE":
            continue
        try:
            msg = json.loads(r.response_body).get("message", "")
        except Exception:
            continue
        for line in str(msg).splitlines():
            m = _LEMMA_LINE.match(line.strip())
            if not m:
                continue
            name = m.group(1).split(".")[-1]
            freq[name] = freq.get(name, 0) + 1
    return [k for k, _ in sorted(freq.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))][:limit]

# =============================================================================
# Structured variants (induction/cases scaffolds)
# =============================================================================
_IND_STEP = re.compile(r"^\s*apply\s*\(\s*induct(?:ion)?\s+([^)]+)\)\s*$", re.IGNORECASE)
_CASES_STEP = re.compile(r"^\s*apply\s*\(\s*cases\s+([^)]+)\)\s*$", re.IGNORECASE)

def _booleanish_goal(s: str) -> bool:
    s = s.lower()
    return any(tok in s for tok in ["true", "false", "¬", "not", "∧", "∨"])

def _case_finishers(facts: List[str]) -> List[str]:
    base = ["by simp", "by auto", "by blast"]
    extras: List[str] = []
    for f in facts[:4]:
        extras.append(f"by (simp add: {f})")
        extras.append(f"by (metis {f})")
    seen, out = set(), []
    for x in extras + base:
        if x not in seen:
            seen.add(x)
            out.append(x)
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
    ind_var = None
    cases_var = None
    for s in final_steps:
        if ind_var is None:
            m = _IND_STEP.match(s);  ind_var = m.group(1) if m else ind_var
        if cases_var is None:
            c = _CASES_STEP.match(s); cases_var = c.group(1) if c else cases_var
        if ind_var or cases_var:
            break
    if not ind_var and not cases_var:
        return None

    facts = list(facts_seed) if facts_seed else []
    t0 = time.monotonic()
    def left() -> float: return timeout_s - (time.monotonic() - t0)
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

    if cases_var and _booleanish_goal(goal):
        finishers = _case_finishers(facts)
        for t_fin in finishers:
            for f_fin in finishers:
                if tries >= max_tries or left() <= 0: return None
                steps = _build_structured_cases_bool_steps(goal, cases_var, t_fin, f_fin)
                ok, _ = finished_ok(run_theory(isabelle, session_id, build_theory([steps[0]] + steps[1:], False, None)))
                tries += 1
                if ok: return steps

    return None

# =============================================================================
# Macros (continuation suggestions) — moved from macros.py
# =============================================================================
import os
import json
from collections import Counter, defaultdict
from typing import Iterable

# Environment-driven defaults
MACRO_MIN_COUNT = int(os.environ.get("MACRO_MIN_COUNT", "2"))
MACRO_MAX_PER_HEAD = int(os.environ.get("MACRO_MAX_PER_HEAD", "5"))

from .config import RUNS_LOG

ContinuationMap = Dict[str, List[Tuple[str, int]]]

def _iter_success_runs(paths: Iterable[str]) -> Iterable[List[str]]:
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    if rec.get("success") and isinstance(rec.get("final_steps"), list):
                        yield rec["final_steps"]
                    elif rec.get("success") and isinstance(rec.get("steps"), list):
                        yield rec["steps"]
        except FileNotFoundError:
            continue

def mine_two_step_macros(
    paths: Optional[List[str]] = None,
    min_count: Optional[int] = None,
    max_per_head: Optional[int] = None,
) -> ContinuationMap:
    """
    Build a mapping: first_step -> list of (second_step, count), sorted by freq.

    Args:
      paths         : runs.jsonl paths (defaults to [RUNS_LOG])
      min_count     : minimum frequency to keep a pair (env MACRO_MIN_COUNT if None)
      max_per_head  : cap continuations per head (env MACRO_MAX_PER_HEAD if None)

    Returns:
      Dict[str, List[Tuple[str, int]]]
    """
    paths = paths or [RUNS_LOG]
    min_count = MACRO_MIN_COUNT if min_count is None else int(min_count)
    max_per_head = MACRO_MAX_PER_HEAD if max_per_head is None else int(max_per_head)

    pair_freq: Dict[Tuple[str, str], int] = Counter()
    for steps in _iter_success_runs(paths):
        # Ignore lemma line (index 0) and finisher (last)
        if not steps or len(steps) < 3:
            continue
        mid = [s for s in steps[1:-1] if s.strip().startswith("apply")]
        for a, b in zip(mid, mid[1:]):
            pair_freq[(a, b)] += 1

    conts: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for (a, b), c in pair_freq.items():
        if c >= min_count:
            conts[a].append((b, c))

    # sort and truncate per head
    for a in list(conts.keys()):
        conts[a].sort(key=lambda t: (-t[1], len(t[0]), t[0]))
        conts[a] = conts[a][:max_per_head]
    return dict(conts)

def _norm(s: str) -> str:
    return " ".join((s or "").strip().split())

def suggest_continuations(step: str,
                          macro_map: Optional[Dict[str, List[Tuple[str, int]]]],
                          k: int = 1) -> List[str]:
    """
    Suggest at most k macro continuations for a just-applied step.

    macro_map format (unchanged):
      {
        "apply simp":        [("done", 3), ("by simp", 2)],
        "apply (induction":  [("apply simp_all", 1)],
        "apply (cases":      [("apply simp", 1), ("by auto", 1)],
        ...
      }

    Matching:
      - exact key match first
      - then prefix match (e.g., "apply (induction" matches "apply (induction xs)")
      - ties broken by weight (descending), then by shorter continuation text
    """
    if not macro_map:
        return []
    key = _norm(step)
    # exact
    choices = list(macro_map.get(key, []))
    # prefix (only if nothing exact)
    if not choices:
        for mk, lst in macro_map.items():
            mk_n = _norm(mk)
            if key.startswith(mk_n):
                choices.extend(lst)
    if not choices:
        return []
    # weight sort & de-dup
    seen, ordered = set(), []
    for cont, w in sorted(choices, key=lambda x: (-int(x[1] if isinstance(x[1], int) else 1), len(_norm(x[0])), _norm(x[0]))):
        c = _norm(cont)
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)
            if len(ordered) >= max(1, int(k)):
                break
    return ordered
