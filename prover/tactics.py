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

import os
import json
import re
import time
import textwrap
from collections import Counter, defaultdict
from typing import List, Optional, Dict, Tuple, Iterable

from .isabelle_api import build_theory, run_theory, finished_ok  # reuse shared API

# --- Small internal helpers to robustly read Isabelle responses (no API changes) ---
def _normalize_response_type(rt: object) -> str:
    """Return 'NOTE', 'FINISHED', etc., tolerating Enums/strings/objects."""
    try:
        if hasattr(rt, "name"):
            return str(getattr(rt, "name")).strip().upper()
        if isinstance(rt, str):
            return rt.strip().upper()
        s = str(rt).strip().upper()
        if "NOTE" in s:
            return "NOTE"
        if "FINISHED" in s:
            return "FINISHED"
        return s
    except Exception:
        return ""

def _decode_body(obj: object) -> Optional[dict]:
    """Decode response_body which may be dict / JSON string / bytes."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        try:
            obj = obj.decode("utf-8", "replace")
        except Exception:
            obj = str(obj)
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except Exception:
            return None
    return None

def _iter_note_messages(responses) -> Iterable[str]:
    """Yield Sledge/Find/etc. messages from direct and FINISHED responses.

    Several isabelle-client versions consolidate command output inside the
    final node JSON instead of emitting separate NOTE records.  Ignoring that
    representation made theorem mining return an empty list even when
    ``find_theorems`` had found useful elimination rules.
    """
    for r in (responses or []):
        typ = _normalize_response_type(getattr(r, "response_type", None))
        body = _decode_body(getattr(r, "response_body", None))
        if isinstance(body, dict):
            if typ in ("NOTE", "WRITELN", "INFORMATION", "TRACING"):
                msg = body.get("message") or body.get("text") or body.get("content")
                if msg is not None:
                    yield str(msg)
            for node in body.get("nodes", []) or []:
                for item in node.get("messages", []) or []:
                    kind = str(item.get("kind", "")).lower()
                    if kind in ("writeln", "information", "tracing", "warning"):
                        message = item.get("message") or item.get("text")
                        if message is not None:
                            yield str(message)
        elif isinstance(body, str):
            # Some clients may return a plain string
            yield body

# --- Variant step templates (tiny helper) ---
VAR_TOKEN = re.compile(r"\b([A-Za-z][A-Za-z0-9_']*)\b")
_FACT_REFERENCE_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)?$"
)
_LLM_TYPE_RECORD = re.compile(
    r"^\[LLM_TYPE\]\s+([A-Za-z][A-Za-z0-9_']*)\s+::\s+"
    r"(.*?)(?=^\[[A-Z0-9_]+\]|\Z)",
    re.M | re.S,
)

def _likely_vars_from_state_hint(state_hint: str, k: int = 3) -> List[str]:
    if not state_hint:
        return []
    toks = VAR_TOKEN.findall(state_hint)
    if not toks:
        return []
    prefer = ["n", "m", "k", "xs", "ys", "zs", "x", "y", "z"]
    seen, ordered = set(), []
    for p in prefer:
        if p in toks and p not in seen:
            seen.add(p); ordered.append(p)
    for t in sorted(toks, key=lambda s: (len(s), s)):
        if t not in seen:
            seen.add(t); ordered.append(t)
    return ordered[:k]


def _typed_vars_from_state_hint(state_hint: str) -> List[tuple[str, str]]:
    """Read variables and types emitted by the planner's Isabelle method."""

    typed: List[tuple[str, str]] = []
    for match in _LLM_TYPE_RECORD.finditer(state_hint or ""):
        name = match.group(1).rstrip("_")
        typ = re.sub(r"\s+", " ", match.group(2)).strip()
        item = (name, typ)
        if name and typ and item not in typed:
            typed.append(item)
    return typed


def _structural_commands_for_type(name: str, typ: str) -> List[str]:
    low = typ.lower()
    # Functions, propositions, sets and unconstrained type variables are poor
    # induction targets.  Lists, naturals and user datatypes are strong ones.
    if any(arrow in typ for arrow in ("⇒", "=>", "\\<Rightarrow>")) or low in {"bool", "prop", "int", "real"}:
        return []
    if low.endswith(" set") or low == "set" or re.fullmatch(r"'[a-z][a-z0-9_']*", low):
        return []
    if " list" in low or low.endswith("list") or low == "nat":
        return [f"apply (induction {name})", f"apply (cases {name})"]
    # A concrete, non-function type is usually a datatype.  Isabelle will
    # reject a proposal when it is not, so this remains a safe search hint.
    return [f"apply (induction {name})", f"apply (cases {name})"]

def variant_step_templates(state_hint: str, limit: int = 4) -> List[str]:
    typed = _typed_vars_from_state_hint(state_hint)
    if typed:
        ranked = sorted(
            typed,
            key=lambda item: (
                0 if " list" in item[1].lower() or item[1].lower().endswith("list") else
                1 if item[1].lower() == "nat" else 2,
            ),
        )
        typed_out: List[str] = []
        for name, typ in ranked:
            for command in _structural_commands_for_type(name, typ):
                if command not in typed_out:
                    typed_out.append(command)
                if len(typed_out) >= limit:
                    return typed_out
        # Presence of type records is authoritative.  Do not fall back to
        # token guessing and accidentally induce on a function or theorem name.
        return typed_out

    vars_ = _likely_vars_from_state_hint(state_hint, k=3)
    out, seen = [], set()
    for v in vars_:
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


def fact_composition_finishers(
    goal_or_state: str,
    facts: Iterable[str] = (),
    *,
    limit: int = 10,
) -> List[str]:
    """Build bounded, rule-aware finishers for composing available facts.

    Automation often extracts the right intermediate facts but does not chain
    them.  These candidates select a small set of standard elimination and
    transitivity rules from the actual goal shape, then let Isabelle's
    ``auto``/``meson``/``metis`` instantiate them.  Every candidate remains
    verifier checked, so the lexical trigger affects search cost, not soundness.
    """

    text = str(goal_or_state or "")
    rules: List[str] = []
    if re.search(r"\binj(?:_on)?\b", text):
        rules.append("injD")
    if re.search(r"(?:≤|\\<le>|<=)", text):
        rules.append("order_trans")
    if re.search(r"\btrans\b", text):
        rules.append("transD")
    if re.search(r"(?:⊆|\\<subseteq>).*?(?:⊆|\\<subseteq>)", text, re.S):
        rules.append("subset_trans")
    rules = list(dict.fromkeys(rules))
    if not rules or limit <= 0:
        return []

    named = [
        str(fact).strip() for fact in facts
        if _FACT_REFERENCE_RE.fullmatch(str(fact).strip())
    ]
    named = list(dict.fromkeys(named))[:6]
    rule_text = " ".join(rules)
    out: List[str] = []

    def add(method: str) -> None:
        if method not in out and len(out) < limit:
            out.append(method)

    if "injD" in rules:
        add("by (auto dest: injD)")
    add(f"by (meson {rule_text})")
    add(f"by (metis {rule_text})")
    if named:
        evidence = " ".join(named)
        add(f"by (metis {evidence} {rule_text})")
        add(f"by (meson {evidence} {rule_text})")
    return out[:limit]

# =============================================================================
# Prechecks (Quickcheck / Nitpick)
# =============================================================================
_QC_HIT = re.compile(r"(?i)(counterexample\s+found|Quickcheck\s+found\s+a\s+counterexample)")
_NP_HIT = re.compile(r"(?i)(Nitpick\s+found\s+a\s+counterexample|genuine\s+counterexample)")

def precheck_quickcheck_refutes(isabelle, session_id: str, steps_with_candidate: List[str], timeout_s: int) -> bool:
    cmd = f"quickcheck[timeout = {int(timeout_s)}]"
    thy = build_theory(steps_with_candidate + [cmd], add_print_state=False, end_with="sorry")
    for msg in _iter_note_messages(run_theory(isabelle, session_id, thy, timeout_s=max(1, int(timeout_s) + 2))):
        if _QC_HIT.search(msg):
            return True
    return False

def precheck_nitpick_refutes(isabelle, session_id: str, steps_with_candidate: List[str], timeout_s: int) -> bool:
    cmd = f"nitpick[timeout = {int(timeout_s)}]"
    thy = build_theory(steps_with_candidate + [cmd], add_print_state=False, end_with="sorry")
    for msg in _iter_note_messages(run_theory(isabelle, session_id, thy, timeout_s=max(1, int(timeout_s) + 2))):
        if _NP_HIT.search(msg):
            return True
    return False

# =============================================================================
# Sledgehammer finishers
# =============================================================================
import re
from typing import List

# More comprehensive regex patterns for Sledgehammer output
_SLEDGE_BY = re.compile(r"(?i)(?:try this:\s*)?(by\s+\([^)]+\)|by\s+\w+(?:\s+[^)\n]*)?)", re.MULTILINE)
_SLEDGE_APPLY = re.compile(r"(?i)(?:try this:\s*)?(apply\s+\([^)]+\)|apply\s+\w+(?:\s+[^)\n]*)?)", re.MULTILINE)
_SLEDGE_METIS_LINE = re.compile(r"(?i)^.*?metis\b.*?:\s*(.*)$", re.MULTILINE)
_SLEDGE_SIMP_LINE = re.compile(r"(?i)(?:try this:\s*)?(simp(?:\s+add:\s*[^)\n]*)?)", re.MULTILINE)
_SLEDGE_AUTO_LINE = re.compile(r"(?i)(?:try this:\s*)?(auto(?:\s+[^)\n]*)?)", re.MULTILINE)
# Strip timing tails like " (0.6 ms" / " (1 ms)" / " (2.3 s)" at line end (closing ')' optional)
_TIMING_TAIL = re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)?\s*$", re.IGNORECASE)
_TRY_THIS = re.compile(r"(?i)^\s*try this:\s*")
_TRY_THIS_ANYWHERE = re.compile(r"(?i)\btry this:\s*(by\s+.+)$")

def _normalize_finisher_text(s: str) -> str:
    # Remove "Try this:" prefix
    s = _TRY_THIS.sub("", s or "")
    # Remove trailing timing annotation; safe because real tactic args are "(letter...)" not "(number...)"
    s = _TIMING_TAIL.sub("", s)
    # Trim trailing punctuation and normalize whitespace
    s = s.strip().rstrip(",;")
    s = " ".join(s.split())
    return s


def _safe_sledge_candidate(candidate: str) -> bool:
    """Reject transport/JSON fragments accidentally captured as tactics."""
    if not candidate or not (candidate.startswith("by ") or candidate == "done"):
        return False
    bad_fragments = (
        "{\"", "\"pos\"", "\"file\"", "response_body", "theory_name",
        "node_name", "session_id", "tmp/", "/tmp", ".thy\"", "exports",
    )
    if any(fragment in candidate for fragment in bad_fragments):
        return False
    lowered = candidate.lower()
    if any(fragment in lowered for fragment in (
        "try this:", "found a proof", "duplicate proof", "theory processed",
        "percentage", "unprocessed", "consolidated", "canceled",
    )):
        return False
    # Nested method arguments such as ``simp add: foo(1)`` are valid, but the
    # former regex stopped at the first closing parenthesis and emitted broken
    # tactics.  A lightweight quote-aware balance check rejects those
    # truncations as well as serialized diagnostic tails.
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: List[str] = []
    quoted = False
    escaped = False
    for char in candidate:
        if escaped:
            escaped = False
            continue
        if char == "\\" and quoted:
            escaped = True
            continue
        if char == '"':
            quoted = not quoted
            continue
        if quoted:
            continue
        if char in "([{":
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                return False
    if quoted or stack:
        return False
    # Genuine suggestions are short single commands.  This also prevents a
    # greedy regex from swallowing a serialized Isabelle response.
    return "\n" not in candidate and len(candidate) <= 500

def _extract_sledge_by_lines(text: str) -> List[str]:
    out: List[str] = []

    # Prefer complete message lines.  This preserves nested parentheses and
    # faithfully handles output such as
    # ``metis: Try this: by (metis foo(1)) (3 ms)``.
    for raw_line in text.splitlines():
        match = _TRY_THIS_ANYWHERE.search(raw_line)
        if match:
            suggestion = _normalize_finisher_text(match.group(1))
        else:
            stripped = raw_line.strip()
            suggestion = _normalize_finisher_text(stripped) if stripped.startswith("by ") else ""
        if _safe_sledge_candidate(suggestion) and suggestion not in out:
            out.append(suggestion)

    # Compatibility fallback for older clients that concatenate messages.
    # The safety predicate discards any truncated/unbalanced match.
    for match in _SLEDGE_BY.finditer(text):
        suggestion = _normalize_finisher_text(match.group(1))
        if _safe_sledge_candidate(suggestion) and suggestion not in out:
            out.append(suggestion)

    # "apply ..." lines
    for match in _SLEDGE_APPLY.finditer(text):
        suggestion = _normalize_finisher_text(match.group(1))
        if _safe_sledge_candidate(suggestion) and suggestion not in out:
            out.append(suggestion)

    # simp/auto one-liners → "by simp"/"by auto"
    for match in _SLEDGE_SIMP_LINE.finditer(text):
        suggestion = _normalize_finisher_text(f"by {match.group(1).strip()}")
        if _safe_sledge_candidate(suggestion) and suggestion not in out:
            out.append(suggestion)

    for match in _SLEDGE_AUTO_LINE.finditer(text):
        suggestion = _normalize_finisher_text(f"by {match.group(1).strip()}")
        if _safe_sledge_candidate(suggestion) and suggestion not in out:
            out.append(suggestion)

    # Heuristic sweep for common finishers (kept, but normalized)
    simple_finishers = ["by simp", "by auto", "by blast", "by force", "by fastforce"]
    for line in text.split('\n'):
        line_lower = line.lower().strip()
        if any(word in line_lower for word in ["simp", "auto", "blast", "force", "fastforce"]):
            for fin in simple_finishers:
                if fin.split()[1] in line_lower and fin not in out:
                    out.append(fin)
                    break

    return out

def sledgehammer_finishers(isabelle, session_id: str, steps: List[str], timeout_s: int = 5, limit: int = 5) -> List[str]:
    """Get sledgehammer finisher suggestions."""
    from .isabelle_api import build_theory, run_theory
    
    sh_cmd = f"sledgehammer [timeout = {int(timeout_s)}]"
    thy = build_theory(steps + [sh_cmd], add_print_state=False, end_with="sorry")
    
    responses = run_theory(isabelle, session_id, thy, timeout_s=max(1, int(timeout_s) + 2))
    # Parse only Isabelle message fields.  Stringifying whole response objects
    # leaks JSON metadata into the tactic regex and previously produced giant
    # candidates containing temporary filenames and status objects.
    all_text = "\n".join(_iter_note_messages(responses))
    
    # Extract suggestions
    cands = _extract_sledge_by_lines(all_text)
    
    # Filter and return valid finishers
    out = []
    seen = set()
    for c in cands:
        clean = c.strip()
        if _safe_sledge_candidate(clean) and clean not in seen:
            seen.add(clean)
            out.append(clean)
            if len(out) >= limit:
                break
    
    return out

# =============================================================================
# Mining lemmas / prioritized facts
# =============================================================================
FIND_NAME_TOKENS = re.compile(r"[A-Za-z][A-Za-z0-9_']{2,}")
_FIND_TOKEN_NOISE = {
    "proof", "prove", "goal", "subgoal", "subgoals", "llm", "raw",
    "vars", "params", "frees", "schematics", "fixes", "assm", "fact",
    "true", "false", "case", "thesis",
}
# Isabelle versions differ on whether ``find_theorems`` numbers results and
# whether names contain theory qualifiers or apostrophes.  Accept both forms;
# the previous mandatory ``1.`` prefix silently discarded every result on
# clients that print ``List.distinct_map: ...``.
_LEMMA_LINE = re.compile(r"^\s*(?:\d+\.\s*)?([A-Za-z0-9_.'-]+):")


def _find_name_tokens_from_state(state_hint: str, limit: int = 12) -> List[str]:
    """Prioritise constants from marker payloads over print-state boilerplate."""
    if not state_hint:
        return []
    marker_payloads = re.findall(
        r"(?ms)^\[LLM_(?:SUBGOAL|ASSM|FACT)\]\s+(.*?)(?=^\[|\Z)",
        state_hint,
    )
    sources = [*marker_payloads, state_hint]
    ordered: List[str] = []
    seen = set()
    for source in sources:
        for token in FIND_NAME_TOKENS.findall(source):
            if token.endswith("__"):
                continue
            bare = token.rstrip("_")
            low = bare.lower()
            if low in _FIND_TOKEN_NOISE:
                continue
            if bare not in seen:
                seen.add(bare)
                ordered.append(bare)
                if len(ordered) >= limit:
                    return ordered
    return ordered

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
    toks = _find_name_tokens_from_state(state_hint)
    if not toks:
        return []
    theory_text = _build_find_theorems_theory(toks)
    lemmas, seen = [], set()
    for msg in _iter_note_messages(run_theory(isabelle, session_id, theory_text, timeout_s=10)):
        for line in msg.splitlines():
            m = _LEMMA_LINE.match(line.strip())
            if m:
                name = m.group(1).split(".")[-1]
                if name not in seen:
                    seen.add(name); lemmas.append(name)
                    if len(lemmas) >= max_lemmas:
                        return lemmas
    return lemmas

def mine_facts_prioritized(
    isabelle,
    session_id: str,
    state_hint: str,
    limit: int = 6,
    timeout_s: int = 12,
) -> List[str]:
    toks = _find_name_tokens_from_state(state_hint)
    if not toks:
        return []
    theory_text = _build_find_theorems_filtered(
        toks, ["simp", "intro", "elim", "dest", ""]
    )
    # Prefer *_def by name (cheap extra pass)
    definition_queries = [f"find_theorems name:{stem}_def" for stem in toks[:6]]
    if definition_queries:
        # _build_find_theorems_filtered returns a complete theory.  Insert
        # extra queries before ``end``; appending them after ``end`` silently
        # made the mining theory invalid on strict Isabelle clients.
        head, separator, tail = theory_text.rpartition("\nend")
        if separator:
            theory_text = head + "\n" + "\n".join(definition_queries) + separator + tail
    freq: Dict[str, int] = {}
    for msg in _iter_note_messages(
        run_theory(
            isabelle,
            session_id,
            theory_text,
            timeout_s=max(1, int(timeout_s)),
        )
    ):
        for line in msg.splitlines():
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
                ok, _ = finished_ok(
                    run_theory(
                        isabelle,
                        session_id,
                        build_theory([steps[0]] + steps[1:], False, None),
                        timeout_s=max(1, int(min(left(), 30)))
                    )
                )
                tries += 1
                if ok: return steps

    if cases_var and _booleanish_goal(goal):
        finishers = _case_finishers(facts)
        for t_fin in finishers:
            for f_fin in finishers:
                if tries >= max_tries or left() <= 0: return None
                steps = _build_structured_cases_bool_steps(goal, cases_var, t_fin, f_fin)
                ok, _ = finished_ok(
                    run_theory(
                        isabelle,
                        session_id,
                        build_theory([steps[0]] + steps[1:], False, None),
                        timeout_s=max(1, int(min(left(), 30)))
                    )
                )
                tries += 1
                if ok: return steps

    return None

# =============================================================================
# Macros (continuation suggestions) — moved from macros.py
# =============================================================================
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
            seen.add(c); ordered.append(c)
            if len(ordered) >= max(1, int(k)):
                break
    return ordered
