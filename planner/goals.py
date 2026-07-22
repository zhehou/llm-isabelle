import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout
from typing import Iterable, List, Optional, Tuple

from prover.isabelle_api import (
    build_theory, last_call_timed_out, last_print_state_block, run_theory, finished_ok,
    shutdown_isabelle,
)

# --- Constants ----------------------------------------------------------------
_LLM_SUBGOAL_MARK = "[LLM_SUBGOAL]"
_LLM_SUBGOAL_RAW_MARK = "[LLM_SUBGOAL_RAW]"
_LLM_VARS_MARK = "[LLM_VARS]"
_LLM_ASSM_MARK = "[LLM_ASSM]"
_LLM_FACT_MARK = "[LLM_FACT]"
_LLM_TYPE_MARK = "[LLM_TYPE]"
_ISA_FAST_TIMEOUT_S = int(os.getenv("ISABELLE_FAST_TIMEOUT_S", "12"))
_ISA_VERIFY_TIMEOUT_S = int(os.getenv("ISABELLE_VERIFY_TIMEOUT_S", "30"))

# === Isabelle interaction ======================================================

def _run_theory_with_timeout(isabelle, session: str, thy: List[str], *, timeout_s: Optional[int]) -> List:
    """Execute through the single timeout boundary in ``run_theory``."""
    timeout_s = timeout_s or _ISA_VERIFY_TIMEOUT_S
    result = run_theory(isabelle, session, thy, timeout_s=timeout_s)
    if last_call_timed_out():
        raise TimeoutError("isabelle_run_timeout")
    return result


def _verify_full_proof(
    isabelle,
    session: str,
    text: str,
    *,
    timeout_s: Optional[int] = None,
) -> bool:
    """Return True iff the full Isar text checks.

    All planner paths use this verifier so contextual fill, fresh-goal replay,
    proof surgery and final verification agree about acceptance.  Callers may
    use a smaller timeout when they are inside a bounded search.
    """
    try:
        thy = build_theory(text.splitlines(), add_print_state=False, end_with=None)
        result = _run_theory_with_timeout(
            isabelle,
            session,
            thy,
            timeout_s=int(timeout_s or _ISA_VERIFY_TIMEOUT_S),
        )
        ok, _ = finished_ok(result)
        return ok
    except Exception:
        return False


def _cleanup_resources(isa, proc) -> None:
    """Best-effort shutdown through the shared Isabelle lifecycle."""
    shutdown_isabelle(isa, proc, timeout_s=2)

# === Utilities =================================================================

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _strip(txt: str) -> str:
    return _ANSI.sub("", (txt or "").replace("\u00A0", " "))


def _looks_truncated(txt: str) -> bool:
    return bool(
        txt
        and (
            "…" in txt
            or " ..." in txt
            or any(re.search(r"\\<[^>]*$", l.strip()) for l in txt.splitlines())
        )
    )


def _log_state_block(prefix: str, block: str, trace: bool = True) -> None:
    if not trace:
        return
    b = block or ""
    print(f"[{prefix}] State block (length={len(b)}):")
    print(b if b.strip() else "  (empty or whitespace only)")


# === Goal extraction (Isabelle-markers only; no heuristics) ====================

def _extract_goal_from_lemma_line(lemma_line: str) -> str:
    q1, q2 = lemma_line.find('"'), lemma_line.rfind('"')
    if q1 == -1 or q2 <= q1:
        raise ValueError(f"Cannot parse lemma line: {lemma_line!r}")
    return lemma_line[q1 + 1 : q2]


def _extract_marker_records(state_block: str, marker: str) -> List[str]:
    """Extract possibly multi-line payloads emitted by an ML marker.

    Isabelle pretty-prints long terms on continuation lines.  Treating marker
    output as one physical line truncated propositions at operators such as
    ``=`` and produced malformed standalone goals.  A record ends only at the
    next marker, and whitespace inside a term is semantically immaterial.
    """
    out: List[str] = []
    for match in re.finditer(
        rf"^{re.escape(marker)}\s+(.*?)(?=^\[[A-Z0-9_]+\]|\Z)",
        state_block or "",
        flags=re.M | re.S,
    ):
        payload = re.sub(r"\s+", " ", match.group(1)).strip()
        if payload and payload not in out:
            out.append(payload)
    return out


def _extract_local_assumptions(state_block: str) -> List[str]:
    """Extract alpha-renamed local assumptions emitted by llm_print_vars."""
    return _extract_marker_records(state_block, _LLM_ASSM_MARK)


def _extract_chained_facts(state_block: str) -> List[str]:
    """Extract unnamed/chained facts supplied to the instrumentation method."""
    return _extract_marker_records(state_block, _LLM_FACT_MARK)


_UNRESOLVED_SCHEMATIC_RE = re.compile(
    r"(?<![A-Za-z0-9_'])\?[A-Za-z][A-Za-z0-9_']*"
)


def _usable_chained_facts(state_block: str) -> List[str]:
    """Return standalone-safe chained facts.

    Current facts may contain unresolved schematics (for example ``?n``)
    whose instantiation belongs to the original proof state.  Replaying those
    as theorem premises changes the goal and can make fresh-goal search prove
    an irrelevant schematic instance.  Local assumptions are retained; only
    chained facts with unresolved schematics are excluded.
    """
    assumptions = {
        re.sub(r"\s+", " ", prop).strip()
        for prop in _extract_local_assumptions(state_block)
    }
    usable: List[str] = []
    for fact in _extract_chained_facts(state_block):
        normalized = re.sub(r"\s+", " ", fact).strip()
        if not normalized or normalized in assumptions:
            continue
        if _UNRESOLVED_SCHEMATIC_RE.search(normalized):
            continue
        if normalized not in usable:
            usable.append(normalized)
    return usable


def _first_lemma_line(full_text: str) -> str:
    # A theory preamble may itself contain proved helper lemmas.  The target is
    # the final theorem declaration in the assembled candidate.
    declarations = [
        line for line in (full_text or "").splitlines()
        if re.match(r"^\s*(?:lemma|theorem|corollary)\b", line)
    ]
    return declarations[-1] if declarations else ""


def _extract_subgoal_from_markers(clean_state: str) -> Optional[str]:
    """Return subgoal text strictly from Isabelle writeln markers.
    Prefers the consistently alpha-renamed form, then RAW as a compatibility
    fallback. Multi-line safe: capture
    until the next marker line that begins with '[' or EOF.
    """
    for tag in (_LLM_SUBGOAL_MARK, _LLM_SUBGOAL_RAW_MARK):
        pat = re.compile(rf"^{re.escape(tag)}\s+(.*?)(?=^\[|\Z)", re.M | re.S)
        m = pat.search(clean_state or "")
        if m:
            return m.group(1).strip()
    return None


# === Variable information (Isabelle-markers only; no heuristics) ===============

def _extract_fixes(state_block: str) -> List[str]:
    m = re.search(r"^\[LLM_FIXES\]\s+(.*)$", state_block or "", flags=re.M)
    if not m:
        return []
    return [t for t in m.group(1).strip().split() if t]


def _extract_variable_info(state_block: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Parse "[LLM_VARS] params: ... | frees: ... | schematics: ..." strictly.
    Returns (params, frees, schems, skolems) where skolems are identified by the
    '__' suffix (from ML alpha-renaming). We DO NOT guess if markers are absent.
    """
    if not state_block:
        return [], [], [], []
    m = re.search(rf"^{re.escape(_LLM_VARS_MARK)}\s+(.*)$", state_block, flags=re.M)
    if not m:
        return [], [], [], []

    line = m.group(1)

    def get(tag: str) -> List[str]:
        mm = re.search(tag + r"\s*([^|]*)", line)
        return (mm.group(1).strip().split() if mm else [])

    params, frees, schems = get("params:"), get("frees:"), get("schematics:")

    # Merge in fixes (context-fixed names) without duplication
    fixes = _extract_fixes(state_block)
    if fixes:
        seen, merged = set(), []
        for v in fixes + frees:
            if v and v not in seen:
                seen.add(v)
                merged.append(v)
        frees = merged

    # Skolems are the names ending with '__' (strip the suffix for presentation)
    skolems = [v.rstrip('_') for v in frees if v.endswith('__')]
    true_frees = [v for v in frees if not v.endswith('__')]

    return params, true_frees, schems, skolems


# === Building the effective goal ==============================================

def _effective_goal_from_state(
    state_block: str,
    fallback_goal: str,
    full_text: str = "",
    hole_span: Tuple[int, int] = (0, 0),
    trace: bool = False,
) -> str:
    """Build effective goal from the alpha-renamed subgoal reported by Isabelle.

    The subgoal printed under [LLM_SUBGOAL*] has the form: f1 ⟹ … ⟹ fn ⟹ g.
    We reconstruct a single meta-level goal and annotate variable classes.

    Design choices (systematic, non-heuristic):
    - Use Isabelle-provided markers only; no textual guessing.
    - Treat *both* case-parameters (params) and skolemised names as meta-parameters:
      we introduce leading ⋀-quantifiers for them. Context frees stay free.
    """
    if not state_block or not state_block.strip():
        return fallback_goal

    clean = _strip(state_block)

    # Extract alpha-renamed subgoal (multi-line safe)
    renamed_subgoal = _extract_subgoal_from_markers(clean) or ""
    if not renamed_subgoal:
        return fallback_goal

    params, frees, schems, skolems = _extract_variable_info(state_block)

    # Replace any skolemised tokens ending with '__' in the *subgoal text* itself
    def clean_renamed_term(term: str) -> str:
        clean_term = term
        for tok in sorted(set(re.findall(r"\b([A-Za-z0-9_]+__)\b", term))):
            clean_term = re.sub(
                r"\b" + re.escape(tok) + r"\b", tok.rstrip("_"), clean_term
            )
        return clean_term

    clean_subgoal = clean_renamed_term(renamed_subgoal)

    # if trace:
    #     print("\n" + "=" * 60)
    #     print(f"DEBUG params={params}, frees={frees}, schems={schems}, skolems={skolems}")
    #     print(f"DEBUG renamed_subgoal= {renamed_subgoal}")
    #     print(f"DEBUG clean_subgoal= {clean_subgoal}")

    # Build the goal with meta-level quantification for params and skolems
    core = clean_subgoal
    # Isabelle moves meta-implication premises into the local proof context.
    # Reattach those assumptions when constructing a standalone prover goal;
    # otherwise ``A ⟹ B`` was incorrectly sent as just ``B``.
    # A theorem premise can live either in the persistent proof context or in
    # Isabelle's current/chained facts.  Both are logically available at the
    # hole, but only the former appeared in earlier standalone goals.
    context_props = [
        clean_renamed_term(prop)
        for prop in (
            _extract_local_assumptions(clean) + _usable_chained_facts(clean)
        )
    ]
    context_props = list(dict.fromkeys(p for p in context_props if p))
    for proposition in reversed(context_props):
        core = f"({proposition}) ⟹ ({core})"
    meta_binders: List[str] = []
    # Preserve order, dedup: params first, then skolems
    for v in list(dict.fromkeys((params or []) + (skolems or []))):
        if v:
            meta_binders.append(v)
    if meta_binders:
        core = f"⋀{' '.join(meta_binders)}. {core}"

    result = f"({core})"
    return result

# === Printing state before a hole (ML-assisted) ================================

def _build_ml_prolog() -> List[str]:
    prolog = """declare [[show_question_marks = true]]
declare [[show_types = false, show_sorts = false]]

(* Custom method to extract variable information *)
method_setup llm_print_vars = ‹
  Scan.succeed (fn ctxt =>
    Method.METHOD (fn facts => fn st =>
      if Thm.nprems_of st < 1 then (
        writeln "[LLM_NOSUBGOAL]"; Seq.single st
      ) else let
        val subgoal_term = Thm.term_of (Thm.cprem_of st 1);
        val assm_terms = map Thm.term_of (Assumption.all_assms_of ctxt);
        val fact_terms = map Thm.prop_of facts;
        val context_terms = assm_terms @ fact_terms;
        val frees_with_types = fold Term.add_frees (subgoal_term :: context_terms) [];
        val frees = map fst frees_with_types;
        val param_list = Logic.strip_params subgoal_term;
        val param_names = map fst param_list;
        val vars = Term.add_vars subgoal_term [] |> map (fn ((n, j), _) => (n, j));
        fun fmt_var (n, 0) = "?" ^ n | fmt_var (n, j) = "?" ^ n ^ Int.toString j;
        val fixes = Variable.dest_fixes ctxt |> map #1;
        
        (* Create name context and rename both params and frees *)
        val name_ctx = Name.make_context (frees @ param_names);
        val (renamed_params, name_ctx2) = fold_map Name.variant param_names name_ctx;
        val (renamed_frees, _) = fold_map Name.variant frees name_ctx2;
        
        (* Alpha-rename the term: first params, then frees *)
        val term_params_renamed = Logic.list_rename_params renamed_params subgoal_term;
        (* Build substitution with actual types from frees_with_types *)
        val free_subst = map2 (fn (old, ty) => fn new => (Free (old, ty), Free (new, ty))) 
                              frees_with_types renamed_frees;
        val fully_renamed_term = Term.subst_free free_subst term_params_renamed;
        val renamed_assms = map (Term.subst_free free_subst) assm_terms;
        val renamed_facts = map (Term.subst_free free_subst) fact_terms;
        
        val term_str = Syntax.string_of_term ctxt fully_renamed_term;
        val thy = Proof_Context.theory_of ctxt;
        val ctxt_raw = Proof_Context.init_global thy;
        val term_raw_str = Syntax.string_of_term ctxt_raw subgoal_term;
        val _ = writeln ("[LLM_SUBGOAL] " ^ term_str);
        val _ = writeln ("[LLM_SUBGOAL_RAW] " ^ term_raw_str);
        val _ = app (fn t => writeln ("[LLM_ASSM] " ^ Syntax.string_of_term ctxt t)) renamed_assms;
        val _ = app (fn t => writeln ("[LLM_FACT] " ^ Syntax.string_of_term ctxt t)) renamed_facts;
        val _ = map2 (fn (_, ty) => fn name =>
                         writeln ("[LLM_TYPE] " ^ name ^ " :: " ^
                                  Syntax.string_of_typ ctxt ty))
                     frees_with_types renamed_frees;
        val _ = writeln ("[LLM_VARS] params: " ^ (space_implode " " renamed_params) ^
                         " | frees: " ^ (space_implode " " renamed_frees) ^
                         " | schematics: " ^ (space_implode " " (map fmt_var vars)));
        val _ = writeln ("[LLM_FIXES] " ^ (space_implode " " fixes));
      in Seq.single st end))
› "extract variable information for LLM"
"""
    return [prolog.strip()]


def _inject_var_extraction(proof_lines: List[str]) -> List[str]:
    return [*proof_lines, "  apply llm_print_vars"]


def _extract_print_state_from_responses(resps: List, *, trace: bool = False) -> str:
    standard = last_print_state_block(resps) or ""
    llm_lines, debug_writeln_count, debug_llm_found = [], 0, False

    for resp in (resps or []):
        resp_type = str(getattr(resp, "response_type", "")).upper()
        if resp_type == "NOTE":
            try:
                body = json.loads(getattr(resp, "response_body", "") or "{}")
            except Exception:
                body = {}
            if isinstance(body, dict) and body.get("kind") == "writeln":
                text = str(body.get("message", "") or ""); debug_writeln_count += 1
                # print(f"[DEBUG writeln #{debug_writeln_count}]: {text[:100]}")
                if any(m in text for m in (_LLM_SUBGOAL_MARK, _LLM_SUBGOAL_RAW_MARK, _LLM_VARS_MARK, _LLM_ASSM_MARK, _LLM_FACT_MARK, _LLM_TYPE_MARK, "[LLM_FIXES]", "[LLM_TEST]")):
                    debug_llm_found = True
                    # print(f"[DEBUG] *** FOUND LLM MARKER in writeln #{debug_writeln_count}: {text[:150]}")
                    if text.strip() != "[LLM_NOSUBGOAL]":
                        llm_lines.append(text)
                elif ("goal" in text and "subgoal" in text and not standard):
                    llm_lines.append(text); standard = text
            continue

        body = getattr(resp, "response_body", None)
        if isinstance(body, bytes):
            body = body.decode(errors="replace")
        try:
            data = json.loads(body) if isinstance(body, str) and body.strip().startswith("{") else body
            if not isinstance(data, dict):
                continue
        except (json.JSONDecodeError, TypeError):
            continue

        for node in data.get("nodes", []) or []:
            for msg in node.get("messages", []) or []:
                kind, text = msg.get("kind"), msg.get("message", "") or ""
                if kind == "writeln":
                    debug_writeln_count += 1
                    # print(f"[DEBUG writeln #{debug_writeln_count}]: {text[:100]}")
                    if any(m in text for m in (_LLM_SUBGOAL_MARK, _LLM_SUBGOAL_RAW_MARK, _LLM_VARS_MARK, _LLM_ASSM_MARK, _LLM_FACT_MARK, _LLM_TYPE_MARK, "[LLM_FIXES]", "[LLM_TEST]")):
                        debug_llm_found = True
                        # print(f"[DEBUG] *** FOUND LLM MARKER in writeln #{debug_writeln_count}: {text[:150]}")
                        llm_lines.append(text)
                    elif ("goal" in text and "subgoal" in text and not standard):
                        llm_lines.append(text); standard = text
                elif kind == "error":
                    benign = (
                        'Bad context for command "end"' in text
                        or text.strip().startswith('Undefined fact: "assms"')
                        or text.strip().startswith('Undefined fact: "set_empty_conv"')
                    )
                    if not benign and trace:
                        print(f"[DEBUG ERROR]: {text[:300]}")

    # print(f"[DEBUG] Total writeln messages: {debug_writeln_count}, LLM markers found: {debug_llm_found}")
    return (standard + "\n" + "\n".join(llm_lines)) if (llm_lines and standard) else (standard or "\n".join(llm_lines))


def _print_state_before_hole(
    isabelle,
    session: str,
    full_text: str,
    hole_span: Tuple[int, int],
    trace: bool = False,
    timeout_s: Optional[int] = None,
) -> str:
    s, _ = hole_span
    lines = full_text[:s].rstrip().splitlines()
    lemma_starts = [
        i for i, line in enumerate(lines)
        if re.match(r"^\s*(?:lemma|theorem|corollary)\b", line)
    ]
    lemma_start = lemma_starts[-1] if lemma_starts else -1
    if lemma_start == -1:
        return ""

    # Preserve every declaration before the focused lemma.  In benchmark
    # theories this contains the datatypes and recursive functions that give
    # the goal its meaning.  Dropping it silently reinterpreted those constants
    # as arbitrary free variables in the fresh-goal fallback.
    theory_prefix = lines[:lemma_start]
    proof_lines = lines[lemma_start:]
    try:
        thy = build_theory(
            theory_prefix + _build_ml_prolog() + _inject_var_extraction(proof_lines),
            add_print_state=True,
            end_with="oops",
        )
        # if trace:
        #     print("[DEBUG] Theory text being sent to Isabelle:")
        #     print("=" * 60)
        #     for i, ln in enumerate(thy.splitlines()[:30]):
        #         print(f"{i:3d}: {ln}")
        #     tl = thy.splitlines()
        #     if len(tl) > 40:
        #         print("  …")
        #         for i, ln in enumerate(tl[-10:], start=len(tl) - 10):
        #             print(f"{i:3d}: {ln}")
        #     print("=" * 60)
        effective_timeout = max(1, int(timeout_s or _ISA_FAST_TIMEOUT_S))
        resps = _run_theory_with_timeout(
            isabelle, session, thy, timeout_s=effective_timeout
        )
        state = _extract_print_state_from_responses(resps, trace=trace)
        # if trace:
        #     print(f"[DEBUG] State block contains [LLM_VARS]: {_LLM_VARS_MARK in state}")
        if _looks_truncated(state):
            thy2 = build_theory(
                theory_prefix
                + ["ML ‹Pretty.setmargin 100000›"]
                + _build_ml_prolog()
                + _inject_var_extraction(proof_lines),
                add_print_state=True,
                end_with="oops",
            )
            resps2 = _run_theory_with_timeout(
                isabelle, session, thy2, timeout_s=effective_timeout
            )
            state2 = _extract_print_state_from_responses(resps2, trace=trace)
            if state2 and len(state2) > len(state):
                state = state2
        return state
    except Exception as e:
        if trace:
            print(f"[DEBUG] Exception: {e}")
        return ""
