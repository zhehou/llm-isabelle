import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout
from typing import Iterable, List, Optional, Tuple

from prover.isabelle_api import build_theory, last_print_state_block, run_theory

# --- Constants ----------------------------------------------------------------
_LLM_SUBGOAL_MARK = "[LLM_SUBGOAL]"
_LLM_SUBGOAL_RAW_MARK = "[LLM_SUBGOAL_RAW]"
_LLM_VARS_MARK = "[LLM_VARS]"
_ISA_FAST_TIMEOUT_S = int(os.getenv("ISABELLE_FAST_TIMEOUT_S", "12"))
_ISA_VERIFY_TIMEOUT_S = int(os.getenv("ISABELLE_VERIFY_TIMEOUT_S", "30"))

# === Isabelle interaction ======================================================

def _run_theory_with_timeout(isabelle, session: str, thy: List[str], *, timeout_s: Optional[int]) -> List:
    """Execute theory with a hard timeout, interrupting Isabelle if needed."""
    timeout_s = timeout_s or _ISA_VERIFY_TIMEOUT_S
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(run_theory, isabelle, session, thy)
        try:
            return fut.result(timeout=timeout_s)
        except _FuturesTimeout:
            try:
                getattr(isabelle, "interrupt", lambda: None)()
            except Exception:
                pass
            raise TimeoutError("isabelle_run_timeout")


def _verify_full_proof(isabelle, session: str, text: str) -> bool:
    """Return True iff the full Isar text checks under _ISA_VERIFY_TIMEOUT_S."""
    try:
        thy = build_theory(text.splitlines(), add_print_state=False, end_with=None)
        _run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_VERIFY_TIMEOUT_S)
        return True
    except Exception:
        return False


def _cleanup_resources(isa, proc) -> None:
    """Best-effort shutdown/cleanup for Isabelle + spawned process."""
    for action in (
        lambda: isa.shutdown(),
        lambda: getattr(__import__("planner.experiments"), "_close_client_loop_safely")(isa),
        lambda: proc.terminate(),
        lambda: proc.kill(),
        lambda: proc.wait(timeout=2),
    ):
        try:
            action()
        except Exception:
            pass

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


def _first_lemma_line(full_text: str) -> str:
    return next((ln for ln in (full_text or "").splitlines() if ln.strip().startswith("lemma ")), "")


def _extract_subgoal_from_markers(clean_state: str) -> Optional[str]:
    """Return subgoal text strictly from Isabelle writeln markers.
    Prefers RAW (internal names) then pretty form. Multi-line safe: capture
    until the next marker line that begins with '[' or EOF.
    """
    for tag in (_LLM_SUBGOAL_RAW_MARK, _LLM_SUBGOAL_MARK):
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
    clean_subgoal = renamed_subgoal
    for tok in sorted(set(re.findall(r"\b([A-Za-z0-9_]+__)\b", renamed_subgoal))):
        clean_subgoal = re.sub(r"\b" + re.escape(tok) + r"\b", tok.rstrip('_'), clean_subgoal)

    # if trace:
    #     print("\n" + "=" * 60)
    #     print(f"DEBUG params={params}, frees={frees}, schems={schems}, skolems={skolems}")
    #     print(f"DEBUG renamed_subgoal= {renamed_subgoal}")
    #     print(f"DEBUG clean_subgoal= {clean_subgoal}")

    # Build the goal with meta-level quantification for params and skolems
    core = clean_subgoal
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
    SIMPLE_METHOD' (fn i => fn st =>
      if Thm.nprems_of st < i orelse i <= 0 then (
        writeln "[LLM_NOSUBGOAL]"; Seq.single st
      ) else let
        val subgoal_term = Thm.term_of (Thm.cprem_of st i);
        val frees_with_types = Term.add_frees subgoal_term [];
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
        
        val term_str = Syntax.string_of_term ctxt fully_renamed_term;
        val thy = Proof_Context.theory_of ctxt;
        val ctxt_raw = Proof_Context.init_global thy;
        val term_raw_str = Syntax.string_of_term ctxt_raw subgoal_term;
        val _ = writeln ("[LLM_SUBGOAL] " ^ term_str);
        val _ = writeln ("[LLM_SUBGOAL_RAW] " ^ term_raw_str);
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


def _extract_print_state_from_responses(resps: List) -> str:
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
                if any(m in text for m in (_LLM_SUBGOAL_MARK, _LLM_SUBGOAL_RAW_MARK, _LLM_VARS_MARK, "[LLM_FIXES]", "[LLM_TEST]")):
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
                    if any(m in text for m in (_LLM_SUBGOAL_MARK, _LLM_VARS_MARK, "[LLM_TEST]")):
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
                    if not benign:
                        print(f"[DEBUG ERROR]: {text[:300]}")

    # print(f"[DEBUG] Total writeln messages: {debug_writeln_count}, LLM markers found: {debug_llm_found}")
    return (standard + "\n" + "\n".join(llm_lines)) if (llm_lines and standard) else (standard or "\n".join(llm_lines))


def _print_state_before_hole(isabelle, session: str, full_text: str, hole_span: Tuple[int, int], trace: bool = False) -> str:
    s, _ = hole_span
    lines = full_text[:s].rstrip().splitlines()
    lemma_start = next((i for i, ln in enumerate(lines) if ln.strip().startswith("lemma ")), -1)
    if lemma_start == -1:
        return ""

    proof_lines = lines[lemma_start:]
    try:
        thy = build_theory(_build_ml_prolog() + _inject_var_extraction(proof_lines), add_print_state=True, end_with="oops")
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
        resps = _run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_FAST_TIMEOUT_S)
        state = _extract_print_state_from_responses(resps)
        # if trace:
        #     print(f"[DEBUG] State block contains [LLM_VARS]: {_LLM_VARS_MARK in state}")
        if _looks_truncated(state):
            thy2 = build_theory(["ML ‹Pretty.setmargin 100000›"] + _build_ml_prolog() + _inject_var_extraction(proof_lines), add_print_state=True, end_with="oops")
            resps2 = _run_theory_with_timeout(isabelle, session, thy2, timeout_s=_ISA_FAST_TIMEOUT_S)
            state2 = _extract_print_state_from_responses(resps2)
            if state2 and len(state2) > len(state):
                state = state2
        return state
    except Exception as e:
        if trace:
            print(f"[DEBUG] Exception: {e}")
        return ""
