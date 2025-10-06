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
_SCHEM = re.compile(r"\?([a-z][A-Za-z0-9_']*)")
_IDENT = re.compile(r"\b([a-z][A-Za-z0-9_']*)\b")


def _strip(txt: str) -> str:
    return _ANSI.sub("", (txt or "").replace("\u00A0", " "))


def _looks_truncated(txt: str) -> bool:
    return bool(txt and ("…" in txt or " ..." in txt or any(re.search(r"\\<[^>]*$", l.strip()) for l in txt.splitlines())))


def _log_state_block(prefix: str, block: str, trace: bool = True) -> None:
    if not trace:
        return
    b = block or ""
    print(f"[{prefix}] State block (length={len(b)}):")
    print(b if b.strip() else "  (empty or whitespace only)")


# === Goal extraction ===========================================================

def _extract_goal_from_lemma_line(lemma_line: str) -> str:
    q1, q2 = lemma_line.find('"'), lemma_line.rfind('"')
    if q1 == -1 or q2 <= q1:
        raise ValueError(f"Cannot parse lemma line: {lemma_line!r}")
    return lemma_line[q1 + 1 : q2]


def _first_lemma_line(full_text: str) -> str:
    return next((ln for ln in (full_text or "").splitlines() if ln.strip().startswith("lemma ")), "")


def _extract_subgoal(lines: List[str], m_llm) -> Optional[str]:
    if m_llm:
        return m_llm.group(1).strip()
    for i, ln in enumerate(lines):
        m = re.match(r"\s*\d+\.\s*(\S.*)$", ln)
        if not m:
            continue
        parts = [m.group(1).strip()]
        j = i + 1
        while j < len(lines):
            lj = lines[j]
            if re.match(r"\s*\d+\.\s", lj) or lj.lstrip().startswith(("goal", "using this:")):
                break
            if lj.startswith(" "):
                parts.append(lj.strip()); j += 1; continue
            break
        sub = re.sub(r"\s+", " ", " ".join(parts)).strip()
        return re.sub(r"\\<[^>]*$", "", sub.replace("…", "")).rstrip()
    return None


# === Using facts parsing =======================================================

def _extract_using_facts(lines: List[str]) -> List[str]:
    for i, ln in enumerate(lines):
        if ln.strip() != "using this:":
            continue
        raw, i = [], i + 1
        while i < len(lines):
            s = lines[i].strip()
            if not s or s.startswith("goal") or s.startswith("using this:"):
                break
            raw.append(lines[i]); i += 1
        return _process_using_block(raw)
    return []


def _process_using_block(raw_block: List[str]) -> List[str]:
    if not raw_block:
        return []
    INFIX_TAIL = re.compile(r"(=|⟹|⇒|->|→|<->|↔|⟷|∧|∨|@|::|≤|≥|≠|,)\s*$")

    def lead(s: str) -> int: return len(s) - len(s.lstrip(" "))
    def bal(s: str) -> int: return s.count("(") + s.count("[") + s.count("{") - s.count(")") - s.count("]") - s.count("}")

    items, cur, head_indent, paren, infix = [], [], 0, 0, False

    def flush():
        nonlocal cur
        if cur:
            txt = re.sub(r"\s+", " ", " ".join(x.strip() for x in cur)).strip()
            txt = re.sub(r"\\<[^>]*$", "", txt.replace("…", "")).rstrip()
            if txt:
                items.append(txt)
        cur = []

    for r in raw_block:
        ind, delta, has_infix = lead(r), bal(r), bool(INFIX_TAIL.search(r))
        if not cur:
            cur = [r.strip()]; head_indent, paren, infix = ind, delta, has_infix; continue
        if ind > head_indent or paren > 0 or infix:
            cur.append(r.strip()); paren += delta; infix = has_infix; continue
        flush(); cur = [r.strip()]; head_indent, paren, infix = ind, delta, has_infix
    flush(); return items


# === Variable information ======================================================

def _extract_fixes(state_block: str) -> List[str]:
    m = re.search(r"^\[LLM_FIXES\]\s+(.*)$", state_block, flags=re.M)
    if not m:
        return []
    return [t for t in m.group(1).strip().split() if t]


def _extract_variable_info(state_block: str) -> Tuple[List[str], List[str], List[str]]:
    """Parse "[LLM_VARS] params: ... | frees: ... | schematics: ...".
    Falls back to lightweight heuristics if the marker is absent.
    """
    if not state_block:
        return [], [], []

    # Preferred: explicit ML markers
    m = re.search(rf"^{re.escape(_LLM_VARS_MARK)}\s+(.*)$", state_block, flags=re.M)
    if m:
        line = m.group(1)
        get = lambda k: (re.search(k + r"\s*([^|]*)", line) or [None, ""]) [1].strip().split()
        params, frees, schems = get("params:"), get("frees:"), get("schematics:")
        fixes = _extract_fixes(state_block)
        if fixes:
            seen, merged = set(), []
            for v in fixes + frees:
                if v and v not in seen:
                    seen.add(v); merged.append(v)
            frees = merged
        return params, frees, schems

    # Heuristic fallback (no ML markers): infer from "using this:" and the first subgoal
    clean = _strip(state_block)
    lines = clean.splitlines()
    sub = _extract_subgoal(lines, None) or ""
    using = set(_IDENT.findall("\n".join(_extract_using_facts(lines))))
    ids = set(_IDENT.findall(sub))
    for kw in {"in","if","then","else","let","case","of","where","and","or","not","set","True","False","Nil","Cons"}:
        using.discard(kw); ids.discard(kw)
    schems = list(set(_SCHEM.findall(sub)))
    frees = list(using & ids)
    params = list(ids - using - set(schems))
    return params, frees, schems


# === Building the effective goal ==============================================

def _effective_goal_from_state(state_block: str, fallback_goal: str, full_text: str = "", 
                               hole_span: Tuple[int, int] = (0, 0), trace: bool = False) -> str:
    if not state_block or not state_block.strip():
        return fallback_goal

    clean = _strip(state_block)
    # Prefer RAW (internal names), then pretty-print variant
    m_llm = re.search(rf"^{re.escape(_LLM_SUBGOAL_RAW_MARK)}\s+(.*)$", clean, flags=re.M) or \
            re.search(rf"^{re.escape(_LLM_SUBGOAL_MARK)}\s+(.*)$", clean, flags=re.M)

    lines = clean.splitlines()
    using_facts = _extract_using_facts(lines)
    subgoal = _extract_subgoal(lines, m_llm)
    if not subgoal:
        return fallback_goal

    params, frees, schems = _extract_variable_info(state_block)
    if trace:
        print("\n" + "=" * 60)
        print(f"DEBUG params={params}, frees={frees}, schematics={schems}")
        print(f"DEBUG subgoal= {subgoal}")

    # (optionally) insert explicit binders if Isabelle provided params
    core = f"⋀{' '.join(params)}. {subgoal}" if params and not subgoal.strip().startswith("⋀") else subgoal
    facts = [f"({f.strip()})" for f in using_facts] if using_facts else []
    result = " ⟹ ".join(facts + [f"({core})"]) if facts else f"({core})"

    # Attach a compact variable summary for downstream prompts
    if params or frees or schems:
        parts = []
        if params: parts.append("BOUND:" + ",".join(params))
        if frees: parts.append("FREE:" + ",".join(frees))
        if schems: parts.append("SCHEM:" + ",".join(schems))
        result = f"{result}\n[VAR_TYPES: {' | '.join(parts)}]"

    if trace:
        print(f"DEBUG final goal: {result[:150]}…\n" + "=" * 60 + "\n")
    return result


# Convenience variants kept for backwards compatibility ------------------------

def _annotate_goal_with_var_types(goal: str, params: set, frees: set, schematics: set) -> str:
    if not (params or frees or schematics):
        return goal
    def repl(m):
        v = m.group(1)
        if v in params: return f"[BOUND:{v}]"
        if v in frees: return f"[FREE:{v}]"
        if (v in schematics) or (f"?{v}" in schematics): return f"[SCHEM:{v}]"
        return v
    return re.sub(r"\b([a-z][A-Za-z0-9_']*)\b", repl, goal)


def _effective_goal_from_state_with_types(state_block: str, fallback_goal: str, full_text: str = "", 
                                          hole_span: Tuple[int, int] = (0, 0), trace: bool = False) -> str:
    goal = _effective_goal_from_state(state_block, fallback_goal, full_text, hole_span, trace)
    params, frees, schems = _extract_variable_info(state_block)
    return _annotate_goal_with_var_types(goal, set(params), set(frees), set(schems))


def _format_goal_with_metadata(goal: str, params: set, frees: set, schematics: set) -> str:
    if not (params or frees or schematics):
        return goal
    meta = []
    if params: meta.append(f"- Bound (∀-quantified): {', '.join(sorted(params))}")
    if frees: meta.append(f"- Free (from context): {', '.join(sorted(frees))}")
    if schematics: meta.append(f"- Schematic (unification vars): {', '.join(sorted(schematics))}")
    return f"{goal}\n\nVariable types:\n" + "\n".join(meta)


def _effective_goal_from_state_alt(state_block: str, fallback_goal: str, full_text: str = "", 
                                   hole_span: Tuple[int, int] = (0, 0), trace: bool = False) -> str:
    params, frees, schems = _extract_variable_info(state_block)
    goal = _effective_goal_from_state(state_block, fallback_goal, full_text, hole_span, trace)
    return _format_goal_with_metadata(goal, set(params), set(frees), set(schems))


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
        val frees = Term.add_frees subgoal_term [] |> map (fn (n, _) => n);
        val param_list = Logic.strip_params subgoal_term;
        val param_names = map fst param_list;
        val vars = Term.add_vars subgoal_term [] |> map (fn ((n, j), _) => (n, j));
        fun fmt_var (n, 0) = "?" ^ n | fmt_var (n, j) = "?" ^ n ^ Int.toString j;
        val fixes = Variable.dest_fixes ctxt |> map #1;
        val name_ctx = Name.make_context (frees @ param_names);
        val (renamed_params, _) = fold_map Name.variant param_names name_ctx;
        val renamed_term = Logic.list_rename_params renamed_params subgoal_term;
        val term_str = Syntax.string_of_term ctxt renamed_term;
        val thy = Proof_Context.theory_of ctxt;
        val ctxt_raw = Proof_Context.init_global thy;
        val term_raw_str = Syntax.string_of_term ctxt_raw subgoal_term;
        val _ = writeln ("[LLM_SUBGOAL] " ^ term_str);
        val _ = writeln ("[LLM_SUBGOAL_RAW] " ^ term_raw_str);
        val _ = writeln ("[LLM_VARS] params: " ^ (space_implode " " renamed_params) ^
                         " | frees: " ^ (space_implode " " frees) ^
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
                print(f"[DEBUG writeln #{debug_writeln_count}]: {text[:100]}")
                if any(m in text for m in (_LLM_SUBGOAL_MARK, _LLM_SUBGOAL_RAW_MARK, _LLM_VARS_MARK, "[LLM_FIXES]", "[LLM_TEST]")):
                    debug_llm_found = True
                    print(f"[DEBUG] *** FOUND LLM MARKER in writeln #{debug_writeln_count}: {text[:150]}")
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
                    print(f"[DEBUG writeln #{debug_writeln_count}]: {text[:100]}")
                    if any(m in text for m in (_LLM_SUBGOAL_MARK, _LLM_VARS_MARK, "[LLM_TEST]")):
                        debug_llm_found = True
                        print(f"[DEBUG] *** FOUND LLM MARKER in writeln #{debug_writeln_count}: {text[:150]}")
                        llm_lines.append(text)
                    elif ("goal" in text and "subgoal" in text and not standard):
                        llm_lines.append(text); standard = text
                elif kind == "error":
                    print(f"[DEBUG ERROR]: {text[:300]}")

    print(f"[DEBUG] Total writeln messages: {debug_writeln_count}, LLM markers found: {debug_llm_found}")
    return (standard + "\n" + "\n".join(llm_lines)) if (llm_lines and standard) else (standard or "\n".join(llm_lines))


def _print_state_before_hole(isabelle, session: str, full_text: str, hole_span: Tuple[int, int], trace: bool = False) -> str:
    s, _ = hole_span
    lines = full_text[:s].rstrip().splitlines()
    lemma_start = next((i for i, ln in enumerate(lines) if ln.strip().startswith("lemma ")), -1)
    if lemma_start == -1:
        return ""

    proof_lines = lines[lemma_start:]
    try:
        thy = build_theory(_build_ml_prolog() + _inject_var_extraction(proof_lines), add_print_state=True, end_with="sorry")
        if trace:
            print("[DEBUG] Theory text being sent to Isabelle:")
            print("=" * 60)
            for i, ln in enumerate(thy.splitlines()[:30]):
                print(f"{i:3d}: {ln}")
            tl = thy.splitlines()
            if len(tl) > 40:
                print("  …")
                for i, ln in enumerate(tl[-10:], start=len(tl) - 10):
                    print(f"{i:3d}: {ln}")
            print("=" * 60)
        resps = _run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_FAST_TIMEOUT_S)
        state = _extract_print_state_from_responses(resps)
        if trace:
            print(f"[DEBUG] State block contains [LLM_VARS]: {_LLM_VARS_MARK in state}")
        if _looks_truncated(state):
            thy2 = build_theory(["ML ‹Pretty.setmargin 100000›"] + _build_ml_prolog() + _inject_var_extraction(proof_lines), add_print_state=True, end_with="sorry")
            resps2 = _run_theory_with_timeout(isabelle, session, thy2, timeout_s=_ISA_FAST_TIMEOUT_S)
            state2 = _extract_print_state_from_responses(resps2)
            if state2 and len(state2) > len(state):
                state = state2
        return state
    except Exception as e:
        if trace:
            print(f"[DEBUG] Exception: {e}")
        return ""


# --- Legacy helpers retained (lightweight wrappers / no-ops) ------------------

def _original_goal_from_state(state_block: str, fallback_goal: Optional[str] = None, full_text: str = "", 
                               hole_span: Tuple[int, int] = (0, 0), trace: bool = False) -> str:
    """Legacy alias for compatibility with older driver imports.
    Accepts the old single-argument call form. If fallback_goal is omitted,
    default to an empty string.
    """
    if fallback_goal is None:
        fallback_goal = ""
    return _effective_goal_from_state(state_block, fallback_goal, full_text, hole_span, trace)



def _inject_ml_in_proof(proof_lines: List[str]) -> List[str]:
    """Legacy alias: previous variant attempted extra ML; now identical to _inject_var_extraction."""
    return _inject_var_extraction(proof_lines)


def _print_state_before_hole_with_ml(isabelle, session: str, full_text: str, hole_span: Tuple[int, int], trace: bool = False) -> str:
    """Legacy alias for compatibility; delegates to _print_state_before_hole."""
    return _print_state_before_hole(isabelle, session, full_text, hole_span, trace)
