import json
import re
import os
from typing import Iterable, List, Optional, Tuple
from prover.isabelle_api import (
    build_theory, last_print_state_block, run_theory
)
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout

_LLM_SUBGOAL_MARK = "[LLM_SUBGOAL]"
_LLM_SUBGOAL_RAW_MARK = "[LLM_SUBGOAL_RAW]"
_LLM_VARS_MARK = "[LLM_VARS]"
_ISA_FAST_TIMEOUT_S = int(os.getenv("ISABELLE_FAST_TIMEOUT_S", "12"))
_ISA_VERIFY_TIMEOUT_S = int(os.getenv("ISABELLE_VERIFY_TIMEOUT_S", "30"))

# ============================================================================
# Isabelle Interaction
# ============================================================================

def _run_theory_with_timeout(isabelle, session: str, thy: List[str], *, timeout_s: Optional[int]) -> List:
    """Execute theory with hard timeout."""
    timeout_s = timeout_s or _ISA_VERIFY_TIMEOUT_S
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(run_theory, isabelle, session, thy)
        try:
            return fut.result(timeout=timeout_s)
        except _FuturesTimeout:
            if hasattr(isabelle, "interrupt"):
                try:
                    isabelle.interrupt()
                except Exception:
                    pass
            raise TimeoutError("isabelle_run_timeout")


def _verify_full_proof(isabelle, session: str, text: str) -> bool:
    """Verify complete proof validity."""
    try:
        thy = build_theory(text.splitlines(), add_print_state=False, end_with=None)
        _run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_VERIFY_TIMEOUT_S)
        return True
    except Exception:
        return False


def _cleanup_resources(isa, proc):
    """Clean up Isabelle resources."""
    for action in [
        lambda: isa.shutdown(),
        lambda: getattr(__import__('planner.experiments'), '_close_client_loop_safely')(isa),
        lambda: proc.terminate(),
        lambda: proc.kill(),
        lambda: proc.wait(timeout=2)
    ]:
        try:
            action()
        except Exception:
            pass

# ============================================================================
# Proof State Extraction
# ============================================================================

def _inject_ml_in_proof(proof_lines: List[str]) -> List[str]:
    """Inject ML code to extract variables at the proof point.
    
    This adds an ML command right before 'sorry' that will execute
    in the proof context and extract variable information.
    """
    # Find where to inject (right before sorry/qed/done)
    injection_point = len(proof_lines)
    
    ml_extract = [
        "  ML_val ‹",
        "    let",
        "      val state = Proof.assert_backward (Toplevel.proof_of @{Isar.state});",
        "      val {context = ctxt, goal = goal_thm, ...} = Proof.goal state;",
        "      val subgoal = Thm.cprem_of goal_thm 1;",
        "      val term = Thm.term_of subgoal;",
        "      val frees = Term.add_frees term [] |> map fst;",
        "      val (params, _) = Logic.strip_params term;",
        "      val param_names = map (fst o fst) params;",
        "      val vars = Term.add_vars term [];",
        "      fun fmt_var (n, 0) = \"?\" ^ n | fmt_var (n, i) = \"?\" ^ n ^ Int.toString i;",
        "      val ctx = Name.make_context (frees @ param_names);",
        "      val (renamed, _) = fold_map Name.variant param_names ctx;",
        "      val renamed_term = Logic.list_rename_params renamed term;",
        "      val term_str = Syntax.string_of_term ctxt renamed_term;",
        "    in",
        "      writeln (\"[LLM_SUBGOAL] \" ^ term_str);",
        "      writeln (\"[LLM_VARS] params: \" ^ (space_implode \" \" renamed) ^",
        "               \" | frees: \" ^ (space_implode \" \" frees) ^",
        "               \" | schematics: \" ^ (space_implode \" \" (map fmt_var vars)))",
        "    end",
        "  ›",
    ]
    
    return proof_lines[:injection_point] + ml_extract + proof_lines[injection_point:]

def _looks_truncated(txt: str) -> bool:
    """Check if text appears truncated."""
    return bool(txt and ("…" in txt or " ..." in txt or 
                any(re.search(r"\\<[^>]*$", line.strip()) for line in txt.splitlines())))


def _log_state_block(prefix: str, block: str, trace: bool = True) -> None:
    """Pretty-print the exact proof state block we consumed."""
    if not trace:
        return
    b = block or ""
    print(f"[{prefix}] State block (length={len(b)}):")
    if b.strip():
        print(b)
    else:
        print("  (empty or whitespace only)")


def _original_goal_from_state(state_block: str) -> Optional[str]:
    """Extract the original subgoal exactly as printed under `goal (…):` (or via [LLM_SUBGOAL])."""
    if not state_block or not state_block.strip():
        return None
    clean = re.sub(r"\x1b\[[0-9;]*m", "", state_block).replace("\u00A0", " ")
    m_llm = re.search(rf"^{re.escape(_LLM_SUBGOAL_MARK)}\s+(.*)$", clean, flags=re.M)
    lines = clean.splitlines()
    return _extract_subgoal(lines, m_llm)


# ============================================================================
# Goal Extraction & Processing
# ============================================================================

def _extract_goal_from_lemma_line(lemma_line: str) -> str:
    """Extract goal text from lemma line."""
    q1, q2 = lemma_line.find('"'), lemma_line.rfind('"')
    if q1 == -1 or q2 == -1 or q2 <= q1:
        raise ValueError(f"Cannot parse lemma line: {lemma_line!r}")
    return lemma_line[q1 + 1:q2]


def _first_lemma_line(full_text: str) -> str:
    """Find first lemma line."""
    return next((line for line in full_text.splitlines() if line.strip().startswith("lemma ")), "")

def _extract_vars_from_term_analysis(state_block: str, subgoal: str) -> Tuple[List[str], List[str], List[str]]:
    """Extract variable type information by analyzing the proof state.
    
    Since the ML approach isn't working, use heuristics based on:
    1. Variables in 'using this:' facts are likely free
    2. Variables only in the subgoal are likely bound
    3. Variables with ? are schematic
    
    Returns: (params, frees, schematics)
    """
    if not state_block or not subgoal:
        return [], [], []
    
    # Extract identifiers from subgoal
    ID_PATTERN = re.compile(r'\b([a-z][A-Za-z0-9_\']*)\b')
    subgoal_vars = set(ID_PATTERN.findall(subgoal))
    
    # Extract variables from 'using this:' facts
    using_vars = set()
    clean = re.sub(r"\x1b\[[0-9;]*m", "", state_block).replace("\u00A0", " ")
    lines = clean.splitlines()
    
    in_using = False
    for line in lines:
        if line.strip() == "using this:":
            in_using = True
            continue
        if in_using:
            if line.strip().startswith("goal") or not line.strip():
                break
            # Extract variables from this line
            using_vars.update(ID_PATTERN.findall(line))
    
    # Filter out common keywords and type constructors
    KEYWORDS = {"in", "if", "then", "else", "let", "case", "of", "where", "and", "or", "not", 
                "set", "True", "False", "Nil", "Cons"}
    subgoal_vars -= KEYWORDS
    using_vars -= KEYWORDS
    
    # Extract schematics (variables starting with ?)
    SCHEM_PATTERN = re.compile(r'\?([a-z][A-Za-z0-9_\']*)')
    schematics = list(set(SCHEM_PATTERN.findall(subgoal)))
    
    # Heuristic: variables in both using and subgoal are likely free
    # Variables only in subgoal are likely params (bound)
    frees = list(using_vars & subgoal_vars)
    params = list(subgoal_vars - using_vars - set(schematics))
    
    return params, frees, schematics


def _print_state_before_hole_with_ml(isabelle, session: str, full_text: str, hole_span: Tuple[int, int], trace: bool = False) -> str:
    """Capture proof state WITH ML-based variable extraction.
    
    This version attempts to inject ML code to extract variable information.
    """
    s, _ = hole_span
    lines = full_text[:s].rstrip().splitlines()
    lemma_start = next((i for i, line in enumerate(lines) if line.strip().startswith('lemma ')), -1)
    
    if lemma_start == -1:
        return ""
    
    proof_lines = lines[lemma_start:]
    
    ml_code = [
        "apply -",  # Dummy apply to ensure we're in proof mode
        "apply (print_state)",  # This doesn't exist, we'll handle it differently
    ]
    
    try:
        # Standard approach
        thy = build_theory(proof_lines, add_print_state=True, end_with="sorry")
        resps = _run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_FAST_TIMEOUT_S)
        state_block = _extract_print_state_from_responses(resps)
        
        if _looks_truncated(state_block):
            thy2 = build_theory(proof_lines + ["ML ‹Pretty.setmargin 100000›"], 
                              add_print_state=True, end_with="sorry")
            resps2 = _run_theory_with_timeout(isabelle, session, thy2, timeout_s=_ISA_FAST_TIMEOUT_S)
            state_block2 = _extract_print_state_from_responses(resps2)
            if state_block2:
                state_block = state_block2
        
        return state_block
    except Exception:
        return ""

def _extract_using_facts(lines: List[str]) -> List[str]:
    """Extract 'using this:' facts."""
    for i, line in enumerate(lines):
        if line.strip() == "using this:":
            raw_block = []
            i += 1
            while i < len(lines):
                s = lines[i].strip()
                if not s or s.startswith("goal") or s.startswith("using this:"):
                    break
                raw_block.append(lines[i])
                i += 1
            return _process_using_block(raw_block)
    return []


def _process_using_block(raw_block: List[str]) -> List[str]:
    """Process raw using block into facts."""
    if not raw_block:
        return []
    
    INFIX_TAIL = re.compile(r"(=|⟹|⇒|->|→|<->|↔|⟷|∧|∨|@|::|≤|≥|≠|,)\s*$")
    items, cur, head_indent, paren_balance, head_infix = [], [], 0, 0, False
    
    def lead_spaces(s): return len(s) - len(s.lstrip(" "))
    def delta_parens(s): return s.count("(") + s.count("[") + s.count("{") - s.count(")") - s.count("]") - s.count("}")
    
    def flush():
        nonlocal cur, items
        if cur:
            txt = re.sub(r"\s+", " ", " ".join(x.strip() for x in cur)).strip()
            txt = re.sub(r"\\<[^>]*$", "", txt.replace("…", "")).rstrip()
            if txt:
                items.append(txt)
        cur = []
    
    for raw in raw_block:
        if re.match(r"\s*(?:[•∙·\-\*]|\(\d+\))\s+", raw):
            flush()
            cur = [raw.split(maxsplit=1)[1].strip() if len(raw.split(maxsplit=1)) > 1 else raw.strip()]
            head_indent, paren_balance, head_infix = lead_spaces(raw), delta_parens(raw), bool(INFIX_TAIL.search(raw))
            continue
        
        if not cur:
            cur = [raw.strip()]
            head_indent, paren_balance, head_infix = lead_spaces(raw), delta_parens(raw), bool(INFIX_TAIL.search(raw))
            continue
        
        ind, delta = lead_spaces(raw), delta_parens(raw)
        if ind > head_indent or paren_balance > 0 or head_infix:
            cur.append(raw.strip())
            paren_balance += delta
            head_infix = bool(INFIX_TAIL.search(raw))
        else:
            flush()
            cur = [raw.strip()]
            head_indent, paren_balance, head_infix = ind, delta, bool(INFIX_TAIL.search(raw))
    
    flush()
    return items


def _extract_subgoal(lines: List[str], m_llm) -> Optional[str]:
    """Extract first numbered subgoal."""
    if m_llm:
        return m_llm.group(1).strip()
    
    for idx, line in enumerate(lines):
        m = re.match(r"\s*\d+\.\s*(\S.*)$", line)
        if m:
            parts = [m.group(1).strip()]
            j = idx + 1
            while j < len(lines):
                lj = lines[j]
                if re.match(r"\s*\d+\.\s", lj) or lj.lstrip().startswith(("goal", "using this:")):
                    break
                if lj.startswith(" "):
                    parts.append(lj.strip())
                    j += 1
                    continue
                break
            
            subgoal = re.sub(r"\s+", " ", " ".join(parts)).strip()
            return re.sub(r"\\<[^>]*$", "", subgoal.replace("…", "")).rstrip()
    
    return None

# ============================================================================
# Alternative variable type detection approach
# ============================================================================

def _extract_variable_types(state_block: str) -> Tuple[set, set, set]:
    """Extract sets of bound parameters, free variables, and schematic variables.
    
    Returns:
        (params_set, frees_set, schematics_set)
    """
    if not state_block:
        return set(), set(), set()
    
    # Look for [LLM_VARS] line
    m = re.search(rf"^{re.escape(_LLM_VARS_MARK)}\s+(.*)$", state_block, flags=re.M)
    if not m:
        return set(), set(), set()
    
    vars_line = m.group(1)
    
    # Parse: "params: x y | frees: a b | schematics: ?z"
    params, frees, schematics = set(), set(), set()
    
    # Extract params
    m_params = re.search(r"params:\s*([^|]*)", vars_line)
    if m_params:
        params = set(m_params.group(1).strip().split())
    
    # Extract frees
    m_frees = re.search(r"frees:\s*([^|]*)", vars_line)
    if m_frees:
        frees = set(m_frees.group(1).strip().split())
    
    # Extract schematics
    m_schems = re.search(r"schematics:\s*([^|]*)", vars_line)
    if m_schems:
        schematics = set(m_schems.group(1).strip().split())
    
    return params, frees, schematics


def _annotate_goal_with_var_types(goal: str, params: set, frees: set, schematics: set) -> str:
    """Annotate goal string to distinguish variable types.
    
    Example output:
        "⋀x_bound xs_bound. x_free ∈ set ((x_bound # xs_bound) @ ys_free)"
    """
    if not (params or frees or schematics):
        return goal
    
    # We'll use a simple suffix approach: _bound, _free, _schem
    # More sophisticated: use special markers like [BOUND:x] that LLM can understand
    
    def replace_var(match):
        var = match.group(1)
        if var in params:
            return f"[BOUND:{var}]"
        elif var in frees:
            return f"[FREE:{var}]"
        elif f"?{var}" in schematics or var in schematics:
            return f"[SCHEM:{var}]"
        return var
    
    # Match identifiers (simplified - may need refinement for Isabelle syntax)
    # This pattern matches variable names, being careful about word boundaries
    pattern = r'\b([a-z][A-Za-z0-9_\']*)\b'
    
    annotated = re.sub(pattern, replace_var, goal)
    return annotated


def _effective_goal_from_state_with_types(state_block: str, fallback_goal: str, 
                                          full_text: str = "", 
                                          hole_span: Tuple[int, int] = (0, 0), 
                                          trace: bool = False) -> str:
    """Build goal from local print_state with variable type annotations."""
    if not state_block or not state_block.strip():
        return fallback_goal
    
    # Extract variable type information
    params, frees, schematics = _extract_variable_types(state_block)
    
    # Get the base goal (existing logic)
    goal = _effective_goal_from_state(state_block, fallback_goal, full_text, hole_span, trace)
    
    # Annotate with variable types
    if params or frees or schematics:
        goal = _annotate_goal_with_var_types(goal, params, frees, schematics)
    
    return goal


# Alternative: Add variable type info as metadata
def _format_goal_with_metadata(goal: str, params: set, frees: set, schematics: set) -> str:
    """Format goal with explicit variable type metadata.
    
    Example:
        Goal: x ∈ set ((x # xs) @ ys)
        Variable types:
        - Bound (∀-quantified): x, xs
        - Free (from context): ys
    """
    if not (params or frees or schematics):
        return goal
    
    metadata = []
    if params:
        metadata.append(f"- Bound (∀-quantified): {', '.join(sorted(params))}")
    if frees:
        metadata.append(f"- Free (from context): {', '.join(sorted(frees))}")
    if schematics:
        metadata.append(f"- Schematic (unification vars): {', '.join(sorted(schematics))}")
    
    return f"{goal}\n\nVariable types:\n" + "\n".join(metadata)

def _effective_goal_from_state_alt(state_block: str, fallback_goal: str, full_text: str = "", 
                              hole_span: Tuple[int, int] = (0, 0), trace: bool = False) -> str:
    params, frees, schematics = _extract_variable_types(state_block)
    goal = _effective_goal_from_state(state_block, fallback_goal, full_text, hole_span, trace)
    return _format_goal_with_metadata(goal, params, frees, schematics)

# ============================================================================
# Debug code
# ============================================================================

def _build_ml_prolog() -> List[str]:
    """Build ML prolog that sets up variable extraction method.
    
    IMPORTANT: build_theory only keeps the FIRST line unindented.
    Uses ‹ › (cartouche) delimiters which are the modern Isabelle standard.
    """
    prolog_text = """declare [[show_question_marks = true]]
declare [[show_types = false, show_sorts = false]]

(* Custom method to extract variable information *)
method_setup llm_print_vars = ‹
  Scan.succeed (fn ctxt =>
    SIMPLE_METHOD' (fn i => fn st =>
      if Thm.nprems_of st < i orelse i <= 0 then (
        (* changed tag so parser can ignore this safely *)
        writeln "[LLM_NOSUBGOAL]";
        Seq.single st
      ) else let
        val subgoal_term = Thm.term_of (Thm.cprem_of st i);
        val frees = Term.add_frees subgoal_term [] |> map (fn (n, _) => n);
        val param_list = Logic.strip_params subgoal_term;
        val param_names = map fst param_list;
        val vars = Term.add_vars subgoal_term [] |> map (fn ((n, j), _) => (n, j));
        fun fmt_var (n, 0) = "?" ^ n | fmt_var (n, j) = "?" ^ n ^ Int.toString j;

        (* NEW: fixed variables from the proof context (Isar “case” fixes) *)
        val fixes = Variable.dest_fixes ctxt |> map #1;

        (* pretty print subgoal with renamed params (unchanged) *)
        val name_ctx = Name.make_context (frees @ param_names);
        val (renamed_params, _) = fold_map Name.variant param_names name_ctx;
        val renamed_term = Logic.list_rename_params renamed_params subgoal_term;
        val term_str = Syntax.string_of_term ctxt renamed_term;

        (* NEW: also print the subgoal from a RAW global context, which preserves internal names like xa__/xsa__ *)
        val thy = Proof_Context.theory_of ctxt;
        val ctxt_raw = Proof_Context.init_global thy;
        val term_raw_str = Syntax.string_of_term ctxt_raw subgoal_term;        

        val params_str = space_implode " " renamed_params;
        val frees_str  = space_implode " " frees;
        val fixes_str  = space_implode " " fixes;
        val vars_str   = space_implode " " (map fmt_var vars);

        val _ = writeln ("[LLM_SUBGOAL] " ^ term_str);
        val _ = writeln ("[LLM_SUBGOAL_RAW] " ^ term_raw_str);
        val _ = writeln ("[LLM_VARS] params: " ^ params_str ^ " | frees: " ^ frees_str ^ " | schematics: " ^ vars_str);
        (* NEW: separate line for fixes so Python can prefer these names *)
        val _ = writeln ("[LLM_FIXES] " ^ fixes_str);
      in Seq.single st end))
› "extract variable information for LLM"
"""
    return [prolog_text.strip()]


def _inject_var_extraction(proof_lines: List[str]) -> List[str]:
    """Inject variable extraction method call."""
    result = proof_lines[:]
    result.append("  apply llm_print_vars")
    return result


def _print_state_before_hole(isabelle, session: str, full_text: str, hole_span: Tuple[int, int], trace: bool = False) -> str:
    """Capture proof state before hole with ML-extracted variable info."""
    s, _ = hole_span
    lines = full_text[:s].rstrip().splitlines()
    lemma_start = next((i for i, line in enumerate(lines) if line.strip().startswith('lemma ')), -1)
    
    if lemma_start == -1:
        return ""
    
    proof_lines = lines[lemma_start:]
    
    try:
        prolog = _build_ml_prolog()
        enhanced_proof = _inject_var_extraction(proof_lines)
        
        thy = build_theory(prolog + enhanced_proof, add_print_state=True, end_with="sorry")
        
        if trace:
            print("[DEBUG] Theory text being sent to Isabelle:")
            print("=" * 60)
            # Print first 30 lines and last 10 lines
            thy_lines = thy.splitlines()
            for i, line in enumerate(thy_lines[:30]):
                print(f"{i:3d}: {line}")
            if len(thy_lines) > 40:
                print("  ...")
                for i, line in enumerate(thy_lines[-10:], start=len(thy_lines)-10):
                    print(f"{i:3d}: {line}")
            print("=" * 60)
        
        resps = _run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_FAST_TIMEOUT_S)
        state_block = _extract_print_state_from_responses(resps)
        
        if trace:
            print(f"[DEBUG] State block contains [LLM_VARS]: {_LLM_VARS_MARK in state_block}")
        
        if _looks_truncated(state_block):
            margin_prolog = ["ML ‹Pretty.setmargin 100000›"] + prolog
            thy2 = build_theory(margin_prolog + _inject_var_extraction(proof_lines), add_print_state=True, end_with="sorry")
            resps2 = _run_theory_with_timeout(isabelle, session, thy2, timeout_s=_ISA_FAST_TIMEOUT_S)
            state_block2 = _extract_print_state_from_responses(resps2)
            if state_block2 and len(state_block2) > len(state_block):
                state_block = state_block2
        
        return state_block
    except Exception as e:
        if trace:
            print(f"[DEBUG] Exception: {e}")
        return ""


def _extract_print_state_from_responses(resps: List) -> str:
    """Extract print_state output INCLUDING our LLM markers from Isabelle responses."""
    standard = last_print_state_block(resps) or ""
    llm_lines = []
    
    debug_writeln_count = 0
    debug_llm_found = False
    
    for resp in (resps or []):
        resp_type = str(getattr(resp, "response_type", "")).upper()

        # Handle NOTE (where writeln usually appears)
        if resp_type == "NOTE":
            try:
                body = json.loads(getattr(resp, "response_body", "") or "{}")
            except Exception:
                body = {}
            if isinstance(body, dict) and body.get("kind") == "writeln":
                text = str(body.get("message", "") or "")
                debug_writeln_count += 1
                print(f"[DEBUG writeln #{debug_writeln_count}]: {text[:100]}")
                if any(m in text for m in (_LLM_SUBGOAL_MARK, _LLM_SUBGOAL_RAW_MARK, _LLM_VARS_MARK, "[LLM_FIXES]", "[LLM_TEST]")):
                    debug_llm_found = True
                    print(f"[DEBUG] *** FOUND LLM MARKER in writeln #{debug_writeln_count}: {text[:150]}")
                    if text.strip() != "[LLM_NOSUBGOAL]":
                        llm_lines.append(text)
                elif ("goal" in text and "subgoal" in text and not standard):
                    llm_lines.append(text)
                    standard = text
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
        
        for node in data.get("nodes", []):
            for msg in node.get("messages", []) or []:
                kind = msg.get("kind")
                text = msg.get("message", "") or ""
                
                if kind == "writeln":
                    debug_writeln_count += 1
                    print(f"[DEBUG writeln #{debug_writeln_count}]: {text[:100]}")
                    
                    if _LLM_SUBGOAL_MARK in text or _LLM_VARS_MARK in text or "[LLM_TEST]" in text:
                        debug_llm_found = True
                        print(f"[DEBUG] *** FOUND LLM MARKER in writeln #{debug_writeln_count}: {text[:150]}")
                        llm_lines.append(text)
                    elif ("goal" in text and "subgoal" in text and not standard):
                        llm_lines.append(text)
                        standard = text
                        
                elif kind == "error":
                    # Print ALL errors, not just llm_print_vars
                    print(f"[DEBUG ERROR]: {text[:300]}")
    
    print(f"[DEBUG] Total writeln messages: {debug_writeln_count}, LLM markers found: {debug_llm_found}")
    
    return (standard + "\n" + "\n".join(llm_lines)) if (llm_lines and standard) else (standard or "\n".join(llm_lines))


def _effective_goal_from_state(state_block: str, fallback_goal: str, full_text: str = "", 
                              hole_span: Tuple[int, int] = (0, 0), trace: bool = False) -> str:
    """Build goal from local print_state with proper variable handling."""
    if not state_block or not state_block.strip():
        return fallback_goal
    
    clean = re.sub(r"\x1b\[[0-9;]*m", "", state_block).replace("\u00A0", " ")
    # Prefer RAW (internal names), then pretty
    m_llm = re.search(rf"^{re.escape(_LLM_SUBGOAL_RAW_MARK)}\s+(.*)$", clean, flags=re.M)
    if not m_llm:
        m_llm = re.search(rf"^{re.escape(_LLM_SUBGOAL_MARK)}\s+(.*)$", clean, flags=re.M)
    
    lines = clean.splitlines()
    using_facts = _extract_using_facts(lines)
    subgoal = _extract_subgoal(lines, m_llm)
    
    if not subgoal:
        return fallback_goal
    
    # Extract variable information
    params, frees, schematics = _extract_variable_info(state_block)
    
    if trace:
        print(f"\n{'='*60}")
        print(f"DEBUG: Extracted from Isabelle:")
        print(f"  params (bound): {params}")
        print(f"  frees (context): {frees}")
        print(f"  schematics: {schematics}")
        print(f"DEBUG: subgoal: {subgoal}")
    
    # Build goal
    facts = [f"({f.strip()})" for f in using_facts] if using_facts else []
    
    # Add binders only if we have params from Isabelle
    if params and not subgoal.strip().startswith("⋀"):
        goal_core = f"⋀{' '.join(params)}. {subgoal}"
    else:
        goal_core = subgoal
    
    result = " ⟹ ".join(facts + [f"({goal_core})"]) if facts else f"({goal_core})"
    
    # Add variable type annotations for LLM
    if params or frees or schematics:
        var_info = []
        if params:
            var_info.append(f"BOUND: {','.join(params)}")
        if frees:
            var_info.append(f"FREE: {','.join(frees)}")
        if schematics:
            var_info.append(f"SCHEM: {','.join(schematics)}")
        result = f"{result}\n[VAR_TYPES: {' | '.join(var_info)}]"
    
    if trace:
        print(f"DEBUG: Final goal: {result[:150]}...")
        print(f"{'='*60}\n")
    
    return result

def _extract_fixes(state_block: str) -> List[str]:
    m = re.search(r"^\[LLM_FIXES\]\s+(.*)$", state_block, flags=re.M)
    if not m:
        return []
    raw = m.group(1).strip()
    return [t for t in raw.split() if t]

def _extract_variable_info(state_block: str) -> Tuple[List[str], List[str], List[str]]:
    """Extract variable information from [LLM_VARS] line."""
    if not state_block:
        return [], [], []
    
    m = re.search(rf"^{re.escape(_LLM_VARS_MARK)}\s+(.*)$", state_block, flags=re.M)
    if not m:
        return [], [], []
    
    vars_line = m.group(1)
    params, frees, schematics = [], [], []
    
    m_params = re.search(r"params:\s*([^|]*)", vars_line)
    if m_params and m_params.group(1).strip():
        params = m_params.group(1).strip().split()
    
    m_frees = re.search(r"frees:\s*([^|]*)", vars_line)
    if m_frees and m_frees.group(1).strip():
        frees = m_frees.group(1).strip().split()
    
    m_schems = re.search(r"schematics:\s*([^|]*)", vars_line)
    if m_schems and m_schems.group(1).strip():
        schematics = m_schems.group(1).strip().split()
    
    # Prefer FIXES for display-stable “context frees”
    fixes = _extract_fixes(state_block)
    if fixes:
        seen, merged = set(), []
        for v in fixes + frees:
            if v and v not in seen:
                seen.add(v); merged.append(v)
        frees = merged
    return params, frees, schematics

