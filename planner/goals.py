import json
import re
import os
from typing import Iterable, List, Optional, Tuple
from prover.isabelle_api import (
    build_theory, last_print_state_block, run_theory
)
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout

_LLM_SUBGOAL_MARK = "[LLM_SUBGOAL]"
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

def _extract_print_state_from_responses(resps: List) -> str:
    """Extract print_state output from Isabelle responses."""
    standard = last_print_state_block(resps) or ""
    llm_lines = []
    
    for resp in (resps or []):
        if str(getattr(resp, "response_type", "")).upper() != "FINISHED":
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
                if msg.get("kind") != "writeln":
                    continue
                text = msg.get("message", "") or ""
                if any(mark in text for mark in [_LLM_SUBGOAL_MARK, _LLM_VARS_MARK]) or \
                   ("goal" in text and "subgoal" in text and not standard):
                    llm_lines.append(text)
                    if "goal" in text and "subgoal" in text:
                        standard = text
    
    return (standard + "\n" + "\n".join(llm_lines)) if (llm_lines and standard) else (standard or "\n".join(llm_lines))


def _print_state_before_hole(isabelle, session: str, full_text: str, hole_span: Tuple[int, int], trace: bool = False) -> str:
    """Capture proof state before hole."""
    s, _ = hole_span
    lines = full_text[:s].rstrip().splitlines()
    lemma_start = next((i for i, line in enumerate(lines) if line.strip().startswith('lemma ')), -1)
    
    if lemma_start == -1:
        return ""
    
    proof_lines = lines[lemma_start:]
    
    try:
        prolog = _build_ml_prolog()
        thy = build_theory(prolog + proof_lines, add_print_state=True, end_with="sorry")
        resps = _run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_FAST_TIMEOUT_S)
        state_block = _extract_print_state_from_responses(resps)
        
        if _looks_truncated(state_block):
            thy2 = build_theory(prolog + proof_lines + ["ML ‹Pretty.setmargin 100000›"], 
                              add_print_state=True, end_with="sorry")
            resps2 = _run_theory_with_timeout(isabelle, session, thy2, timeout_s=_ISA_FAST_TIMEOUT_S)
            state_block2 = _extract_print_state_from_responses(resps2)
            if state_block2:
                state_block = state_block2
        
        return state_block
    except Exception:
        return ""


def _build_ml_prolog() -> List[str]:
    """Build ML prolog for state capture."""
    return [
        "declare [[show_question_marks = true]]",
        "declare [[show_types = false, show_sorts = false]]",
        "ML ‹",
        "  Pretty.setmargin 100000;",
        "  let",
        "    val st = Toplevel.proof_of @{Isar.state};",
        "    val th = #goal (Proof.goal st);",
        "    val sg = Thm.cprem_of th 1;",
        "    val t = Thm.term_of sg;",
        "    val ctxt = Proof.context_of st;",
        "    val frees = Term.add_frees t [] |> map #1;",
        "    val (params, _) = Logic.strip_params t;",
        "    val pnames = map (fn ((n,_),_) => n) params;",
        "    val ctx0 = Name.make_context (frees @ pnames);",
        "    val (new_names, _) = fold_map Name.variant pnames ctx0;",
        "    val t' = Logic.list_rename_params new_names t;",
        "    val s = Syntax.string_of_term ctxt t';",
        "    val schem_vnames = Term.add_vars t [] |> map #1;",
        "    fun string_of_vname (x, i) = if i = 0 then \"?\" ^ x else \"?\" ^ x ^ Int.toString i;",
        "    val s_params = String.concatWith \" \" new_names;",
        "    val s_frees = String.concatWith \" \" frees;",
        "    val s_schems = String.concatWith \" \" (map string_of_vname schem_vnames);",
        "  in",
        f"    writeln (\"{_LLM_SUBGOAL_MARK} \" ^ s);",
        f"    writeln (\"{_LLM_VARS_MARK} params: \" ^ s_params ^ \" | frees: \" ^ s_frees ^ \" | schematics: \" ^ s_schems)",
        "  end",
        "›",
    ]


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


def _effective_goal_from_state(state_block: str, fallback_goal: str, full_text: str = "", 
                              hole_span: Tuple[int, int] = (0, 0), trace: bool = False) -> str:
    """Build goal from local print_state."""
    if not state_block or not state_block.strip():
        return fallback_goal
    
    clean = re.sub(r"\x1b\[[0-9;]*m", "", state_block).replace("\u00A0", " ")
    m_llm = re.search(rf"^{re.escape(_LLM_SUBGOAL_MARK)}\s+(.*)$", clean, flags=re.M)
    
    lines = clean.splitlines()
    using_facts = _extract_using_facts(lines)
    subgoal = _extract_subgoal(lines, m_llm)
    
    if subgoal:
        facts = [f"({f.strip()})" for f in using_facts] if using_facts else []
        
        if m_llm:
            return " ⟹ ".join(facts + [f"({subgoal})"]) if facts else f"({subgoal})"
        
        subgoal = _recover_binders(subgoal)
        return " ⟹ ".join(facts + [f"({subgoal})"]) if facts else f"({subgoal})"
    
    return fallback_goal


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


def _recover_binders(subgoal: str) -> str:
    """Recover binders for subgoal."""
    ID = re.compile(r"\b([a-z][A-Za-z0-9_']*)\b")
    HEAD_APP = re.compile(r"\b([a-z][A-Za-z0-9_']*)\b\s*(?=\(|[A-Za-z])")
    KEYWORDS = {"in", "if", "then", "else", "let", "case", "of", "where", "and", "or", "not"}
    
    def idents(s):
        seen, out = set(), []
        for m in ID.finditer(s):
            v = m.group(1)
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out
    
    m_bind = re.match(r"\s*⋀\s*([A-Za-z0-9_'\s]+)\.\s*(.*)$", subgoal)
    if m_bind:
        params = m_bind.group(1).strip().split()
        sub_core = m_bind.group(2).strip()
    else:
        sub_core = subgoal.strip()
        head_syms = {m.group(1) for m in HEAD_APP.finditer(sub_core)}
        params = [v for v in idents(sub_core) if v not in head_syms and v not in KEYWORDS]
    
    return f"⋀{' '.join(params)}. {sub_core}" if params else sub_core