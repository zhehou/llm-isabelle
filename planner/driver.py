from __future__ import annotations

import time
import re
import json
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from planner.skeleton import (
    Skeleton, find_sorry_spans, propose_isar_skeleton, propose_isar_skeleton_diverse_best,
)
from planner.repair import try_cegis_repairs, _APPLY_OR_BY as _TACTIC_LINE_RE
from prover.config import ISABELLE_SESSION
from prover.isabelle_api import (
    build_theory, get_isabelle_client, last_print_state_block,
    run_theory, start_isabelle_server,
)
from prover.prover import prove_goal
from prover.utils import parse_subgoals

# Constants
_LLM_SUBGOAL_MARK = "[LLM_SUBGOAL]"
_LLM_VARS_MARK = "[LLM_VARS]"
_INLINE_BY_TAIL = re.compile(r"\s+by\s+.+$")
_BARE_DOT = re.compile(r"(?m)^\s*\.\s*$")

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

@dataclass(slots=True)
class PlanAndFillResult:
    success: bool
    outline: str
    fills: List[str]
    failed_holes: List[int]

def _proof_bounds_top_level(text: str) -> Optional[Tuple[int, int]]:
    """Return (start,end) offsets of the last top-level proof..qed block."""
    qed_matches = list(re.finditer(r"(?m)^\s*qed\b", text))
    if not qed_matches:
        return None
    
    end = qed_matches[-1].end()
    proof_matches = list(re.finditer(r"(?m)^\s*proof\b.*$", text[:qed_matches[-1].start()]))
    if not proof_matches:
        return None
    
    return (proof_matches[-1].start(), end)

def _tactic_spans_topdown(text: str) -> List[Tuple[int, int]]:
    """Top-down tactic line spans within the last proof..qed block."""
    bounds = _proof_bounds_top_level(text)
    if not bounds:
        return []
    
    b0, b1 = bounds
    seg = text[b0:b1]
    lines = seg.splitlines(True)
    spans = []
    off = b0
    
    for line in lines:
        if _TACTIC_LINE_RE.match(line or "") or _INLINE_BY_TAIL.search(line or ""):
            spans.append((off, off + len(line.rstrip("\n"))))
        off += len(line)
    
    return spans

def _repair_failed_proof_topdown(isa, session, full: str, goal_text: str, model: str,
                                 left_s, max_repairs_per_hole: int, trace: bool) -> Tuple[str, bool]:
    """Walk tactics from the top; CEGIS-repair the first failing one."""
    t_spans = _tactic_spans_topdown(full)
    if not t_spans:
        if trace:
            print("[planner]   (failed-tactic) no tactic lines found", flush=True)
        return full, False
    
    for i, span in enumerate(t_spans):
        if left_s() <= 3.0:
            break
        
        if trace:
            print(f"[planner]   (failed-tactic) probing tactic #{i+1}/{len(t_spans)}", flush=True)
        
        st = _print_state_before_hole(isa, session, full, span, trace)
        eff_goal = _effective_goal_from_state(st, goal_text, full, span, trace)
        per_budget = min(30.0, max(15.0, left_s() * 0.33))
        
        patched, applied, _reason = try_cegis_repairs(
            full_text=full, hole_span=span, goal_text=eff_goal, model=model,
            isabelle=isa, session=session, repair_budget_s=per_budget,
            max_ops_to_try=max_repairs_per_hole, beam_k=2, 
            allow_whole_fallback=False, trace=trace,
        )
        
        if applied and patched != full:
            full = patched
            if trace:
                print("[planner]   (failed-tactic) patch applied, re-verifying…", flush=True)
            if _verify_full_proof(isa, session, full):
                return full, True
            # Rebuild spans after patch
            t_spans = _tactic_spans_topdown(full)
            if i >= len(t_spans):
                i = max(0, len(t_spans) - 1)
    
    return full, False

def _extract_goal_from_lemma_line(lemma_line: str) -> str:
    """Extract goal text from lemma line."""
    q1, q2 = lemma_line.find('"'), lemma_line.rfind('"')
    if q1 == -1 or q2 == -1 or q2 <= q1:
        raise ValueError(f"Cannot parse lemma line: {lemma_line!r}")
    return lemma_line[q1 + 1:q2]

def _first_lemma_line(full_text: str) -> str:
    """Find the first lemma line in the text."""
    for line in full_text.splitlines():
        if line.strip().startswith("lemma "):
            return line
    return ""

def _extract_print_state_from_responses(resps: List) -> str:
    """Extract print_state output from Isabelle responses."""
    standard_result = last_print_state_block(resps) or ""
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
                if (text.startswith(_LLM_SUBGOAL_MARK) or 
                    text.startswith(_LLM_VARS_MARK) or 
                    ("goal" in text and "subgoal" in text and not standard_result)):
                    llm_lines.append(text)
                    if "goal" in text and "subgoal" in text:
                        standard_result = text
    
    if llm_lines and standard_result:
        return standard_result + "\n" + "\n".join(llm_lines)
    return standard_result or "\n".join(llm_lines)

def _original_goal_from_state(state_block: str) -> Optional[str]:
    """Extract the original subgoal exactly as printed under `goal (…):` (or via [LLM_SUBGOAL])."""
    if not state_block or not state_block.strip():
        return None
    clean = re.sub(r"\x1b\[[0-9;]*m", "", state_block).replace("\u00A0", " ")
    m_llm = re.search(rf"^{re.escape(_LLM_SUBGOAL_MARK)}\s+(.*)$", clean, flags=re.M)
    lines = clean.splitlines()
    return _extract_subgoal(lines, m_llm)

def _print_state_before_hole(isabelle, session: str, full_text: str, hole_span: Tuple[int, int], trace: bool = False) -> str:
    """Capture the proof state right before the hole."""
    s, e = hole_span
    prefix_text = full_text[:s].rstrip()
    if not prefix_text:
        return ""
    
    lines = prefix_text.splitlines()
    lemma_start = next((i for i, line in enumerate(lines) if line.strip().startswith('lemma ')), -1)
    if lemma_start == -1:
        return ""
    
    proof_lines = lines[lemma_start:]
    
    try:
        # ML prolog for better state capture
        prolog = [
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
            "    fun string_of_vname (x, i) =",
            "      if i = 0 then \"?\" ^ x else \"?\" ^ x ^ Int.toString i;",
            "    val s_params = String.concatWith \" \" new_names;",
            "    val s_frees = String.concatWith \" \" frees;",
            "    val s_schems = String.concatWith \" \" (map string_of_vname schem_vnames);",
            "  in",
            f"    writeln (\"{_LLM_SUBGOAL_MARK} \" ^ s);",
            f"    writeln (\"{_LLM_VARS_MARK} params: \" ^ s_params ^ \" | frees: \" ^ s_frees ^ \" | schematics: \" ^ s_schems)",
            "  end",
            "›",
        ]
        
        proof_lines1 = prolog + proof_lines
        thy = build_theory(proof_lines1, add_print_state=True, end_with="sorry")
        resps = run_theory(isabelle, session, thy)        
        state_block = _extract_print_state_from_responses(resps)
        
        # Handle truncation
        if _looks_truncated(state_block):
            proof_lines_wide = prolog + proof_lines + ["ML ‹Pretty.setmargin 100000›"]
            thy2 = build_theory(proof_lines_wide, add_print_state=True, end_with="sorry")
            resps2 = run_theory(isabelle, session, thy2)
            state_block2 = _extract_print_state_from_responses(resps2)
            if state_block2:
                state_block = state_block2
        
        return state_block
            
    except Exception:
        return ""

def _looks_truncated(txt: str) -> bool:
    """Check if text appears truncated."""
    if not txt or "…" in txt or " ..." in txt:
        return "…" in txt or " ..." in txt
    return any(re.search(r"\\\\<[^>]*$", line.strip()) for line in txt.splitlines())

def _effective_goal_from_state(state_block: str, fallback_goal: str, full_text: str = "", 
                              hole_span: Tuple[int, int] = (0, 0), trace: bool = False) -> str:
    """Build the goal to send to the prover from the local print_state."""
    if not state_block or not state_block.strip():
        return fallback_goal
    
    clean = re.sub(r"\x1b\[[0-9;]*m", "", state_block).replace("\u00A0", " ")
    
    # Extract ML-rendered subgoal
    m_llm = re.search(rf"^{re.escape(_LLM_SUBGOAL_MARK)}\s+(.*)$", clean, flags=re.M)
    m_vars = re.search(
        rf"^{re.escape(_LLM_VARS_MARK)}\s*params:\s*(.*?)\s*\|\s*frees:\s*(.*?)\s*\|\s*schematics:\s*(.*)$",
        clean, flags=re.M
    )
    
    lines = clean.splitlines()
    using_facts = _extract_using_facts(lines)
    subgoal = _extract_subgoal(lines, m_llm)
    
    if subgoal:
        facts = [f"({f.strip()})" for f in using_facts] if using_facts else []
        
        if m_llm:
            subgoal_fixed = f"({subgoal})"
            return " ⟹ ".join(facts + [subgoal_fixed]) if facts else subgoal_fixed
        
        # Binder recovery
        subgoal = _recover_binders(subgoal)
        chain = " ⟹ ".join(facts + [f"({subgoal})"]) if facts else f"({subgoal})"
        return chain
    
    return fallback_goal

def _extract_using_facts(lines: List[str]) -> List[str]:
    """Extract 'using this:' facts from state lines."""
    using_facts = []
    i = 0
    
    while i < len(lines):
        if lines[i].strip() != "using this:":
            i += 1
            continue
        
        i += 1
        raw_block = []
        
        while i < len(lines):
            raw = lines[i]
            s = raw.strip()
            if not s or s.startswith("goal") or s.startswith("using this:"):
                break
            raw_block.append(raw)
            i += 1
        
        using_facts = _process_using_block(raw_block)
        break
    
    return using_facts

def _process_using_block(raw_block: List[str]) -> List[str]:
    """Process raw using block into individual facts."""
    if not raw_block:
        return []
    
    def _lead_spaces(s: str) -> int:
        return len(s) - len(s.lstrip(" "))
    
    def _delta_parens(s: str) -> int:
        opens = s.count("(") + s.count("[") + s.count("{")
        closes = s.count(")") + s.count("]") + s.count("}")
        return opens - closes
    
    _INFIX_TAIL = re.compile(r"(=|⟹|⇒|->|→|<->|↔|⟷|∧|∨|@|::|≤|≥|≠|,)\s*$")
    
    items = []
    cur = []
    head_indent = 0
    paren_balance = 0
    head_ended_with_infix = False
    
    def _flush():
        nonlocal cur, items
        if cur:
            txt = " ".join(x.strip() for x in cur)
            txt = re.sub(r"\s+", " ", txt).strip()
            txt = txt.replace("…", "").rstrip()
            txt = re.sub(r"\\\\<[^>]*$", "", txt).rstrip()
            if txt:
                items.append(txt)
        cur = []
    
    for raw in raw_block:
        if re.match(r"\s*(?:[•∙·\-\*]|\(\d+\))\s+", raw):
            _flush()
            cur = [raw.split(maxsplit=1)[1].strip() if len(raw.split(maxsplit=1)) > 1 else raw.strip()]
            head_indent = _lead_spaces(raw)
            paren_balance = _delta_parens(raw)
            head_ended_with_infix = bool(_INFIX_TAIL.search(raw))
            continue
        
        if not cur:
            cur = [raw.strip()]
            head_indent = _lead_spaces(raw)
            paren_balance = _delta_parens(raw)
            head_ended_with_infix = bool(_INFIX_TAIL.search(raw))
            continue
        
        ind = _lead_spaces(raw)
        this_delta = _delta_parens(raw)
        
        if ind > head_indent or paren_balance > 0 or head_ended_with_infix:
            cur.append(raw.strip())
            paren_balance += this_delta
            head_ended_with_infix = bool(_INFIX_TAIL.search(raw))
        else:
            _flush()
            cur = [raw.strip()]
            head_indent = ind
            paren_balance = this_delta
            head_ended_with_infix = bool(_INFIX_TAIL.search(raw))
    
    _flush()
    return items

def _extract_subgoal(lines: List[str], m_llm) -> Optional[str]:
    """Extract the first numbered subgoal."""
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
            subgoal = subgoal.replace("…", "").rstrip()
            subgoal = re.sub(r"\\\\<[^>]*$", "", subgoal).rstrip()
            return subgoal
    
    return None

def _recover_binders(subgoal: str) -> str:
    """Recover binders for the subgoal."""
    ID = re.compile(r"\b([a-z][A-Za-z0-9_']*)\b")
    HEAD_APP = re.compile(r"\b([a-z][A-Za-z0-9_']*)\b\s*(?=\(|[A-Za-z])")
    KEYWORDS = {"in", "if", "then", "else", "let", "case", "of", "where", "and", "or", "not"}
    
    def _idents(s: str) -> List[str]:
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
        params = [v for v in _idents(sub_core) if v not in head_syms and v not in KEYWORDS]
    
    return f"⋀{' '.join(params)}. {sub_core}" if params else sub_core

def _fill_one_hole(isabelle, session: str, full_text: str, hole_span: Tuple[int, int], 
                  goal_text: str, model: Optional[str], per_hole_timeout: int, *, trace: bool = False) -> Tuple[str, bool, str]:
    """Fill a single hole in the proof."""
    
    def _quick_subgoals(text: str) -> int:
        try:
            thy = build_theory(text.splitlines(), add_print_state=True, end_with="sorry")
            resps = run_theory(isabelle, session, thy)
            block = _extract_print_state_from_responses(resps) or ""
            return int(parse_subgoals(block)) if block.strip() else 9999
        except Exception:
            return 9999
    
    def _insert_above_hole_keep_sorry(text: str, hole: Tuple[int, int], lines_to_insert: List[str]) -> str:
        s, _ = hole
        ls = text.rfind("\n", 0, s) + 1
        le = text.find("\n", s)
        hole_line = text[ls:(le if le != -1 else len(text))]
        indent = hole_line[:len(hole_line) - len(hole_line.lstrip(" "))]
        payload = "".join(f"{indent}{ln.strip()}\n" for ln in lines_to_insert if ln.strip())
        return text[:s] + payload + text[s:]
    
    # Check if it's a stale hole
    try:
        s_line_start = full_text.rfind("\n", 0, hole_span[0]) + 1
        prev_line_end = s_line_start - 1
        prev_prev_nl = full_text.rfind("\n", 0, prev_line_end) + 1
        prev_line = full_text[prev_prev_nl:prev_line_end+1]
    except Exception:
        prev_line = ""
    
    prev_is_finisher = (bool(_INLINE_BY_TAIL.search(prev_line)) or
                       bool(_TACTIC_LINE_RE.match(prev_line)) or
                       prev_line.strip() in {"done", "."})
    
    if prev_is_finisher:
        s, e = hole_span
        return full_text[:s] + "\n" + full_text[e:], True, "(stale-hole)"
    
    state_block = _print_state_before_hole(isabelle, session, full_text, hole_span, trace)
    # Show the exact proof state block we will parse
    _log_state_block("fill", state_block, trace=trace)
    # Parse and print the original goal straight from the state block
    orig_goal = _original_goal_from_state(state_block)
    eff_goal  = _effective_goal_from_state(state_block, goal_text, full_text, hole_span, trace)
    if trace:
        if orig_goal:
            print(f"[fill] Original goal: {orig_goal}")
        print(f"\n[fill] Effective goal: {eff_goal}")
    
    res = prove_goal(
        isabelle, session, eff_goal, model_name_or_ensemble=model,
        beam_w=3, max_depth=6, hint_lemmas=6, timeout=per_hole_timeout,
        models=None, save_dir=None, use_sledge=True, sledge_timeout=10,
        sledge_every=1, trace=trace, use_color=False, use_qc=False,
        qc_timeout=2, qc_every=1, use_np=False, np_timeout=5, np_every=2,
        facts_limit=8, do_minimize=False, minimize_timeout=8,
        do_variants=False, variant_timeout=6, variant_tries=24,
        enable_reranker=True, initial_state_hint=state_block,
    )
    
    steps = [str(s) for s in res.get("steps", [])]
    if not steps:
        return full_text, False, "no-steps"
    
    applies = [s for s in steps if s.startswith("apply")]
    fin = next((s for s in steps if s.startswith("by ") or s.strip() == "done"), "")
    
    # Handle finisher case
    if fin:
        script_lines = applies + [fin]
        insert = "\n  " + "\n  ".join(script_lines) + "\n"
        s, e = hole_span
        new_text = full_text[:s] + insert + full_text[e:]
        
        before, after = _quick_subgoals(full_text), _quick_subgoals(new_text)
        if trace:
            print(f"[fill] subgoals: {before} → {after} (finisher present)")
        
        if after == 9999:
            return (new_text, True, "\n".join(script_lines)) if _verify_full_proof(isabelle, session, new_text) else (full_text, False, "finisher-ambiguous-noverify")
        
        return (new_text, True, "\n".join(script_lines)) if after <= before else (full_text, False, "finisher-regressed")
    
    # Handle apply-only case
    if applies:
        s, _ = hole_span
        win_s = max(0, full_text.rfind("\n", 0, max(0, s-256)) + 1)
        window = full_text[win_s:s]
        dedup = [a for a in applies if a not in window]
        
        if not dedup:
            return full_text, False, "apply-duplicate"
        
        probe_text = _insert_above_hole_keep_sorry(full_text, hole_span, dedup)
        before, after = _quick_subgoals(full_text), _quick_subgoals(probe_text)
        
        if trace:
            print(f"[fill] subgoals: {before} → {after} (apply-only)")
        
        return (probe_text, False, "\n".join(dedup)) if after != 9999 and after < before else (full_text, False, "apply-no-effect")
    
    return full_text, False, "no-tactics"

def _verify_full_proof(isabelle, session: str, text: str) -> bool:
    """Verify that a complete proof is valid."""
    try:
        thy = build_theory(text.splitlines(), add_print_state=False, end_with=None)
        run_theory(isabelle, session, thy)
        return True
    except Exception:
        return False

def _open_minimal_sorries(isabelle, session: str, text: str) -> Tuple[str, bool]:
    """Localize a failing finisher with minimal opening."""
    def _runs(ts: List[str]) -> bool:
        try:
            thy = build_theory(ts, add_print_state=True, end_with="sorry")
            run_theory(isabelle, session, thy)
            return True
        except Exception:
            return False
    
    lines = text.splitlines()
    
    # Try whole-line tactic candidates first (top-down)
    for i, line in enumerate(lines):
        if not (_TACTIC_LINE_RE.match(line) or line.strip() == "done" or _BARE_DOT.match(line)):
            continue
        
        indent = line[:len(line) - len(line.lstrip(" "))]
        trial = lines[:i] + [f"{indent}sorry"] + lines[i + 1:]
        
        if _runs(trial):
            lines[i] = f"{indent}sorry"
            return "\n".join(lines) + ("" if text.endswith("\n") else "\n"), True
    
    # Try inline '... by TACTIC' patterns (top-down)
    for i, line in enumerate(lines):
        if not line:
            continue
        
        m = _INLINE_BY_TAIL.search(line)
        if not m:
            continue
        
        indent = line[:len(line) - len(line.lstrip(" "))]
        header = line[:m.start()].rstrip()
        trial = lines[:i] + [header, f"{indent}sorry"] + lines[i + 1:]
        
        if _runs(trial):
            lines[i] = header
            lines.insert(i + 1, f"{indent}sorry")
            return "\n".join(lines) + ("" if text.endswith("\n") else "\n"), True
    
    return (text if text.endswith("\n") else text + "\n"), False

def _cleanup_resources(isa, proc):
    """Clean up Isabelle resources."""
    try:
        isa.shutdown()
        try:
            from planner.experiments import _close_client_loop_safely
            _close_client_loop_safely(isa)
        except Exception:
            pass
    except Exception:
        pass
    
    try:
        if hasattr(proc, "terminate"):
            proc.terminate()
        if hasattr(proc, "kill"):
            proc.kill()
        if all(hasattr(proc, attr) for attr in ("poll", "communicate", "pid")):
            proc.wait(timeout=2)
    except Exception:
        pass

def plan_outline(goal: str, *, model: Optional[str] = None, outline_k: Optional[int] = None,
                outline_temps: Optional[Iterable[float]] = None, legacy_single_outline: bool = False,
                priors_path: Optional[str] = None, context_hints: bool = False,
                lib_templates: bool = False, alpha: float = 1.0, beta: float = 0.5,
                gamma: float = 0.2, hintlex_path: Optional[str] = None, hintlex_top: int = 8) -> str:
    """Generate an Isar outline with 'sorry' placeholders."""
    server_info, proc = start_isabelle_server(name="planner", log_file="logs/planner_ui.log")
    isa = get_isabelle_client(server_info)
    session = isa.session_start(session=ISABELLE_SESSION)
    
    try:
        if legacy_single_outline:
            return propose_isar_skeleton(goal, model=model, temp=0.35, force_outline=True).text
        
        temps = tuple(outline_temps) if outline_temps else (0.35, 0.55, 0.85)
        k = int(outline_k) if outline_k is not None else 3
        
        best, _diag = propose_isar_skeleton_diverse_best(
            goal, isabelle=isa, session_id=session, model=model, temps=temps, k=k,
            force_outline=True, priors_path=priors_path, context_hints=context_hints,
            lib_templates=lib_templates, alpha=alpha, beta=beta, gamma=gamma,
            hintlex_path=hintlex_path, hintlex_top=hintlex_top,
        )
        return best.text
    finally:
        _cleanup_resources(isa, proc)

def plan_and_fill(goal: str, model: Optional[str] = None, timeout: int = 100, *, mode: str = "auto",
                 outline_k: Optional[int] = None, outline_temps: Optional[Iterable[float]] = None,
                 legacy_single_outline: bool = False, repairs: bool = True,
                 max_repairs_per_hole: int = 2, trace: bool = False,
                 priors_path: Optional[str] = None, context_hints: bool = False,
                 lib_templates: bool = False, alpha: float = 1.0, beta: float = 0.5,
                 gamma: float = 0.2, hintlex_path: Optional[str] = None,
                 hintlex_top: int = 8) -> PlanAndFillResult:
    """Plan and fill holes in Isar proofs."""
    force_outline = (mode == "outline")
    
    server_info, proc = start_isabelle_server(name="planner", log_file="logs/planner_ui.log")
    if trace:
        print("[planner] starting Isabelle server…", flush=True)
    
    isa = get_isabelle_client(server_info)    
    session = isa.session_start(session=ISABELLE_SESSION)
    if trace:
        print(f"[planner] session started: {session}", flush=True)
    
    t0 = time.monotonic()
    left_s = lambda: max(0.0, timeout - (time.monotonic() - t0))
    
    try:
        # 1) Generate outline
        if legacy_single_outline:
            if trace:
                print("[planner] outline: single low-temp", flush=True)
            full = propose_isar_skeleton(goal, model=model, temp=0.35, force_outline=force_outline).text
        else:
            temps = tuple(outline_temps) if outline_temps else (0.35, 0.55, 0.85)
            k = int(outline_k) if outline_k is not None else 3
            if trace:
                print(f"[planner] outline: diverse k={k} temps={temps}", flush=True)
            
            best, _diag = propose_isar_skeleton_diverse_best(
                goal, isabelle=isa, session_id=session, model=model, temps=temps, k=k,
                force_outline=force_outline, priors_path=priors_path,
                context_hints=context_hints, lib_templates=lib_templates,
                alpha=alpha, beta=beta, gamma=gamma, hintlex_path=hintlex_path,
                hintlex_top=hintlex_top,
            )
            full = best.text
        
        spans = find_sorry_spans(full)
        if trace:
            print(f"[planner] outline ready: {len(full)} chars; holes={len(spans)}", flush=True)
        
        if mode == "outline":
            return PlanAndFillResult(True, full, [], [])
        
        # 2) Handle complete proofs
        if not spans:
            if trace:
                print("[planner] no holes detected — verifying complete proof", flush=True)
            if _verify_full_proof(isa, session, full):
                return PlanAndFillResult(True, full, [], [])
            
            if trace:
                print("[planner] complete proof failed — attempting repair", flush=True)
            if repairs and left_s() > 6.0:
                full, ok = _repair_failed_proof_topdown(
                    isa, session, full, goal, model, left_s, max_repairs_per_hole, trace
                )
                if ok:
                    return PlanAndFillResult(True, full, [], [])
            
            if trace:
                print("[planner] fallback — opening minimal holes", flush=True)
            full2, opened = _open_minimal_sorries(isa, session, full)
            if opened:
                full = full2
            else:
                return PlanAndFillResult(False, full, [], [0])
        
        # 3) Fill holes
        lemma_line = _first_lemma_line(full)
        if not lemma_line:
            return PlanAndFillResult(False, full, [], [0])
        
        goal_text = _extract_goal_from_lemma_line(lemma_line)
        fills, failed = [], []
        hole_idx = 0
        
        while "sorry" in full and left_s() > 0:
            spans = find_sorry_spans(full)
            if not spans:
                break
            
            span = spans[0]
            remaining = max(1, len(spans))
            per_hole_budget = int(max(5, left_s() / remaining))
            
            if trace:
                s, e = span
                print(f"[planner] fill hole #{hole_idx} span=({s},{e}) budget={per_hole_budget}s", flush=True)
            
            full2, ok, script = _fill_one_hole(
                isa, session, full, span, goal_text, model, 
                per_hole_timeout=per_hole_budget, trace=trace
            )
            
            if ok:
                full = full2
                fills.append(script)
                if trace:
                    n_steps = script.count("\n") + 1
                    print(f"[planner]   ✓ filled with {n_steps} step(s)", flush=True)
                hole_idx += 1
                continue
            
            # Try CEGIS repairs
            if repairs and left_s() > 6:
                _state = _print_state_before_hole(isa, session, full, span, trace)
                _eff_goal = _effective_goal_from_state(_state, goal_text, full, span, trace)
                
                if trace:
                    print("[planner]   → trying local repairs (CEGIS)…", flush=True)
                
                patched, applied, _reason = try_cegis_repairs(
                    full_text=full, hole_span=span, goal_text=_eff_goal, model=model,
                    isabelle=isa, session=session, 
                    repair_budget_s=min(30.0, max(15.0, left_s() * 0.33)),
                    max_ops_to_try=max_repairs_per_hole, beam_k=2, 
                    allow_whole_fallback=True, trace=trace,
                )
                
                if applied and patched != full:
                    full = patched
                    if trace:
                        print("[planner]   ✓ repair patched snippet — retrying hole", flush=True)
                    continue
            
            failed.append(hole_idx)
            if trace:
                print("[planner]   ✗ hole still failing", flush=True)
            hole_idx += 1
        
        # Final verification and cleanup
        success = ("sorry" not in full)
        if success:
            if _verify_full_proof(isa, session, full):
                if trace:
                    print("[planner] finished: success=True", flush=True)
                return PlanAndFillResult(True, full, fills, failed)
            
            if repairs and left_s() > 6.0:
                if trace:
                    print("[planner] final verification failed — attempting repair", flush=True)
                full, ok = _repair_failed_proof_topdown(
                    isa, session, full, goal_text, model, left_s, max_repairs_per_hole, trace
                )
                if ok:
                    if trace:
                        print("[planner] finished: success=True (after repair)", flush=True)
                    return PlanAndFillResult(True, full, fills, failed)
            
            if trace:
                print("[planner] finished: success=False", flush=True)
            return PlanAndFillResult(False, full, fills, failed)
        
        if trace:
            print("[planner] finished: success=False", flush=True)
        return PlanAndFillResult(False, full, fills, failed)
    
    finally:
        _cleanup_resources(isa, proc)