from __future__ import annotations

import time
import re
import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout
import hashlib

from planner.skeleton import (
    Skeleton, find_sorry_spans, propose_isar_skeleton, propose_isar_skeleton_diverse_best,
)
from planner.repair import try_cegis_repairs, regenerate_whole_proof, _APPLY_OR_BY as _TACTIC_LINE_RE
from prover.config import ISABELLE_SESSION
from prover.isabelle_api import (
    build_theory, get_isabelle_client, last_print_state_block,
    run_theory, start_isabelle_server,
)
from prover.prover import prove_goal

def _hole_fingerprint(full_text: str, span: tuple[int, int], context: int = 80) -> str:
    """Stable key for a hole: hash a small window around the 'sorry'."""
    s, e = span
    lo = max(0, s - context)
    hi = min(len(full_text), e + context)
    snippet = full_text[lo:hi]
    return hashlib.sha1(snippet.encode("utf-8")).hexdigest()[:16]

# Constants
_LLM_SUBGOAL_MARK = "[LLM_SUBGOAL]"
_LLM_VARS_MARK = "[LLM_VARS]"
_INLINE_BY_TAIL = re.compile(r"\s+by\s+.+$")
_BARE_DOT = re.compile(r"(?m)^\s*\.\s*$")
_ISA_FAST_TIMEOUT_S = int(os.getenv("ISABELLE_FAST_TIMEOUT_S", "12"))
_ISA_VERIFY_TIMEOUT_S = int(os.getenv("ISABELLE_VERIFY_TIMEOUT_S", "30"))

@dataclass(slots=True)
class PlanAndFillResult:
    success: bool
    outline: str
    fills: List[str]
    failed_holes: List[int]


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


# ============================================================================
# Hole Filling
# ============================================================================

def _fill_one_hole(isabelle, session: str, full_text: str, hole_span: Tuple[int, int], 
                  goal_text: str, model: Optional[str], per_hole_timeout: int, *, trace: bool = False) -> Tuple[str, bool, str]:
    """Fill single hole in proof."""
    
    # Check for stale hole
    try:
        s_line_start = full_text.rfind("\n", 0, hole_span[0]) + 1
        prev_line_end = s_line_start - 1
        prev_prev_nl = full_text.rfind("\n", 0, prev_line_end) + 1
        prev_line = full_text[prev_prev_nl:prev_line_end+1]
    except Exception:
        prev_line = ""
    
    if (_INLINE_BY_TAIL.search(prev_line) or _TACTIC_LINE_RE.match(prev_line) or 
        prev_line.strip() in {"done", "."}):
        s, e = hole_span
        return full_text[:s] + "\n" + full_text[e:], True, "(stale-hole)"
    
    state_block = _print_state_before_hole(isabelle, session, full_text, hole_span, trace)
    _log_state_block("fill", state_block, trace=trace)
    
    orig_goal = _original_goal_from_state(state_block)
    eff_goal = _effective_goal_from_state(state_block, goal_text, full_text, hole_span, trace)
    
    if trace:
        if orig_goal:
            print(f"[fill] Original goal: {orig_goal}")
        print(f"[fill] Effective goal: {eff_goal}")
    
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
    
    # Handle finisher
    if fin:
        script_lines = applies + [fin]
        insert = "\n  " + "\n  ".join(script_lines) + "\n"
        s, e = hole_span
        new_text = full_text[:s] + insert + full_text[e:]
        
        if _verify_full_proof(isabelle, session, new_text):
            return new_text, True, "\n".join(script_lines)
        return full_text, False, "finisher-unverified"
    
    # Handle apply-only
    if applies:
        s, _ = hole_span
        win_s = max(0, full_text.rfind("\n", 0, max(0, s-256)) + 1)
        window = full_text[win_s:s]
        dedup = [a for a in applies if a not in window]
        
        if not dedup:
            return full_text, False, "apply-duplicate"
        
        probe_text = _insert_above_hole_keep_sorry(full_text, hole_span, dedup)
        return probe_text, False, "\n".join(dedup)
    
    return full_text, False, "no-tactics"


def _insert_above_hole_keep_sorry(text: str, hole: Tuple[int, int], lines_to_insert: List[str]) -> str:
    """Insert lines above hole while keeping sorry."""
    s, _ = hole
    ls = text.rfind("\n", 0, s) + 1
    le = text.find("\n", s)
    hole_line = text[ls:(le if le != -1 else len(text))]
    indent = hole_line[:len(hole_line) - len(hole_line.lstrip(" "))]
    payload = "".join(f"{indent}{ln.strip()}\n" for ln in lines_to_insert if ln.strip())
    return text[:s] + payload + text[s:]


# ============================================================================
# Repair
# ============================================================================

def _proof_bounds_top_level(text: str) -> Optional[Tuple[int, int]]:
    """Return (start,end) offsets of last top-level proof..qed block."""
    qed_matches = list(re.finditer(r"(?m)^\s*qed\b", text))
    if not qed_matches:
        return None
    
    end = qed_matches[-1].end()
    proof_matches = list(re.finditer(r"(?m)^\s*proof\b.*$", text[:qed_matches[-1].start()]))
    if not proof_matches:
        return None
    
    return (proof_matches[-1].start(), end)


def _tactic_spans_topdown(text: str) -> List[Tuple[int, int]]:
    """Top-down tactic line spans within last proof..qed block."""
    bounds = _proof_bounds_top_level(text)
    if not bounds:
        return []
    
    b0, b1 = bounds
    seg = text[b0:b1]
    lines = seg.splitlines(True)
    spans, off = [], b0
    
    for line in lines:
        if _TACTIC_LINE_RE.match(line or "") or _INLINE_BY_TAIL.search(line or ""):
            spans.append((off, off + len(line.rstrip("\n"))))
        off += len(line)
    
    return spans


def _repair_failed_proof_topdown(isa, session, full: str, goal_text: str, model: str,
                                 left_s, max_repairs_per_hole: int, trace: bool) -> Tuple[str, bool]:
    """Walk tactics from top; CEGIS-repair first failing one."""
    t_spans = _tactic_spans_topdown(full)
    if not t_spans:
        return full, False
    
    i = 0
    while i < len(t_spans) and left_s() > 3.0:
        span = t_spans[i]
        st = _print_state_before_hole(isa, session, full, span, trace)
        eff_goal = _effective_goal_from_state(st, goal_text, full, span, trace)
        per_budget = min(30.0, max(15.0, left_s() * 0.33))
        
        patched, applied, _ = try_cegis_repairs(
            full_text=full, hole_span=span, goal_text=eff_goal, model=model,
            isabelle=isa, session=session, repair_budget_s=per_budget,
            max_ops_to_try=max_repairs_per_hole, beam_k=2,
            allow_whole_fallback=False, trace=trace, resume_stage=0,
        )
        
        if applied and patched != full:
            full = patched
            if _verify_full_proof(isa, session, full):
                return full, True
            t_spans = _tactic_spans_topdown(full)
            continue
        i += 1
    
    return full, False


def _open_minimal_sorries(isabelle, session: str, text: str) -> Tuple[str, bool]:
    """Localize failing finisher with minimal opening."""
    def runs(ts):
        try:
            thy = build_theory(ts, add_print_state=True, end_with="sorry")
            _run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_VERIFY_TIMEOUT_S)
            return True
        except Exception:
            return False
    
    lines = text.splitlines()
    
    # Try whole-line tactics
    for i, line in enumerate(lines):
        if not (_TACTIC_LINE_RE.match(line) or line.strip() == "done" or _BARE_DOT.match(line)):
            continue
        
        indent = line[:len(line) - len(line.lstrip(" "))]
        if runs(lines[:i] + [f"{indent}sorry"] + lines[i + 1:]):
            lines[i] = f"{indent}sorry"
            return "\n".join(lines) + ("" if text.endswith("\n") else "\n"), True
    
    # Try inline 'by TACTIC' patterns
    for i, line in enumerate(lines):
        m = _INLINE_BY_TAIL.search(line) if line else None
        if not m:
            continue
        
        indent = line[:len(line) - len(line.lstrip(" "))]
        header = line[:m.start()].rstrip()
        if runs(lines[:i] + [header, f"{indent}sorry"] + lines[i + 1:]):
            lines[i] = header
            lines.insert(i + 1, f"{indent}sorry")
            return "\n".join(lines) + ("" if text.endswith("\n") else "\n"), True
    
    return (text if text.endswith("\n") else text + "\n"), False


# ============================================================================
# Public API
# ============================================================================

def plan_outline(goal: str, *, model: Optional[str] = None, outline_k: Optional[int] = None,
                outline_temps: Optional[Iterable[float]] = None, legacy_single_outline: bool = False,
                priors_path: Optional[str] = None, context_hints: bool = False,
                lib_templates: bool = False, alpha: float = 1.0, beta: float = 0.5,
                gamma: float = 0.2, hintlex_path: Optional[str] = None, hintlex_top: int = 8) -> str:
    """Generate Isar outline with 'sorry' placeholders."""
    server_info, proc = start_isabelle_server(name="planner", log_file="logs/planner_ui.log")
    isa = get_isabelle_client(server_info)
    session = isa.session_start(session=ISABELLE_SESSION)
    
    try:
        if legacy_single_outline:
            return propose_isar_skeleton(goal, model=model, temp=0.35, force_outline=True).text
        
        temps = tuple(outline_temps) if outline_temps else (0.35, 0.55, 0.85)
        k = int(outline_k) if outline_k is not None else 3
        
        best, _ = propose_isar_skeleton_diverse_best(
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
    server_info, proc = start_isabelle_server(name="planner", log_file="logs/planner_ui.log")
    isa = get_isabelle_client(server_info)
    session = isa.session_start(session=ISABELLE_SESSION)
    
    t0 = time.monotonic()
    left_s = lambda: max(0.0, timeout - (time.monotonic() - t0))
    
    try:
        # Generate outline
        if legacy_single_outline:
            full = propose_isar_skeleton(goal, model=model, temp=0.35, force_outline=(mode=="outline")).text
        else:
            temps = tuple(outline_temps) if outline_temps else (0.35, 0.55, 0.85)
            k = int(outline_k) if outline_k is not None else 3
            best, _ = propose_isar_skeleton_diverse_best(
                goal, isabelle=isa, session_id=session, model=model, temps=temps, k=k,
                force_outline=(mode=="outline"), priors_path=priors_path,
                context_hints=context_hints, lib_templates=lib_templates,
                alpha=alpha, beta=beta, gamma=gamma, hintlex_path=hintlex_path,
                hintlex_top=hintlex_top,
            )
            full = best.text
        
        spans = find_sorry_spans(full)
        
        if mode == "outline":
            return PlanAndFillResult(True, full, [], [])
        
        # Handle complete proofs
        if not spans:
            if _verify_full_proof(isa, session, full):
                return PlanAndFillResult(True, full, [], [])
            
            if repairs and left_s() > 6.0:
                full, ok = _repair_failed_proof_topdown(isa, session, full, goal, model, left_s, max_repairs_per_hole, trace)
                if ok:
                    return PlanAndFillResult(True, full, [], [])
            
            full2, opened = _open_minimal_sorries(isa, session, full)
            full = full2 if opened else full
            if not opened:
                return PlanAndFillResult(False, full, [], [0])
        
        # Fill holes
        lemma_line = _first_lemma_line(full)
        if not lemma_line:
            return PlanAndFillResult(False, full, [], [0])
        
        goal_text = _extract_goal_from_lemma_line(lemma_line)
        fills, failed = [], []
        # map from stable hole key (byte offset of the hole start) -> resume stage (0..3)
        repair_progress: dict[int, int] = {}
        # track attempts per (hole, stage) so we can cap stage-2 and trigger regeneration
        stage_tries: dict[Tuple[int, int], int] = {}
        STAGE2_CAP = 3  # after 3 failed stage-2 attempts, regenerate whole proof
        # log de-dup for "Skipping fill..." so we don't spam
        _skip_fill_logged_once: set[Tuple[str, int]] = set()       
        
        while "sorry" in full and left_s() > 0:
            spans = find_sorry_spans(full)
            if not spans:
                break
            
            span = spans[0]
            hole_key = _hole_fingerprint(full, span)  # stable identity across minor shifts
            per_hole_budget = int(max(5, left_s() / max(1, len(spans))))

            # If this hole already entered repair escalation, skip fill and go straight to repairs.
            start_stage = repair_progress.get(hole_key, 0)
            if start_stage == 0:
                # Try to fill this hole once.
                full2, ok, script = _fill_one_hole(
                    isa, session, full, span, goal_text, model,
                    per_hole_timeout=per_hole_budget, trace=trace
                )

                # Strict policy: commit only verified changes (ok=True).
                if ok and full2 != full:
                    full = full2
                    fills.append(script)
                    # hole consumed; forget its progress and re-scan from the top
                    repair_progress.pop(hole_key, None)
                    continue
            else:
                # Only print this once per (hole,stage) to avoid log spam
                if trace and (hole_key, start_stage) not in _skip_fill_logged_once:
                    print(
                        f"[fill] Skipping fill for hole @{hole_key}; escalate repairs from stage {start_stage}"
                    )
                    _skip_fill_logged_once.add((hole_key, start_stage))

            # Try CEGIS repairs (possibly starting at an escalated stage)
            if repairs and left_s() > 6:
                state = _print_state_before_hole(isa, session, full, span, trace)
                eff_goal = _effective_goal_from_state(state, goal_text, full, span, trace)
                
                patched, applied, _ = try_cegis_repairs(
                    full_text=full, hole_span=span, goal_text=eff_goal, model=model,
                    isabelle=isa, session=session, 
                    repair_budget_s=min(30.0, max(15.0, left_s() * 0.33)),
                    max_ops_to_try=max_repairs_per_hole, beam_k=2, 
                    allow_whole_fallback=False,  # <-- remove stage-3 from the inner loop
                    trace=trace, resume_stage=start_stage,
                )
                # Commit only verified patches; otherwise escalate stage.
                if applied and patched != full:
                    full = patched
                    repair_progress.clear()   # restart top-down after accepting a repair
                    stage_tries.clear()
                    continue
                # No commit → count this attempt at the current stage
                key = (hole_key, start_stage)
                stage_tries[key] = stage_tries.get(key, 0) + 1
                # If we are at stage < 2, escalate within local/subproof track
                if start_stage < 2:
                    repair_progress[hole_key] = min(start_stage + 1, 2)
                else:
                    # We are at stage==2 (case/subproof track). If cap reached → whole regeneration.
                    if stage_tries.get((hole_key, 2), 0) >= STAGE2_CAP:
                        if trace:
                            print(f"[repair] Stage-2 cap reached for hole @{hole_key}. Regenerating whole proof…")
                        regen_budget = min(40.0, max(8.0, left_s() * 0.8))
                        new_full, ok_re, _ = regenerate_whole_proof(
                            full_text=full, goal_text=goal_text, model=model,
                            isabelle=isa, session=session, budget_s=regen_budget,
                            trace=trace, prior_outline_text=full
                        )
                        if ok_re and new_full != full:
                            full = new_full
                            # Restart everything from top after replacement
                            repair_progress.clear()
                            stage_tries.clear()
                            continue
                        # If regeneration didn’t yield a verified patch, fall back to a BRAND-NEW outline
                        if trace:
                            print("[repair] Whole regeneration failed to verify; proposing a fresh outline…")
                        temps = tuple(outline_temps) if outline_temps else (0.35, 0.55, 0.85)
                        k = int(outline_k) if outline_k is not None else 3
                        best, _ = propose_isar_skeleton_diverse_best(
                            goal_text, isabelle=isa, session_id=session, model=model, temps=temps, k=k,
                            force_outline=True, priors_path=priors_path, context_hints=context_hints,
                            lib_templates=lib_templates, alpha=alpha, beta=beta, gamma=gamma,
                            hintlex_path=hintlex_path, hintlex_top=hintlex_top,
                        )
                        full = best.text
                        repair_progress.clear()
                        stage_tries.clear()
                        continue
                    # else: stay on stage 2 and try again (bounded by the cap)
                    repair_progress[hole_key] = 2
            
            # Keep focusing on the same top hole unless it was actually removed.
            # (No hole_idx increment; re-enter loop to either escalate further or try next strategy.)
        
        # Final verification
        success = ("sorry" not in full)
        if success:
            if _verify_full_proof(isa, session, full):
                return PlanAndFillResult(True, full, fills, failed)
        
        return PlanAndFillResult(False, full, fills, failed)
    
    finally:
        _cleanup_resources(isa, proc)