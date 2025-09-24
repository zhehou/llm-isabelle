from __future__ import annotations

import time
import re
import json
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from planner.skeleton import (
    Skeleton,
    find_sorry_spans,
    propose_isar_skeleton,
    propose_isar_skeleton_diverse_best,
)
from planner.repair import try_cegis_repairs

from prover.config import ISABELLE_SESSION
from prover.isabelle_api import (
    build_theory,
    get_isabelle_client,
    last_print_state_block,
    run_theory,
    start_isabelle_server,
)
from prover.prover import prove_goal
from planner.repair import _APPLY_OR_BY as _TACTIC_LINE_RE

# Marker lines emitted by our ML helpers inside print_state capture
_LLM_SUBGOAL_MARK = "[LLM_SUBGOAL]"
_LLM_VARS_MARK = "[LLM_VARS]"

_INLINE_BY_TAIL = re.compile(r"\s+by\s+.+$")
_BARE_DOT = re.compile(r"(?m)^\s*\.\s*$")


@dataclass(slots=True)
class PlanAndFillResult:
    success: bool
    outline: str
    fills: List[str]
    failed_holes: List[int]


def _extract_goal_from_lemma_line(lemma_line: str) -> str:
    q1 = lemma_line.find('"')
    q2 = lemma_line.rfind('"')
    if q1 == -1 or q2 == -1 or q2 <= q1:
        raise ValueError(f"Cannot parse lemma line: {lemma_line!r}")
    return lemma_line[q1 + 1 : q2]


def _first_lemma_line(full_text: str) -> str:
    for L in full_text.splitlines():
        if L.strip().startswith("lemma "):
            return L
    return ""


def _extract_print_state_from_responses(resps: List) -> str:
    """Extract print_state output from Isabelle responses."""
    # Collect the standard print_state block (for numbered goal etc.)
    standard_result = last_print_state_block(resps) or ""

    # Also collect our ML 'writeln' markers (authoritative subgoal + var classes)
    llm_lines: List[str] = []
    for resp in (resps or []):
        resp_type = getattr(resp, "response_type", None)
        if str(resp_type).upper() == "FINISHED":
            body = getattr(resp, "response_body", None)
            if isinstance(body, bytes):
                body = body.decode(errors="replace")
            data = None
            try:
                if isinstance(body, str) and body.strip().startswith("{"):
                    data = json.loads(body)
                elif isinstance(body, dict):
                    data = body
            except (json.JSONDecodeError, TypeError):
                data = None
            if not data:
                continue
            nodes = data.get("nodes", [])
            for node in nodes:
                for msg in node.get("messages", []) or []:
                    if msg.get("kind") != "writeln":
                        continue
                    text = msg.get("message", "") or ""
                    # Keep our markers; keep extra goal text if present
                    if text.startswith(_LLM_SUBGOAL_MARK) or text.startswith(_LLM_VARS_MARK):
                        llm_lines.append(text)
                    elif ("goal" in text and "subgoal" in text and not standard_result):
                        standard_result = text

    if llm_lines and standard_result:
        return standard_result + "\n" + "\n".join(llm_lines)
    return standard_result or ("\n".join(llm_lines) if llm_lines else "")


# -----------------------------------------------------------------------------
# Local proof-state capture with robust pretty-print settings
# -----------------------------------------------------------------------------
def _print_state_before_hole(isabelle, session: str, full_text: str, hole_span: Tuple[int, int], trace: bool = False) -> str:
    """Capture the proof state right before the hole."""
    s, e = hole_span
    prefix_text = full_text[:s].rstrip()
    if not prefix_text:
        return ""
    
    # Extract lemma and proof content from full text
    lines = prefix_text.splitlines()
    lemma_start = -1
    
    for i, line in enumerate(lines):
        if line.strip().startswith('lemma '):
            lemma_start = i
            break
    
    if lemma_start == -1:
        return ""
    
    proof_lines = lines[lemma_start:]
    
    # if trace:
    #     print(f"[DEBUG] Proof lines being sent to build_theory:")
    #     for i, line in enumerate(proof_lines[:10]):
    #         print(f"[DEBUG]   {i}: {repr(line)}")
    #     if len(proof_lines) > 10:
    #         print(f"[DEBUG]   ... and {len(proof_lines) - 10} more lines")
    
    try:
        # Help Isabelle show distinctions we need:
        #  - show_question_marks: prints schematics as ?x
        prolog = [
            "declare [[show_question_marks = true]]",
            "declare [[show_types = false, show_sorts = false]]",
            # Print an ML-computed, alpha-renamed subgoal with fresh parameter names
            # that cannot clash with frees. Also set an extremely wide margin to
            # avoid elision of binders / parentheses.
            "ML ‹",
            "  Pretty.setmargin 100000;",
            "  let",
            "    val st   = Toplevel.proof_of @{Isar.state};",
            "    val th   = #goal (Proof.goal st);",
            "    val sg   = Thm.cprem_of th 1;",
            "    val t    = Thm.term_of sg;",
            "    val ctxt = Proof.context_of st;",
            "    val frees = Term.add_frees t [] |> map #1;",  
            "    val (params, _) = Logic.strip_params t;",
            "    val pnames = map (fn ((n,_),_) => n) params;",
            "    val ctx0 = Name.make_context (frees @ pnames);",
            "    val (new_names, _) = fold_map Name.variant pnames ctx0;",
            "    val t' = Logic.list_rename_params new_names t;",
            "    val s  = Syntax.string_of_term ctxt t';",
            "    (* classify schematic variables as well *)",
            "    val schem_vnames = Term.add_vars t [] |> map #1;",  
            "    fun string_of_vname (x, i) =",
            "      if i = 0 then \"?\" ^ x else \"?\" ^ x ^ Int.toString i;",
            "    val s_params = String.concatWith \" \" new_names;",
            "    val s_frees  = String.concatWith \" \" frees;",
            "    val s_schems = String.concatWith \" \" (map string_of_vname schem_vnames);",
            "  in",
            "    writeln (\"" + _LLM_SUBGOAL_MARK + " \" ^ s);",
            "    writeln (\"" + _LLM_VARS_MARK + " params: \" ^ s_params ^ \" | frees: \" ^ s_frees ^ \" | schematics: \" ^ s_schems)",
            "  end",
            "›",
        ]
        proof_lines1 = prolog + proof_lines
        thy = build_theory(proof_lines1, add_print_state=True, end_with="sorry")
        resps = run_theory(isabelle, session, thy)        
        state_block = _extract_print_state_from_responses(resps)
        # Heuristic: Isabelle pretty-printer may truncate with "…" (or cut tokens mid-symbol).
        def _looks_truncated(txt: str) -> bool:
            if not txt:
                return False
            if "…" in txt or " ..." in txt:
                return True
            # any line ending with an incomplete Isabelle symbol like "\<Long"
            for L in txt.splitlines():
                if re.search(r"\\\\<[^>]*$", L.strip()):
                    return True
            return False
        if _looks_truncated(state_block):
            # Retry with a very wide margin to avoid elision (keep the ML printer)
            proof_lines_wide = prolog + proof_lines + [
                "  ML ‹Pretty.setmargin 100000›"
            ]
            thy2 = build_theory(proof_lines_wide, add_print_state=True, end_with="sorry")
            resps2 = run_theory(isabelle, session, thy2)
            state_block2 = _extract_print_state_from_responses(resps2)
            if state_block2:
                state_block = state_block2
        return state_block
            
    except Exception as e:
        # if trace:
        #     print(f"[DEBUG] _print_state_before_hole failed: {e}")
        return ""

# -----------------------------------------------------------------------------
# Goal synthesis from proof-state with safe parenthesization and binder recovery
# -----------------------------------------------------------------------------
def _effective_goal_from_state(state_block: str, fallback_goal: str, full_text: str = "", hole_span: Tuple[int, int] = (0, 0), trace: bool = False) -> str:
    """Build the goal to send to the prover from the local print_state."""
    if state_block and state_block.strip():
        clean = re.sub(r"\x1b\[[0-9;]*m", "", state_block)
        clean = clean.replace("\u00A0", " ")
        # Try to grab the ML-rendered, alpha-renamed subgoal first.
        m_llm = re.search(rf"^{re.escape(_LLM_SUBGOAL_MARK)}\s+(.*)$", clean, flags=re.M)
        # try to fish var classification (for logging/annotation only)
        m_vars = re.search(
            rf"^{re.escape(_LLM_VARS_MARK)}\s*params:\s*(.*?)\s*\|\s*frees:\s*(.*?)\s*\|\s*schematics:\s*(.*)$",
            clean, flags=re.M)

        lines = clean.splitlines()

        # Extract "using this:" facts - default to MULTIPLE premises,
        # but MERGE consecutive lines when we have strong wrap signal:
        #   (a) increased indentation vs the item head,
        #   (b) unbalanced parentheses/brackets/braces after the head line,
        #   (c) the head line ends with an infix/connector (e.g., '=', '⟹', '∧', '@', '::', '≤', '≥', '→', '↔', '⟷', ',').
        # NOTE: When converting to a prover goal, each 'using' fact must be parenthesized and chained as "(f1) ⟹ (f2) ⟹ … ⟹ subgoal".
        # Without parentheses, facts that themselves contain ⟹ can parse incorrectly (e.g., "A ⟹ B ⟹ C" vs "(A ⟹ B) ⟹ C").        
        using_facts: List[str] = []
        i = 0
        while i < len(lines):
            if lines[i].strip() != "using this:":
                i += 1
                continue
            i += 1
            raw_block: List[str] = []
            # 1) Collect raw block until next header/blank
            while i < len(lines):
                raw = lines[i]
                s = raw.strip()
                if (not s) or s.startswith("goal") or s.startswith("using this:"):
                    break
                raw_block.append(raw)
                i += 1

            # 2) Token helpers
            def _lead_spaces(s: str) -> int:
                return len(s) - len(s.lstrip(" "))
            _INFIX_TAIL = re.compile(r"(=|⟹|⇒|->|→|<->|↔|⟷|∧|∨|@|::|≤|≥|≠|,)\s*$")

            # Parenthesis balance across Isabelle text (no strings expected here)
            def _delta_parens(s: str) -> int:
                opens = s.count("(") + s.count("[") + s.count("{")
                closes = s.count(")") + s.count("]") + s.count("}")
                return opens - closes

            # 3) Split into items; start new item by default, merge only with strong wrap signals
            items: List[str] = []
            cur: List[str] = []
            head_indent = 0
            paren_balance = 0
            head_ended_with_infix = False

            def _flush():
                nonlocal cur, items
                if cur:
                    txt = " ".join(x.strip() for x in cur)
                    txt = re.sub(r"\s+", " ", txt).strip()
                    if "…" in txt:
                        txt = txt.replace("…", "").rstrip()
                    if re.search(r"\\\\<[^>]*$", txt):
                        txt = re.sub(r"\\\\<[^>]*$", "", txt).rstrip()
                    if txt:
                        items.append(txt)
                cur = []

            for raw in raw_block:
                # reliable bullet/enumeration → force a new item
                if re.match(r"\s*(?:[•∙·\-\*]|\(\d+\))\s+", raw):
                    _flush()
                    cur = [raw.split(maxsplit=1)[1].strip()] if raw.split(maxsplit=1)[0] else [raw.strip()]
                    head_indent = _lead_spaces(raw)
                    paren_balance = _delta_parens(raw)
                    head_ended_with_infix = bool(_INFIX_TAIL.search(raw))
                    continue

                if not cur:
                    # start a new item
                    cur = [raw.strip()]
                    head_indent = _lead_spaces(raw)
                    paren_balance = _delta_parens(raw)
                    head_ended_with_infix = bool(_INFIX_TAIL.search(raw))
                    continue

                ind = _lead_spaces(raw)
                this_delta = _delta_parens(raw)
                # Strong wrap signals → merge with current item
                if ind > head_indent or paren_balance > 0 or head_ended_with_infix:
                    cur.append(raw.strip())
                    paren_balance += this_delta
                    head_ended_with_infix = bool(_INFIX_TAIL.search(raw))
                else:
                    # new item boundary
                    _flush()
                    cur = [raw.strip()]
                    head_indent = ind
                    paren_balance = this_delta
                    head_ended_with_infix = bool(_INFIX_TAIL.search(raw))
            _flush()

            using_facts = items
            break  # only the first "using this:" block is considered

        # --- Find first numbered subgoal and accumulate wrapped continuation lines ---
        subgoal: Optional[str] = None
        start_idx = -1
        head_tail = ""
        for idx, L in enumerate(lines):
            m = re.match(r"\s*\d+\.\s*(\S.*)$", L)
            if m:
                start_idx = idx
                head_tail = m.group(1).strip()
                break
        if start_idx != -1:
            parts = [head_tail]
            j = start_idx + 1
            # Accumulate wrapped continuation lines of the numbered subgoal
            while j < len(lines):
                Lj = lines[j]
                # stop at next numbered goal or a new section header
                if re.match(r"\s*\d+\.\s", Lj):
                    break
                if Lj.lstrip().startswith(("goal", "using this:")):
                    break
                # continuation lines are indented (pretty printer wraps them)
                if Lj.startswith(" "):
                    parts.append(Lj.strip())
                    j += 1
                    continue
                break
            subgoal = re.sub(r"\s+", " ", " ".join(parts)).strip()
            # last-ditch: if still elided, drop trailing ellipsis and incomplete Isabelle symbol fragments
            if "…" in (subgoal or ""):
                subgoal = subgoal.replace("…", "").rstrip()
            if re.search(r"\\\\<[^>]*$", subgoal or ""):
                subgoal = re.sub(r"\\\\<[^>]*$", "", subgoal or "").rstrip()

        # If ML provided the authoritative subgoal, prefer it.
        if m_llm:
            subgoal = m_llm.group(1).strip()

        if subgoal:
            # -------------------------------
            # 1) Parenthesize 'using' facts.
            # -------------------------------
            def _paren(s: str) -> str:
                t = s.strip()
                return t if (t.startswith("(") and t.endswith(")")) else f"({t})"

            facts = [_paren(f) for f in using_facts] if using_facts else []

            # If ML provided the authoritative subgoal, trust it verbatim:
            # Just chain facts and the (parenthesized) subgoal. Do NOT run
            # any binder-recovery or identifier heuristics here.
            if m_llm:
                subgoal_fixed = f"({subgoal})"
                return " ⟹ ".join(facts + [subgoal_fixed]) if facts else subgoal_fixed          

            # ----------------------------------------------------
            # 2) Binder recovery with HEAD detection (no alpha-rename)
            #    Goal shape we build is:
            #      ⋀params. (f1) ⟹ (f2) ⟹ … ⟹ (subgoal_core)
            #    where params are variable-like identifiers in the subgoal
            #    that are NOT function heads. This mirrors Isabelle’s local
            #    meta-quantified parameters without misclassifying constants.
            # ----------------------------------------------------
            ID = re.compile(r"\b([a-z][A-Za-z0-9_']*)\b")
            HEAD_APP = re.compile(r"\b([a-z][A-Za-z0-9_']*)\b\s*(?=\(|[A-Za-z])")
            # Minimal keyword guard for the rare case we lack ML markers.
            KEYWORDS = {"in","if","then","else","let","case","of","where","and","or","not"}

            def _idents(s: str) -> List[str]:
                seen, out = set(), []
                for m in ID.finditer(s):
                    v = m.group(1)
                    if v in seen:
                        continue
                    seen.add(v)
                    out.append(v)
                return out

            # Extract existing binders if present
            m_bind = re.match(r"\s*⋀\s*([A-Za-z0-9_'\s]+)\.\s*(.*)$", subgoal)
            if m_bind:
                params = [v for v in m_bind.group(1).strip().split() if v]
                sub_core = m_bind.group(2).strip()
            else:
                sub_core = subgoal.strip()
                head_syms = {m.group(1) for m in HEAD_APP.finditer(sub_core)}
                # params = lower-case identifiers that are not function heads
                params = [v for v in _idents(sub_core) if v not in head_syms and v not in KEYWORDS]

            # --------------------------------------------
            # 3) Parenthesize the final subgoal when chaining facts
            #    and lift binders to the very front:
            #      ⋀params. (f1) ⟹ … ⟹ (sub_core)
            # --------------------------------------------
            chain = " ⟹ ".join(facts + [f"({sub_core})"]) if facts else f"({sub_core})"
            return (f"⋀{' '.join(params)}. {chain}") if params else chain
            
    if full_text and hole_span != (0, 0):
        if trace:
            print(f"[fill] Failed to extract goal from state")
    return fallback_goal

def _parse_llm_vars(state_block: str) -> Optional[dict]:
    """Parse [LLM_VARS] classification from a state block."""
    if not state_block:
        return None
    for L in state_block.splitlines():
        Ls = L.strip()
        if not Ls.startswith(_LLM_VARS_MARK):
            continue
        m = re.match(
            rf"^{re.escape(_LLM_VARS_MARK)}\s*params:\s*(.*?)\s*\|\s*frees:\s*(.*?)\s*\|\s*schematics:\s*(.*)$",
            Ls)
        if not m:
            continue
        split_ws = lambda s: [x for x in re.split(r"\s+", s.strip()) if x]
        return {"params": split_ws(m.group(1)),
                "frees": split_ws(m.group(2)),
                "schematics": split_ws(m.group(3))}
    return None

def _annotate_goal_for_log(goal: str, vars_info: Optional[dict]) -> str:
    """Purely cosmetic: prefix frees with '?' in logs; never sent to the prover."""
    if not vars_info:
        return goal
    g = goal
    m = re.match(r"\s*⋀\s*([A-Za-z0-9_'\s]+)\.\s*(.*)$", g)
    binder, tail = ("", g)
    if m:
        binder = f"⋀{m.group(1).strip()}. "
        tail = m.group(2).strip()
    def mark(text: str, name: str) -> str:
        return re.sub(rf"\b{re.escape(name)}\b", f"?{name}", text)
    for name in vars_info.get("frees", []):
        tail = mark(tail, name)
    return binder + tail

def _fill_one_hole(
    isabelle, session: str, full_text: str, hole_span: Tuple[int, int], goal_text: str,
    model: Optional[str], per_hole_timeout: int, *, trace: bool = False,    
) -> Tuple[str, bool, str]:
    # If this 'sorry' sits immediately after a finisher/tactic, it's a stale hole:
    # delete it without asking the prover to avoid duplicates like an extra 'by force'.
    try:
        s_line_start = full_text.rfind("\n", 0, hole_span[0]) + 1
        prev_line_end = s_line_start - 1
        prev_prev_nl = full_text.rfind("\n", 0, prev_line_end) + 1
        prev_line = full_text[prev_prev_nl:prev_line_end+1]
    except Exception:
        prev_line = ""
    prev_is_finisher = (
        bool(_INLINE_BY_TAIL.search(prev_line)) or
        bool(_TACTIC_LINE_RE.match(prev_line)) or
        prev_line.strip() in {"done", "."}
    )
    if prev_is_finisher:
        s, e = hole_span
        new_text = full_text[:s] + "\n" + full_text[e:]
        return new_text, True, "(stale-hole)"

    # Normal hole filling    
    state_block = _print_state_before_hole(isabelle, session, full_text, hole_span, trace)
    eff_goal = _effective_goal_from_state(state_block, goal_text, full_text, hole_span, trace)
    vars_info = _parse_llm_vars(state_block)
    eff_goal_log = _annotate_goal_for_log(eff_goal, vars_info)
    
    if trace:
        print(f"[fill] State block (length={len(state_block)}):")
        if state_block.strip():
            print(state_block[:200] + ("..." if len(state_block) > 200 else ""))
        else:
            print("  (empty or whitespace only)")
        if vars_info:
            print(f"[fill] Vars: params={vars_info.get('params',[])} "
                  f"frees={vars_info.get('frees',[])} schematics={vars_info.get('schematics',[])}")
            print(f"[fill] Effective goal (annotated): {eff_goal_log}")
        print(f"[fill] Effective goal: {eff_goal}")
        print(f"[fill] Original goal: {goal_text}")

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

    steps: List[str] = [str(s) for s in res.get("steps", [])]
    if not steps:
        return full_text, False, "no-steps"

    applies = [s for s in steps if s.startswith("apply")]
    fin = next((s for s in steps if s.startswith("by ") or s.strip() == "done"), "")
    script_lines: List[str] = applies + ([fin] if fin else [])
    
    if not script_lines:
        return full_text, False, "no-tactics"

    insert = "\n  " + "\n  ".join(script_lines) + "\n"
    s, e = hole_span
    new_text = full_text[:s] + insert + full_text[e:]
    return new_text, True, "\n".join(script_lines)


def _verify_full_proof(isabelle, session: str, text: str) -> bool:
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
    
    # IMPORTANT: process TOP-DOWN — Isabelle stops at the first error.
    # 1) Try whole-line tactic candidates first.
    for i, L in enumerate(lines):
        if not (_TACTIC_LINE_RE.match(L) or L.strip() == "done" or _BARE_DOT.match(L)):
            continue
        indent = L[: len(L) - len(L.lstrip(" "))]
        trial = lines[:i] + [f"{indent}sorry"] + lines[i + 1 :]
        if _runs(trial):
            lines[i] = f"{indent}sorry"
            return ("\n".join(lines) + ("" if text.endswith("\n") else "\n")), True
    
    # 2) Try inline '... by TACTIC' patterns, TOP-DOWN.
    for i, L in enumerate(lines):
        if not L:
            continue
        m = _INLINE_BY_TAIL.search(L)
        if not m:
            continue
        indent = L[: len(L) - len(L.lstrip(" "))]
        # Split the line: keep header (e.g., 'show ?case') and open a 'sorry' line.
        header = L[: m.start()].rstrip()
        trial = lines[:i] + [header, f"{indent}sorry"] + lines[i + 1 :]
        if _runs(trial):
            lines[i] = header
            lines.insert(i + 1, f"{indent}sorry")
            return ("\n".join(lines) + ("" if text.endswith("\n") else "\n")), True         
            
    return (text if text.endswith("\n") else text + "\n"), False


def plan_outline(goal: str, *, model: Optional[str] = None, outline_k: Optional[int] = None,
                outline_temps: Optional[Iterable[float]] = None, legacy_single_outline: bool = False,
                priors_path: Optional[str] = None, context_hints: bool = False,
                lib_templates: bool = False, alpha: float = 1.0, beta: float = 0.5,
                gamma: float = 0.2, hintlex_path: Optional[str] = None, hintlex_top: int = 8) -> str:
    """Generate an Isar outline with 'sorry' placeholders and return it."""
    server_info, proc = start_isabelle_server(name="planner", log_file="logs/planner_ui.log")
    isa = get_isabelle_client(server_info)
    session = isa.session_start(session=ISABELLE_SESSION)
    try:
        if legacy_single_outline:
            skel = propose_isar_skeleton(goal, model=model, temp=0.35, force_outline=True)
            return skel.text
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
                try: proc.terminate()
                except Exception: pass
            if hasattr(proc, "kill"):
                try: proc.kill()
                except Exception: pass
        except Exception:
            pass


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
    def left_s() -> float:
        return max(0.0, timeout - (time.monotonic() - t0))

    try:
        # 1) Outline
        if legacy_single_outline:
            if trace:
                print("[planner] outline: single low-temp", flush=True)            
            skel = propose_isar_skeleton(goal, model=model, temp=0.35, force_outline=force_outline)
            full = skel.text
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
            
        if trace:
            holes_now = find_sorry_spans(full)
            print(f"[planner] outline ready: {len(full)} chars; holes={len(holes_now)}", flush=True)
            
        if mode == "outline":
            return PlanAndFillResult(True, full, [], [])

        # 2) Verify if already complete
        spans = find_sorry_spans(full)
        if not spans:
            if trace:
                print("[planner] no holes detected – verifying complete proof", flush=True)
            if _verify_full_proof(isa, session, full):
                return PlanAndFillResult(True, full, [], [])
            if trace:
                print("[planner] complete proof did not verify – opening minimal holes", flush=True)
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
        fills: List[str] = []
        failed: List[int] = []

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
                print(f"[planner] fill hole #{hole_idx} span=({s},{e}) "
                      f"budget={per_hole_budget}s remaining_holes≈{remaining}", flush=True)            

            full2, ok, script = _fill_one_hole(isa, session, full, span, goal_text, model, per_hole_timeout=per_hole_budget, trace=trace)
            if ok:
                full = full2
                fills.append(script)
                if trace:
                    n_steps = script.count("\n") + 1
                    print(f"[planner]   ✓ filled with {n_steps} step(s)", flush=True)                
                hole_idx += 1
                continue

            # CEGIS repairs
            if repairs and left_s() > 6:
                _state = _print_state_before_hole(isa, session, full, span, trace)
                _eff_goal = _effective_goal_from_state(_state, goal_text, full, span, trace)
                if trace:
                    print("[planner]   → trying local repairs (CEGIS)…", flush=True)
                patched, applied, _reason = try_cegis_repairs(
                    full_text=full, hole_span=span, goal_text=_eff_goal, model=model,
                    isabelle=isa, session=session, repair_budget_s=min(10.0, max(5.0, left_s() * 0.33)),
                    max_ops_to_try=max_repairs_per_hole, beam_k=2, allow_whole_fallback=True, trace=trace,
                )
                if applied and patched != full:
                    full = patched
                    if trace:
                        print("[planner]   ✓ repair patched snippet – retrying hole", flush=True)                    
                    continue

            failed.append(hole_idx)
            if trace:
                print("[planner]   ✗ hole still failing", flush=True)            
            hole_idx += 1

        success = ("sorry" not in full)
        if trace:
            print(f"[planner] finished: success={success}", flush=True)        
        return PlanAndFillResult(success, full, fills, failed)

    finally:
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
                try: proc.terminate()
                except Exception: pass
            if hasattr(proc, "kill"):
                try: proc.kill()
                except Exception: pass
            is_popen = all(hasattr(proc, attr) for attr in ("poll", "communicate", "pid"))
            if is_popen:
                try: proc.wait(timeout=2)
                except Exception: pass
        except Exception:
            pass