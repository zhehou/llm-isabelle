from __future__ import annotations

import time
import re
import os
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout
import hashlib

from planner.skeleton import (
    Skeleton, find_sorry_spans, propose_isar_skeleton, propose_isar_skeleton_diverse_best,
)
from planner.contextual import (
    _fact_replay_groups,
    fill_hole_in_context,
    splice_hole_commands,
    splice_hole_command_variants,
    _local_fact_hints,
)
from planner.repair import try_cegis_repairs, regenerate_whole_proof, _APPLY_OR_BY as _TACTIC_LINE_RE
from planner.frontier import frontier_advanced, frontier_score
from prover.config import ISABELLE_SESSION
from prover.isabelle_api import (
    build_theory, finished_ok, get_isabelle_client, last_print_state_block, start_isabelle_server,
)
from prover.prover import prove_goal
from planner.goals import _print_state_before_hole, _log_state_block, _effective_goal_from_state, _first_lemma_line, _extract_goal_from_lemma_line, _cleanup_resources, _verify_full_proof, _run_theory_with_timeout, _usable_chained_facts

def _hole_fingerprint(full_text: str, span: tuple[int, int], context: int = 80) -> str:
    """Stable key based on the enclosing lemma, case and statement head.

    The previous key hashed nearby text, so inserting a verified ``apply`` line
    made the same hole appear new and reset all repair limits.
    """
    s, _ = span
    before = full_text[:s].splitlines()
    lemma = next((ln.strip() for ln in reversed(before) if ln.strip().startswith("lemma ")), "")
    case = next((ln.strip() for ln in reversed(before) if ln.strip().startswith("case ")), "")
    head = next(
        (
            ln.strip()
            for ln in reversed(before)
            if re.match(r"^\s*(?:have|show|obtain|thus|hence)\b", ln)
        ),
        "",
    )
    key = "\n".join((lemma, case, head))
    if not key.strip():
        key = full_text[max(0, s - context): min(len(full_text), s + context)]
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

# Constants
_INLINE_BY_TAIL = re.compile(r"\s+by\s+.+$")
_BARE_DOT = re.compile(r"(?m)^\s*\.\s*$")
_METHOD_MODIFIER_LINE = re.compile(r"^\s*(?:using|unfolding)\b")
_HEAD_CMD_RE = re.compile(r"^\s*(have|show|obtain)\b")  # local copy to avoid new imports
_ISA_VERIFY_TIMEOUT_S = int(os.getenv("ISABELLE_VERIFY_TIMEOUT_S", "30"))

@dataclass(slots=True)
class PlannerMetrics:
    """Explicit counters for planner control-flow outcomes."""

    deterministic_prepass_successes: int = 0
    local_repairs: int = 0
    subproof_repairs: int = 0
    stage3_repairs: int = 0
    verified_aux_obligations: int = 0
    complete_generated_outlines: int = 0
    outline_backtracks: int = 0


@dataclass(slots=True)
class PlanAndFillResult:
    success: bool
    outline: str
    fills: List[str]
    failed_holes: List[int]
    metrics: PlannerMetrics = field(default_factory=PlannerMetrics)


def _theory_preamble_from_text(full_text: str) -> str:
    """Return outer-syntax declarations preceding the first target lemma."""

    lines = (full_text or "").splitlines()
    lemma_starts = [
        i for i, line in enumerate(lines)
        if re.match(r"^\s*(?:lemma|theorem|corollary)\b", line)
    ]
    lemma_start = lemma_starts[-1] if lemma_starts else len(lines)
    return "\n".join(lines[:lemma_start]).strip()


# ============================================================================
# Hole Filling
# ============================================================================

def _fill_one_hole(isabelle, session: str, full_text: str, hole_span: Tuple[int, int], 
                  goal_text: str, model: Optional[str], per_hole_timeout: int, *,
                  fill_beam: int = 4, fill_depth: int = 8, fill_facts_limit: int = 8,
                  fill_sledge: bool = True, fill_sledge_timeout: int = 10,
                  trace: bool = False) -> Tuple[str, bool, str]:
    """Fill single hole in proof."""
    fill_started = time.monotonic()
    
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
    
    # First search in the complete Isar document.  This preserves case facts,
    # local assumptions and proof mode, unlike reconstruction as a fresh lemma.
    # Bound exact-context beam search so an unchanged state cannot consume the
    # entire hole budget.  Context-complete fresh-goal search and repair retain
    # independent opportunities afterwards.
    hole_count = max(1, len(find_sorry_spans(full_text)))
    # Multi-hole outlines need enough time left to replay a standalone proof
    # and visit later independent holes.  A fixed 25-second contextual budget
    # consumed almost the entire per-hole allocation when an outline had four
    # leaves.  Single-hole searches retain a larger exact-context window.
    contextual_fraction = 0.45 if hole_count > 1 else 0.60
    contextual_cap = 20 if hole_count > 1 else 28
    contextual_budget = min(
        contextual_cap,
        max(6, int(max(4, per_hole_timeout) * contextual_fraction)),
    )
    contextual = fill_hole_in_context(
        isabelle,
        session,
        full_text,
        hole_span,
        goal_text,
        model,
        contextual_budget,
        beam_w=fill_beam,
        max_depth=fill_depth,
        facts_limit=fill_facts_limit,
        use_sledge=fill_sledge,
        sledge_timeout=fill_sledge_timeout,
        trace=trace,
    )
    if contextual.success:
        context_tag = (
            "[contextual-fast]"
            if contextual.reason in {
                "context-fast-verified", "context-apply-done-verified"
            }
            else "[contextual-search]"
        )
        return contextual.text, True, context_tag + "\n" + contextual.script
    if contextual.reason == "context-partial-progress" and contextual.text != full_text:
        return contextual.text, False, "[contextual-partial]\n" + contextual.script
    if trace:
        print(
            f"[fill-context] {contextual.reason}; falling back to fresh-goal "
            f"search (rank {contextual.initial_rank} -> {contextual.final_rank})."
        )
    # The fresh-goal prover is also useful when exact-context search was
    # exhausted, especially for an outer ``show ?thesis`` whose effective goal
    # is the original theorem.  Candidate scripts are always spliced back and
    # verified in the complete Isar document, so losing local context here
    # cannot create a false success.
    state_block = _print_state_before_hole(isabelle, session, full_text, hole_span, trace)
    _log_state_block("fill", state_block, trace=trace)
    
    # orig_goal = _original_goal_from_state(state_block)
    eff_goal = _effective_goal_from_state(state_block, goal_text, full_text, hole_span, trace)
    if trace:
        from planner.goals import _extract_chained_facts, _extract_local_assumptions
        assumptions = _extract_local_assumptions(state_block)
        raw_chained = _extract_chained_facts(state_block)
        chained = _usable_chained_facts(state_block)
        print(
            f"[fill] Context-complete goal ({len(assumptions)} local assumptions, "
            f"{len(chained)}/{len(raw_chained)} usable chained facts): "
            f"{eff_goal[:1200]}"
        )
    
    fallback_timeout = max(
        1,
        min(35, int(per_hole_timeout - (time.monotonic() - fill_started))),
    )
    if fallback_timeout <= 1:
        return full_text, False, contextual.reason

    res = prove_goal(
        isabelle, session, eff_goal, model_name_or_ensemble=model,
        beam_w=max(1, fill_beam), max_depth=max(1, fill_depth), hint_lemmas=6,
        timeout=fallback_timeout,
        models=None, save_dir=None, use_sledge=fill_sledge,
        sledge_timeout=max(1, fill_sledge_timeout),
        sledge_every=1, trace=trace, use_color=False, use_qc=True,
        qc_timeout=2, qc_every=1, use_np=True, np_timeout=5, np_every=2,
        facts_limit=max(1, fill_facts_limit), do_minimize=False, minimize_timeout=8,
        do_variants=True, variant_timeout=6, variant_tries=24,
        enable_reranker=True, initial_state_hint=state_block,
        theory_preamble=_theory_preamble_from_text(full_text),
    )
    
    steps = [str(s) for s in res.get("steps", [])]

    # Fallbacks: some backends return finishers/applies in separate keys
    fin_candidates = []
    # singular fields
    for k in ("finisher", "finish", "final"):
        v = res.get(k)
        if isinstance(v, str):
            fin_candidates.append(v)
    # list fields
    for k in ("finishers", "sledge_finishers"):
        vs = res.get(k)
        if isinstance(vs, (list, tuple)):
            fin_candidates.extend([str(x) for x in vs if isinstance(x, str)])
    applies_from_keys = []
    for k in ("applies", "apply_steps"):
        vs = res.get(k)
        if isinstance(vs, (list, tuple)):
            applies_from_keys.extend([str(x) for x in vs if isinstance(x, str) and x.startswith("apply")])

    applies = [s for s in steps if s.startswith("apply")]
    if applies_from_keys:
        applies = applies or applies_from_keys  # prefer explicit list if steps were empty

    fin = next((s for s in steps if s.startswith("by ") or s.strip() == "done"), "")
    if not fin:
        fin = next((x for x in fin_candidates if isinstance(x, str) and (x.startswith("by ") or x.strip() == "done")), "")

    # If neither steps nor recognized finishers were returned, report no-steps
    if not (applies or fin):
        return full_text, False, "no-steps"
    
    # Handle finisher
    if fin:
        script_variants = [applies + [fin]]
        if fin.startswith("by "):
            # ``by`` closes a goal in prove mode; after an existing apply-style
            # command Isabelle may instead require another apply followed by
            # done.  Verify both encodings in the complete Isar document.
            script_variants.append(applies + ["apply " + fin[3:].strip(), "done"])
            # The standalone goal makes local context explicit as meta-level
            # premises.  When replaying its method inside structured Isar,
            # explicitly supply the named case/intermediate facts that denote
            # those premises.  Verification filters stale or inapplicable
            # names, so this is a general replay bridge rather than a theorem-
            # specific rule.
            local_facts = _local_fact_hints(
                full_text, hole_span, max(1, fill_facts_limit)
            )
            if local_facts:
                body = fin[3:].strip()
                for group in _fact_replay_groups(local_facts, limit=16):
                    # Try both the method-combinator form and native Isar
                    # modifiers.  The latter is particularly robust for
                    # ``then show`` leaves because it preserves case chaining.
                    script_variants.extend([
                        applies + [f"using {group}", fin],
                        applies + [f"by (use {group} in {body})"],
                    ])
        for script_lines in script_variants:
            for new_text, dropped_modifiers in splice_hole_command_variants(
                full_text, hole_span, script_lines
            ):
                if _verify_full_proof(isabelle, session, new_text):
                    replay = "[replaced-modifiers]\n" if dropped_modifiers else ""
                    return (
                        new_text,
                        True,
                        "[fresh-goal]\n" + replay + "\n".join(script_lines),
                    )
        return full_text, False, "finisher-unverified"
    
    # Handle apply-only  (NEVER mark success for apply-only scripts)
    if applies:
        # Decide if the hole sits under a have/show/obtain head; if so, we must NOT
        # leave a bare 'apply' there (illegal in 'prove' mode). Replace the hole with
        # a tiny subproof instead of inserting above the hole.
        s, e = hole_span
        # scan a small window upwards to find the enclosing head line
        head_line_start = full_text.rfind("\n", 0, s) + 1
        scan_start = max(0, full_text.rfind("\n", 0, max(0, head_line_start - 512)) + 1)
        segment = full_text[scan_start:s]
        lines = segment.splitlines()
        head_idx = None
        for i in range(len(lines) - 1, -1, -1):
            if _HEAD_CMD_RE.match(lines[i] or ""):
                head_idx = i
                break

        # Deduplicate against already-present lines in the local window
        dedup_window = segment
        dedup = [a for a in applies if a not in dedup_window]
        if not dedup:
            return full_text, False, "apply-duplicate"

        if head_idx is not None:
            # Apply-only inside have/show is illegal in 'prove' mode unless closed by 'by ...'.
            # Do NOT fabricate 'proof ... qed'. Leave the hole as-is and let the caller escalate to repair.
            if trace:
                print("[fill] apply-only inside have/show; not inserting proof/qed; escalating to repair.")
            return full_text, False, "apply-inside-have/show"
        else:
            # Non have/show context — keep existing behaviour (insert above, keep sorry)
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

# --- helper: pick the sorry-span nearest a target offset (to preserve focus) ---
def _nearest_sorry_span(spans: List[Tuple[int, int]], target_s: int) -> Optional[Tuple[int, int]]:
    if not spans:
        return None
    return min(spans, key=lambda sp: abs(sp[0] - target_s))

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

def _repair_failed_proof_topdown(isa, session, full: str, goal_text: str, model: Optional[str],
                                 left_s, max_repairs_per_hole: int, trace: bool) -> Tuple[str, bool]:
    """Walk tactics from top; attempt CEGIS-repair on the first failing one.

    This must never crash the UI route. Timeouts / broken Isabelle responses are treated as
    'repair failed', and the caller may decide to fall back (e.g. open minimal sorries).
    """
    t_spans = _tactic_spans_topdown(full)
    if not t_spans:
        return full, False

    i = 0
    while i < len(t_spans) and left_s() > 3.0:
        span = t_spans[i]
        try:
            st = _print_state_before_hole(isa, session, full, span, trace)
            eff_goal = _effective_goal_from_state(st, goal_text, full, span, trace)
        except Exception as ex:
            if trace:
                print(f"[repair] Could not extract state/goal before tactic (skipping): {ex}")
            i += 1
            continue

        per_budget = min(30.0, max(15.0, left_s() * 0.33))

        try:
            patched, applied, _ = try_cegis_repairs(
                full_text=full, hole_span=span, goal_text=eff_goal, model=model,
                isabelle=isa, session=session, repair_budget_s=per_budget,
                max_ops_to_try=max_repairs_per_hole, beam_k=2,
                allow_whole_fallback=False, trace=trace, resume_stage=0,
            )
        except (TimeoutError, _FuturesTimeout, ValueError) as ex:
            # TimeoutError: verifier timed out; ValueError: isabelle_client returned unexpected/empty response
            if trace:
                print(f"[repair] CEGIS repair aborted (treat as failed): {type(ex).__name__}: {ex}")
            return full, False
        except Exception as ex:
            if trace:
                print(f"[repair] CEGIS repair crashed (treat as failed): {type(ex).__name__}: {ex}")
            return full, False

        if applied and patched != full:
            if _verify_full_proof(isa, session, patched):
                return patched, True

            # Partial progress: keep it, then try to open the failing spot into a 'sorry'
            if trace:
                print("[repair] Partial progress in topdown repair (unverified). Opening sorries...")
            full = patched
            full2, opened = _open_minimal_sorries(isa, session, full)
            if opened:
                full = full2
                t_spans = _tactic_spans_topdown(full)
                i = 0
                continue

        i += 1

    return full, False

def _quick_state_and_errors(isabelle, session: str, text: str, *, timeout_s: Optional[int] = None) -> Tuple[str, List[str]]:
    """Run a theory quickly and return (last_state_block, error_messages).

    Best-effort utility used only to locate an error line for opening with 'sorry'.
    It must be robust: on any exception, return empty state and a single error string.
    """
    try:
        ts = text.splitlines()
        thy = build_theory(ts, add_print_state=True, end_with=None)
        out = _run_theory_with_timeout(
            isabelle, session, thy,
            timeout_s=int(timeout_s) if timeout_s is not None else min(_ISA_VERIFY_TIMEOUT_S, 15),
        )
        state = ""
        try:
            state = last_print_state_block(out)
        except Exception:
            state = ""

        # Normalize messages to strings for simple scanning
        if isinstance(out, (list, tuple)):
            msgs = [str(m) for m in out]
        else:
            msgs = [str(out)]

        errs = [m for m in msgs if any(tok in m.lower() for tok in ("error", "exception", "failed"))]
        return state, errs
    except Exception as ex:
        return "", [str(ex)]
    
def _extract_error_lines(errs: List[str]) -> List[int]:
    """Extract 1-based line numbers from Isabelle error messages (best-effort)."""
    if not errs:
        return []

    patts = [
        re.compile(r"(?i)\bline\s+(\d+)\b"),          # 'line 23'
        re.compile(r"(?i)\bLine\s+(\d+)\b"),          # 'Line 23'
        re.compile(r":(\d+):(\d+)\b"),                # 'Scratch.thy:23:5'
        re.compile(r"\((\d+),(\d+)\)"),               # '(23,5)'
    ]

    found: set[int] = set()
    for raw in errs:
        s = str(raw)
        for p in patts:
            for m in p.finditer(s):
                try:
                    n = int(m.group(1))
                    if n > 0:
                        found.add(n)
                except Exception:
                    pass

    return sorted(found)

def _open_minimal_sorries(isabelle, session: str, text: str) -> Tuple[str, bool]:
    """Turn the smallest diagnosed set of failing tactics into fillable holes.

    This is deterministic proof surgery, not proof generation.  First try each
    tactic in isolation.  If a proof has several bad tactics, admit all tactic
    leaves and greedily restore every command whose restoration still checks.
    The result is verified with Isabelle before it is returned.  Error-line
    localisation remains a last-resort fallback for client versions that
    cannot verify the all-open diagnostic variant.

    Returns ``(new_text, opened)`` and never raises.
    """
    def _ensure_nl(s: str) -> str:
        return s if s.endswith("\n") else s + "\n"

    def runs(candidate: str) -> bool:
        return _verify_full_proof(
            isabelle,
            session,
            candidate,
            timeout_s=min(_ISA_VERIFY_TIMEOUT_S, 15),
        )

    lines = text.splitlines()
    tactic_lines = [
        i
        for i, line in enumerate(lines)
        if (
            _TACTIC_LINE_RE.match(line or "")
            or _INLINE_BY_TAIL.search(line or "")
            or line.strip() == "done"
            or _BARE_DOT.match(line or "")
        )
    ]

    def atom_start(tactic_line: int) -> int:
        """Include contiguous method modifiers in a repair atom."""
        start = tactic_line
        cursor = tactic_line - 1
        pending_blank: List[int] = []
        saw_modifier = False
        while cursor >= 0:
            line = lines[cursor]
            if not line.strip():
                pending_blank.append(cursor)
                cursor -= 1
                continue
            if _METHOD_MODIFIER_LINE.match(line):
                saw_modifier = True
                start = cursor
                if pending_blank:
                    start = min(start, min(pending_blank))
                    pending_blank.clear()
                cursor -= 1
                continue
            break
        return start if saw_modifier else tactic_line

    atom_starts = {line_no: atom_start(line_no) for line_no in tactic_lines}

    def opened_variant(opened_lines: set[int]) -> str:
        skipped: set[int] = set()
        for tactic_line in opened_lines:
            skipped.update(range(atom_starts[tactic_line], tactic_line))
        result: List[str] = []
        for i, line in enumerate(lines):
            if i in skipped:
                continue
            if i not in opened_lines:
                result.append(line)
                continue
            indent = line[:len(line) - len(line.lstrip(" "))]
            # A standalone indented ``by`` also matches _INLINE_BY_TAIL; test
            # the tactic-line form first to avoid creating a blank line and an
            # over-indented replacement.
            if _TACTIC_LINE_RE.match(line) or line.strip() == "done" or _BARE_DOT.match(line):
                result.append(f"{indent}sorry")
            elif (inline := _INLINE_BY_TAIL.search(line)):
                result.append(line[:inline.start()].rstrip())
                result.append(f"{indent}  sorry")
            else:
                result.append(f"{indent}sorry")
        return _ensure_nl("\n".join(result))

    try:
        if runs(text):
            return _ensure_nl(text), False
    except Exception:
        # If even 'runs' crashes, do nothing.
        return _ensure_nl(text), False

    # Most failures are a single wrong fact/method.  Diagnose those with one
    # verification per candidate, starting at the end where leaf tactics live.
    for line_no in reversed(tactic_lines):
        candidate = opened_variant({line_no})
        if runs(candidate):
            return candidate, True

    # Multiple independent bad leaves: start with all tactic leaves admitted,
    # then restore every command that is not necessary for a valid document.
    if tactic_lines:
        opened = set(tactic_lines)
        all_open = opened_variant(opened)
        if runs(all_open):
            for line_no in tactic_lines:
                trial = set(opened)
                trial.discard(line_no)
                candidate = opened_variant(trial)
                if runs(candidate):
                    opened = trial
            candidate = opened_variant(opened)
            if opened and runs(candidate):
                return candidate, True

    # Last resort: find the first reported error line and open the nearest
    # preceding tactic.  The subsequent exact-context filler still verifies
    # the complete document before accepting any replacement.
    try:
        _, errs = _quick_state_and_errors(isabelle, session, text)
        err_lines = _extract_error_lines(errs)
    except Exception:
        err_lines = []

    if not err_lines:
        return _ensure_nl(text), False

    failing_line_1based = min(err_lines)
    failing_idx = failing_line_1based - 1

    for i in range(min(failing_idx, len(lines) - 1), -1, -1):
        line = lines[i]

        if _TACTIC_LINE_RE.match(line) or line.strip() == "done" or _BARE_DOT.match(line):
            indent = line[:len(line) - len(line.lstrip(" "))]
            lines[i] = f"{indent}sorry"
            return _ensure_nl("\n".join(lines)), True

        m = _INLINE_BY_TAIL.search(line)
        if m:
            indent = line[:len(line) - len(line.lstrip(" "))]
            header = line[:m.start()].rstrip()
            lines[i] = header
            lines.insert(i + 1, f"{indent}sorry")
            return _ensure_nl("\n".join(lines)), True

    return _ensure_nl(text), False

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
                 max_repairs_per_hole: int = 2, trace: bool = False, repair_trace: bool = False,
                 priors_path: Optional[str] = None, context_hints: bool = False,
                 lib_templates: bool = False, alpha: float = 1.0, beta: float = 0.5,
                 gamma: float = 0.2, hintlex_path: Optional[str] = None,
                 hintlex_top: int = 8, fill_beam: int = 4, fill_depth: int = 8,
                 fill_facts_limit: int = 8, fill_sledge: bool = True,
                 fill_sledge_timeout: int = 10, repair_beam_k: int = 2,
                 whole_fallback: bool = True, whole_repair: bool = True,
                 isabelle=None,
                 session_id: Optional[str] = None,
                 initial_outline: Optional[str] = None,
                 theory_preamble: str = "") -> PlanAndFillResult:
    """Plan and fill holes in Isar proofs.

    Notes:
      - 'repair_trace' is a backwards-compatible alias used by the UI. It enables 'trace'.
      - Repair/verification timeouts or broken Isabelle responses must not crash the caller.
        We treat them as repair failures, and (best-effort) restart Isabelle for subsequent calls.
    """
    if repair_trace and not trace:
        trace = True
    if mode not in {"auto", "outline", "plan-fill"}:
        raise ValueError(f"Unsupported planner mode: {mode!r}")

    metrics = PlannerMetrics()

    def _result(
        success: bool,
        outline: str,
        fills: List[str],
        failed_holes: List[int],
    ) -> PlanAndFillResult:
        return PlanAndFillResult(
            success, outline, fills, failed_holes, metrics=metrics
        )

    def _record_local_repair(reason: str = "") -> None:
        if "stage=2" in str(reason):
            metrics.subproof_repairs += 1
        else:
            metrics.local_repairs += 1

    if (isabelle is None) != (session_id is None):
        raise ValueError("isabelle and session_id must be supplied together")
    owns_isabelle = isabelle is None
    if owns_isabelle:
        server_info, proc = start_isabelle_server(name="planner", log_file="logs/planner_ui.log")
        isa = get_isabelle_client(server_info)
        session = isa.session_start(session=ISABELLE_SESSION)
    else:
        # Experiment runners already own a warm Isabelle server.  Reusing it
        # avoids a process and event-loop teardown for every benchmark goal.
        proc = None
        isa = isabelle
        session = str(session_id)

    t0 = time.monotonic()
    left_s = lambda: max(0.0, timeout - (time.monotonic() - t0))

    restart_count = 0

    def _restart_isabelle(reason: str, ex: Optional[BaseException] = None) -> None:
        nonlocal isa, session, proc, restart_count
        if not owns_isabelle:
            if trace:
                print(f"[planner] Shared Isabelle unavailable during {reason}; caller retains ownership.")
            return
        if restart_count >= 2:
            return
        restart_count += 1
        if trace:
            msg = f"[planner] Restarting Isabelle (#{restart_count}) due to {reason}"
            if ex is not None:
                msg += f": {type(ex).__name__}: {ex}"
            print(msg)
        try:
            _cleanup_resources(isa, proc)
        except Exception:
            pass
        server_info2, proc2 = start_isabelle_server(name="planner", log_file="logs/planner_ui.log")
        isa2 = get_isabelle_client(server_info2)
        session2 = isa2.session_start(session=ISABELLE_SESSION)
        isa, session, proc = isa2, session2, proc2

    try:
        outline_alternatives: List[str] = []
        # In auto mode, run a short deterministic portfolio before paying for
        # outline generation.  It preserves the theory preamble and permits
        # Isabelle automation, definition/library structural macros and
        # Sledgehammer, but never constructs a model or retrieval stack.
        if mode == "auto" and not initial_outline and left_s() > 6.0:
            direct_budget = int(max(6, min(20, left_s() * 0.15)))
            try:
                direct = prove_goal(
                    isa,
                    session,
                    goal,
                    model_name_or_ensemble=model,
                    beam_w=max(1, fill_beam),
                    max_depth=max(1, fill_depth),
                    hint_lemmas=6,
                    timeout=direct_budget,
                    models=None,
                    save_dir=None,
                    use_sledge=fill_sledge,
                    sledge_timeout=max(1, fill_sledge_timeout),
                    sledge_every=1,
                    trace=trace,
                    use_color=False,
                    use_qc=False,
                    qc_timeout=2,
                    qc_every=1,
                    use_np=False,
                    np_timeout=5,
                    np_every=2,
                    facts_limit=max(1, fill_facts_limit),
                    do_minimize=False,
                    minimize_timeout=8,
                    do_variants=False,
                    variant_timeout=6,
                    variant_tries=24,
                    enable_reranker=False,
                    theory_preamble=theory_preamble,
                    deterministic_only=True,
                )
                if bool(direct.get("success")):
                    direct_text = "\n".join(str(s) for s in direct.get("steps", []))
                    if direct_text and not direct_text.endswith("\n"):
                        direct_text += "\n"
                    if direct_text and not find_sorry_spans(direct_text) \
                            and _verify_full_proof(isa, session, direct_text):
                        metrics.deterministic_prepass_successes += 1
                        if direct.get("origin") == "definition-obligation":
                            metrics.verified_aux_obligations += 1
                        return _result(True, direct_text, [], [])
            except (TimeoutError, _FuturesTimeout, ValueError) as ex:
                _restart_isabelle("direct_prover", ex)
            except Exception as ex:
                if trace:
                    print(f"[planner] Direct prover portfolio member failed: {type(ex).__name__}: {ex}")

        # Generate outline.  ``plan-fill`` is deliberately distinct from
        # ``outline``: both force placeholders, but only outline returns early.
        force_outline = mode in {"outline", "plan-fill"}
        if initial_outline:
            outline_budget = 0.0
        elif mode == "outline":
            outline_budget = max(1.0, left_s())
        else:
            # Reserve most of the total request for filling and repair.  The
            # previous independent K-generation and state-check timeouts could
            # consume the entire request before the first hole was visited.
            outline_budget = min(
                max(1.0, left_s() - 5.0),
                45.0,
                max(8.0, left_s() * 0.35),
            )
        if trace and initial_outline:
            print(
                f"[planner] fixed-outline phase: supplied fixture "
                f"remaining={left_s():.1f}s"
            )
        elif trace:
            print(
                f"[planner] outline phase: budget={outline_budget:.1f}s "
                f"remaining={left_s():.1f}s"
            )
        if initial_outline:
            full = initial_outline.strip() + "\n"
            fixture_lemma = _first_lemma_line(full)
            if not fixture_lemma or _extract_goal_from_lemma_line(fixture_lemma) != goal:
                raise ValueError("initial outline lemma does not match requested goal")
        elif legacy_single_outline:
            full = propose_isar_skeleton(
                goal, model=model, temp=0.35, force_outline=force_outline,
                timeout_s=max(1, int(outline_budget)),
                theory_preamble=theory_preamble,
            ).text
        else:
            temps = tuple(outline_temps) if outline_temps else (0.35, 0.55, 0.85)
            k = int(outline_k) if outline_k is not None else 3
            best, outline_diagnostics = propose_isar_skeleton_diverse_best(
                goal, isabelle=isa, session_id=session, model=model, temps=temps, k=k,
                force_outline=force_outline, priors_path=priors_path,
                context_hints=context_hints, lib_templates=lib_templates,
                alpha=alpha, beta=beta, gamma=gamma, hintlex_path=hintlex_path,
                hintlex_top=hintlex_top, total_budget_s=outline_budget,
                theory_preamble=theory_preamble,
            )
            full = best.text
            outline_alternatives = [
                str(candidate)
                for candidate in outline_diagnostics.get("ranked_candidates", [])
                if str(candidate).strip() and str(candidate).strip() != full.strip()
            ]

        if trace:
            print(
                f"[planner] fill phase: holes={len(find_sorry_spans(full))} "
                f"remaining={left_s():.1f}s"
            )

        spans = find_sorry_spans(full)
        surgery_hole_keys: set[str] = set()

        if mode == "outline":
            return _result(True, full, [], [])

        # Handle complete proofs
        if not spans:
            try:
                if _verify_full_proof(isa, session, full):
                    if not initial_outline:
                        metrics.complete_generated_outlines += 1
                    return _result(True, full, [], [])
            except (TimeoutError, _FuturesTimeout, ValueError) as ex:
                _restart_isabelle("verify_full_proof", ex)

            if not repairs:
                return _result(False, full, [], [0])

            # Deterministic proof surgery comes before generative block repair:
            # identify the smallest bad tactic set, admit it, then send those
            # exact holes through the normal verified contextual filler.
            full2, opened = _open_minimal_sorries(isa, session, full)
            if opened:
                full = full2
                surgery_hole_keys = {
                    _hole_fingerprint(full, span)
                    for span in find_sorry_spans(full)
                }
                if trace:
                    print(
                        f"[repair] proof surgery opened "
                        f"{len(surgery_hole_keys)} failing tactic(s)."
                    )
            elif left_s() > 6.0:
                full, ok = _repair_failed_proof_topdown(
                    isa, session, full, goal, model, left_s,
                    max_repairs_per_hole, trace,
                )
                if ok:
                    metrics.local_repairs += 1
                    return _result(True, full, ["[repair]\n[block]"], [])
                return _result(False, full, [], [0])
            else:
                return _result(False, full, [], [0])

        # Fill holes
        lemma_line = _first_lemma_line(full)
        if not lemma_line:
            return _result(False, full, [], [0])

        goal_text = _extract_goal_from_lemma_line(lemma_line)
        fills: List[str] = []
        partial_tries: dict[str, int] = {}
        repair_memory: dict[str, List[str]] = {}
        tried_outlines: set[str] = {full.strip()}

        def _verified_hole_reduction(before: str, after: str) -> bool:
            return (
                after != before
                and len(find_sorry_spans(after)) < len(find_sorry_spans(before))
                and _verify_full_proof(isa, session, after)
            )

        def _verified_frontier_progress(before: str, after: str) -> bool:
            return (
                after != before
                and frontier_advanced(before, after)
                and _verify_full_proof(isa, session, after)
            )

        def _regenerate_or_reoutline() -> bool:
            """Stage 3, then bounded backtracking to a distinct outline branch."""
            nonlocal full
            if left_s() <= 8.0:
                return False
            before = full
            new_full, ok_re = before, False
            regen_reason = ""
            if repairs and whole_repair:
                try:
                    new_full, ok_re, regen_reason = regenerate_whole_proof(
                        full_text=before,
                        goal_text=goal_text,
                        model=model,
                        isabelle=isa,
                        session=session,
                        budget_s=min(40.0, max(8.0, left_s() * 0.55)),
                        trace=trace,
                        prior_outline_text=before,
                    )
                except (TimeoutError, _FuturesTimeout, ValueError) as ex:
                    _restart_isabelle("regenerate_whole_proof", ex)
                except Exception as ex:
                    if trace:
                        print(f"[repair] Whole regeneration failed: {type(ex).__name__}: {ex}")

            if ok_re and _verified_frontier_progress(before, new_full):
                full = new_full
                partial_tries.clear()
                metrics.stage3_repairs += 1
                if regen_reason == "regen:verified-obligation":
                    metrics.verified_aux_obligations += 1
                if trace:
                    print(
                        f"[repair] Stage 3 advanced source frontier "
                        f"{frontier_score(before)} -> {frontier_score(new_full)}."
                    )
                return True

            # The diverse planner is a small script-level beam, not merely an
            # ensemble whose losing candidates are discarded.  Once a branch
            # is exhausted, backtrack once to the next Isabelle-admitted
            # candidate.  ``tried_outlines`` prevents cycles.
            while outline_alternatives:
                alternate = outline_alternatives.pop(0)
                key = alternate.strip()
                if not key or key in tried_outlines:
                    continue
                tried_outlines.add(key)
                if _verify_full_proof(isa, session, alternate):
                    full = alternate
                    partial_tries.clear()
                    metrics.outline_backtracks += 1
                    if trace:
                        print("[planner] Backtracking to next verified outline branch.")
                    return True

            if not (repairs and whole_repair) or left_s() <= 6.0:
                return False
            try:
                temps = tuple(outline_temps) if outline_temps else (0.45, 0.7, 0.95)
                k = int(outline_k) if outline_k is not None else 3
                best, regenerated_diagnostics = propose_isar_skeleton_diverse_best(
                    goal_text,
                    isabelle=isa,
                    session_id=session,
                    model=model,
                    temps=temps,
                    k=k,
                    force_outline=True,
                    priors_path=priors_path,
                    context_hints=context_hints,
                    lib_templates=lib_templates,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    hintlex_path=hintlex_path,
                    hintlex_top=hintlex_top,
                    total_budget_s=min(20.0, max(4.0, left_s() * 0.35)),
                    theory_preamble=theory_preamble,
                )
                candidate = best.text
                outline_alternatives.extend(
                    str(item)
                    for item in regenerated_diagnostics.get("ranked_candidates", [])
                    if str(item).strip() and str(item).strip() != candidate.strip()
                )
            except Exception:
                return False
            if not candidate.strip() or candidate.strip() in tried_outlines:
                return False
            tried_outlines.add(candidate.strip())
            # This is a new branch, so it need not dominate the failed branch's
            # textual frontier.  It must be distinct and Isabelle-admitted.
            if _verify_full_proof(isa, session, candidate):
                full = candidate
                partial_tries.clear()
                metrics.outline_backtracks += 1
                if trace:
                    print("[planner] Switching to a newly generated verified outline branch.")
                return True
            return False

        while find_sorry_spans(full) and left_s() > 0:
            spans = find_sorry_spans(full)
            if not spans:
                break

            # Maintain a verified admitted document at every iteration.  A bad
            # non-hole command is converted into a hole at its earliest failing
            # tactic, so Fill and Repair share one deterministic source frontier.
            if not _verify_full_proof(isa, session, full):
                if not repairs:
                    if trace:
                        print("[planner] Earliest source failure is not a hole and repairs are disabled.")
                    break
                opened_text, opened = _open_minimal_sorries(isa, session, full)
                if opened and opened_text != full and _verify_full_proof(isa, session, opened_text):
                    full = opened_text
                    surgery_hole_keys.update(
                        _hole_fingerprint(full, item) for item in find_sorry_spans(full)
                    )
                    if trace:
                        print("[repair] Opened earliest failing tactic into the Fill frontier.")
                    continue
                # Syntax/type/statement failures that cannot be converted by
                # tactic surgery go through the same Stage 1/2 repair engine.
                # That engine retargets from this hole to Isabelle's earliest
                # reported failure line.
                if left_s() > 6.0:
                    repair_reason = ""
                    try:
                        patched, applied, repair_reason = try_cegis_repairs(
                            full_text=full,
                            hole_span=spans[0],
                            goal_text=goal_text,
                            model=model,
                            isabelle=isa,
                            session=session,
                            repair_budget_s=min(25.0, max(8.0, left_s() * 0.30)),
                            max_ops_to_try=max_repairs_per_hole,
                            beam_k=repair_beam_k,
                            allow_whole_fallback=False,
                            trace=trace,
                            resume_stage=0,
                            prior_store=repair_memory,
                            failure_notes=["Isabelle rejected a command before the first remaining hole."],
                        )
                    except Exception as ex:
                        if trace:
                            print(f"[repair] Earliest-error repair failed: {type(ex).__name__}: {ex}")
                        patched, applied = full, False
                    if applied and patched != full and _verify_full_proof(isa, session, patched):
                        full = patched
                        fills.append("[repair]\n[earliest-error]")
                        _record_local_repair(repair_reason)
                        continue
                if _regenerate_or_reoutline():
                    continue
                break

            # Source order is intentional: never repair a later symptom while
            # an earlier admission remains unresolved.
            span = spans[0]

            hole_key = _hole_fingerprint(full, span)
            divisor = max(1, min(len(spans), 3))
            remaining = left_s()
            reserve = (
                min(30.0, max(8.0, remaining * 0.25))
                if repairs or whole_repair or whole_fallback
                else 2.0
            )
            searchable = max(4.0, remaining - reserve)
            per_hole_budget = int(max(4, min(searchable, searchable / divisor)))
            before_fill = full
            try:
                full2, ok, script = _fill_one_hole(
                    isa,
                    session,
                    full,
                    span,
                    goal_text,
                    model,
                    per_hole_timeout=per_hole_budget,
                    fill_beam=fill_beam,
                    fill_depth=fill_depth,
                    fill_facts_limit=fill_facts_limit,
                    fill_sledge=fill_sledge,
                    fill_sledge_timeout=fill_sledge_timeout,
                    trace=trace,
                )
            except (TimeoutError, _FuturesTimeout, ValueError) as ex:
                _restart_isabelle("fill_one_hole", ex)
                full2, ok, script = full, False, "fill-exception"
            except Exception as ex:
                if trace:
                    print(f"[fill] _fill_one_hole crashed: {type(ex).__name__}: {ex}")
                full2, ok, script = full, False, "fill-exception"

            if ok and _verified_hole_reduction(before_fill, full2):
                full = full2
                if hole_key in surgery_hole_keys:
                    fills.append("[repair]\n[surgery]\n" + script)
                    surgery_hole_keys.discard(hole_key)
                    metrics.local_repairs += 1
                else:
                    fills.append(script)
                partial_tries.pop(hole_key, None)
                continue

            if script.startswith("[contextual-partial]") and full2 != before_fill \
                    and len(find_sorry_spans(full2)) == len(find_sorry_spans(before_fill)) \
                    and _verify_full_proof(isa, session, full2):
                partial_tries[hole_key] = partial_tries.get(hole_key, 0) + 1
                full = full2
                if trace:
                    print(
                        f"[fill] Accepted exact-state partial progress for @{hole_key} "
                        f"(step {partial_tries[hole_key]})."
                    )
                # Recompute and revisit the first source hole immediately.
                continue

            # Local CEGIS is now tried once as a bounded search.  Its stages
            # only return changes that close at least one hole.
            if repairs and left_s() > 6.0:
                try:
                    state = _print_state_before_hole(isa, session, full, span, trace)
                    eff_goal = _effective_goal_from_state(state, goal_text, full, span, trace)
                except (TimeoutError, _FuturesTimeout, ValueError) as ex:
                    _restart_isabelle("print_state_before_hole", ex)
                    state, eff_goal = "", goal_text
                except Exception as ex:
                    if trace:
                        print(f"[repair] Could not compute effective goal: {type(ex).__name__}: {ex}")
                    state, eff_goal = "", goal_text

                repair_reason = ""
                try:
                    patched, applied, repair_reason = try_cegis_repairs(
                        full_text=full, hole_span=span, goal_text=eff_goal, model=model,
                        isabelle=isa, session=session,
                        repair_budget_s=min(30.0, max(15.0, left_s() * 0.33)),
                        max_ops_to_try=max_repairs_per_hole, beam_k=repair_beam_k,
                        allow_whole_fallback=False, trace=trace, resume_stage=0,
                        prior_store=repair_memory,
                        failure_notes=[f"Fill failed at the earliest hole: {script}"],
                    )
                except (TimeoutError, _FuturesTimeout, ValueError) as ex:
                    _restart_isabelle("try_cegis_repairs", ex)
                    patched, applied = full, False
                except Exception as ex:
                    if trace:
                        print(f"[repair] try_cegis_repairs crashed: {type(ex).__name__}: {ex}")
                    patched, applied = full, False

                if applied and _verified_frontier_progress(full, patched):
                    before_repair = full
                    full = patched
                    fills.append("[repair]")
                    _record_local_repair(repair_reason)
                    partial_tries.clear()
                    if trace:
                        print(
                            f"[repair] Local/subproof repair advanced source frontier "
                            f"{frontier_score(before_repair)} -> {frontier_score(full)}."
                        )
                    continue

            if _regenerate_or_reoutline():
                continue

            if trace:
                print(
                    f"[planner] Earliest hole @{hole_key} remains blocked after "
                    f"Fill and all enabled repair stages; stopping at source frontier."
                )
            break

        # A generated outline can contain an unhelpful or even false
        # intermediate statement.  Before returning holes, give the strong
        # theorem-level prover the remaining budget and accept its result only
        # after complete verification.  This is a general portfolio fallback,
        # not a theorem-specific template.
        if whole_fallback and find_sorry_spans(full) and left_s() > 5.0:
            direct_budget = int(max(5, min(35, left_s())))
            if trace:
                print(
                    f"[planner] structured fill exhausted; trying verified "
                    f"whole-goal fallback ({direct_budget}s)."
                )
            try:
                direct = prove_goal(
                    isa,
                    session,
                    goal_text,
                    model_name_or_ensemble=model,
                    beam_w=max(1, fill_beam),
                    max_depth=max(1, fill_depth),
                    hint_lemmas=6,
                    timeout=direct_budget,
                    models=None,
                    save_dir=None,
                    use_sledge=fill_sledge,
                    sledge_timeout=max(1, fill_sledge_timeout),
                    sledge_every=1,
                    trace=trace,
                    use_color=False,
                    use_qc=True,
                    qc_timeout=2,
                    qc_every=1,
                    use_np=True,
                    np_timeout=5,
                    np_every=2,
                    facts_limit=max(1, fill_facts_limit),
                    do_minimize=False,
                    minimize_timeout=8,
                    do_variants=True,
                    variant_timeout=6,
                    variant_tries=24,
                    enable_reranker=True,
                    theory_preamble=_theory_preamble_from_text(full),
                )
                if bool(direct.get("success")):
                    direct_text = "\n".join(str(s) for s in direct.get("steps", []))
                    if direct_text and not direct_text.endswith("\n"):
                        direct_text += "\n"
                    if direct_text and not find_sorry_spans(direct_text) \
                            and _verify_full_proof(isa, session, direct_text):
                        fills.append("[whole-goal]\n" + direct_text.strip())
                        if direct.get("origin") == "definition-obligation":
                            metrics.verified_aux_obligations += 1
                        return _result(True, direct_text, fills, [])
            except (TimeoutError, _FuturesTimeout, ValueError) as ex:
                _restart_isabelle("whole_goal_fallback", ex)
            except Exception as ex:
                if trace:
                    print(
                        f"[planner] Whole-goal fallback failed: "
                        f"{type(ex).__name__}: {ex}"
                    )

        # Final verification
        success = not find_sorry_spans(full)
        if success:
            try:
                if _verify_full_proof(isa, session, full):
                    return _result(True, full, fills, [])
            except (TimeoutError, _FuturesTimeout, ValueError) as ex:
                _restart_isabelle("final_verify_full_proof", ex)

        failed = list(range(len(find_sorry_spans(full))))
        return _result(False, full, fills, failed)

    finally:
        if owns_isabelle:
            _cleanup_resources(isa, proc)
