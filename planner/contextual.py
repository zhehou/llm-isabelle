from __future__ import annotations

"""Exact-context proof search for planner holes.

The ordinary prover starts from a fresh ``lemma`` command.  That is ideal for
top-level goals, but it loses case assumptions, local fact names, fixes and
proof mode when it is used to fill an Isar ``sorry``.  This module reuses the
prover's candidate generators while evaluating every command after splicing it
into the complete outline at the target hole.
"""

import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from prover.config import MODEL as DEFAULT_MODEL
from prover.isabelle_api import (
    build_theory,
    run_theory,
)
from prover.llm import propose_finishers, propose_steps
from prover.tactics import (
    _extract_sledge_by_lines,
    _iter_note_messages,
    mine_facts_prioritized,
    variant_step_templates,
    fact_composition_finishers,
)
from prover.utils import parse_subgoals, state_fingerprint
from prover.strategies import (
    definition_guided_finishers,
    definition_guided_helper_scripts,
    definition_guided_steps,
    generic_structural_finishers,
    generic_structural_steps,
)
from planner.goals import _print_state_before_hole, _verify_full_proof


_NAMED_FACT_RE = re.compile(
    r"(?m)^\s*(?:have|assume|note|define)\s+([A-Za-z_][A-Za-z0-9_']*)\s*:"
)
_PROOF_HEAD_RE = re.compile(
    r"(?m)^(?P<indent>[ \t]*)(?:have|show|thus|hence|obtain)\b[^\n]*$"
)
_CASE_LINE_RE = re.compile(
    r"(?m)^(\s*)case\s+(?:\()?([A-Za-z_][A-Za-z0-9_']*)"
)


@dataclass(slots=True)
class ContextFillResult:
    text: str
    success: bool
    script: str
    reason: str
    initial_rank: Tuple[int, int]
    final_rank: Tuple[int, int]


@dataclass(slots=True)
class _Probe:
    ok: bool
    text: str
    state: str
    n_subgoals: Optional[int]
    rank: Tuple[int, int]
    error: str = ""


def _line_indent_at(text: str, offset: int) -> str:
    line_start = text.rfind("\n", 0, max(0, offset)) + 1
    line = text[line_start:offset]
    return line[: len(line) - len(line.lstrip(" "))]


def splice_hole_commands(
    full_text: str,
    hole_span: Tuple[int, int],
    commands: Sequence[str],
) -> str:
    """Replace the target ``sorry`` token with equally indented commands."""
    s, e = hole_span
    if not (0 <= s <= e <= len(full_text)):
        return full_text
    cleaned = [str(c).strip() for c in commands if str(c).strip()]
    if not cleaned:
        return full_text
    line_start = full_text.rfind("\n", 0, s) + 1
    line_prefix = full_text[line_start:s]
    base_indent = _line_indent_at(full_text, s)
    prefix_end = s
    if line_prefix.strip():
        # Inline leaf such as ``show ?case sorry``.  Proof commands must begin
        # on a new, nested line rather than producing ``show ?case apply ...``.
        indent = base_indent + "  "
        while prefix_end > line_start and full_text[prefix_end - 1] in " \t":
            prefix_end -= 1
        replacement = "\n" + indent + cleaned[0]
        replacement += "".join("\n" + indent + c for c in cleaned[1:])
    else:
        indent = base_indent
        replacement = cleaned[0] + "".join("\n" + indent + c for c in cleaned[1:])
    return full_text[:prefix_end] + replacement + full_text[e:]


_LOCAL_MODIFIER_BLOCK_RE = re.compile(
    r"(?ms)(?:^[ \t]*(?:using|unfolding)\b[^\n]*\n(?:^[ \t]*\n)*)+$"
)


def _drop_local_modifiers(
    full_text: str,
    hole_span: Tuple[int, int],
) -> Tuple[str, Tuple[int, int], bool]:
    """Drop the modifier block immediately before a proof-method hole.

    ``using`` and ``unfolding`` affect the method that follows them.  A fresh
    or repaired method must sometimes replace that bundle rather than inherit
    a stale fact selection.  The returned span points at the same hole in the
    shortened document.
    """
    s, e = hole_span
    line_start = full_text.rfind("\n", 0, max(0, s)) + 1
    match = _LOCAL_MODIFIER_BLOCK_RE.search(full_text[:line_start])
    if not match:
        return full_text, hole_span, False
    removed = line_start - match.start()
    shortened = full_text[:match.start()] + full_text[line_start:]
    return shortened, (s - removed, e - removed), True


def splice_hole_command_variants(
    full_text: str,
    hole_span: Tuple[int, int],
    commands: Sequence[str],
) -> List[Tuple[str, bool]]:
    """Return normal and modifier-replacing splices, in that order."""
    normal = splice_hole_commands(full_text, hole_span, commands)
    variants: List[Tuple[str, bool]] = [(normal, False)]
    shortened, shifted, dropped = _drop_local_modifiers(full_text, hole_span)
    if dropped:
        replacement = splice_hole_commands(shortened, shifted, commands)
        if replacement != normal:
            variants.append((replacement, True))
    return variants


def _state_complexity(state: str) -> int:
    """Cheap, deterministic tie-breaker below Isabelle's subgoal count."""
    clean = " ".join((state or "").split())
    if not clean:
        return 10**6
    logical = sum(clean.count(tok) for tok in ("⟹", "∧", "∨", "∀", "∃", " = "))
    return min(10**6, len(clean) + 20 * logical)


def _rank(state: str, n_subgoals: Optional[int]) -> Tuple[int, int]:
    return (n_subgoals if isinstance(n_subgoals, int) else 9999, _state_complexity(state))


def _structural_progress(before: _Probe, after: _Probe) -> bool:
    """Return true only for proof-state progress that Isabelle can justify.

    Pretty-printed goal length is useful for beam ordering, but is not a sound
    progress criterion: simplification can merely rename or reformat one
    unsolved goal.  A resumable partial script must discharge at least one
    subgoal.  Same-cardinality rewrites stay inside the beam and may still lead
    to a complete proof, but are never committed back to the planner.
    """
    return (
        before.ok
        and after.ok
        and isinstance(before.n_subgoals, int)
        and isinstance(after.n_subgoals, int)
        and after.n_subgoals < before.n_subgoals
    )


def _local_fact_hints(full_text: str, hole_span: Tuple[int, int], limit: int) -> List[str]:
    prefix = full_text[: hole_span[0]]
    # Do not expose the name of the fact whose proof is the current hole.  A
    # declaration such as ``have take_map: ... sorry`` is not in the local
    # context yet.  Replaying ``take_map`` can accidentally resolve an
    # unrelated global theorem of the same name and makes benchmark results
    # look far stronger than the actual filler.  Compare source lines rather
    # than statement text so multi-line propositions remain harmless.
    current_head_line: Optional[int] = None
    proof_heads = list(_PROOF_HEAD_RE.finditer(prefix))
    if proof_heads:
        current_head_line = prefix.count("\n", 0, proof_heads[-1].start())

    facts: List[str] = []
    for match in reversed(list(_NAMED_FACT_RE.finditer(prefix))):
        fact_line = prefix.count("\n", 0, match.start())
        if current_head_line is not None and fact_line == current_head_line:
            continue
        facts.append(match.group(1))
    case_matches = list(_CASE_LINE_RE.finditer(prefix))
    if case_matches:
        # The active nested cases are the most recent case at each indentation
        # level.  This excludes a preceding sibling such as ``case 0`` while
        # retaining its enclosing ``case Cons`` for a later ``case Suc``.
        latest_by_indent: Dict[int, str] = {}
        for match in case_matches:
            latest_by_indent[len(match.group(1).expandtabs(2))] = match.group(2)
        active_cases = [
            latest_by_indent[indent]
            for indent in sorted(latest_by_indent, reverse=True)
        ]
        # Bare case facts carry equations selected by ``then``.  Aliases cover
        # premises and induction hypotheses.  Exact verification filters any
        # alias not supplied by a particular case package.
        # Round-robin aliases by kind rather than exhausting every alias of
        # the innermost case first.  In a nested ``Cons``/``Suc`` proof the
        # useful pair is commonly ``Suc`` + ``Cons.IH``; the old ordering put
        # ``Cons.IH`` beyond the replay cutoff.  Explicitly named local facts
        # also precede speculative aliases because they are guaranteed to be
        # in scope.  All candidates remain subject to complete-proof checking.
        aliases = [
            f"{case}.{suffix}"
            for suffix in ("prems", "IH", "hyps")
            for case in active_cases
        ]
        facts = [*active_cases, *facts, *aliases]
    return list(dict.fromkeys(facts))[: max(0, limit)]


_FACT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)?$")


def _fact_replay_groups(facts: Sequence[str], limit: int = 16) -> List[str]:
    """Build small, ordered fact portfolios for replay in the real context.

    A proof of the exported standalone goal may depend on several premises at
    once.  Replaying just the first named fact is therefore incomplete.  The
    portfolios below prioritise the combinations created by nested Isar
    cases: the innermost case equation together with an outer induction
    hypothesis/premise, followed by named local facts and individual facts.

    Undefined speculative aliases are harmless because every resulting
    method is checked in the complete Isar document before it is accepted.
    """
    ordered = [
        str(f).strip() for f in facts
        if _FACT_NAME_RE.fullmatch(str(f).strip())
    ]
    ordered = list(dict.fromkeys(ordered))
    if not ordered or limit <= 0:
        return []

    dotted_prefixes = {
        fact.split(".", 1)[0] for fact in ordered if "." in fact
    }
    case_names = [
        fact for fact in ordered
        if "." not in fact and fact in dotted_prefixes
    ]
    named = [
        fact for fact in ordered
        if "." not in fact and fact not in case_names
    ]
    groups: List[str] = []

    def add(*items: str) -> None:
        group = " ".join(dict.fromkeys(x for x in items if x))
        if group and group not in groups:
            groups.append(group)

    if case_names:
        inner = case_names[0]
        outer = case_names[1:]
        # Outer IHs are the highest-value facts in nested induction/case
        # leaves; outer premises are next.  Pair them with the inner equation.
        for suffix in ("IH", "prems", "hyps"):
            for case in outer:
                alias = f"{case}.{suffix}"
                if alias in ordered:
                    add(inner, alias)
        for fact in named:
            add(inner, fact)
        add(inner)
        for case in outer:
            add(case)

    # In assumption-style and calculation proofs there is no case package.
    # The useful evidence is commonly the two most recent named facts (for
    # example two inequalities used for transitivity).  Put their small
    # conjunction portfolio before singles so a tight fast-path limit does
    # not silently discard the only useful combination.
    if not case_names and len(named) >= 2:
        add(named[0], named[1])
        if len(named) >= 3:
            add(named[0], named[1], named[2])

    for fact in named:
        add(fact)
    for fact in ordered:
        add(fact)

    # A few general pairs cover two named assumptions without exploding the
    # verifier portfolio quadratically.
    pair_pool = ordered[:6]
    for i, left in enumerate(pair_pool):
        for right in pair_pool[i + 1:]:
            add(left, right)
            if len(groups) >= limit:
                return groups[:limit]
    return groups[:limit]


def _responses_text(responses: Iterable[object]) -> str:
    try:
        return "\n".join(str(x) for x in _iter_note_messages(responses))
    except Exception:
        return "\n".join(str(x) for x in (responses or []))


def fill_hole_in_context(
    isabelle,
    session: str,
    full_text: str,
    hole_span: Tuple[int, int],
    goal_text: str,
    model: Optional[str],
    timeout_s: int,
    *,
    beam_w: int = 4,
    max_depth: int = 8,
    facts_limit: int = 8,
    use_sledge: bool = True,
    sledge_timeout: int = 10,
    trace: bool = False,
) -> ContextFillResult:
    """Search for a continuation while checking every step in the real outline.

    A successful result removes the target hole.  On failure, a verified best
    partial continuation is returned only when it strictly improves the exact
    state rank, allowing the driver to continue from genuine progress.
    """
    started = time.monotonic()
    budget = max(1, int(timeout_s))

    def left() -> float:
        return max(0.0, budget - (time.monotonic() - started))

    cache: Dict[Tuple[str, ...], _Probe] = {}

    def target_sorry_span(text: str) -> Optional[Tuple[int, int]]:
        spans = [(m.start(), m.end()) for m in re.finditer(r"\bsorry\b", text)]
        if not spans:
            return None
        return min(spans, key=lambda span: abs(span[0] - hole_span[0]))

    def probe(continuation: Sequence[str]) -> _Probe:
        key = tuple(continuation)
        if key in cache:
            return cache[key]
        if continuation:
            candidate = splice_hole_commands(
                full_text, hole_span, [*continuation, "sorry"]
            )
            target = target_sorry_span(candidate)
        else:
            # Inspect the real admitted outline directly.  Re-splicing the
            # same token can shift an inline hole or select a neighbouring
            # ``sorry`` in multi-hole outlines.
            candidate = full_text
            target = hole_span
        try:
            if target is None:
                raise ValueError("instrumented hole disappeared")
            # Inspect the prefix immediately before the admitted command.  The
            # previous implementation inserted ``apply -`` and ``print_state``
            # into the proof itself, which is illegal for some Isabelle proof
            # modes.  The shared ML-assisted extractor preserves the exact
            # local context without guessing the current outer-syntax mode.
            state = _print_state_before_hole(
                isabelle,
                session,
                candidate,
                target,
                trace=False,
                timeout_s=max(1, int(min(left() or 1.0, 15.0))),
            )
            n_subgoals = parse_subgoals(state)
            ok = bool(state.strip()) and isinstance(n_subgoals, int)
            error = "" if ok else (
                "empty-state" if not state.strip() else "unparsed-subgoal-count"
            )
            out = _Probe(
                ok, candidate, state, n_subgoals,
                _rank(state, n_subgoals), error,
            )
        except Exception as exc:
            out = _Probe(
                False, candidate, "", None, (9999, 10**6),
                f"{type(exc).__name__}: {exc}",
            )
        cache[key] = out
        return out

    def try_finish(continuation: Sequence[str], finisher: str) -> Optional[str]:
        if "sorry" in finisher:
            return None
        commands = [*continuation, finisher]
        for candidate, dropped in splice_hole_command_variants(
            full_text, hole_span, commands
        ):
            timeout = max(1, int(min(left() or 1.0, 10.0)))
            if _verify_full_proof(
                isabelle, session, candidate, timeout_s=timeout
            ):
                return candidate
        return None

    def try_method(
        continuation: Sequence[str],
        method: str,
        *,
        include_apply_done: bool = True,
    ) -> Tuple[Optional[str], str]:
        """Try equivalent Isar encodings for prove and state modes."""
        method = method.strip()
        if method.startswith("by "):
            body = method[3:].strip()
        elif method.startswith("apply "):
            body = method[6:].strip()
        else:
            body = method
        scripts = [[*continuation, f"by {body}"]]
        if include_apply_done:
            scripts.append([*continuation, f"apply {body}", "done"])
        for script in scripts:
            solved = try_finish(script[:-1], script[-1])
            if solved is not None:
                return solved, "\n".join(script)
        return None, ""

    def contextual_sledge(continuation: Sequence[str]) -> List[str]:
        if not use_sledge or left() <= 2.0:
            return []
        command = f"sledgehammer [timeout = {max(1, min(sledge_timeout, int(left())))}]"
        candidate = splice_hole_commands(
            full_text, hole_span, [*continuation, command, "sorry"]
        )
        try:
            thy = build_theory(candidate.splitlines(), add_print_state=False, end_with=None)
            responses = run_theory(
                isabelle,
                session,
                thy,
                timeout_s=max(1, min(sledge_timeout + 3, int(left()) or 1)),
            )
            return _extract_sledge_by_lines(_responses_text(responses))[:5]
        except Exception:
            return []

    deterministic_finishers = [
        "by assumption",
        "by simp",
        "by clarsimp",
        "by auto",
        "by fastforce",
        "by blast",
    ]

    local_facts = _local_fact_hints(full_text, hole_span, facts_limit)

    # Reserve the exact proof state before method trials consume the contextual
    # budget.  This state drives theorem mining and multi-step search even when
    # every cheap finisher fails.
    initial = probe([])
    if trace and not initial.ok:
        print(
            f"[fill-context] initial probe unavailable: {initial.error}; "
            f"state_length={len(initial.state)}"
        )

    # Fast path that requires no successful state parse.  Put explicit local
    # facts ahead of the broad automation portfolio: structured leaves often
    # differ from an easy theorem only because their premises have case names.
    priority_finishers = ["by assumption", "by simp", "by auto"]

    # Treat a top-level recursive hole as a strategy-selection problem, not a
    # long sequence of unrelated tactic guesses.  The source declaration tells
    # us which argument is structurally decreasing; candidates are still
    # replayed in the exact Isar context and accepted only after full checking.
    prefix = full_text[:hole_span[0]]
    lemma_matches = list(re.finditer(
        r"(?m)^\s*(?:lemma|theorem|corollary)\b", prefix
    ))
    lemma_start = lemma_matches[-1].start() if lemma_matches else len(prefix)
    local_source = prefix[lemma_start:]
    inside_case = bool(re.search(r"(?m)^\s*case\b", local_source))
    preamble = prefix[:lemma_start].strip()
    guided_finishers = [] if inside_case else definition_guided_finishers(
        goal_text, preamble, initial.state, limit=4
    )
    standard_finishers = [] if inside_case else generic_structural_finishers(
        goal_text, initial.state, limit=4
    )
    composition_finishers = fact_composition_finishers(
        f"{goal_text}\n{initial.state}", local_facts, limit=5
    )
    helper_scripts = [] if inside_case else definition_guided_helper_scripts(
        goal_text, preamble, initial.state, limit=2
    )
    priority_finishers.extend(guided_finishers)
    priority_finishers.extend(standard_finishers)
    priority_finishers.extend(composition_finishers)
    priority_finishers = list(dict.fromkeys(priority_finishers))
    fast_finishers: List[str] = []
    if local_facts:
        # Replay premise *portfolios*, not just individual names.  Interleave
        # auto and simp so witness/subset reasoning and constructor equations
        # both get a bounded opportunity before generative search.
        for index, group in enumerate(_fact_replay_groups(local_facts, limit=6)):
            fast_finishers.extend([
                f"by (use {group} in auto)",
                f"by (use {group} in simp)",
            ])
            # Stronger combinators are reserved for the highest-ranked fact
            # bundles.  This closes transitivity/witness leaves without
            # multiplying every verifier call in a large context.
            if index < 2:
                fast_finishers.extend([
                    f"by (use {group} in fastforce)",
                    f"by (use {group} in blast)",
                ])
    fast_finishers.extend(["by clarsimp", *deterministic_finishers[4:]])

    # Direct closure and definition-derived induction run first.  Reserve one
    # verifier transaction for a synthesized helper lemma when applicable.
    for finisher in priority_finishers:
        search_reserve = min(6.0, budget * 0.35) if initial.ok else 0.0
        helper_reserve = 4.0 if helper_scripts else 0.5
        if left() <= search_reserve + helper_reserve:
            break
        # At an admitted Isar leaf, ``by`` is the native encoding.  Trying an
        # equivalent apply/done pair after every failed method doubled replay
        # cost and exhausted the budget before stateful search could begin.
        solved, verified_script = try_method(
            [], finisher, include_apply_done=False
        )
        if solved is not None:
            if trace:
                if finisher in guided_finishers:
                    source = "definition strategy"
                elif finisher in standard_finishers:
                    source = "library structural strategy"
                elif finisher in composition_finishers:
                    source = "fact composition"
                else:
                    source = "exact-context fast path"
                print(f"[fill-context] solved by {source}: {verified_script}")
            if finisher in guided_finishers:
                reason = "context-definition-verified"
            elif finisher in standard_finishers:
                reason = "context-library-structure-verified"
            elif finisher in composition_finishers:
                reason = "context-fact-composition-verified"
            else:
                reason = "context-fast-verified"
            return ContextFillResult(
                solved, True, verified_script, reason,
                (9999, 10**6), (0, 0)
            )

    for script in helper_scripts:
        search_reserve = min(6.0, budget * 0.35) if initial.ok else 0.0
        if left() <= search_reserve + 0.5:
            break
        solved = try_finish(script[:-1], script[-1])
        if solved is not None:
            verified_script = "\n".join(script)
            if trace:
                print(f"[fill-context] solved by synthesized helper lemma:\n{verified_script}")
            return ContextFillResult(
                solved, True, verified_script, "context-helper-verified",
                (9999, 10**6), (0, 0),
            )

    # Fact replay and the remaining broad methods follow structural search.
    # This retains exact local context on Isabelle-client variants where the
    # inserted ``print_state`` message is not exposed in the expected shape.
    for finisher in fast_finishers:
        search_reserve = min(6.0, budget * 0.35) if initial.ok else 0.0
        if left() <= search_reserve + 0.5:
            break
        solved, verified_script = try_method(
            [], finisher, include_apply_done=False
        )
        if solved is not None:
            if trace:
                print(f"[fill-context] solved by exact-context fast path: {verified_script}")
            return ContextFillResult(
                solved, True, verified_script, "context-fast-verified",
                (9999, 10**6), (0, 0),
            )

    if not initial.ok:
        # If state extraction itself is unavailable, retain a small
        # compatibility fallback for clients/proof modes that require
        # apply/done rather than a terminal ``by`` command.
        for finisher in deterministic_finishers[:3]:
            body = finisher[3:].strip() if finisher.startswith("by ") else finisher
            solved = try_finish([f"apply {body}"], "done")
            if solved is not None:
                script = f"apply {body}\ndone"
                return ContextFillResult(
                    solved, True, script, "context-apply-done-verified",
                    initial.rank, (0, 0),
                )
        return ContextFillResult(
            full_text, False, "",
            f"context-probe-failed:{initial.error or 'unknown'}",
            initial.rank, initial.rank,
        )

    model_list = [str(model or DEFAULT_MODEL)]
    beam: List[Tuple[List[str], _Probe]] = [([], initial)]
    best_commands: List[str] = []
    best_probe = initial
    seen_states = {state_fingerprint(initial.state or "")}

    for depth in range(max(1, max_depth)):
        if left() <= 1.0 or not beam:
            break
        next_nodes: List[Tuple[List[str], _Probe]] = []

        for node_index, (continuation, current) in enumerate(beam):
            if left() <= 1.0:
                break

            # Isabelle's cheap deterministic automation wins surprisingly
            # often, especially for the leaves of induction and calculation
            # outlines.  Try it before fact mining, Sledgehammer or an LLM
            # request so easy holes cost only a few local verifier calls.
            # The empty continuation was already tried by the fast path.
            cheap_finishers = list(deterministic_finishers) if continuation else []
            if current.n_subgoals == 0:
                cheap_finishers.insert(0, "done")
            for finisher in cheap_finishers:
                if left() <= 0.5:
                    break
                solved, verified_script = try_method(continuation, finisher)
                if solved is not None:
                    script = verified_script
                    if trace:
                        print(
                            f"[fill-context] solved by deterministic automation "
                            f"at depth {depth + 1}:\n{script}"
                        )
                    return ContextFillResult(
                        solved, True, script, "context-verified", initial.rank, (0, 0)
                    )

            # On an outer theorem hole with no local assumptions, typed
            # induction/case moves are the deterministic analogue of a root
            # decision.  Explore them before remote generation so a slow model
            # cannot consume the budget before Isabelle sees the obvious
            # recursive split.  Branch leaves are handled on the next depth.
            structural_children = False
            if depth == 0 and not continuation and "[LLM_ASSM]" not in current.state:
                structural_commands = definition_guided_steps(
                    goal_text, preamble, current.state, limit=4
                )
                structural_commands.extend(
                    generic_structural_steps(goal_text, current.state, limit=4)
                )
                structural_commands.extend(
                    variant_step_templates(current.state, limit=4)
                )
                for command in dict.fromkeys(structural_commands):
                    if left() <= 1.0:
                        break
                    child_commands = [command]
                    solved = try_finish(child_commands, "done")
                    if solved is not None:
                        script = "\n".join([command, "done"])
                        return ContextFillResult(
                            solved, True, script, "context-step-verified",
                            initial.rank, (0, 0),
                        )
                    child = probe(child_commands)
                    if not child.ok:
                        continue
                    fp = state_fingerprint(child.state or "")
                    if fp in seen_states:
                        continue
                    seen_states.add(fp)
                    next_nodes.append((child_commands, child))
                    structural_children = True
                    if child.rank < best_probe.rank:
                        best_commands, best_probe = child_commands, child
            if structural_children:
                # Preserve budget for simplifying the generated branches.
                continue

            facts = list(local_facts)
            try:
                mined = mine_facts_prioritized(
                    isabelle,
                    session,
                    current.state,
                    limit=max(0, facts_limit),
                    timeout_s=max(1, int(min(5.0, max(1.0, left() * 0.35)))),
                )
                # Reserve room for theorem discovery; speculative case aliases
                # must not crowd out relevant library elimination rules.
                local_quota = max(1, facts_limit // 2)
                facts = list(dict.fromkeys(
                    facts[:local_quota] + mined
                ))[: max(0, facts_limit)]
            except Exception:
                pass

            # Named local facts are often the only difference between a leaf
            # that simp can close and one it cannot.  Exercise them
            # deterministically before asking a model.  Every candidate is
            # still checked in the complete outline, so stale case fact names
            # are harmless.
            replay_groups = _fact_replay_groups(facts, limit=10)
            for finisher in fact_composition_finishers(
                f"{goal_text}\n{current.state}", facts, limit=5
            ):
                if left() <= 0.5:
                    break
                solved, verified_script = try_method(continuation, finisher)
                if solved is not None:
                    if trace:
                        print(
                            "[fill-context] solved by fact composition: "
                            f"{verified_script}"
                        )
                    return ContextFillResult(
                        solved, True, verified_script,
                        "context-fact-composition-verified",
                        initial.rank, (0, 0),
                    )
            if replay_groups and left() > 0.5:
                for group_index, grouped_facts in enumerate(replay_groups):
                    if left() <= 0.5:
                        break
                    methods = ("fastforce", "blast") if group_index < 3 else ("blast",)
                    for method in methods:
                        finisher = f"by (use {grouped_facts} in {method})"
                        solved, verified_script = try_method(continuation, finisher)
                        if solved is not None:
                            if trace:
                                print(
                                    "[fill-context] solved with verified fact group "
                                    f"{grouped_facts}:\n{verified_script}"
                                )
                            return ContextFillResult(
                                solved, True, verified_script, "context-facts-verified",
                                initial.rank, (0, 0),
                            )

            for fact in facts[:6]:
                if left() <= 0.5:
                    break
                for finisher in (
                    f"by (auto simp add: {fact})",
                    f"by (metis {fact})",
                ):
                    solved, verified_script = try_method(continuation, finisher)
                    if solved is not None:
                        if trace:
                            print(
                                "[fill-context] solved with verified local fact "
                                f"{fact}:\n{verified_script}"
                            )
                        return ContextFillResult(
                            solved, True, verified_script, "context-fact-verified",
                            initial.rank, (0, 0),
                        )

            # Generative fallback must see the same local definitions as
            # Isabelle.  Supplying only the lemma made fresh recursive
            # constants semantically opaque to the model, even though the
            # verifier had their equations in scope.
            prompt_steps = [
                *preamble.splitlines(),
                f'lemma "{goal_text}"',
                *continuation,
            ]
            finishers = contextual_sledge(continuation) if node_index == 0 else []
            finishers += propose_finishers(
                model_list,
                goal_text,
                prompt_steps,
                current.state,
                [],
                0,
                facts=facts,
                allow_done=bool(current.n_subgoals == 0),
                temp=min(0.55, 0.2 + 0.05 * depth),
                reranker=None,
                timeout_s=max(1, int(min(8.0, max(1.0, left() * 0.40)))),
            )
            finishers = list(dict.fromkeys(finishers))
            for finisher in finishers:
                if left() <= 0.5:
                    break
                solved, verified_script = try_method(continuation, finisher)
                if solved is not None:
                    script = verified_script
                    if trace:
                        print(f"[fill-context] solved at depth {depth + 1}:\n{script}")
                    return ContextFillResult(
                        solved, True, script, "context-verified", initial.rank, (0, 0)
                    )

            proposals = definition_guided_steps(
                goal_text, preamble, current.state, limit=4
            )
            proposals += generic_structural_steps(
                goal_text, current.state, limit=4
            )
            proposals += variant_step_templates(current.state, limit=4)
            proposals += propose_steps(
                model_list,
                goal_text,
                prompt_steps,
                current.state,
                facts=facts,
                reranker=None,
                depth=depth,
                temp=min(0.9, 0.5 + 0.08 * depth),
                timeout_s=max(1, int(min(8.0, max(1.0, left() * 0.40)))),
            )
            for command in list(dict.fromkeys(proposals)):
                if left() <= 0.5:
                    break
                child_commands = [*continuation, command]
                # An apply-style proposal may itself discharge every subgoal.
                # Probe instrumentation would then be illegal because there
                # is no remaining proof state, so test ``done`` first.
                solved = try_finish(child_commands, "done")
                if solved is not None:
                    script = "\n".join([*child_commands, "done"])
                    if trace:
                        print(f"[fill-context] proposal closed the proof:\n{script}")
                    return ContextFillResult(
                        solved, True, script, "context-step-verified",
                        initial.rank, (0, 0),
                    )
                child = probe(child_commands)
                if not child.ok:
                    continue
                fp = state_fingerprint(child.state or "")
                if fp in seen_states:
                    continue
                seen_states.add(fp)
                next_nodes.append((child_commands, child))
                if child.rank < best_probe.rank:
                    best_commands, best_probe = child_commands, child

        next_nodes.sort(key=lambda item: (item[1].rank, len(item[0])))
        beam = next_nodes[: max(1, beam_w)]

    if best_commands and _structural_progress(initial, best_probe):
        partial = splice_hole_commands(full_text, hole_span, [*best_commands, "sorry"])
        script = "\n".join(best_commands)
        if trace:
            print(
                f"[fill-context] verified partial progress {initial.rank} -> "
                f"{best_probe.rank}:\n{script}"
            )
        return ContextFillResult(
            partial,
            False,
            script,
            "context-partial-progress",
            initial.rank,
            best_probe.rank,
        )

    return ContextFillResult(
        full_text, False, "", "context-search-exhausted", initial.rank, best_probe.rank
    )


__all__ = [
    "ContextFillResult",
    "fill_hole_in_context",
    "_fact_replay_groups",
    "splice_hole_commands",
    "splice_hole_command_variants",
]
