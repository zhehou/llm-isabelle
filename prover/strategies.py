from __future__ import annotations

"""Definition-directed proof strategy synthesis.

The step prover normally sees only a proposition and a proof state.  For fresh
theories that throws away the most useful search signal: the equations of the
recursive constants occurring in the proposition.  This module recovers that
signal without relying on benchmark names.  It identifies the structurally
decreasing argument of each ``fun``/``primrec`` declaration, follows nested
calls in the goal, and proposes small *complete* induction portfolios.

All returned methods are merely candidates.  Isabelle remains the sole
checker, so an imperfect parse can cost an attempt but can never certify an
invalid proof.
"""

from dataclasses import dataclass, replace
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


_OUTER_COMMAND = re.compile(
    r"(?m)^\s*(?:datatype|codatatype|fun|function|primrec|definition|"
    r"abbreviation|inductive|inductive_set|lemma|theorem|corollary|end)\b"
)
_FUN_HEAD = re.compile(
    r"(?m)^\s*(?:fun|primrec)\s+([A-Za-z_][A-Za-z0-9_'.]*)\b"
)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_']*")
_TYPE_RECORD = re.compile(
    r"^\[LLM_TYPE\]\s+([A-Za-z][A-Za-z0-9_']*)\s+::\s+"
    r"(.*?)(?=^\[[A-Z0-9_]+\]|\Z)",
    re.M | re.S,
)

_RESERVED = {
    "and", "arbitrary", "assumes", "begin", "by", "case", "cases",
    "else", "end", "fixes", "for", "fun", "if", "in", "induction",
    "lemma", "let", "of", "primrec", "proof", "qed", "shows", "then",
    "theorem", "where", "True", "False",
}


@dataclass(frozen=True)
class RecursiveDefinition:
    name: str
    arity: int
    recursive_positions: Tuple[int, ...]
    equations: Tuple[str, ...]


@dataclass(frozen=True)
class StrategyPlan:
    target: str
    score: float
    generalize: Tuple[str, ...]
    sources: Tuple[str, ...]


@dataclass(frozen=True)
class AuxiliaryObligation:
    """A conjecture that must be proved before it may guide the main proof.

    ``main_methods`` contain an ``{aux}`` placeholder.  Keeping the auxiliary
    theorem and the consumer methods separate lets the prover verify an
    obligation DAG rather than gambling on one large generated script.
    """

    kind: str
    name_hint: str
    binders: Tuple[str, ...]
    body: str
    proof_methods: Tuple[str, ...]
    main_methods: Tuple[str, ...]
    score: float = 0.0

    @property
    def statement(self) -> str:
        """The locally usable, explicitly quantified proposition.

        Local Isar ``have`` blocks need explicit meta-quantification because
        the synthesized placeholder names are not part of the surrounding
        context.  A standalone outer ``lemma`` must use :attr:`body` instead:
        Isabelle then fixes the free variables before the proof method runs,
        so commands such as ``induction llm_xs`` see a genuine local variable.
        """

        if not self.binders:
            return self.body
        return f"⋀{' '.join(self.binders)}. {self.body}"

    @property
    def top_level_statement(self) -> str:
        """Binder-free statement suitable for an outer ``lemma`` command."""

        return self.body


# Recursive argument positions for common HOL library constants.  Fresh local
# definitions are recovered from their equations below; library constants have
# no source equations in the generated Scratch theory, so this small signature
# table supplies the corresponding structural signal.  It is deliberately
# about argument positions rather than theorem names or benchmark statements.
_STANDARD_RECURSIVE_SIGNATURES: Dict[str, Tuple[int, Tuple[int, ...]]] = {
    "append": (2, (0,)),
    "butlast": (1, (0,)),
    "concat": (1, (0,)),
    "distinct": (1, (0,)),
    "drop": (2, (1, 0)),
    "filter": (2, (1,)),
    "fold": (3, (1,)),
    "foldl": (3, (2,)),
    "foldr": (3, (1,)),
    "last": (1, (0,)),
    "length": (1, (0,)),
    "list_all": (2, (1,)),
    "list_all2": (3, (1, 2)),
    "map": (2, (1,)),
    "nth": (2, (0,)),
    "partition": (2, (1,)),
    "prod_list": (1, (0,)),
    "remove1": (2, (1,)),
    "replicate": (2, (0,)),
    "rev": (1, (0,)),
    "set": (1, (0,)),
    "sum_list": (1, (0,)),
    "take": (2, (1, 0)),
    "zip": (2, (0, 1)),
}


def _standard_recursive_definitions() -> Dict[str, RecursiveDefinition]:
    return {
        name: RecursiveDefinition(name, arity, positions, ())
        for name, (arity, positions) in _STANDARD_RECURSIVE_SIGNATURES.items()
    }


def _balanced_end(text: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    quoted = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index + 1
    return len(text)


def _application_args(rest: str, arity: int) -> List[str]:
    """Read Isabelle application atoms, respecting grouped terms."""

    args: List[str] = []
    index = 0
    pairs = {"(": ")", "[": "]", "{": "}"}
    while index < len(rest) and len(args) < arity:
        while index < len(rest) and rest[index].isspace():
            index += 1
        if index >= len(rest):
            break
        start = index
        if rest[index] in pairs:
            index = _balanced_end(rest, index, rest[index], pairs[rest[index]])
        else:
            while index < len(rest) and not rest[index].isspace():
                # An ungrouped meta/object connective terminates an
                # application.  Most declaration equations group compound
                # patterns, so keeping this conservative is preferable.
                if rest.startswith(("==>", "=>", "=", "\u27f9", "\u27f6"), index):
                    break
                index += 1
        atom = rest[start:index].strip()
        if not atom:
            break
        args.append(atom)
    return args


def _quoted_strings(block: str) -> Iterable[str]:
    index = 0
    while index < len(block):
        if block[index] != '"':
            index += 1
            continue
        index += 1
        chars: List[str] = []
        escaped = False
        while index < len(block):
            char = block[index]
            index += 1
            if escaped:
                chars.extend(("\\", char))
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                break
            else:
                chars.append(char)
        yield "".join(chars)


def _datatype_constructors(preamble: str) -> set[str]:
    constructors: set[str] = set()
    for match in re.finditer(
        r"(?ms)^\s*(?:datatype|codatatype)\b.*?=(.*?)(?="
        r"^\s*(?:datatype|codatatype|fun|primrec|definition|lemma|theorem|end)\b|\Z)",
        preamble or "",
    ):
        rhs = match.group(1)
        for branch in re.split(r"\s+\|\s+", rhs):
            head = _IDENT.search(branch.strip())
            if head:
                constructors.add(head.group(0))
    return constructors


def _constructor_strength(term: str, constructors: set[str]) -> float:
    stripped = term.strip()
    while stripped.startswith("(") and stripped.endswith(")"):
        stripped = stripped[1:-1].strip()
    if stripped in {"[]", "0"}:
        return 3.0
    if "#" in stripped or (stripped.startswith("[") and stripped.endswith("]")):
        return 3.0
    head = _IDENT.match(stripped)
    if head and (head.group(0) in constructors or head.group(0) == "Suc"):
        return 3.0
    # A plain variable contributes no structural evidence.
    return 0.0


def parse_recursive_definitions(preamble: str) -> Dict[str, RecursiveDefinition]:
    """Extract recursive argument positions from local defining equations."""

    text = preamble or ""
    constructors = _datatype_constructors(text)
    heads = list(_FUN_HEAD.finditer(text))
    definitions: Dict[str, RecursiveDefinition] = {}
    for head_index, head in enumerate(heads):
        name = head.group(1)
        next_outer = _OUTER_COMMAND.search(text, head.end())
        block_end = next_outer.start() if next_outer else len(text)
        # If another fun head begins before a generic outer-command match,
        # use it as the boundary (mainly defensive for unusual indentation).
        if head_index + 1 < len(heads):
            block_end = min(block_end, heads[head_index + 1].start())
        block = text[head.start():block_end]
        equations: List[str] = []
        lhs_args: List[List[str]] = []
        for quoted in _quoted_strings(block):
            if "=" not in quoted:
                continue
            lhs = quoted.split("=", 1)[0].strip()
            match = re.match(rf"{re.escape(name)}\b(.*)$", lhs, re.S)
            if not match:
                continue
            # Arity is not known yet.  A generous upper bound is harmless;
            # parsing stops at '=' because only the left side is supplied.
            args = _application_args(match.group(1), 32)
            if args:
                equations.append(quoted.strip())
                lhs_args.append(args)
        if not lhs_args:
            continue
        arity = max(len(args) for args in lhs_args)
        scores = [0.0] * arity
        coverage = [0] * arity
        for args in lhs_args:
            for position, term in enumerate(args):
                scores[position] += _constructor_strength(term, constructors)
                if _constructor_strength(term, constructors) > 0:
                    coverage[position] += 1
        if not any(scores):
            continue
        # Prefer a position that is structurally matched in the largest share
        # of equations.  This distinguishes the instruction stream from the
        # stack in interpreters where both occasionally use list patterns.
        ranked = sorted(
            range(arity),
            key=lambda pos: (
                coverage[pos] / max(1, len(lhs_args)), scores[pos], -pos
            ),
            reverse=True,
        )
        best = ranked[0]
        best_ratio = coverage[best] / max(1, len(lhs_args))
        positions = tuple(
            pos for pos in ranked
            if scores[pos] > 0
            and coverage[pos] / max(1, len(lhs_args)) >= max(0.75, best_ratio - 0.05)
        )
        definitions[name] = RecursiveDefinition(
            name=name,
            arity=arity,
            recursive_positions=positions or (best,),
            equations=tuple(equations),
        )
    return definitions


def _typed_names(state_hint: str) -> List[str]:
    names: List[str] = []
    for match in _TYPE_RECORD.finditer(state_hint or ""):
        name = match.group(1)
        typ = re.sub(r"\s+", " ", match.group(2)).strip()
        # Isabelle's escaped arrow is the common representation in traces.
        if any(arrow in typ for arrow in ("\u21d2", "=>", "\\<Rightarrow>")):
            continue
        names.append(name)
        trimmed = name.rstrip("_")
        if trimmed and trimmed != name:
            names.append(trimmed)
    return list(dict.fromkeys(names))


def _function_occurrences(expr: str, definition: RecursiveDefinition):
    pattern = re.compile(rf"\b{re.escape(definition.name)}\b")
    for match in pattern.finditer(expr or ""):
        args = _application_args(expr[match.end():], definition.arity)
        if len(args) >= definition.arity:
            yield match.start(), args


def _term_identifiers(term: str, known_constants: set[str]) -> List[str]:
    out: List[str] = []
    for ident in _IDENT.findall(term or ""):
        if ident in known_constants or ident in _RESERVED or ident[0].isupper():
            continue
        if ident not in out:
            out.append(ident)
    return out


def _all_freeish_identifiers(
    goal: str,
    definitions: Dict[str, RecursiveDefinition],
    state_hint: str,
) -> List[str]:
    typed = _typed_names(state_hint)
    goal_tokens = set(_IDENT.findall(goal or ""))
    matching_typed = [name for name in typed if name in goal_tokens]
    if matching_typed:
        return matching_typed
    known = set(definitions) | _RESERVED
    return [
        token for token in dict.fromkeys(_IDENT.findall(goal or ""))
        if token not in known and not token[0].isupper()
    ]


def _bound_identifiers(goal: str) -> set[str]:
    """Collect visibly bound lambda/meta/object-quantifier variables."""

    bound: set[str] = set()
    binder = re.compile(
        r"(?:λ|⋀|∀|∃|\\<lambda>|\\<And>|\\<forall>|\\<exists>)\s*([^.]*)\."
    )
    for match in binder.finditer(goal or ""):
        bound.update(_IDENT.findall(match.group(1)))
    return bound


def _plans_from_definitions(
    goal: str,
    definitions: Dict[str, RecursiveDefinition],
    state_hint: str = "",
    *,
    limit: int = 4,
) -> List[StrategyPlan]:
    """Rank induction targets using supplied recursive call positions."""

    if not definitions:
        return []
    known_constants = set(definitions) | _RESERVED
    score: Dict[str, float] = {}
    sources: Dict[str, List[str]] = {}
    dependents: Dict[str, List[str]] = {}
    equality = (goal or "").find("=")

    for definition in definitions.values():
        for offset, args in _function_occurrences(goal, definition):
            side_bonus = 1.5 if equality < 0 or offset < equality else 0.5
            for position in definition.recursive_positions:
                if position >= len(args):
                    continue
                identifiers = _term_identifiers(args[position], known_constants)
                for order, identifier in enumerate(identifiers):
                    direct = bool(re.fullmatch(rf"\(?\s*{re.escape(identifier)}\s*\)?", args[position]))
                    points = (6.0 if direct else max(2.0, 5.0 - order)) + side_bonus
                    score[identifier] = score.get(identifier, 0.0) + points
                    sources.setdefault(identifier, []).append(definition.name)
                    others: List[str] = []
                    # Accumulator/state parameters and the unchosen side of a
                    # composite recursive argument are useful generalizations.
                    for arg_index, arg in enumerate(args):
                        if arg_index != position:
                            others.extend(_term_identifiers(arg, known_constants))
                    for other in identifiers:
                        if other != identifier:
                            others.append(other)
                    dependents.setdefault(identifier, []).extend(others)

    if not score:
        return []
    freeish = _all_freeish_identifiers(goal, definitions, state_hint)
    free_set = set(freeish)
    bound_names = _bound_identifiers(goal)
    if any(name in free_set for name in score):
        score = {name: value for name, value in score.items() if name in free_set}
    ranked = sorted(score, key=lambda name: (-score[name], len(name), name))
    plans: List[StrategyPlan] = []
    for target in ranked[: max(1, limit)]:
        generalize: List[str] = []
        for name in dependents.get(target, []):
            if name == target or name in generalize:
                continue
            if name not in free_set and free_set:
                continue
            # Never place constants or locally bound meta-quantifier names in
            # ``arbitrary`` merely because a lightweight lexer saw them.
            if name in known_constants or name[0].isupper():
                continue
            if name in bound_names:
                continue
            generalize.append(name)
        plans.append(StrategyPlan(
            target=target,
            score=score[target],
            generalize=tuple(generalize[:6]),
            sources=tuple(dict.fromkeys(sources.get(target, []))),
        ))
    return plans


def definition_guided_plans(
    goal: str,
    theory_preamble: str,
    state_hint: str = "",
    *,
    limit: int = 4,
) -> List[StrategyPlan]:
    """Rank induction targets using local recursive equations in ``goal``."""

    return _plans_from_definitions(
        goal,
        parse_recursive_definitions(theory_preamble),
        state_hint,
        limit=limit,
    )


def generic_structural_plans(
    goal: str,
    state_hint: str = "",
    *,
    limit: int = 4,
) -> List[StrategyPlan]:
    """Rank induction targets for standard HOL recursive constants.

    This complements definition-directed search when the target uses imported
    functions such as ``filter`` or ``map`` whose equations are not present in
    the local theory preamble.
    """

    return _plans_from_definitions(
        goal,
        _standard_recursive_definitions(),
        state_hint,
        limit=limit,
    )


def _induction_head(plan: StrategyPlan, with_generalization: bool) -> str:
    if with_generalization and plan.generalize:
        return f"induction {plan.target} arbitrary: {' '.join(plan.generalize)}"
    return f"induction {plan.target}"


def definition_guided_finishers(
    goal: str,
    theory_preamble: str,
    state_hint: str = "",
    *,
    limit: int = 8,
) -> List[str]:
    """Return complete, Isabelle-checkable structural proof methods."""

    plans = definition_guided_plans(
        goal, theory_preamble, state_hint, limit=max(1, min(3, limit))
    )
    if not plans:
        return []
    preamble = theory_preamble or ""
    has_conditionals = "if " in preamble
    has_append = "@" in (goal or "") or "@" in preamble
    has_arithmetic = bool(re.search(r"(?:\+|\*|\^|sum_list|length)", goal or ""))
    out: List[str] = []
    for plan in plans:
        heads = [_induction_head(plan, True)]
        if plan.generalize:
            heads.append(_induction_head(plan, False))
        for head in heads:
            methods = [f"by ({head}) simp_all"]
            if has_conditionals:
                methods.append(f"by ({head}) (auto split: if_splits)")
            if has_append:
                methods.append(f"by ({head}) (simp_all add: append_assoc)")
            if has_arithmetic:
                methods.append(f"by ({head}) (simp_all add: algebra_simps)")
            methods.append(f"by ({head}) auto")
            for method in methods:
                if method not in out:
                    out.append(method)
                if len(out) >= limit:
                    return out
    return out


def generic_structural_finishers(
    goal: str,
    state_hint: str = "",
    *,
    limit: int = 6,
) -> List[str]:
    """Return atomic induction methods for imported HOL library functions."""

    plans = generic_structural_plans(
        goal, state_hint, limit=max(1, min(3, limit))
    )
    out: List[str] = []
    has_branching = bool(re.search(r"\b(?:filter|partition)\b|\bif\b", goal or ""))
    has_injectivity = bool(re.search(r"\binj(?:_on)?\b", goal or ""))
    has_append = "@" in (goal or "")
    for plan in plans:
        # Imported library theorems normally keep predicates/functions fixed,
        # so try the plain induction before its generalised fallback.
        heads = [_induction_head(plan, False)]
        if plan.generalize:
            heads.append(_induction_head(plan, True))
        for head in heads:
            methods: List[str] = []
            if has_injectivity:
                methods.append(f"by ({head}) (auto dest: injD)")
            if has_branching:
                methods.append(f"by ({head}) auto")
                methods.append(f"by ({head}) (auto split: if_splits)")
            methods.append(f"by ({head}) simp_all")
            if has_append:
                methods.append(f"by ({head}) (simp_all add: append_assoc)")
            methods.append(f"by ({head}) auto")
            for method in methods:
                if method not in out:
                    out.append(method)
                if len(out) >= limit:
                    return out
    return out


def generic_structural_steps(
    goal: str,
    state_hint: str = "",
    *,
    limit: int = 4,
) -> List[str]:
    """Return standard-library induction moves for beam search."""

    out: List[str] = []
    for plan in generic_structural_plans(goal, state_hint, limit=limit):
        out.append(f"apply ({_induction_head(plan, False)})")
        if plan.generalize:
            out.append(f"apply ({_induction_head(plan, True)})")
        if len(out) >= limit:
            break
    return out[:limit]


def definition_guided_steps(
    goal: str,
    theory_preamble: str,
    state_hint: str = "",
    *,
    limit: int = 4,
) -> List[str]:
    """Return structural expansion commands for multi-step beam search."""

    out: List[str] = []
    for plan in definition_guided_plans(goal, theory_preamble, state_hint, limit=limit):
        out.append(f"apply ({_induction_head(plan, True)})")
        if plan.generalize:
            out.append(f"apply ({_induction_head(plan, False)})")
        if len(out) >= limit:
            break
    return out[:limit]


def _is_list_homomorphism_candidate(definition: RecursiveDefinition) -> bool:
    saw_nil = False
    saw_cons = False
    saw_append_result = False
    for equation in definition.equations:
        lhs, rhs = equation.split("=", 1)
        match = re.match(rf"{re.escape(definition.name)}\b(.*)$", lhs.strip(), re.S)
        if not match:
            continue
        args = _application_args(match.group(1), definition.arity)
        for position in definition.recursive_positions:
            if position >= len(args):
                continue
            term = args[position].replace(" ", "")
            saw_nil = saw_nil or term in {"[]", "([])"}
            saw_cons = saw_cons or "#" in term
        saw_append_result = saw_append_result or (
            "@" in rhs and re.search(rf"\b{re.escape(definition.name)}\b", rhs) is not None
        )
    return saw_nil and saw_cons and saw_append_result


def _render_call(name: str, args: Sequence[str]) -> str:
    return " ".join([name, *[arg.strip() for arg in args]])


def _strip_outer_parens(term: str) -> str:
    value = (term or "").strip()
    while value.startswith("(") and value.endswith(")"):
        end = _balanced_end(value, 0, "(", ")")
        if end != len(value):
            break
        value = value[1:-1].strip()
    return value


def _rhs_guarantees_nonempty_list(rhs: str) -> bool:
    """Conservatively recognize equations that always construct a list cell.

    This is deliberately syntactic: false negatives merely omit a strategy,
    while every emitted invariant is subsequently checked by Isabelle.  The
    recognizer handles constructor literals, cons, and conditionals whose two
    branches both have the property.
    """

    value = _strip_outer_parens(rhs)
    # A top-level list cons is enough.  Ignore hashes nested inside a function
    # argument or conditional branch until the recursive checks below.
    depth = 0
    quoted = False
    escaped = False
    saw_cons = False
    saw_product = False
    for char in value:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == "#" and depth == 0:
            saw_cons = True
        elif char == "," and depth == 0:
            saw_product = True

    # Do not mistake a constructor nested in a pair component for the result
    # type itself, e.g. ``([x], ys)`` or ``(x # xs, ys)``.
    if saw_product:
        return False
    if re.match(r"^\[\s*[^\]\s]", value) or saw_cons:
        return True

    # Isabelle's equation RHSs commonly use a single, un-nested if.  Split it
    # without attempting to parse the condition; both result branches must be
    # independently constructor-producing.
    match = re.match(r"(?s)^if\b.*?\bthen\b(.*?)\belse\b(.*)$", value)
    if match:
        return (
            _rhs_guarantees_nonempty_list(match.group(1))
            and _rhs_guarantees_nonempty_list(match.group(2))
        )
    return False


def _always_nonempty_list_result(definition: RecursiveDefinition) -> bool:
    if not definition.equations:
        return False
    return all(
        "=" in equation
        and _rhs_guarantees_nonempty_list(equation.split("=", 1)[1])
        for equation in definition.equations
    )


def _obligation_main_methods(
    plan: StrategyPlan,
    *,
    has_append: bool,
    has_arithmetic: bool,
    generalize_first: bool = True,
) -> Tuple[str, ...]:
    generalized = _induction_head(plan, True)
    plain = _induction_head(plan, False)
    if plan.generalize:
        heads = [generalized, plain] if generalize_first else [plain, generalized]
    else:
        heads = [plain]
    methods: List[str] = []
    for head in heads:
        simp_facts = ["{aux}"]
        if has_append:
            simp_facts.append("append_assoc")
        if has_arithmetic:
            simp_facts.append("algebra_simps")
        methods.extend([
            f"by ({head}) (simp_all add: {' '.join(simp_facts)})",
            f"by ({head}) (auto simp add: {' '.join(simp_facts)})",
        ])
    return tuple(dict.fromkeys(methods))


def definition_guided_obligations(
    goal: str,
    theory_preamble: str,
    state_hint: str = "",
    *,
    limit: int = 4,
) -> List[AuxiliaryObligation]:
    """Infer a small verified-obligation DAG from recursive equations.

    Two general schemas are currently useful across domains:

    * append homomorphisms for recursive list consumers whose step equation
      combines results with ``@``;
    * constructor/shape invariants when every defining equation guarantees a
      nonempty list result.  These unlock simplification of partial operations
      such as ``butlast`` after accumulator-generalized induction.

    The schemas never establish truth by themselves.  Callers must prove the
    returned statement first and only then try a main method that cites it.
    """

    definitions = parse_recursive_definitions(theory_preamble)
    plans = definition_guided_plans(goal, theory_preamble, state_hint, limit=3)
    if not definitions or not plans:
        return []
    has_append = "@" in (goal or "") or "@" in (theory_preamble or "")
    has_arithmetic = bool(re.search(r"(?:\+|\*|\^|sum_list|length)", goal or ""))
    obligations: List[AuxiliaryObligation] = []

    for definition in definitions.values():
        occurrences = list(_function_occurrences(goal, definition))
        if not occurrences:
            continue

        if _is_list_homomorphism_candidate(definition):
            selected_args: Optional[List[str]] = None
            selected_position: Optional[int] = None
            for _, args in occurrences:
                for position in definition.recursive_positions:
                    if position >= len(args):
                        continue
                    term = args[position]
                    if "@" in term or any(
                        re.search(rf"\b{re.escape(name)}\b", term)
                        for name in definitions
                    ):
                        selected_args, selected_position = list(args), position
                        break
                if selected_args is not None:
                    break
            if selected_args is not None and selected_position is not None:
                left_args = list(selected_args)
                first_args = list(selected_args)
                second_args = list(selected_args)
                left_args[selected_position] = "(llm_xs @ llm_ys)"
                first_args[selected_position] = "llm_xs"
                second_args[selected_position] = "llm_ys"
                body = (
                    f"{_render_call(definition.name, left_args)} = "
                    f"{_render_call(definition.name, first_args)} @ "
                    f"{_render_call(definition.name, second_args)}"
                )
                for plan in plans:
                    obligations.append(AuxiliaryObligation(
                        kind="append-homomorphism",
                        name_hint="llm_hom",
                        binders=("llm_xs", "llm_ys"),
                        body=body,
                        proof_methods=(
                            "by (induction llm_xs) (simp_all add: append_assoc)",
                            "by (induction llm_xs) auto",
                        ),
                        main_methods=_obligation_main_methods(
                            plan,
                            has_append=True,
                            has_arithmetic=has_arithmetic,
                            # A homomorphism about the current function
                            # parameters should normally stay fixed across the
                            # consumer induction.  Generalization remains a
                            # checked fallback.
                            generalize_first=False,
                        ),
                        score=20.0 + plan.score,
                    ))

        if _always_nonempty_list_result(definition):
            for position in definition.recursive_positions:
                if position >= definition.arity:
                    continue
                args = [f"llm_p{i}" for i in range(definition.arity)]
                args[position] = "llm_xs"
                parameters = [arg for arg in args if arg != "llm_xs"]
                binders = tuple([*parameters, "llm_xs"])
                arbitrary = f" arbitrary: {' '.join(parameters)}" if parameters else ""
                body = f"{_render_call(definition.name, args)} ≠ []"
                for plan in plans:
                    obligations.append(AuxiliaryObligation(
                        kind="nonempty-result",
                        name_hint="llm_nonempty",
                        binders=binders,
                        body=body,
                        proof_methods=(
                            f"by (induction llm_xs{arbitrary}) simp_all",
                            f"by (induction llm_xs{arbitrary}) auto",
                        ),
                        main_methods=_obligation_main_methods(
                            plan,
                            has_append=has_append,
                            has_arithmetic=has_arithmetic,
                        ),
                        score=30.0 + plan.score,
                    ))

    # Prefer shape invariants: they usually indicate a genuinely blocked
    # simplifier rather than a merely optional strengthening.  One obligation
    # may be consumed by several induction plans, so merge those consumers
    # instead of re-proving the same helper theorem for every plan.
    obligations.sort(key=lambda item: (-item.score, item.kind, item.body))
    unique: List[AuxiliaryObligation] = []
    positions: Dict[Tuple[object, ...], int] = {}
    for obligation in obligations:
        key = (
            obligation.kind,
            obligation.name_hint,
            obligation.binders,
            obligation.body,
            obligation.proof_methods,
        )
        existing_index = positions.get(key)
        if existing_index is not None:
            existing = unique[existing_index]
            merged_methods = tuple(dict.fromkeys(
                (*existing.main_methods, *obligation.main_methods)
            ))
            unique[existing_index] = replace(
                existing,
                main_methods=merged_methods,
                score=max(existing.score, obligation.score),
            )
            continue
        positions[key] = len(unique)
        unique.append(obligation)
    return unique[:max(0, limit)]


def definition_guided_helper_scripts(
    goal: str,
    theory_preamble: str,
    state_hint: str = "",
    *,
    limit: int = 3,
) -> List[List[str]]:
    """Render inferred obligations as local Isar helper-proof candidates.

    The underlying schemas include recursive append homomorphisms and result
    shape invariants.  This renderer is used when Fill must remain inside an
    existing Isar context; standalone proving and Stage 3 instead verify the
    auxiliary theorem in a separate transaction before consuming it.
    """

    obligations = definition_guided_obligations(
        goal, theory_preamble, state_hint, limit=max(1, limit)
    )
    scripts: List[List[str]] = []
    for obligation in obligations:
        for proof_method in obligation.proof_methods[:2]:
            if not obligation.main_methods:
                continue
            main_method = obligation.main_methods[0].format(aux=obligation.name_hint)
            script = ["proof -"]
            if obligation.binders:
                # Explicitly introduce the meta-bound variables before the
                # induction method.  Calling ``induction`` directly on a
                # ``⋀`` proposition leaves the target name unavailable to the
                # method in several Isabelle versions.
                script.extend([
                    f'  have {obligation.name_hint}: "{obligation.statement}"',
                    "  proof -",
                    f"    fix {' '.join(obligation.binders)}",
                    f'    show "{obligation.body}"',
                    f"      {proof_method}",
                    "  qed",
                ])
            else:
                script.extend([
                    f'  have {obligation.name_hint}: "{obligation.body}"',
                    f"    {proof_method}",
                ])
            script.extend([
                "  show ?thesis",
                f"    {main_method}",
                "qed",
            ])
            if script not in scripts:
                scripts.append(script)
            if len(scripts) >= limit:
                return scripts
    return scripts


__all__ = [
    "RecursiveDefinition",
    "StrategyPlan",
    "AuxiliaryObligation",
    "parse_recursive_definitions",
    "definition_guided_plans",
    "definition_guided_finishers",
    "definition_guided_steps",
    "generic_structural_plans",
    "generic_structural_finishers",
    "generic_structural_steps",
    "definition_guided_obligations",
    "definition_guided_helper_scripts",
]
