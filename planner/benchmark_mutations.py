from __future__ import annotations

"""Generate planner benchmarks from verified Isar proof oracles.

The planner should be evaluated on proof states, not on a short list of
hand-written final tactics.  This tool masks method leaves to create structured
fill tasks or injects controlled defects to create repair tasks.  It stores an
oracle hash rather than the oracle proof in the generated suite.

Input records are JSONL objects with ``id``, ``goal`` and ``proof`` fields.
Additional labels are copied through.  Generated suites are still checked by
``planner.experiments``: hole suites must contain holes, and repair suites must
be rejected by Isabelle before the planner is allowed to repair them.
"""

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


_STANDALONE_METHOD = re.compile(r"(?m)^(?P<indent>[ \t]*)(?P<method>by\b[^\n]*)$")
_INLINE_METHOD = re.compile(
    r"(?m)^(?P<head>[ \t]*(?:have|show|thus|hence)\b[^\n]*?)[ \t]+(?P<method>by\b[^\n]*)$"
)
_USING_LINE = re.compile(r"(?m)^(?P<indent>[ \t]*)using\b(?P<facts>[^\n]*)$")
_INDUCTION_ARBITRARY = re.compile(r"\s+arbitrary\s*:\s*[A-Za-z0-9_' ]+")
_HAVE_QUOTED = re.compile(
    r'(?m)^(?P<prefix>[ \t]*have(?:\s+[A-Za-z_][A-Za-z0-9_\']*\s*:)?\s*)"(?P<prop>[^"]+)"'
)


@dataclass(frozen=True, slots=True)
class MethodLeaf:
    start: int
    end: int
    indent: str
    inline: bool
    method: str
    score: int


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not str(record.get("goal", "")).strip():
                raise ValueError(f"Missing goal at {path}:{line_no}")
            if not str(record.get("proof", "")).strip():
                raise ValueError(f"Missing proof at {path}:{line_no}")
            records.append(record)
    return records


def _method_leaves(proof: str) -> List[MethodLeaf]:
    leaves: List[MethodLeaf] = []
    inline_ranges: List[Tuple[int, int]] = []
    for match in _INLINE_METHOD.finditer(proof):
        start, end = match.span("method")
        indent = match.group("head")[: len(match.group("head")) - len(match.group("head").lstrip())]
        prefix = proof[max(0, start - 160):start]
        score = 30 + len(indent) + (20 if "using" in prefix else 0)
        leaves.append(MethodLeaf(start, end, indent, True, match.group("method"), score))
        inline_ranges.append((start, end))
    for match in _STANDALONE_METHOD.finditer(proof):
        start, end = match.span("method")
        if any(a <= start < b for a, b in inline_ranges):
            continue
        indent = match.group("indent")
        prefix = proof[max(0, match.start() - 160):match.start()]
        score = 20 + len(indent) + (25 if re.search(r"(?m)^\s*(?:using|then|from)\b", prefix) else 0)
        leaves.append(MethodLeaf(start, end, indent, False, match.group("method"), score))
    return sorted(leaves, key=lambda leaf: (leaf.score, leaf.start), reverse=True)


def _replace_spans(text: str, replacements: Sequence[Tuple[int, int, str]]) -> str:
    out = text
    for start, end, replacement in sorted(replacements, reverse=True):
        out = out[:start] + replacement + out[end:]
    return out


def _dependency_depth(proof: str) -> int:
    depth = 0
    maximum = 0
    for line in proof.splitlines():
        stripped = line.strip()
        if stripped.startswith("proof"):
            depth += 1
            maximum = max(maximum, depth)
        elif stripped == "qed":
            depth = max(0, depth - 1)
    return maximum


def mask_method_holes(
    proof: str,
    count: int,
    *,
    rng: random.Random,
) -> Tuple[str, int]:
    """Replace dependency-rich method leaves by admitted planner holes."""
    leaves = _method_leaves(proof)
    if not leaves:
        raise ValueError("proof has no maskable method leaves")
    count = min(max(1, count), len(leaves))
    # Select from the high-value prefix, but vary ties and nearby leaves across
    # generated versions.  The seed makes the suite fully reproducible.
    pool = leaves[: max(count, min(len(leaves), count * 3))]
    selected = rng.sample(pool, count)
    outline = _replace_spans(
        proof,
        [(leaf.start, leaf.end, "sorry") for leaf in selected],
    )
    return outline, len(selected)


def _replace_using_fact(proof: str) -> Optional[str]:
    matches = list(_USING_LINE.finditer(proof))
    if not matches:
        return None
    match = matches[-1]
    replacement = f"{match.group('indent')}using __planner_missing_fact__"
    return proof[:match.start()] + replacement + proof[match.end():]


def _drop_arbitrary(proof: str) -> Optional[str]:
    mutated, n = _INDUCTION_ARBITRARY.subn("", proof, count=1)
    return mutated if n else None


def _corrupt_intermediate(proof: str) -> Optional[str]:
    matches = list(_HAVE_QUOTED.finditer(proof))
    for match in reversed(matches):
        prop = match.group("prop")
        if " = " in prop:
            bad = prop.replace(" = ", " ≠ ", 1)
        elif " ⟷ " in prop:
            bad = prop.replace(" ⟷ ", " ⟹ ", 1)
        elif " ∈ " in prop:
            bad = prop.replace(" ∈ ", " ∉ ", 1)
        else:
            continue
        start, end = match.span("prop")
        return proof[:start] + bad + proof[end:]
    return None


def inject_repair_mutation(
    proof: str,
    operator: str,
    *,
    rng: random.Random,
) -> Tuple[str, str]:
    """Inject one controlled defect and return ``(mutant, actual_operator)``."""
    operator = operator.strip().lower().replace("-", "_")
    if operator == "wrong_fact":
        mutated = _replace_using_fact(proof)
        if mutated is not None:
            return mutated, operator
    elif operator == "drop_arbitrary":
        mutated = _drop_arbitrary(proof)
        if mutated is not None:
            return mutated, operator
    elif operator == "corrupt_intermediate":
        mutated = _corrupt_intermediate(proof)
        if mutated is not None:
            return mutated, operator
    elif operator != "wrong_method":
        raise ValueError(f"unknown mutation operator: {operator}")

    leaves = _method_leaves(proof)
    if not leaves:
        raise ValueError("proof has no mutable method leaves")
    leaf = rng.choice(leaves[: min(4, len(leaves))])
    mutant = _replace_spans(proof, [(leaf.start, leaf.end, "by (rule FalseE)")])
    return mutant, "wrong_method"


def generate_records(
    records: Iterable[Dict[str, object]],
    *,
    kind: str,
    per_proof: int,
    holes: int,
    operators: Sequence[str],
    seed: int,
) -> List[Dict[str, object]]:
    generated: List[Dict[str, object]] = []
    for index, source in enumerate(records):
        goal = str(source["goal"]).strip()
        proof = str(source["proof"]).strip() + "\n"
        source_id = str(source.get("id") or f"oracle_{index + 1}")
        oracle_hash = hashlib.sha256(proof.encode("utf-8")).hexdigest()
        for variant in range(max(1, per_proof)):
            rng = random.Random(f"{seed}:{source_id}:{variant}:{kind}")
            record: Dict[str, object] = {
                key: value for key, value in source.items()
                if key not in {"proof", "outline"}
            }
            record["id"] = f"{source_id}_{kind}_{variant + 1}"
            record["goal"] = goal
            record["oracle_sha256"] = oracle_hash
            record["dependency_depth"] = _dependency_depth(proof)
            record["fixture_kind"] = kind
            if kind == "holes":
                outline, masked = mask_method_holes(proof, holes, rng=rng)
                record["outline"] = outline
                record["holes"] = masked
                record["mutation"] = "masked_methods"
                record.setdefault("required_mechanism", "contextual_fill")
            elif kind == "repair":
                requested = operators[variant % len(operators)]
                outline, actual = inject_repair_mutation(proof, requested, rng=rng)
                record["outline"] = outline
                record["mutation"] = actual
                record.setdefault("required_mechanism", "structural_repair")
            else:
                raise ValueError("kind must be 'holes' or 'repair'")
            generated.append(record)
    return generated


def _write_jsonl(path: Path, records: Sequence[Dict[str, object]], kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    contract = "holes" if kind == "holes" else "invalid"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# fixture_expect: {contract}\n")
        handle.write("# generated by planner.benchmark_mutations; oracle bodies intentionally omitted\n")
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fill/repair fixtures from Isar proof oracles")
    parser.add_argument("--input", required=True, help="JSONL containing goal + verified proof")
    parser.add_argument("--output", required=True, help="Generated fixture JSONL")
    parser.add_argument("--kind", choices=("holes", "repair"), default="holes")
    parser.add_argument("--per-proof", type=int, default=1)
    parser.add_argument("--holes", type=int, default=2, help="Method leaves to mask per hole fixture")
    parser.add_argument(
        "--operators",
        default="wrong_method,wrong_fact,drop_arbitrary,corrupt_intermediate",
        help="Comma-separated repair mutations",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    operators = [x.strip() for x in args.operators.split(",") if x.strip()]
    if not operators:
        raise SystemExit("at least one mutation operator is required")
    source = _read_jsonl(Path(args.input))
    generated = generate_records(
        source,
        kind=args.kind,
        per_proof=max(1, args.per_proof),
        holes=max(1, args.holes),
        operators=operators,
        seed=args.seed,
    )
    _write_jsonl(Path(args.output), generated, args.kind)
    print(f"wrote {len(generated)} {args.kind} fixtures to {args.output}")


if __name__ == "__main__":
    main()
