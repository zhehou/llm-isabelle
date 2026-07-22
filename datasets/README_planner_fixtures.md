# Fixed-outline planner benchmarks

These suites measure planner mechanisms independently of outline generation.
Each JSONL record supplies a theorem and a fixed Isar document.

- `planner_structured.jsonl` contains reachable outlines with 2–4 holes. Run
  with repairs and whole-goal fallback disabled to measure contextual and
  fresh-goal filling.
- `planner_repair.jsonl` contains complete-looking but invalid proofs. Run with
  repairs enabled and whole-goal fallback disabled. The planner must localise
  the failure, open or replace the faulty block, and verify the result.

The first comment in each suite is a machine-checked contract:

- `fixture_expect: holes` rejects records that contain no `sorry`.
- `fixture_expect: invalid` compiles every input before planning and rejects
  records Isabelle already accepts. Such records are reported as fixture
errors and can no longer inflate the repair rate.

Hole fixtures are also compiled with their admissions before planning. This
guarantees that every measured hole is reachable in a structurally valid Isar
document; malformed generated fixtures are reported as fixture errors.

Completed invalid proofs are repaired in two stages. Deterministic proof
surgery first identifies the smallest set of bad tactic leaves and replaces
only those commands with verified holes. A tactic and its immediately
preceding `using`/`unfolding` lines form one repair atom, so a stale fact
selection is removed with the bad method. The ordinary exact-context filler
then solves those holes. Generative block repair is used only when surgery
cannot produce a valid admitted document.

Do not compare these directly with goal-only suite rates. Report success,
median fills, fill origins, repair-stage counters, and median time together.
The repair counters are `local_repairs`, `subproof_repairs` and
`stage3_repairs`; `verified_aux_obligations`, `complete_generated_outlines`
and `outline_backtracks` expose other control-loop contributions that are not
ordinary hole fills.

## Novel-theory planning suite

`planner_novel.jsonl` is the primary end-to-end planning benchmark. Each
record supplies a goal plus a self-contained theory preamble, but no proof
outline. The preambles introduce fresh recursive functions, datatypes,
semantics and abstract machines. This prevents an exact theorem already in
`Main` from solving the benchmark by name, while still allowing ordinary HOL
library facts such as append associativity to be discovered and used.

The suite covers accumulator generalisation, tree induction, compiler
correctness, substitution semantics, helper-lemma invention, fold fusion,
semiring normalisation, multiset refinement, finite-map updates, state-machine
composition and nested homomorphisms. Preambles are compiled before planning;
invalid declarations are reported as fixture errors and excluded from the
success denominator.

Problem records may include an optional `"imports"` string or list (for
example `["HOL-Library.Multiset"]`). The runner carries these imports through
outline checking, exact-context filling, fresh-goal search and repair. This is
important for benchmark validity: an unavailable library constant must not be
silently parsed as a free variable with a different theorem meaning.
Theory names containing punctuation are quoted automatically in the generated
Isabelle header.

For fresh recursive definitions, the structural prover can infer small
auxiliary obligations from the defining equations. It currently recognizes
append homomorphisms and guaranteed-nonempty list results. The standalone
prover and Stage 3 repair first prove such an obligation as a separate theorem,
then attempt the main generalized induction in the enriched theory. Thus the
reported result is a checked proof DAG; an attractive but false invariant is
discarded before it can influence the target proof.

The obligation layer deliberately renders binders in two forms. A local
`have` uses explicit `⋀` quantification followed by a structured `fix`, whereas
an outer helper `lemma` uses a binder-free body so Isabelle fixes the variables
before `induction` is invoked. Obligations shared by several structural plans
are proved once and retain all distinct consumer methods. Trace output includes
the exact helper theorem and the first Isabelle diagnostic on rejection.

Imported HOL recursion is handled even when its equations do not occur in the
problem preamble. The prover uses a small argument-position signature for
standard constants and checks atomic induction-plus-closure methods before
generation. Exact-context leaves also receive rule-aware fact composition for
injection elimination and transitivity, using both local fact names and facts
mined from Isabelle's actual state.

```bash
python -m planner.experiments bench --problem-fixtures datasets/planner_novel.jsonl --mode plan-fill --diverse --k 4 --temps "0.25,0.45,0.70,0.95" --timeout 300 --strict-no-sorry --verify --no-whole-fallback --context-hints --model "gemini:gemini-3-flash-preview" --shuffle --seed 42 --trace
```

For an ablation of repair, run the same command once with `--no-repairs`. A
useful planner result should report nonzero searched contextual fills and
should outperform that ablation on the `very_hard` and helper-invention
strata; a high score produced only by fast contextual replay is not evidence
of effective planning.

## Deep and generated suites

`planner_deep.jsonl` is the broader fixed-outline suite.  It spans lists,
sets, arithmetic, logic, relations, options and functions.  Its labels are
copied into benchmark CSVs, and the terminal report stratifies success by
`difficulty` and `required_mechanism`.

Run it with one command:

```bash
python -m planner.experiments bench --outline-fixtures datasets/planner_deep.jsonl --mode plan-fill --timeout 240 --strict-no-sorry --verify --no-repairs --no-whole-fallback --context-hints --model "gemini:gemini-3-flash-preview" --trace
```

`planner.benchmark_mutations` turns any JSONL corpus of verified proof oracles
(`id`, `goal`, `proof`) into reproducible fill or repair fixtures. Generated
records retain category/difficulty labels and an oracle SHA-256, but omit the
oracle body. Repair output must still pass the `fixture_expect: invalid`
preflight check; a mutation that accidentally remains valid is rejected rather
than counted as a planner attempt.

```bash
python -m planner.benchmark_mutations --input datasets/my_verified_oracles.jsonl --output datasets/generated_fill.jsonl --kind holes --holes 3 --per-proof 2 --seed 42
```

```bash
python -m planner.benchmark_mutations --input datasets/my_verified_oracles.jsonl --output datasets/generated_repair.jsonl --kind repair --operators wrong_method,wrong_fact,drop_arbitrary,corrupt_intermediate --per-proof 4 --seed 42
```
