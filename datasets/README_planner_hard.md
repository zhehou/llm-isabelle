# Planner-focused benchmark

`planner_hard.txt` contains 36 independent Isabelle/HOL goals grouped by the
kind of planning pressure they create: nested transformations, preservation of
local assumptions, induction, auxiliary arithmetic facts, finite-set
reasoning, and quantified implication chains.

`planner_context.txt` is a smaller 12-goal regression suite specifically for
checking that theorem premises, case assumptions, and chained implications
survive hole-state extraction.

For a strict measurement of planner filling, disable theorem-level rescue:

```bash
python -m planner.experiments bench --file datasets/planner_hard.txt --mode plan-fill --diverse --k 3 --temps "0.35,0.55,0.85" --timeout 180 --strict-no-sorry --verify --no-whole-fallback --context-hints --model "gemini:gemini-3-flash-preview" --shuffle --seed 42 --trace
```

Run the same command once with repairs and once with `--no-repairs`. The CSV
separates `contextual_fills`, `fresh_goal_fills`, `repair_fills`, and
`whole_goal_fallbacks`, so a high success rate can no longer conceal reliance
on theorem-level replacement.

Before treating the suite as a stable research benchmark, run the direct
prover with verification once and remove or correct any goal rejected by the
specific Isabelle release and imports used in the experiment.
