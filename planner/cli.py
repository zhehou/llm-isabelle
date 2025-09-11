from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from planner.driver import plan_and_fill


def _parse_temps(s: Optional[str]) -> Optional[List[float]]:
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: List[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            raise argparse.ArgumentTypeError(f"Invalid temperature: {p!r}")
    return out or None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Planner: Plan → Sketch → Fill (Isabelle/HOL)")

    # Accept BOTH a --goal flag and a positional goal (backwards-compatible)
    ap.add_argument("--goal", dest="goal_flag",
                    help='Lemma statement, e.g., map f (xs @ ys) = map f xs @ map f ys')
    ap.add_argument("goal_pos", nargs="?",
                    help='Lemma statement without the --goal flag')

    ap.add_argument("--model", default=None,
                    help="Model id (e.g., 'ollama:qwen2.5:14b', 'hf:meta-llama/...', 'gemini:gemini-2.5-pro')")
    ap.add_argument("--timeout", type=int, default=120,
                    help="Total wall-clock seconds for planning + filling")
    ap.add_argument("--mode", choices=["auto", "outline"], default="auto",
                    help="auto: allow whole proofs; outline: force placeholders and fill")

    # Diverse-outline controls
    ap.add_argument("--diverse-outlines", dest="diverse", action="store_true",
                    help="Enable diverse outline sampling + quick sketch check")
    ap.add_argument("--single-outline", dest="diverse", action="store_false",
                    help="Disable diversity; use a single low-temp outline")
    ap.set_defaults(diverse=True)
    ap.add_argument("--k", type=int, default=3,
                    help="Number of outline candidates (when --diverse-outlines)")
    ap.add_argument("--temps", type=_parse_temps, default=None,
                    help="Comma-separated temps for outline sampling, e.g. '0.35,0.55,0.85'")

    # Local repair controls
    ap.add_argument("--repairs", dest="repairs", action="store_true",
                    help="Enable local LLM-guided repairs if a hole fails to fill (default).")
    ap.add_argument("--no-repairs", dest="repairs", action="store_false",
                    help="Disable local repairs; only try direct fill for each hole.")
    ap.set_defaults(repairs=True)
    ap.add_argument("--max-repairs-per-hole", type=int, default=2,
                    help="Max repair ops to try for each failing hole (default: 2).")
    ap.add_argument("--repair-trace", action="store_true",
                    help="Print repair proposals and decisions for debugging.")

    # Context hints & priors / scoring knobs (all optional; defaults keep old behavior)
    ap.add_argument("--context-hints", dest="context_hints", action="store_true",
                    help="Mine local Isabelle state and feed top facts/defs as outline hints.")
    ap.add_argument("--no-context-hints", dest="context_hints", action="store_false")
    ap.set_defaults(context_hints=False)

    ap.add_argument("--lib-templates", dest="lib_templates", action="store_true",
                    help="Prepend a few tiny library outline templates when they match obvious tokens.")
    ap.add_argument("--no-lib-templates", dest="lib_templates", action="store_false")
    ap.set_defaults(lib_templates=False)

    ap.add_argument("--priors", default=None,
                    help="Optional JSON file of pattern priors / rules.")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Weight on subgoal count (keep ≥1.0).")
    ap.add_argument("--beta", type=float, default=0.5,
                    help="Weight on pattern penalty.")
    ap.add_argument("--gamma", type=float, default=0.2,
                    help="Weight on hint bonus.")

    # NEW: micro-RAG hint lexicon (optional)
    ap.add_argument("--hintlex", default=None,
                    help="Path to token→hints JSON (from planner.priors aggregate).")
    ap.add_argument("--hintlex-top", type=int, default=8,
                    help="Max hints to take per token from the lexicon.")

    args = ap.parse_args(argv)

    # Resolve goal: flag > positional > stdin
    goal = args.goal_flag or args.goal_pos
    if not goal:
        data = sys.stdin.read().strip()
        if data.startswith('lemma "') and data.endswith('"'):
            data = data[len('lemma "'): -1]
        goal = data.strip()

    if not goal:
        print("No goal provided. Use --goal '…', a positional goal, or pipe via stdin.", file=sys.stderr)
        return 2

    res = plan_and_fill(
        goal,
        model=args.model,
        timeout=args.timeout,
        mode=args.mode,
        outline_k=args.k if args.diverse else 1,
        outline_temps=args.temps,
        legacy_single_outline=(not args.diverse),
        repairs=args.repairs,
        max_repairs_per_hole=args.max_repairs_per_hole,
        repair_trace=args.repair_trace,
        # planner scoring/context
        priors_path=args.priors,
        context_hints=args.context_hints,
        lib_templates=args.lib_templates,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        # NEW: hintlex
        hintlex_path=args.hintlex,
        hintlex_top=args.hintlex_top,
    )

    print(res.outline, end="" if res.outline.endswith("\n") else "\n")
    return 0 if res.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
