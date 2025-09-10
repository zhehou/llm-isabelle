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
    if not out:
        return None
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Planner: Plan → Sketch → Fill (Isabelle/HOL)")
    ap.add_argument("goal", nargs="?", help='Lemma statement without quotes, e.g., ALL xs. rev (rev xs) = xs')
    ap.add_argument("--model", default=None, help="Model id (e.g., 'ollama:qwen2.5:14b', 'hf:meta-llama/…', 'gemini:gemini-2.5-pro')")
    ap.add_argument("--timeout", type=int, default=120, help="Total wall-clock seconds for planning + filling")
    ap.add_argument("--mode", choices=["auto", "outline"], default="auto",
                    help="auto: allow whole proofs; outline: force placeholders and fill")
    # Diverse-outline controls
    ap.add_argument("--diverse-outlines", dest="diverse", action="store_true",
                    help="Enable diverse outline sampling + quick sketch check")
    ap.add_argument("--single-outline", dest="diverse", action="store_false",
                    help="Disable diversity; use a single low-temp outline")
    ap.set_defaults(diverse=True)  # default ON to use Hybrid planner flow
    ap.add_argument("--k", type=int, default=3, help="Number of outline candidates (when --diverse-outlines)")
    ap.add_argument("--temps", type=_parse_temps, default=None,
                    help="Comma-separated temps for outline sampling, e.g. '0.35,0.55,0.85'")
    args = ap.parse_args(argv)

    # Goal handling
    goal = args.goal
    if not goal:
        # read from stdin, strip quotes if user pasted the whole lemma line
        data = sys.stdin.read().strip()
        if data.startswith('lemma "') and data.endswith('"'):
            data = data[len('lemma "'): -1]
        goal = data.strip()
    if not goal:
        print("No goal provided. Pass as an argument or via stdin.", file=sys.stderr)
        return 2

    res = plan_and_fill(
        goal,
        model=args.model,
        timeout=args.timeout,
        mode=args.mode,
        # pass-through diverse-outline controls
        outline_k=args.k if args.diverse else 1,
        outline_temps=args.temps,
        legacy_single_outline=(not args.diverse),
    )

    print(res.outline, end="" if res.outline.endswith("\n") else "\n")
    return 0 if res.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
