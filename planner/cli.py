# planner/cli.py
from __future__ import annotations
import argparse
from planner.driver import plan_and_fill

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Outline planner for Isabelle/HOL proofs (planner.driver.plan_and_fill).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--goal",
        required=True,
        help="Lemma goal (inside quotes), e.g., 'rev (rev xs) = xs'",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Model name to use (None = driver default/env).",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=100,
        help="Planning timeout in seconds.",
    )
    args = ap.parse_args()

    r = plan_and_fill(goal=args.goal, model=args.model, timeout=args.timeout)
    print("SUCCESS:", getattr(r, "success", False))
    print("---- OUTLINE ----")
    print(getattr(r, "outline", "") or "")

if __name__ == "__main__":
    main()