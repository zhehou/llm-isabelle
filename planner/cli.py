# planner/cli.py
from __future__ import annotations
import argparse
from planner.driver import plan_and_fill

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", required=True, help='The lemma goal inside quotes, e.g., \'rev (rev xs) = xs\'')
    ap.add_argument("--model", default=None)
    ap.add_argument("--budget-s", type=int, default=10)
    args = ap.parse_args()
    r = plan_and_fill(goal=args.goal, model=args.model, budget_s=args.budget_s)
    print("SUCCESS:", r.success)
    print("---- OUTLINE ----")
    print(r.outline)

if __name__ == "__main__":
    main()
