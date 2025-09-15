#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Randomly sample N goals (one per line) from an input .txt and write them to a new file.

Usage:
  python sample_goals.py input.txt output.txt --n 500 --seed 42 [--keep-order]

Notes:
- Empty lines are ignored by default.
- If N > number of available goals, all goals are written (no error).
- With --keep-order, the sampled lines are written in their original file order.
"""

import argparse
import random
from typing import List, Tuple

def read_goals(path: str) -> List[Tuple[int, str]]:
    """Read non-empty lines, returning list of (original_index, line_text_without_newline)."""
    goals = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            # Keep the line content as-is except the trailing newline
            text = line.rstrip("\r\n")
            if text.strip() == "":
                continue  # skip empty or whitespace-only lines
            goals.append((idx, text))
    return goals

def write_goals(path: str, items: List[Tuple[int, str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for _, text in items:
            f.write(text + "\n")

def main():
    ap = argparse.ArgumentParser(description="Randomly sample goals from a text file.")
    ap.add_argument("input", help="Path to input .txt (one goal per line)")
    ap.add_argument("output", help="Path to write sampled goals")
    ap.add_argument("--n", type=int, default=500, help="Number of goals to sample (default: 500)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    ap.add_argument("--keep-order", action="store_true",
                    help="Write sampled goals in their original file order")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    goals = read_goals(args.input)
    if not goals:
        # Create/clear output file and exit quietly
        open(args.output, "w", encoding="utf-8").close()
        return

    k = min(args.n, len(goals))
    sampled = random.sample(goals, k)

    if args.keep_order:
        sampled.sort(key=lambda x: x[0])

    write_goals(args.output, sampled)

if __name__ == "__main__":
    main()
