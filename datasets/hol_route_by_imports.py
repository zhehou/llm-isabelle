#!/usr/bin/env python3
"""
Route HOL goals using actual theory imports, with fallback to theory name and
file path. Produces per-bucket .txt files and a TSV of suggested EXTRA_IMPORTS.

Input:
  datasets/hol_goals.jsonl  (from hol_extract_goals_v4.py)

Outputs:
  datasets/hol_main.txt
  datasets/hol_sets_lists.txt
  datasets/hol_complex.txt
  datasets/hol_analysis.txt
  datasets/hol_number_theory.txt
  datasets/hol_binomial.txt
  datasets/hol_algebra.txt
  datasets/hol_word.txt
  datasets/hol_misc.txt
  datasets/hol_bucket_imports.tsv
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from collections import defaultdict

BUCKET_IMPORTS = {
    "main": "",
    "sets_lists": "",
    "complex": "Complex_Main",
    "analysis": "Complex_Main",  # keep as alias bucket if you want to split later
    "number_theory": "Number_Theory",
    "binomial": "Binomial",
    "algebra": "Groups Rings Fields Vector_Spaces",
    "word": "Word",
    "misc": "",
}
ORDER = ["main", "sets_lists", "complex", "analysis", "number_theory", "binomial", "algebra", "word", "misc"]

# Canonical theory-name signals (either as imports or as *current theory*)
T_COMPLEX = {
    "Complex", "Complex_Main", "Complex_Analysis", "Transcendental",
    "Topological_Spaces", "Limits", "Series", "Deriv", "NthRoot",
    "Real", "Real_Vector_Spaces", "Real_Asymp", "Filter"
}
T_NUMBER = {
    "Number_Theory", "Primes", "GCD", "Euclidean_Rings", "Quadratic_Residues",
    "Chinese_Remainder", "IntDiv", "Parity", "Semiring_Normalization", "Presburger"
}
T_BINOM = {"Binomial", "Binomial_Plus"}
T_ALGEBRA = {"Algebra", "Groups", "Groups_Big", "Groups_List", "Rings", "Fields", "Vector_Spaces", "Modules"}
T_WORD = {"Bit_Operations", "Word"}

# Light-weight sets/lists indicators
T_SETS_LISTS = {"Set", "Finite_Set", "Set_Interval", "List", "Map", "Option", "Sum_Type", "Product_Type", "Relation"}

def _has(tokens: list[str] | set[str], names: set[str]) -> bool:
    S = set(tokens)
    return any(x in S for x in names)

def _contains_any(s: str, names: set[str]) -> bool:
    return any(n in s for n in names)

def decide_bucket(imports: list[str], theory: str | None, file_path: str, goal: str) -> str:
    imps = imports or []
    thy = theory or ""
    name = Path(file_path).name
    p = str(Path(file_path))

    # 1) Imports or Theory give strong signal
    if _has(imps, T_COMPLEX) or thy in T_COMPLEX:
        return "complex"
    if _has(imps, T_NUMBER) or thy in T_NUMBER:
        return "number_theory"
    if _has(imps, T_BINOM) or thy in T_BINOM:
        return "binomial"
    if _has(imps, T_ALGEBRA) or thy in T_ALGEBRA:
        return "algebra"
    if _has(imps, T_WORD) or thy in T_WORD:
        return "word"

    # 2) Paths can still disambiguate (sessions often mirror folders)
    if "/Complex_Analysis/" in p or "/Analysis/" in p:
        return "complex"
    if "/Number_Theory/" in p:
        return "number_theory"
    if "/Algebra/" in p:
        return "algebra"
    if name in {"Binomial.thy", "Binomial_Plus.thy"}:
        return "binomial"
    if name in {"Bit_Operations.thy"}:
        return "word"

    # 3) Obvious sets/lists by theory/file or goal tokens
    if thy in T_SETS_LISTS or name in {t + ".thy" for t in T_SETS_LISTS}:
        return "sets_lists"
    if any(tok in goal for tok in ["∈", "⊆", "∪", "∩", "insert", " set ", "@", "[]", "map", "rev", "fold"]):
        return "sets_lists"

    # 4) Default
    return "main"

def uniq(xs):
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="outdir", required=True)
    args = ap.parse_args()

    rows = []
    with Path(args.inp).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    buckets = defaultdict(list)
    for r in rows:
        b = decide_bucket(r.get("imports", []), r.get("theory"), r.get("file", ""), r.get("goal", ""))
        buckets[b].append(r["goal"])

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    written, total = [], 0
    for b in ORDER:
        goals = uniq(buckets.get(b, []))
        (outdir / f"hol_{b}.txt").write_text("\n".join(goals) + ("\n" if goals else ""), encoding="utf-8")
        written.append((b, str(outdir / f"hol_{b}.txt"), len(goals)))
        total += len(goals)

    # Anything not covered goes to misc
    covered = set(g for blist in buckets.values() for g in blist)
    misc = [r["goal"] for r in rows if r["goal"] not in covered]
    (outdir / "hol_misc.txt").write_text("\n".join(misc) + ("\n" if misc else ""), encoding="utf-8")
    written.append(("misc", str(outdir / "hol_misc.txt"), len(misc)))
    total += len(misc)

    with (outdir / "hol_bucket_imports.tsv").open("w", encoding="utf-8") as f:
        for b in ORDER:
            f.write(f"{b}\t{BUCKET_IMPORTS.get(b, '')}\n")
        f.write("misc\t\n")

    print("Routing summary:")
    for b, p, n in written:
        print(f"  {b:14s} {n:6d} -> {p}")
    print(f"  {'total':14s} {total:6d}")
    print(f"Wrote suggested imports → {outdir/'hol_bucket_imports.tsv'}")

if __name__ == "__main__":
    main()