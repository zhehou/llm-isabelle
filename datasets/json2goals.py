#!/usr/bin/env python3
import json, re, sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse
import csv

SYMBOLS = [
    (r'\<And>', '⋀'),
    (r'\<Longrightarrow>', '⟹'),
    (r'\<Rightarrow>', '⟹'),
    (r'\<forall>', '∀'),
    (r'\<exists>', '∃'),
    (r'\<or>', '∨'),
    (r'\<and>', '∧'),
    (r'\<not>', '¬'),
    (r'\<longleftrightarrow>', '⟷'),
    (r'\<rightarrow>', '⟶'),
    (r'\<le>', '≤'),
    (r'\<ge>', '≥'),
    (r'\<in>', '∈'),
    (r'\<subseteq>', '⊆'),
    (r'\<subset>', '⊂'),
    (r'\<emptyset>', '∅'),
    (r'\<nat>', 'ℕ'),
    (r'\<int>', 'ℤ'),
    (r'\<real>', 'ℝ'),
    (r'\<lbrakk>', '⟦'),
    (r'\<rbrakk>', '⟧'),
]

START = re.compile(r'^\s*(lemma|theorem|corollary)\b', re.IGNORECASE | re.DOTALL)
QUOTED = re.compile(r'"([^"]+)"')
THEORY = re.compile(r'^\s*theory\s+([A-Za-z0-9_\'\-]+)\s+imports\s+(.*?)\s*begin', re.IGNORECASE | re.DOTALL)

def iso_norm(s: str) -> str:
    for k,v in SYMBOLS:
        s = s.replace(k, v)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def parse_theory_header(steps: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """Return (theory_name, imports[]) if a header step exists; else ("", [])."""
    for item in steps:
        step = item.get('step') or ''
        m = THEORY.search(step)
        if m:
            name = m.group(1).strip()
            raw = m.group(2).strip()
            toks = re.split(r'[\s"]+', raw)
            imports = [t for t in toks if t]
            return name, imports
    return "", []

def extract_goals_from_steps(steps: List[Dict[str, Any]]) -> List[str]:
    goals: List[str] = []
    for item in steps:
        step = item.get('step') or ''
        if START.match(step):
            cands = QUOTED.findall(step)
            if cands:
                prop = max(cands, key=len)
                goals.append(iso_norm(prop)); continue
            ro = item.get('raw_output') or ''
            m = re.search(r'goal.*?:\s*1\.\s*(.*)$', ro, flags=re.S)
            if m:
                line = m.group(1).splitlines()[0]
                goals.append(iso_norm(line))
    # dedupe while keeping order
    seen = set(); res = []
    for g in goals:
        if g not in seen:
            seen.add(g); res.append(g)
    return res

def collect_json_files(args) -> List[Path]:
    files: List[Path] = []
    if args.root:
        root = Path(args.root)
        files += sorted(root.rglob("*.json"))
    files += [Path(p) for p in args.inputs]
    # unique while preserving order
    seen = set(); uniq = []
    for f in files:
        p = f.resolve()
        if p not in seen:
            seen.add(p); uniq.append(p)
    return uniq

def safe_lemma_name(i: int) -> str:
    return f"FVEL_GOAL_{i:06d}"

def write_wrappers(wrapper_dir: Path, base_imports: List[str], per_theory: Dict[str, Dict[str, Any]]):
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    for thy_name, info in per_theory.items():
        goals: List[str] = info["goals"]
        if not goals: continue
        imports: List[str] = info["imports"]
        # Build imports line: original theory (if provided) + base imports (optional umbrella libs)
        imp_list = []
        # Most JSONs originate from theory 'thy_name' (the file that contained the lemmas).
        if thy_name:
            imp_list.append(thy_name)
        imp_list += base_imports
        # Render
        fn = wrapper_dir / f"{thy_name if thy_name else 'FVEL'}_Wrappers.thy"
        with fn.open("w", encoding="utf-8") as f:
            f.write(f'theory {(thy_name if thy_name else "FVEL")}_Wrappers\n  imports\n')
            for I in imp_list:
                if "-" in I or "." in I:  # keep session-style quoting for safety
                    f.write(f'    "{I}"\n')
                else:
                    f.write(f'    {I}\n')
            f.write("begin\n\n")
            for i, g in enumerate(goals, 1):
                name = safe_lemma_name(i)
                f.write(f'lemma {name}:\n  "{g}"\n  sorry\n\n')
            f.write("end\n")

def main():
    ap = argparse.ArgumentParser(description="Convert FVELER JSON traces to goals.txt (and optional wrappers).")
    ap.add_argument("--root", help="Root directory to search recursively for *.json")
    ap.add_argument("--outfile", required=True, help="Output .txt with one goal per line")
    ap.add_argument("--emit-wrappers", action="store_true", help="Emit wrapper .thy files grouped by source theory")
    ap.add_argument("--wrapper-dir", default="fveler_wrappers", help="Directory for emitted wrapper theories")
    ap.add_argument("--wrapper-base-import", action="append", default=[],
                    help="Extra imports to add to each wrapper (e.g., HOL-Analysis.Analysis); repeatable")
    ap.add_argument("inputs", nargs="*", help="Individual JSON files (optional if --root is set)")
    args = ap.parse_args()

    files = collect_json_files(args)
    if not files:
        print("[ERROR] No JSON files found.", file=sys.stderr)
        sys.exit(2)

    all_goals: List[str] = []
    per_theory: Dict[str, Dict[str, Any]] = {}  # thy_name -> {imports:[], goals:[]}
    meta_rows = []

    for jf in files:
        try:
            steps = json.loads(jf.read_text(encoding='utf-8'))
        except Exception as e:
            print(f"[warn] cannot read {jf}: {e}", file=sys.stderr)
            continue

        thy_name, imports = parse_theory_header(steps)
        goals = extract_goals_from_steps(steps)

        for g in goals:
            all_goals.append(g)
            meta_rows.append({"json": str(jf), "theory": thy_name, "imports": " ".join(imports), "goal": g})

        key = thy_name or "FVEL"
        ent = per_theory.setdefault(key, {"imports": imports, "goals": []})
        ent["goals"].extend(goals)

    # write goals
    out = Path(args.outfile); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(all_goals) + ("\n" if all_goals else ""), encoding="utf-8")

    # write meta CSV
    meta_csv = out.with_suffix(".meta.csv")
    with meta_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["json","theory","imports","goal"])
        w.writeheader(); w.writerows(meta_rows)

    # wrappers
    if args.emit_wrappers:
        write_wrappers(Path(args.wrapper_dir), args.wrapper_base_import, per_theory)

    print(f"Wrote {len(all_goals)} goals -> {out}")
    print(f"Meta: {meta_csv}")
    if args.emit_wrappers:
        print(f"Wrappers in: {args.wrapper_dir}")

if __name__ == "__main__":
    main()
