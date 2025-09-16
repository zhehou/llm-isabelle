#!/usr/bin/env python3
"""
thys2goal.py  —  Unified Isabelle/HOL extractor

Features
- Parses lemma/theorem/corollary blocks from .thy files.
- Supports both quoted propositions and structured `assumes ... shows ...` style.
- Normalizes meta connectives (⋀ / \<And>, ⟹ / ==>).
- Hoists any meta-binders that appear after ⟹ back to the front.
- Parenthesizes occurrences of "X choose Y" as "(X choose Y)".
- Two modes:
    * --mode=minif2f  (backward compatible): expects repo/isabelle/{valid,test} and writes split files + wrappers.
    * --mode=generic   (new): rglob any .thy tree (e.g., PutnamBench/isabelle), write one goals file + optional wrapper.
- Emits index and imports summary CSVs in both modes.

Usage Examples
--------------
# miniF2F (original behavior preserved)
python thys2goal.py --mode minif2f --repo /path/to/miniF2F --outdir datasets/mini_f2f

# PutnamBench or any Isabelle tree
python thys2goal.py --mode generic --repo /path/to/PutnamBench/isabelle \
  --outfile datasets/putnambench_goals.txt --keep-all-props --list-skipped \
  --emit-wrappers --session-import HOL-Analysis

"""
import re
import csv
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Iterable

# ---------- Regexes ----------
QUOTE_OR_CT = r'(?:\"(?P<dq>.*?)\"|‹(?P<ct>.*?)›)'  # "..." or ‹...›

IMPORTS_PAT = re.compile(r'imports\s+(?P<imports>.*?)\s+begin', re.IGNORECASE | re.DOTALL)

THM_START  = re.compile(r'^[ \t]*(lemma|theorem|corollary)\b.*$', re.IGNORECASE | re.MULTILINE)
THM_CUTOFF = re.compile(r'^[ \t]*(lemma|theorem|corollary|qed|sorry|end)\b', re.IGNORECASE | re.MULTILINE)

FIXES_BLOCK   = re.compile(r'\bfixes\b(?P<fixes>.*?)(?=\b(assumes|shows|proof|qed|sorry|end|lemma|theorem|corollary)\b)', re.IGNORECASE | re.DOTALL)
ASSUMES_BLOCK = re.compile(r'\bassumes\b(?P<assumes>.*?)(?=\b(shows|proof|qed|sorry|end|lemma|theorem|corollary)\b)', re.IGNORECASE | re.DOTALL)
SHOWS_ONE     = re.compile(fr'\bshows\b\s*{QUOTE_OR_CT}', re.IGNORECASE | re.DOTALL)
THEOREM_COLON = re.compile(fr'(?:lemma|theorem|corollary)\b[^:\n]*:\s*{QUOTE_OR_CT}', re.IGNORECASE | re.DOTALL)

# fixes: “a b :: real” and “a::real b::real”
FIX_ITEM_GROUP  = re.compile(r'([A-Za-z0-9_\'\s]+?)::\s*([A-Za-z0-9_\.]+)')
FIX_ITEM_SINGLE = re.compile(r'\b([A-Za-z0-9_\'"]+)\s*::\s*([A-Za-z0-9_\.]+)')
AND_SPLIT       = re.compile(r'\band\b', re.IGNORECASE)

AFP_PREFIXES = ("Symmetric_Polynomials.", "Symmetric_Polynomials")

# ---------- helpers ----------
def clean(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

def header_body(text: str) -> Tuple[str, str]:
    m = re.search(r'^(.*?)\bbegin\b(.*)$', text, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
    if m: return m.group(1), m.group(2)
    m2 = re.search(r'^(.*?)\b(?:lemma|theorem|corollary)\b(.*)$', text, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
    if m2: return m2.group(1), m2.group(2)
    return "", text

def imports_from_header(header: str) -> List[str]:
    m = IMPORTS_PAT.search(header)
    if not m: return []
    raw = m.group('imports').strip()
    toks = re.split(r'[\s"]+', raw)
    return [t for t in toks if t]

def normalize_meta(s: str) -> str:
    # meta connectives
    s = s.replace(r'\<And>', '⋀')
    s = s.replace(r'\<Longrightarrow>', '⟹')
    s = s.replace('==>', '⟹')
    return s

# Hoist leading meta binders from an assumption string
HOIST_META = re.compile(r'^\s*(?:⋀|\\<And\>|\!\!)\s*([^.]*)\.\s*(.*)$', re.DOTALL)
def hoist_assumption_meta_binders(s: str) -> Tuple[List[str], str]:
    binders: List[str] = []
    s = normalize_meta(s)
    while True:
        m = HOIST_META.match(s)
        if not m: break
        head, rest = m.group(1).strip(), m.group(2)
        head_toks = [t for t in re.split(r'\s+', head) if t]
        binders.extend(head_toks)
        s = rest.strip()
    return binders, s

def fmt_binder(tok: str) -> str:
    tok = tok.strip()
    return f"({tok})" if '::' in tok and not (tok.startswith('(') and tok.endswith(')')) else tok

def parse_fixes(block: str) -> List[str]:
    out: List[str] = []
    if not block: return out
    parts = AND_SPLIT.split(block)
    for part in parts:
        part = part.strip()
        g = FIX_ITEM_GROUP.search(part)
        if g:
            vars_raw, ty = g.group(1), g.group(2)
            for v in filter(None, re.split(r'\s+', vars_raw.strip())):
                out.append(f"({v}::{ty})")
        for s in FIX_ITEM_SINGLE.finditer(part):
            v, ty = s.group(1), s.group(2)
            p = f"({v}::{ty})"
            if p not in out: out.append(p)
    return out

def wrap_choose_everywhere(s: str) -> str:
    # Wrap ANY "U choose V" (tokens or parenthesized chunks) as "(U choose V)"
    def repl(m):
        u, v = m.group(1), m.group(2)
        return f"({u} choose {v})"
    # token or (...) on each side (non-greedy for (...) )
    return re.sub(r'((?:\([^\)]*\)|[^()\s])+)\s+choose\s+((?:\([^\)]*\)|[^()\s])+)', repl, s)

def parse_assumes_and_hoist(block: str) -> Tuple[List[str], List[str]]:
    assumptions: List[str] = []
    hoisted: List[str] = []
    if not block: return assumptions, hoisted
    for m in re.finditer(QUOTE_OR_CT, block, flags=re.DOTALL):
        p = m.group('dq') if m.group('dq') is not None else m.group('ct')
        if p is None: continue
        p = clean(p)
        bnds, resid = hoist_assumption_meta_binders(p)
        hoisted.extend(bnds)
        resid = normalize_meta(resid)
        resid = wrap_choose_everywhere(resid)
        if resid: assumptions.append(resid)
    return assumptions, hoisted

def iter_statement_blocks(body: str) -> Iterable[str]:
    for m in THM_START.finditer(body):
        start = m.start()
        nxt = THM_CUTOFF.search(body, m.end())
        end = nxt.start() if nxt else len(body)
        yield body[start:end]

def final_hoist_anywhere(goal: str) -> str:
    """As a final safety pass, hoist ANY '⟹ ⋀… .' that slipped through into the front binder."""
    goal = normalize_meta(goal)
    # Extract current front binders
    front_binders: List[str] = []
    m_head = re.match(r'^\s*⋀([^.]*)\.\s*(.*)$', goal)
    core = goal
    if m_head:
        head, core = m_head.group(1), m_head.group(2)
        front_binders = [fmt_binder(t) for t in re.split(r'\s+', head.strip()) if t]

    # Iteratively hoist occurrences after arrows
    pat = re.compile(r'⟹\s*(?:⋀|\\<And\>|\!\!)\s*([^.]*)\.\s*')
    while True:
        m = pat.search(core)
        if not m: break
        head = m.group(1).strip()
        toks = [fmt_binder(t) for t in re.split(r'\s+', head) if t]
        # merge without dupes, keep order
        for t in toks:
            if t not in front_binders:
                front_binders.append(t)
        # remove this binder occurrence
        core = core[:m.start()] + "⟹ " + core[m.end():]

    # Rebuild
    if front_binders:
        return "⋀" + " ".join(front_binders) + ". " + core.strip()
    else:
        return core.strip()

def extract_props_from_block(block: str) -> List[str]:
    # 1) structured 'shows "…"' / 'shows ‹…›'
    mshows = SHOWS_ONE.search(block)
    if mshows:
        fixes = parse_fixes(FIXES_BLOCK.search(block).group('fixes') if FIXES_BLOCK.search(block) else "")
        assms, hoisted = parse_assumes_and_hoist(ASSUMES_BLOCK.search(block).group('assumes')
                                                 if ASSUMES_BLOCK.search(block) else "")
        binders_list: List[str] = []
        for b in fixes + hoisted:
            fb = fmt_binder(b)
            if fb not in binders_list: binders_list.append(fb)

        conc_raw = mshows.group('dq') if mshows.group('dq') is not None else mshows.group('ct')
        conc = clean(normalize_meta(conc_raw))
        conc = wrap_choose_everywhere(conc)

        binder = ("⋀" + " ".join(binders_list) + ". ") if binders_list else ""
        apref  = (" ⟹ ".join(assms) + " ⟹ ") if assms else ""
        goal = binder + apref + conc
        goal = final_hoist_anywhere(goal)
        goal = wrap_choose_everywhere(goal)
        return [goal]

    # 2) colon style  lemma XXX : "…"
    mcolon = THEOREM_COLON.search(block)
    if mcolon:
        conc_raw = mcolon.group('dq') if mcolon.group('dq') is not None else mcolon.group('ct')
        goal = wrap_choose_everywhere(clean(normalize_meta(conc_raw)))
        goal = final_hoist_anywhere(goal)
        goal = wrap_choose_everywhere(goal)
        return [goal]

    # 3) last resort: last quoted/cartouche as conc
    cands = [clean(normalize_meta(m.group('dq') if m.group('dq') is not None else m.group('ct')))
             for m in re.finditer(QUOTE_OR_CT, block, flags=re.DOTALL)]
    if cands:
        goal = wrap_choose_everywhere(cands[-1])
        goal = final_hoist_anywhere(goal)
        goal = wrap_choose_everywhere(goal)
        return [goal]
    return []

def extract_props_with_ctx(text: str) -> List[str]:
    _header, body = header_body(text)
    acc: List[str] = []
    for blk in iter_statement_blocks(body):
        acc.extend(extract_props_from_block(blk))
    if acc: return acc
    # fallback on the whole text for colon style
    mcolon = THEOREM_COLON.search(text)
    if mcolon:
        conc_raw = mcolon.group('dq') if mcolon.group('dq') is not None else mcolon.group('ct')
        goal = wrap_choose_everywhere(clean(normalize_meta(conc_raw)))
        goal = final_hoist_anywhere(goal)
        goal = wrap_choose_everywhere(goal)
        return [goal]
    return []

def write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in lines: f.write(x + "\n")

def write_lemmas_thy(path: Path, session_import: str, goals: List[str], thy_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f'theory {thy_name}\n  imports {session_import}\nbegin\n')
        for g in goals:
            f.write('lemma "' + g.replace('"', '\\"') + '"\n  sorry\n\n')
        f.write('end\n')

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Unified Isabelle extractor (miniF2F and generic trees).")
    ap.add_argument("--repo", required=True, help="Path to dataset root. For --mode=minif2f, this is the miniF2F repo; for --mode=generic, this is any Isabelle tree (e.g., PutnamBench/isabelle).")
    ap.add_argument("--outdir", default="datasets/mini_f2f", help="(minif2f mode) Output directory")
    ap.add_argument("--exclude-afp", action="store_true", help="(minif2f mode) Exclude AFP problems (e.g., Symmetric_Polynomials.Vieta)")
    ap.add_argument("--keep-all-props", action="store_true", help="Keep every parsed proposition per file (default: only first)")
    ap.add_argument("--list-skipped", action="store_true", help="List files with no parsed proposition")
    ap.add_argument("--mode", choices=["minif2f","generic"], default="minif2f",
                    help="minif2f: expect isabelle/{valid,test}. generic: rglob all .thy under --repo and write a single goals file.")
    ap.add_argument("--outfile", default="datasets/putnambench_goals.txt",
                    help="(generic mode) Single output file with one goal per line.")
    ap.add_argument("--emit-wrappers", action="store_true",
                    help="(generic mode) Emit a wrapper .thy containing all goals as lemmas.")
    ap.add_argument("--session-import", default="HOL-Analysis",
                    help="(generic mode) Session import used by wrapper theory (e.g., Main, HOL, HOL-Library, HOL-Analysis).")
    args = ap.parse_args()

    repo = Path(args.repo); outdir = Path(args.outdir)

    # ------------------------------
    # MODE: GENERIC (e.g., PutnamBench)
    # ------------------------------
    if args.mode == "generic":
        root = repo if repo.is_dir() else repo.parent
        # Allow either a repo that IS the isabelle tree, or has an 'isabelle' subdir.
        scan_root = root
        # If a direct .thy tree isn't the given dir, try "<repo>/isabelle"
        if not any(scan_root.glob("*.thy")) and (root / "isabelle").exists():
            scan_root = root / "isabelle"

        if not scan_root.exists():
            raise SystemExit(f"[ERROR] No Isabelle tree at {repo} (looked at {scan_root})")

        files = sorted(scan_root.rglob("*.thy"))
        if not files:
            raise SystemExit(f"[ERROR] No .thy files under {scan_root}")

        all_props: List[str] = []
        import_counter: Dict[str, int] = {}
        index_rows = []
        skipped: List[str] = []

        for thy in files:
            try:
                text = thy.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                skipped.append(str(thy))
                continue

            header, _ = header_body(text)
            imports = imports_from_header(header)
            for im in imports:
                import_counter[im] = import_counter.get(im, 0) + 1

            props = extract_props_with_ctx(text)
            if not props:
                skipped.append(str(thy))
                index_rows.append({"split": "all", "problem_name": thy.stem, "imports": " ".join(imports), "n_props": 0})
                continue
            if not args.keep_all_props and len(props) > 1:
                props = [props[0]]

            index_rows.append({"split": "all", "problem_name": thy.stem, "imports": " ".join(imports), "n_props": len(props)})
            all_props.extend(props)

        # Write single goals file
        out_path = Path(args.outfile)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_lines(out_path, all_props)

        # Write metadata next to outfile
        idx = out_path.with_suffix(".index.csv")
        with idx.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["split","problem_name","imports","n_props"])
            w.writeheader(); w.writerows(index_rows)

        imp = out_path.with_suffix(".imports.csv")
        with imp.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["import_name", "count"])
            for name, cnt in sorted(import_counter.items(), key=lambda x: (-x[1], x[0])):
                w.writerow([name, cnt])

        # Optional wrapper
        if args.emit_wrappers:
            wrap = out_path.with_suffix(".Wrapper.thy")
            # Deduplicate but keep order
            seen = set()
            dedup_goals = []
            for g in all_props:
                if g not in seen:
                    seen.add(g)
                    dedup_goals.append(g)
            write_lemmas_thy(wrap, args.session_import, dedup_goals, wrap.stem)

        print(f"generic: {len(all_props)} -> {out_path}")
        print(f"Index:   {idx}")
        print(f"Imports: {imp}")
        if args.list_skipped and skipped:
            print(f"Skipped ({len(skipped)}):")
            for s in skipped[:60]: print("  -", s)
        return

    # ------------------------------
    # MODE: MINIF2F (existing behavior)
    # ------------------------------
    if not (repo / "isabelle").exists():
        raise SystemExit(f"[ERROR] {repo} missing 'isabelle/'")

    def is_afp_import(imp: str) -> bool:
        return any(imp.startswith(pref) for pref in AFP_PREFIXES)

    splits = {"validation": "valid", "test": "test"}
    all_props: Dict[str, List[str]] = {"validation": [], "test": []}
    index_rows = []
    import_counter: Dict[str, int] = {}
    skipped: List[str] = []

    for split, src in splits.items():
        files = sorted((repo / "isabelle" / src).glob("*.thy"))
        if not files:
            raise SystemExit(f"[ERROR] No .thy in {repo}/isabelle/{src}")
        for thy in files:
            text = thy.read_text(encoding="utf-8", errors="ignore")
            header, _ = header_body(text)
            imports = imports_from_header(header)
            for im in imports:
                import_counter[im] = import_counter.get(im, 0) + 1

            if args.exclude_afp and any(is_afp_import(imp) for imp in imports):
                index_rows.append({"split": split, "problem_name": thy.stem, "imports": " ".join(imports), "n_props": 0})
                continue

            props = extract_props_with_ctx(text)
            if not props:
                skipped.append(f"{split}/{thy.name}")
                index_rows.append({"split": split, "problem_name": thy.stem, "imports": " ".join(imports), "n_props": 0})
                continue
            if not args.keep_all_props and len(props) > 1:
                props = [props[0]]

            index_rows.append({"split": split, "problem_name": thy.stem, "imports": " ".join(imports), "n_props": len(props)})
            all_props[split].extend(props)

    outdir.mkdir(parents=True, exist_ok=True)
    # Output files (preserve original names for backward compatibility)
    write_lines(outdir / "mini_f2f_validation.txt", all_props["validation"])
    write_lines(outdir / "mini_f2f_test.txt",       all_props["test"])

    with (outdir / "mini_f2f_index.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split","problem_name","imports","n_props"])
        w.writeheader(); w.writerows(index_rows)
    with (outdir / "mini_f2f_imports_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["import_name", "count"])
        for name, cnt in sorted(import_counter.items(), key=lambda x: (-x[1], x[0])): w.writerow([name, cnt])

    # Wrapper theories with the exact names used previously
    write_lemmas_thy(outdir / "MiniF2F_Validation_Lemmas.thy", "MiniF2F_Base",
                     all_props["validation"], "MiniF2F_Validation_Lemmas")
    write_lemmas_thy(outdir / "MiniF2F_Test_Lemmas.thy", "MiniF2F_Base",
                     all_props["test"], "MiniF2F_Test_Lemmas")

    print("Done.")
    print(f"validation: {len(all_props['validation'])} -> {outdir/'mini_f2f_validation.txt'}")
    print(f"test:       {len(all_props['test'])} -> {outdir/'mini_f2f_test.txt'}")
    print("Wrappers:   ", outdir/'MiniF2F_Validation_Lemmas.thy', "and", outdir/'MiniF2F_Test_Lemmas.thy')
    print(f"Index:      {outdir/'mini_f2f_index.csv'}")
    print(f"Imports:    {outdir/'mini_f2f_imports_summary.csv'}")
    if args.list_skipped and skipped:
        print(f"Skipped ({len(skipped)}):")
        for s in skipped[:60]: print("  -", s)

if __name__ == "__main__":
    main()