#!/usr/bin/env python3
# datasets/prep_minif2f_isabelle.py
#
# ONE goal file per split (validation/test), with context:
#   ⋀(x::T) (y::U) ... . A1 ⟹ A2 ⟹ ... ⟹ C
#
# Fixes:
#  - Hoist ANY inner meta-binders (⋀ / \<And> / !!) that occur after ⟹ back to the front binder.
#  - Normalize meta arrows to ⟹ inside assumptions/conclusion.
#  - Parenthesize every occurrence of   X choose Y   as   (X choose Y).
#  - Emit wrapper theories:
#      MiniF2F_Validation_Lemmas.thy
#      MiniF2F_Test_Lemmas.thy
#
import re
import csv
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Iterable

# ---------- Regexes ----------
QUOTE_OR_CT = r'(?:\"(?P<dq>.*?)\"|‹(?P<ct>.*?)›)'  # "..." or ‹...›

IMPORTS_PAT = re.compile(r'imports\s+(?P<imports>.*?)\s+begin', re.IGNORECASE | re.DOTALL)

THM_START  = re.compile(r'^[ \t]*(lemma|theorem)\b.*$', re.IGNORECASE | re.MULTILINE)
THM_CUTOFF = re.compile(r'^[ \t]*(lemma|theorem|qed|sorry|end)\b', re.IGNORECASE | re.MULTILINE)

FIXES_BLOCK   = re.compile(r'\bfixes\b(?P<fixes>.*?)(?=\b(assumes|shows|proof|qed|sorry|end|lemma|theorem)\b)', re.IGNORECASE | re.DOTALL)
ASSUMES_BLOCK = re.compile(r'\bassumes\b(?P<assumes>.*?)(?=\b(shows|proof|qed|sorry|end|lemma|theorem)\b)', re.IGNORECASE | re.DOTALL)
SHOWS_ONE     = re.compile(fr'\bshows\b\s*{QUOTE_OR_CT}', re.IGNORECASE | re.DOTALL)
THEOREM_COLON = re.compile(fr'(?:lemma|theorem)\b[^:\n]*:\s*{QUOTE_OR_CT}', re.IGNORECASE | re.DOTALL)

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
    m2 = re.search(r'^(.*?)\b(?:lemma|theorem)\b(.*)$', text, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
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
    props: List[str] = []

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

    mcolon = THEOREM_COLON.search(block)
    if mcolon:
        conc_raw = mcolon.group('dq') if mcolon.group('dq') is not None else mcolon.group('ct')
        goal = wrap_choose_everywhere(clean(normalize_meta(conc_raw)))
        goal = final_hoist_anywhere(goal)
        goal = wrap_choose_everywhere(goal)
        return [goal]

    # last resort: last quoted/cartouche as conc
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
    ap = argparse.ArgumentParser(description="miniF2F Isabelle extractor (one file per split, with context).")
    ap.add_argument("--repo", required=True, help="Path to cloned openai/miniF2F or facebookresearch/miniF2F")
    ap.add_argument("--outdir", default="datasets/mini_f2f", help="Output directory")
    ap.add_argument("--exclude-afp", action="store_true", help="Exclude AFP problems (e.g., Symmetric_Polynomials.Vieta)")
    ap.add_argument("--keep-all-props", action="store_true", help="Keep every parsed proposition per file (default: only first)")
    ap.add_argument("--list-skipped", action="store_true", help="List files with no parsed proposition")
    args = ap.parse_args()

    repo = Path(args.repo); outdir = Path(args.outdir)
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
    write_lines(outdir / "mini_f2f_validation.txt", all_props["validation"])
    write_lines(outdir / "mini_f2f_test.txt",       all_props["test"])

    with (outdir / "mini_f2f_index.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split","problem_name","imports","n_props"])
        w.writeheader(); w.writerows(index_rows)
    with (outdir / "mini_f2f_imports_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["import_name", "count"])
        for name, cnt in sorted(import_counter.items(), key=lambda x: (-x[1], x[0])): w.writerow([name, cnt])

    # Wrapper theories with the exact names you requested
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
