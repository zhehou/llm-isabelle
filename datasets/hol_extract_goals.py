#!/usr/bin/env python3
"""
Extract lemma/theorem statements from Isabelle/HOL theories, with robust theory
header parsing (theory name + imports). Outputs a JSONL with per-goal metadata
and a plain TXT list of goals.

Usage:
  python datasets/hol_extract_goals_v4.py \
    --isabelle-hol /Applications/Isabelle2025.app/src/HOL \
    --out datasets \
    --only List Set Nat Algebra Analysis Complex_Analysis Number_Theory Binomial

Outputs:
  datasets/hol_goals.jsonl
  datasets/hol_goals.txt
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Optional, Tuple, Iterable

# Strip (* ... *) comments (non-nested; good enough for headers)
COMMENT_RE = re.compile(r"\(\*.*?\*\)", re.DOTALL)

# Lemma/theorem starts
DECL_RE = re.compile(r'(?m)^\s*(lemma|theorem)\b')

# Theory header bits
THEORY_RE = re.compile(r'(?m)^\s*theory\s+(\S+)\b')
IMPORTS_BLOCK_RE = re.compile(r'(?s)\bimports\b(.*?)\bbegin\b')

_MIN_LEN, _MAX_LEN = 5, 2000

def _balanced_cartouche(text: str, start: int) -> Optional[Tuple[int, int]]:
    j = text.find('›', start + 1)
    return (start, j + 1) if j != -1 else None

def _quoted_string(text: str, start: int) -> Optional[Tuple[int, int]]:
    i = start + 1
    while True:
        j = text.find('"', i)
        if j == -1: return None
        k, bs = j - 1, 0
        while k >= 0 and text[k] == '\\':
            bs += 1; k -= 1
        if bs % 2 == 0:
            return (start, j + 1)
        i = j + 1

def _first_stmt_after(text: str, pos: int) -> Optional[str]:
    window = text[pos:pos + 12000]
    cands = []
    q = window.find('"')
    if q != -1:
        span = _quoted_string(window, q)
        if span: cands.append((span[0], window[span[0]+1:span[1]-1]))
    c = window.find('‹')
    if c != -1:
        span = _balanced_cartouche(window, c)
        if span: cands.append((span[0], window[span[0]+1:span[1]-1]))
    if not cands: return None
    cands.sort(key=lambda t: t[0])
    stmt = cands[0][1].strip()
    return stmt if _MIN_LEN <= len(stmt) <= _MAX_LEN else None

def _strip_comments(s: str) -> str:
    return COMMENT_RE.sub("", s)

def _split_import_tokens(raw: str) -> list[str]:
    """
    Handle imports like:
      imports Main "HOL-Library.Multiset" "~~/src/HOL/Number_Theory/Primes"
    We normalize by removing quotes and common path punctuation, then split.
    """
    s = raw
    s = s.replace('"', ' ')
    s = s.replace('(', ' ').replace(')', ' ').replace('+', ' ')
    s = s.replace('~', ' ').replace('/', ' ').replace('\\', ' ')
    s = re.sub(r"\s+", " ", s.strip())
    if not s: return []
    toks = s.split(" ")
    # Keep reasonable atoms (theory-ish names)
    cleaned = [t for t in toks if re.match(r"^[A-Za-z0-9_.'-]+$", t)]
    return cleaned

def _parse_header(text: str) -> tuple[Optional[str], list[str]]:
    head = _strip_comments(text[: min(len(text), 20000)])
    m_thy = THEORY_RE.search(head)
    theory = m_thy.group(1) if m_thy else None
    imports: list[str] = []
    m_imp = IMPORTS_BLOCK_RE.search(head)
    if m_imp:
        imports = _split_import_tokens(m_imp.group(1))
    return theory, imports

def _want_dir(sub: str, only: Optional[set[str]]) -> bool:
    if not only: return True
    parts = Path(sub).parts
    return any(p in only for p in parts)

def _extract_file(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    theory, imports = _parse_header(text)
    for m in DECL_RE.finditer(text):
        stmt = _first_stmt_after(text, m.end())
        if stmt:
            yield {"goal": stmt, "file": str(path), "theory": theory, "imports": imports}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isabelle-hol", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=[])
    args = ap.parse_args()

    hol = Path(args.isabelle_hol).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    only = set(args.only) if args.only else None
    thy_files = [p for p in hol.rglob("*.thy") if _want_dir(str(p.parent.relative_to(hol)), only)]
    print(f"Scanning {len(thy_files)} .thy files under {hol}")

    rows = []
    for thy in thy_files:
        try:
            rows.extend(_extract_file(thy))
        except Exception as e:
            print(f"[warn] {thy}: {e}")

    seen, uniq = set(), []
    for r in rows:
        key = (r["goal"], r["file"])
        if key not in seen:
            seen.add(key); uniq.append(r)

    (out / "hol_goals.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in uniq) + ("\n" if uniq else ""),
        encoding="utf-8")
    (out / "hol_goals.txt").write_text(
        "\n".join(r["goal"] for r in uniq) + ("\n" if uniq else ""),
        encoding="utf-8")

    print(f"Wrote {len(uniq)} items → {out/'hol_goals.jsonl'}")
    print(f"Also wrote plain list → {out/'hol_goals.txt'}")

if __name__ == "__main__":
    main()