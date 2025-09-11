from __future__ import annotations
from pathlib import Path
import os
import json
import re
from typing import Iterator, Tuple, Dict, Any, List, Optional

# -----------------------------------------------------------------------------
# Regexes (robust to AFP styles)
# -----------------------------------------------------------------------------
# Start of a lemma-like statement; allow names, locales, attributes.
LEMMA_START_RE = re.compile(
    r'^\s*(?:lemma|theorem|proposition|corollary)\b', re.UNICODE
)
# End markers for a block header / one-liner proofs.
QED_RE   = re.compile(r'^\s*qed\b', re.UNICODE)
BY_RE    = re.compile(r'^\s*by\b', re.UNICODE)
PROOF_RE = re.compile(r'^\s*proof\b', re.UNICODE)

# Theory header bits
IMPORTS_RE    = re.compile(r'^\s*imports\s+(.*)$', re.UNICODE)
THEORY_HDR_RE = re.compile(r'^\s*theory\s+([A-Za-z_][A-Za-z0-9_]*)\b', re.UNICODE)

# Identifiers, defs
ID_RE   = re.compile(r"\b([A-Za-z_][A-Za-z0-9_']*)\b", re.UNICODE)
DEF_RE  = re.compile(r"\b([A-Za-z_][A-Za-z0-9_']*)_def\b", re.UNICODE)

# Quoted formulas (ASCII quotes) and cartouches (‹ … ›)
QUOTE_RE     = re.compile(r'"([^"]+)"', re.UNICODE | re.DOTALL)
CARTOUCHE_RE = re.compile(r'‹(.*?)›', re.UNICODE | re.DOTALL)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _expand_dir(p: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(p)))

def _theory_header_info(thy_text: str) -> Dict[str, Any]:
    theory = None
    imports: List[str] = []
    for line in thy_text.splitlines()[:200]:
        m1 = THEORY_HDR_RE.match(line)
        if m1 and theory is None:
            theory = m1.group(1)
        m2 = IMPORTS_RE.match(line)
        if m2:
            # crude split on whitespace; filter 'begin'
            imports += [tok.strip() for tok in m2.group(1).split()
                        if tok.strip() and tok.strip() != "begin"]
    return {"theory": theory, "imports": imports}

def _premises_from_goal(stmt: str) -> Optional[str]:
    # Very light heuristic: text before the last '⟹' or inside ⟦ ⟧
    if "⟹" in stmt:
        try:
            return stmt.rsplit("⟹", 1)[0].strip()
        except Exception:
            return None
    if "⟦" in stmt and "⟧" in stmt:
        try:
            return stmt.split("⟦", 1)[1].split("⟧", 1)[0].strip()
        except Exception:
            return None
    return None

def _defs_and_names(block: str) -> Dict[str, Any]:
    defs = sorted(set(DEF_RE.findall(block)))
    names = sorted(set(ID_RE.findall(block)))
    return {"defs_in_block": defs, "names_in_block": names}

def _first_quoted_or_cartouche(text: str) -> Optional[str]:
    # Prefer the LAST quote/cartouche in the header (often the "shows" formula)
    qs = QUOTE_RE.findall(text)
    cs = CARTOUCHE_RE.findall(text)
    seq = []
    if qs: seq.extend(qs)
    if cs: seq.extend(cs)
    if seq:
        return seq[-1].strip()
    return None

def _header_region(lines: List[str], start: int, end: int) -> str:
    """
    Return the lemma header region (from 'start' up to the first 'proof'/'by'
    line or up to 'end' if not present). This keeps assumptions/shows text.
    """
    j = start + 1
    while j < end:
        L = lines[j]
        if PROOF_RE.match(L) or BY_RE.match(L) or QED_RE.match(L):
            break
        # Stop header if we hit an empty line followed by indented proof text.
        if not L.strip() and j + 1 < end and lines[j + 1].strip().startswith(("-", "have", "show", "fix", "assume")):
            break
        j += 1
    return "\n".join(lines[start:j])

def _outline_from_header(header_line: str, goal: str) -> str:
    # If the header itself carries a proof mode (rare), keep it; else add minimal skeleton.
    if PROOF_RE.search(header_line):
        proof_mode = header_line.strip()
        return f'lemma "{goal}"\n{proof_mode}\n  sorry\nqed\n'
    return f'lemma "{goal}"\nproof\n  sorry\nqed\n'

# -----------------------------------------------------------------------------
# Main iterator
# -----------------------------------------------------------------------------

def iter_lemmas_with_proofs(thy_text: str) -> Iterator[Tuple[str, str, str]]:
    """
    Yields (full_block_text, lemma_stmt, outline_with_sorry) from a .thy.
    Robust to:
      - named lemmas ('lemma foo:')
      - cartouches (‹ … ›) and ASCII quotes
      - 'shows' on later lines
      - one-liner 'by …' proofs (no 'qed')
    """
    lines = thy_text.splitlines()
    n = len(lines)
    i = 0
    while i < n:
        L = lines[i]
        if not LEMMA_START_RE.match(L):
            i += 1
            continue

        # Find end of this lemma block: next lemma start OR explicit 'qed'
        j = i + 1
        while j < n:
            if LEMMA_START_RE.match(lines[j]):
                break
            if QED_RE.match(lines[j]):
                j += 1  # include qed
                break
            # If it's a one-liner 'by ...', consider this line as block end
            if BY_RE.match(lines[j]) and i + 1 == j:
                j += 1
                break
            j += 1

        block = "\n".join(lines[i:j])  # full lemma region
        header = _header_region(lines, i, j)
        stmt = _first_quoted_or_cartouche(header)

        if not stmt:
            # Fallback: scan whole block for a quoted formula (last one wins)
            stmt = _first_quoted_or_cartouche(block)

        if stmt:
            outline = _outline_from_header(header.splitlines()[0] if header else lines[i], stmt)
            yield (block, stmt, outline)

        i = max(j, i + 1)

# -----------------------------------------------------------------------------
# Public miners
# -----------------------------------------------------------------------------

def mine_afp_corpus(src_dir: str, out_pairs: str) -> None:
    """
    Walk AFP sources, emit JSONL with {"goal": ..., "outline": ...}
    (kept for backward compatibility with existing scripts)
    """
    src = _expand_dir(src_dir)
    out_path = Path(out_pairs)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for thy in src.rglob("*.thy"):
            try:
                text = thy.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for _block, stmt, outline in iter_lemmas_with_proofs(text):
                try:
                    f.write(json.dumps({"goal": stmt, "outline": outline}, ensure_ascii=False) + "\n")
                    n += 1
                except Exception:
                    continue
    print(f"Wrote {n} pairs to {out_pairs}")

def mine_afp_corpus_rich(src_dir: str, out_jsonl: str) -> None:
    """
    Walk AFP sources and emit JSONL with:
      {"goal":..., "outline":..., "theory":..., "imports":[...], "premises":..., "defs_in_block":[...], "names_in_block":[...]}
    NOTE: still keeps 'goal' and 'outline' keys for compatibility.
    """
    src = _expand_dir(src_dir)
    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    for thy in src.rglob("*.thy"):
        try:
            text = thy.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        hdr = _theory_header_info(text)
        recs_written = 0
        with out_path.open("a", encoding="utf-8") as f:
            for block, stmt, outline in iter_lemmas_with_proofs(text):
                rec: Dict[str, Any] = {
                    "goal": stmt,
                    "outline": outline,
                    "theory": hdr.get("theory"),
                    "imports": hdr.get("imports", []),
                    "premises": _premises_from_goal(stmt),
                }
                rec.update(_defs_and_names(block))
                try:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
                    recs_written += 1
                except Exception:
                    continue
        # Optional: simple progress for big trees (comment out if noisy)
        # print(f"{thy}: {recs_written} lemmas")
    print(f"Wrote {n} rich records to {out_jsonl}")
