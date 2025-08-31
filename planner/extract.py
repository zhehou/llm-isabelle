# planner/extract.py
from __future__ import annotations
from pathlib import Path
import re
from typing import Iterator, Tuple

LEMMA_RE = re.compile(r'(?m)^\s*(lemma|theorem)\s+"([^"]+)"')
QED_RE   = re.compile(r'(?m)^\s*qed\b')

def iter_lemmas_with_proofs(thy_text: str) -> Iterator[Tuple[str,str,str]]:
    """
    Yields (full_block_text, lemma_stmt, outline_with_sorry) from a .thy.
    The outline is a quick heuristic: keep the header and (if present) a proof mode line,
    then insert `sorry` and close with `qed`.
    """
    lines = thy_text.splitlines()
    i = 0
    while i < len(lines):
        m = LEMMA_RE.match(lines[i])
        if not m: 
            i += 1; continue
        start = i
        stmt = m.group(2)
        # scan until qed or end-of-file
        j = i + 1
        while j < len(lines) and not QED_RE.match(lines[j]):
            j += 1
        end = j + 1 if j < len(lines) else j
        block = "\n".join(lines[start:end])

        # skeletonize: keep header and the first 'proof' line if it appears soon
        proof_mode = None
        for k in range(i+1, min(i+6, end)):
            if lines[k].strip().startswith("proof"):
                proof_mode = lines[k].strip()
                break
        header = lines[i]
        if proof_mode is None:
            outline = f'{header}\nproof\n  sorry\nqed\n'
        else:
            outline = f'{header}\n{proof_mode}\n  sorry\nqed\n'
        yield (block, stmt, outline)
        i = end

def mine_afp_corpus(src_dir: str, out_pairs: str) -> None:
    """
    Walk AFP sources, emit JSONL with {"goal": ..., "outline": ...}
    """
    import json
    src = Path(src_dir)
    n = 0
    with open(out_pairs, "w", encoding="utf-8") as f:
        for thy in src.rglob("*.thy"):
            try:
                text = thy.read_text(encoding="utf-8", errors="ignore")
                for _block, stmt, outline in iter_lemmas_with_proofs(text):
                    f.write(json.dumps({"goal": stmt, "outline": outline}, ensure_ascii=False) + "\n")
                    n += 1
            except Exception:
                continue
    print(f"Wrote {n} pairs to {out_pairs}")
