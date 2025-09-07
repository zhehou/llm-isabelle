from __future__ import annotations
from pathlib import Path
import re
from typing import Iterator, Tuple

# Anchor at line starts; tolerate spaces before keywords.
LEMMA_RE = re.compile(r'^\s*(?:lemma|theorem)\s+"([^"]+)"')
QED_RE   = re.compile(r'^\s*qed\b')
PROOF_RE = re.compile(r'^\s*proof\b.*')  # catch "proof", "proof -", "proof (cases ...)", etc.

def iter_lemmas_with_proofs(thy_text: str) -> Iterator[Tuple[str, str, str]]:
    """
    Yields (full_block_text, lemma_stmt, outline_with_sorry) from a .thy.
    The outline is a quick heuristic: keep the header and (if present) a proof mode line,
    then insert `sorry` and close with `qed`.
    """
    lines = thy_text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m = LEMMA_RE.match(line)
        if not m:
            i += 1
            continue

        start = i
        stmt = m.group(1)

        # Scan forward until the first 'qed' line or EOF.
        j = i + 1
        while j < n and not QED_RE.match(lines[j]):
            j += 1
        end = j + 1 if j < n else j  # include 'qed' line if found

        block = "\n".join(lines[start:end])

        # Skeletonize: keep header and the first 'proof' line if it appears soon.
        proof_mode = None
        for k in range(i + 1, min(i + 6, end)):
            if PROOF_RE.match(lines[k]):
                proof_mode = lines[k].strip()
                break

        header = line
        if proof_mode is None:
            outline = f"{header}\nproof\n  sorry\nqed\n"
        else:
            outline = f"{header}\n{proof_mode}\n  sorry\nqed\n"

        yield (block, stmt, outline)
        i = end  # jump past this block

def mine_afp_corpus(src_dir: str, out_pairs: str) -> None:
    """
    Walk AFP sources, emit JSONL with {"goal": ..., "outline": ...}
    """
    import json
    src = Path(src_dir)
    out_path = Path(out_pairs)
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
                    # Keep going even if a single record fails to serialize/write
                    continue
    print(f"Wrote {n} pairs to {out_pairs}")