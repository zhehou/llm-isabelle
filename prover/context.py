# =============================================
# File: prover/context.py
# Purpose: Fast, lightweight provider of file/theory-aware context
#          for premise selection. Parses .thy text to collect nearby
#          facts (defs/lemmas/etc.) and returns their IDs for boosting.
# =============================================
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterable
import re

# ---- Simple .thy tokenizer (robust to whitespace/comments) ----
# We aim to be permissive and fast rather than complete.
# Captures: definition/abbreviation/fun/primrec/function/datatype/record
#           lemma/theorem/corollary/lemma\s+\(named_simps?rule\)
#           note/lemmas declarations
# Each fact is recorded with a stable ID and byte offset.

_FACT_PATTERNS = [
    ("definition", re.compile(r"(^|\n)\s*definition\s+(?P<name>[A-Za-z0-9_'.-]+)?\s*[:=]", re.MULTILINE)),
    ("abbreviation", re.compile(r"(^|\n)\s*abbreviation\s+(?P<name>[A-Za-z0-9_'.-]+)?\s*[:=]", re.MULTILINE)),
    ("fun", re.compile(r"(^|\n)\s*fun\s+(?P<name>[A-Za-z0-9_'.-]+)", re.MULTILINE)),
    ("primrec", re.compile(r"(^|\n)\s*primrec\s+(?P<name>[A-Za-z0-9_'.-]+)", re.MULTILINE)),
    ("function", re.compile(r"(^|\n)\s*function\s+(?P<name>[A-Za-z0-9_'.-]+)", re.MULTILINE)),
    ("datatype", re.compile(r"(^|\n)\s*datatype\s+(?P<name>[A-Za-z0-9_'.-]+)?\b", re.MULTILINE)),
    ("record", re.compile(r"(^|\n)\s*record\s+(?P<name>[A-Za-z0-9_'.-]+)\b", re.MULTILINE)),
    ("lemma", re.compile(r"(^|\n)\s*(lemma|theorem|corollary)\s+(?P<name>[A-Za-z0-9_'.-]+)?\s*:\s*\"(?P<stmt>.*?)\"", re.MULTILINE|re.DOTALL)),
    ("note", re.compile(r"(^|\n)\s*(note|lemmas)\s+(?P<name>[A-Za-z0-9_'.-]+)\s*:\s*(?P<stmt>[^\n;]+)", re.MULTILINE)),
]

_COMMENT = re.compile(r"\(\*[^*]*\*+(?:[^)(*][^*]*\*+)*\)")  # (* ... *) nested-light removal


def _strip_comments(s: str) -> str:
    # remove (* ... *) comments quickly; safe for most files
    return re.sub(_COMMENT, " ", s)


def _fact_id(kind: str, name: Optional[str], path: Path, idx: int) -> str:
    base = path.name
    nm = name or f"{kind}_{idx}"
    return f"{base}:{nm}:{idx}"


@dataclass
class Fact:
    fact_id: str
    kind: str
    name: Optional[str]
    text: str  # short text/caption/statement
    byte_offset: int
    file: str


class ContextWindow:
    """Maintain a mapping from file->ordered facts and offer a window query.

    This module is intentionally independent from the retrieval models. It
    just surfaces IDs for boosting and (optionally) texts for seeding an index.
    """
    def __init__(self, window_size: int = 400):
        self.window_size = int(window_size)
        self._facts_by_file: Dict[str, List[Fact]] = {}

    # --- ingestion ---
    def ingest_theory(self, path: str | Path) -> List[Fact]:
        p = Path(path)
        txt = p.read_text(encoding="utf-8", errors="ignore")
        txt = _strip_comments(txt)
        facts: List[Fact] = []
        for kind, pat in _FACT_PATTERNS:
            for m in pat.finditer(txt):
                name = m.groupdict().get("name")
                stmt = m.groupdict().get("stmt")
                start = m.start()
                fid = _fact_id(kind, name, p, len(facts))
                # Take a compact text: name + stmt/head (if any)
                snippet = (name or kind)
                if stmt:
                    snippet += " :: " + " ".join(stmt.split())[:512]
                facts.append(Fact(fid, kind, name, snippet, start, str(p)))
        facts.sort(key=lambda f: f.byte_offset)
        self._facts_by_file[str(p)] = facts
        return facts

    def ingest_many(self, paths: Iterable[str | Path]) -> None:
        for pt in paths:
            try:
                self.ingest_theory(pt)
            except Exception:
                continue

    # --- query ---
    def facts_for(self, file_path: str | Path, byte_offset: int) -> List[str]:
        """Return up to `window_size` fact IDs before `byte_offset` in `file_path`.
        The order is newest-first (closest to the goal), which matches human use.
        """
        key = str(Path(file_path))
        facts = self._facts_by_file.get(key, [])
        if not facts:
            return []
        # binary search could be added; linear scan is ok for few hundred facts
        prev = [f for f in facts if f.byte_offset <= int(byte_offset)]
        prev_ids = [f.fact_id for f in prev][-self.window_size:]
        return list(reversed(prev_ids))  # closest first

    def seed_pairs(self) -> List[Tuple[str, str]]:
        """Yield (fact_id, text) for seeding a retrieval index.
        Optional helper: retrieval can call this to pre-add facts.
        """
        out: List[Tuple[str, str]] = []
        for facts in self._facts_by_file.values():
            for f in facts:
                out.append((f.fact_id, f.text))
        return out