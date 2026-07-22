"""Source-order progress measures for verified, partially admitted Isar proofs.

The planner deliberately checks candidates with ``sorry`` enabled while it is
constructing a proof.  A useful repair may therefore replace one large hole by
several smaller, later holes.  Counting holes alone rejects exactly those
repairs.  This module measures the *verified source frontier*: how much proof
structure Isabelle can read before the first remaining admission.
"""

from __future__ import annotations

import re
from typing import Tuple


_SORRY_RE = re.compile(r"\bsorry\b")
_PROOF_COMMAND_RE = re.compile(
    r"^\s*(?:lemma|theorem|corollary|proposition|proof|qed|next|case|"
    r"fix|assume|presume|obtain|have|show|thus|hence|then|from|with|"
    r"using|unfolding|also|moreover|ultimately|finally|apply|by|done)\b"
)


def frontier_score(text: str) -> Tuple[int, int, int, int]:
    """Return a lexicographic score for a partially admitted proof.

    Complete proofs dominate partial proofs.  For partial proofs, only source
    before the first ``sorry`` contributes positive evidence.  This prevents a
    rewrite later in the document from masquerading as progress and makes the
    scheduler deterministic.
    """

    source = text or ""
    holes = list(_SORRY_RE.finditer(source))
    complete = int(not holes)
    prefix = source if complete else source[: holes[0].start()]
    meaningful_lines = [
        line
        for line in prefix.splitlines()
        if line.strip() and not line.lstrip().startswith(("(*", "--"))
    ]
    commands = sum(bool(_PROOF_COMMAND_RE.match(line)) for line in meaningful_lines)
    return complete, commands, len(meaningful_lines), -len(holes)


def frontier_advanced(before: str, after: str) -> bool:
    """True when ``after`` moves the earliest verified source frontier."""

    return after != before and frontier_score(after) > frontier_score(before)

