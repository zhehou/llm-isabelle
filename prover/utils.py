# prover/utils.py
import re
from typing import Optional

ANSI = {
    "reset": "\x1b[0m","bold": "\x1b[1m","dim": "\x1b[2m",
    "green": "\x1b[32m","red": "\x1b[31m","yellow": "\x1b[33m",
    "blue": "\x1b[34m","cyan": "\x1b[36m","gray": "\x1b[90m",
}

def color(use_color: bool, key: str, s: str) -> str:
    return (ANSI.get(key, "") + s + ANSI["reset"]) if use_color else s

SUBGOALS_PATTERNS = [
    re.compile(r"\b(\d+)\s+subgoals?\b", re.IGNORECASE),
    re.compile(r"(?i)\bgoal\s*\(\s*(\d+)\s+subgoals?\s*\)"),
]

def parse_subgoals(block: str) -> Optional[int]:
    for pat in SUBGOALS_PATTERNS:
        m = pat.search(block)
        if m: return int(m.group(1))
    return None

import hashlib

def state_fingerprint(s: str) -> str:
    """Hash a normalized print_state block to detect revisits."""
    s = " ".join(s.strip().split())
    return hashlib.sha1(s.encode("utf-8")).hexdigest()
