# prover/proof_io.py
import os, re
from typing import List

def slugify_goal(goal: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_]+", "_", goal).strip("_")
    import hashlib
    h = hashlib.sha1(goal.encode("utf-8")).hexdigest()[:8]
    return f"{base[:50]}_{h}" if base else h

def write_theory_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
