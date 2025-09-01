# prover/isabelle_api.py
import os, json, tempfile, textwrap
import re
from typing import List, Tuple, Optional, Dict, Any
from isabelle_client import start_isabelle_server, get_isabelle_client, IsabelleResponse

from .config import EXTRA_IMPORTS

def _header(imports=None):
    imps = ["Main"] + list(imports or []) + list(EXTRA_IMPORTS or [])
    return f"theory Scratch\nimports {' '.join(imps)}\nbegin\n"

FOOTER = "end\n"

_use_calls = 0

# --- Helpers for NOTE parsing / subgoal counting ---
_SUBGOALS_PATTERNS = [
    re.compile(r"\b(\d+)\s+subgoals?\b", re.IGNORECASE),
    re.compile(r"(?i)\bgoal\s*\(\s*(\d+)\s+subgoals?\s*\)"),
]

def note_messages(resps: List[IsabelleResponse]) -> List[str]:
    """Collect 'writeln' NOTE messages from Isabelle responses."""
    out: List[str] = []
    for r in resps:
        if getattr(r, "response_type", "") != "NOTE":
            continue
        try:
            body = json.loads(r.response_body)
        except Exception:
            continue
        if body.get("kind") == "writeln":
            out.append(str(body.get("message", "")))
    return out

def estimate_subgoals(resps: List[IsabelleResponse]) -> Optional[int]:
    """Best-effort parse of the 'N subgoals' count from recent NOTE output."""
    for msg in reversed(note_messages(resps)):
        for pat in _SUBGOALS_PATTERNS:
            m = pat.search(msg)
            if m:
                return int(m.group(1))
    return None

def build_theory(steps: List[str], add_print_state: bool, end_with: Optional[str]) -> str:
    body = [steps[0]] + ["  " + s for s in steps[1:]]
    if add_print_state: body.append("  print_state")
    if end_with: body.append("  " + end_with)
    return textwrap.dedent(_header() + "\n".join(body) + "\n\n" + FOOTER)

def run_theory(isabelle, session_id: str, theory_text: str) -> List[IsabelleResponse]:
    global _use_calls
    _use_calls += 1
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "Scratch.thy")
        with open(p, "w", encoding="utf-8") as f:
            f.write(theory_text)
        return list(isabelle.use_theories(theories=["Scratch"], session_id=session_id, master_dir=tmp))

def finished_ok(resps: List[IsabelleResponse]) -> Tuple[bool, Dict[str, Any]]:
    for r in reversed(resps):
        if getattr(r, "response_type", "") == "FINISHED":
            try:
                obj = json.loads(r.response_body)
                return bool(obj.get("ok", False)), obj
            except Exception:
                return False, {}
    return False, {}

def last_print_state_block(resps: List[IsabelleResponse]) -> str:
    txt = ""
    for r in resps:
        if getattr(r, "response_type", "") == "NOTE":
            try:
                body = json.loads(r.response_body)
                if body.get("kind") == "writeln":
                    msg = str(body.get("message", ""))
                    if "subgoal" in msg or "goal (" in msg or "goal\n" in msg:
                        txt = msg
            except Exception:
                pass
    return txt

def use_calls_count() -> int:
    return _use_calls
