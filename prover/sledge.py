# prover/sledge.py
import json, re
from typing import List
from .isabelle_api import build_theory, run_theory

_SLEDGE_BY = re.compile(r"(?i)(?:try this:\s*)?(by\s+\([^)]+\)|by\s+\w+(?:\s+.+)?)")
_SLEDGE_METIS_LINE = re.compile(r"(?i)^metis\b.*?:\s*(.*)$")

def _extract_sledge_by_lines(text: str) -> List[str]:
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s: continue
        m = _SLEDGE_BY.search(s)
        if m:
            out.append(re.sub(r"\s+", " ", m.group(1).strip()))
            continue
        m2 = _SLEDGE_METIS_LINE.match(s)
        if m2:
            facts = m2.group(1).strip()
            out.append(f"by (metis {facts})" if facts else "by (metis)")
    seen, dedup = set(), []
    for c in out:
        if c not in seen:
            seen.add(c); dedup.append(c)
    return dedup

def sledgehammer_finishers(isabelle, session_id: str, steps: List[str], timeout_s: int = 5, limit: int = 5) -> List[str]:
    sh_cmd = f"sledgehammer [timeout = {int(timeout_s)}]"
    thy = build_theory(steps + [sh_cmd], add_print_state=False, end_with="sorry")
    resps = run_theory(isabelle, session_id, thy)
    texts = []
    for r in resps:
        if getattr(r, "response_type", "") != "NOTE": continue
        try:
            body = json.loads(r.response_body)
        except Exception:
            continue
        msg = str(body.get("message", "") or "")
        if "Try this:" in msg or "by " in msg or "Metis" in msg or "metis" in msg:
            texts.append(msg)
    cands: List[str] = []
    for t in texts:
        cands.extend(_extract_sledge_by_lines(t))
    out, seen = [], set()
    for c in cands:
        if c.startswith("by ") or c == "done":
            if c not in seen:
                seen.add(c); out.append(c)
                if len(out) >= limit: break
    return out
