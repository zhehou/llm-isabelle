# prover/mining.py
import os, re, json, tempfile, textwrap
from typing import List, Dict
from .isabelle_api import run_theory

FIND_NAME_TOKENS = re.compile(r"[A-Za-z][A-Za-z0-9_']{2,}")
_LEMMA_LINE = re.compile(r"^\s*\d+\.\s*([A-Za-z0-9_\.]+):")

def _build_find_theorems_theory(symbols: List[str]) -> str:
    body = "\n".join(f'find_theorems name:{s}' for s in symbols[:8])
    return textwrap.dedent(f"theory FT_Scratch\nimports Main\nbegin\n\n{body}\n\nend").strip()

def _build_find_theorems_filtered(symbols: List[str], filters: List[str]) -> str:
    lines = []
    for s in symbols[:8]:
        for flt in filters:
            lines.append(f"find_theorems {flt} name:{s}" if flt else f"find_theorems name:{s}")
    return textwrap.dedent(
    "theory FT2_Scratch\nimports Main\nbegin\n\n{}\n\nend".format("\n".join(lines))
).strip()

def mine_lemmas_from_state(isabelle, session_id: str, state_hint: str, max_lemmas: int = 6) -> List[str]:
    toks = list(dict.fromkeys(FIND_NAME_TOKENS.findall(state_hint)))
    if not toks: return []
    theory_text = _build_find_theorems_theory(toks)
    resps = run_theory(isabelle, session_id, theory_text)
    lemmas, seen = [], set()
    for r in resps:
        if getattr(r, "response_type", "") != "NOTE": continue
        try:
            body = json.loads(r.response_body)
        except Exception:
            continue
        if body.get("kind") != "writeln": continue
        msg = str(body.get("message", ""))
        for line in msg.splitlines():
            m = _LEMMA_LINE.match(line.strip())
            if m:
                name = m.group(1).split(".")[-1]
                if name not in seen:
                    seen.add(name); lemmas.append(name)
                    if len(lemmas) >= max_lemmas: return lemmas
    return lemmas

def mine_facts_prioritized(isabelle, session_id: str, state_hint: str, limit: int = 6) -> List[str]:
    toks = list(dict.fromkeys(FIND_NAME_TOKENS.findall(state_hint)))
    if not toks: return []
    theory_text = _build_find_theorems_filtered(toks, ["simp", "intro", "rule"])
    resps = run_theory(isabelle, session_id, theory_text)
    freq: Dict[str, int] = {}
    for r in resps:
        if getattr(r, "response_type", "") != "NOTE": continue
        try:
            body = json.loads(r.response_body)
        except Exception:
            continue
        if body.get("kind") != "writeln": continue
        for line in str(body.get("message","")).splitlines():
            m = _LEMMA_LINE.match(line.strip())
            if not m: continue
            name = m.group(1).split(".")[-1]
            freq[name] = freq.get(name, 0) + 1
    return [k for k,_ in sorted(freq.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))][:limit]
