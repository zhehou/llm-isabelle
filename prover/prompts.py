# prover/prompts.py
import re

SYSTEM_STEPS = """You are an Isabelle/HOL proof assistant.
Given a lemma goal, the accepted proof lines so far, and the latest printed subgoals text,
propose 3-8 SHORT next proof steps (one per line), suitable to append inside an Isar proof.
Return ONLY commands like:
- apply simp
- apply simp_all
- apply auto
- apply (cases xs)
- apply (induction xs)
- apply (metis)
No comments, no code fences, one command per line."""

SYSTEM_FINISH = """You are an Isabelle/HOL proof assistant.
Given a lemma goal, the accepted proof lines so far, and the latest printed subgoals text,
propose 3-8 SHORT finishing commands that can close the proof.
Return ONLY commands like:
- done
- by simp
- by auto
- by blast
- by (simp add: SOME_LEMMA)
- by (metis SOME_LEMMA)
No comments, no code fences, one command per line."""

USER_TEMPLATE = """Goal:
{goal}

Accepted steps so far:
{steps}

Latest printed subgoals (may be partial):
{state_hint}

Constraints:
- Output ONLY the commands, one per line.
- 3 to 8 candidates.
"""

_LINE_RE = re.compile(r"^\s*(?:[-*]\s*)?([a-zA-Z].*?)\s*$")

def parse_ollama_lines(text: str, allowed_prefixes, max_items: int):
    if text.startswith("__ERROR__"): return []
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL | re.MULTILINE)
    lines = text.splitlines()
    out, seen = [], set()
    for ln in lines:
        m = _LINE_RE.match(ln)
        if not m: continue
        cand = re.sub(r"^\d+\.\s*", "", m.group(1).strip())
        if not cand or len(cand) > 120 or "#" in cand: continue
        if not any(cand.startswith(p) for p in allowed_prefixes): continue
        cand = re.sub(r"\s+", " ", cand)
        if cand not in seen:
            seen.add(cand); out.append(cand)
        if len(out) >= max_items: break
    return out
