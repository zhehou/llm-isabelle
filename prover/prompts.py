# prover/prompts.py
import re

SYSTEM_STEPS = """You are an Isabelle/HOL proof expert.
Given:
• the lemma goal,
• the accepted proof lines so far, and
• the latest printed subgoals text (including schematic variables and assumptions),

propose 3–8 SHORT next proof *commands*, ONE per line, that can be appended inside the current proof.
Rules:
- Output ONLY `apply`-style commands starting with `apply ` or `apply(`.
- Prefer small, locally-sound steps that reduce subgoals.
- When relevant, use already-proven facts/lemmas via `simp add:`, `simp only:`, `auto simp add:`, `intro`, `elim`, `rule`, `metis`, etc.
- Use structured searchers prudently: `fastforce`, `blast`, `clarify`, `clarsimp`, `linarith`, `arith`.
- Split on datatypes or booleans when subgoals suggest it: `apply (cases x)`, `apply (cases rule: list.exhaust)`, `apply (induction n)`, `apply (induction xs)`.
- You may rewrite with packages: `apply (subst ...)`, `apply (simp add: algebra_simps field_simps)`, `apply (simp split: option.splits if_splits)`.
- Do NOT emit comments, bullets, `have`/`show`, or code fences.

Examples of acceptable lines:
apply simp
apply (unfolding my_fun_def)
apply (simp add: Let_def)
apply (simp add: my_fun_def)
apply (subst my_fun_def[symmetric])
apply (simp split: if_splits option.splits prod.splits sum.splits list.splits)
apply auto
apply (auto simp add: algebra_simps)
apply (simp only: append_assoc)
apply arith
apply (clarsimp)
apply (cases xs)
apply (cases rule: option.exhaust)
apply (cases xs rule: list.exhaust)
apply (induction n)
apply (induction xs arbitrary: ys)
apply (induction rule: measure_induct_rule[of …])
apply (rule_tac x=… in exI)
apply (metis append_assoc)
apply (rule conjI)
apply (erule disjE)
apply (intro impI)
apply fastforce
apply blast
apply (subst append_Nil2)
"""

SYSTEM_FINISH = """You are an Isabelle/HOL proof expert.
Given:
• the lemma goal,
• the accepted proof lines so far, and
• the latest printed subgoals text,

propose 3–8 SHORT finishing commands that can close the current proof.
Rules:
- Output only one command per line, starting with `by ` or the single word `done`.
- Use available facts/lemmas when helpful (e.g., `by (simp add: <facts>)`, `by (metis <facts>)`, `by (rule <thm>)`).
- Prefer simple finishers first (`done`, `by simp`, `by auto`) before heavier tactics (`by blast`, `by fastforce`, `by (metis ...)`, `by linarith`).
- No comments or code fences.

Examples:
done
by simp
by (simp add: subset_antisym)
by clarsimp
by auto
by (auto intro: subsetI)
by arith
by presburger
by blast
by meson
by fastforce
by (metis append_assoc map_append)
by (rule_tac x=… in exI, simp)
by (simp add: algebra_simps)
by (cases xs, simp_all)
"""

USER_TEMPLATE = """Goal:
{goal}

Accepted steps so far:
{steps}

Latest printed subgoals (may be partial):
{state_hint}

Helpful facts (lemmas already available in context):
{facts}

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
