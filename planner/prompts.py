# ========== Prompt Templates for Repair ==========

_LOCAL_SYSTEM = """You are an Isabelle/HOL expert.
You propose a replacement for the provided Isabelle/Isar proof BLOCK that can be verified in Isabelle/HOL.
Return ONLY the new BLOCK text (no JSON, no comments). Preserve all text outside the block.

EDIT SCOPE
- Keep the first "... have ..." line EXACTLY as is
- The proof is ONLY for the first line, don't prove anything else in PROOF_CONTEXT
- Do NOT add additional "have" or "show" statements if you have already proved the opening line
- Maintain indentation and whitespace style of the original.

STRICT RULES
- In `using`/`simp add:`/`unfolding` refer ONLY to named facts (no raw quoted propositions) in PROOF_CONTEXT.
- Respect meta-targets: inside induction branches prefer `show ?case`; otherwise prefer `show ?thesis`.
- Your output must be substantively different from every block in PRIOR FAILED BLOCKS.
- When trivial, close with `by simp` / `by auto` / `by blast` / `by fastforce`, etc, but don't use . as a tactic. 
- Never add "qed" in BLOCK
- Don't copy text from PROOF_CONTEXT. 

LIGHT GRAMMAR (allowed shapes)
<stmt> ::=
  "using" <thms>
| "unfolding" <thms>

<proof> ::=
  "by" <method>
| "sorry"

<method> ::= "simp" ["add:" <thms>] ["only:" <thms>]
           | "auto" | "blast" | "fastforce" | "clarsimp"
           | "intro" <thms> | "elim" <thms> | "rule" <thm>           
           | "subst" <thm> | "-"

OUTPUT
- Keep branch structure intact; every opened branch must end with a `show` and close.
- Do NOT invent new constants or fact names; use only identifiers in LOCAL_CONTEXT or the original BLOCK.
- Output ONLY the revised BLOCK (no fences).
"""

_LOCAL_USER = """WHAT FAILED:
{why}

GOAL:
{goal}

PROOF_CONTEXT (lemma header and all proven statements before the BLOCK - you can reference any named facts from here):
<<<CONTEXT
{proof_context}
CONTEXT

ISABELLE_ERRORS (learn from previous errors and avoid generating proofs that have similar errors):
{errors}

COUNTEREXAMPLE_HINTS (learn from counterexamples of previous goals and avoid generating goals based on the counterexamples):
{ce_hints}

PRIOR FAILED BLOCKS (do **not** repeat these ideas/structures; these are bad examples, not templates):
<<<FAILED_PROOFS
{prior_failed_blocks}
FAILED_PROOFS

ORIGINAL BLOCK TO REPLACE:
<<<BLOCK
{block_text}
BLOCK

Return ONLY the new BLOCK text (no fences)."""

_BLOCK_SYSTEM = """You are an Isabelle/HOL expert.
You propose a replacement for the provided Isabelle/Isar proof BLOCK that can be verified in Isabelle/HOL.
Return ONLY the new BLOCK text (no JSON, no comments). Preserve all text outside the block.

EDIT SCOPE
- Edit ONLY inside the BLOCK; keep lemma header unchanged if it's present.
- You MAY modify the structure (change proof strategy, add intermediate facts)
- Name all new facts like f1, f2, etc.
- Keep existing case/fact names/labels stable
- Maintain indentation and whitespace style of the original.

STRICT RULES
- In `using`/`simp add:`/`unfolding` refer ONLY to named facts (no raw quoted propositions) in PROOF_CONTEXT.
- Respect meta-targets: inside induction branches prefer `show ?case`; otherwise prefer `show ?thesis`.
- Your output must be substantively different from every block in PRIOR FAILED BLOCKS.
- When trivial, close with `by simp` / `by auto` / `by blast` / `by fastforce`, etc, but don't use . as a tactic. 
- Don't add "qed" if there isn't an open "proof".
- Don't copy text from PROOF_CONTEXT. 

LIGHT GRAMMAR (allowed shapes)
<stmt> ::=
  "using" <thms>
| "unfolding" <thms>
| "have" "<prop>" <proof>
| "show ?case" <proof>        // inside induction branches
| "show ?thesis" <proof>      // other branches
| "from" <thms> <goalstmt>
| "with" <thms> <goalstmt>
| "also" | "moreover" | "finally" <goalstmt>
| "next"                       // to separate branches
| "let" <pat> "=" <expr> | "define" <name> "where" "<eqn>"

<goalstmt> ::= "have" "<prop>" <proof> | "show" "<prop>" <proof>

<proof> ::=
  "by" <method>
| "proof" ["(" <method> ")"] <stmts>* "qed"
| "sorry"

<method> ::= "simp" ["add:" <thms>] ["only:" <thms>]
           | "auto" | "blast" | "fastforce" | "clarsimp"
           | "intro" <thms> | "elim" <thms> | "rule" <thm>
           | "cases" <expr> | "induction" <var> ["arbitrary:" <vars>]
           | "subst" <thm> | "-"

OUTPUT
- Keep branch structure intact; every opened branch must end with a `show` and close.
- Do NOT invent new constants or fact names; use only identifiers in LOCAL_CONTEXT or the original BLOCK.
- Output ONLY the revised BLOCK (no fences).
"""

_BLOCK_USER = """WHAT FAILED:
{why}

GOAL:
{goal}

PROOF_CONTEXT (lemma header and all proven statements before the BLOCK - you can reference any named facts from here):
<<<CONTEXT
{proof_context}
CONTEXT

ISABELLE_ERRORS (learn from previous errors and avoid generating proofs that have similar errors):
{errors}

COUNTEREXAMPLE_HINTS (learn from counterexamples of previous goals and avoid generating goals based on the counterexamples):
{ce_hints}

PRIOR FAILED BLOCKS (do **not** repeat these ideas/structures; these are bad examples, not templates):
<<<FAILED_PROOFS
{prior_failed_blocks}
FAILED_PROOFS

ORIGINAL BLOCK TO REPLACE:
<<<BLOCK
{block_text}
BLOCK

Return ONLY the new BLOCK text (no fences)."""

# -----------------------------------------------------------------------------
# Prompt for OUTLINES  (nudged with ?case and calculational patterns)
# -----------------------------------------------------------------------------
SKELETON_PROMPT = """You are an Isabelle/HOL expert. 

TASK
Given a lemma statement, first figure out a proof plan in English INTERNALLY that aims to break the problem into smaller problems so you can divide and conquer. Do NOT reveal your plan. Output ONLY a CLEAN Isabelle/Isar proof outline that corresponds to your English proof plan and is verifiable in Isabelle/HOL. Leave nontrivial reasoning steps as `sorry`.

HARD OUTPUT RULES
- Output ONLY Isabelle/Isar (no prose, no code fences).
- Begin at (or immediately after) the exact header:
  lemma "{goal}"
- Produce exactly ONE lemma..qed block.
- Prefer structured proofs with named intermediate facts (e.g., f1, f2) that are then reused.
- Use the right shell:
  • Induction: `proof (induction <var>)` → branches `case …` with `show ?case …`.
  • Exhaustive cases: `proof (cases <expr>)` or `proof (cases rule: <T>.exhaust)` → branches ending with `show ?thesis …`.
  • Calculational: `proof -` with `have …`, `also`, `moreover`, `finally show ?thesis …`.
- When trivial, close with `by simp` / `by auto` / `by blast` / `by fastforce`, etc, but don't use . as a tactic. 
- Do NOT invent constants or fact names; only use variables/tokens present in the goal or locally introduced facts.

LIGHT GRAMMAR (allowed shapes)
lemma "{goal}"
<refine>* <proof>
<refine> ::= using <thms> | unfolding <thms> | apply <method>
<proof>  ::= proof [<method>] <stmts>* qed | by <method> | sorry | done
<stmts>  ::= fix <vars> | assume <n>: "<prop>" | have "<prop>" <proof>
             | show ?case <proof> | show ?thesis <proof> | then <goal_stmt>
             | from <thms> <goal_stmt> | with <thms> <goal_stmt>
             | also | moreover | finally <goal_stmt> | next
<goal_stmt> ::= have "<prop>" <proof> | show "<prop>" <proof>
<method> ::= "induction" <var> ["arbitrary:" <vars>] | "cases" <expr> | "-"
             | "simp" ["add:" <thms>] ["only:" <thms>] | "auto" | "blast"
             | "fastforce" | "clarsimp" | "intro" <thms> | "elim" <thms>
             | "rule" <thm> | "metis" [<thms>] | "(" <method> ")"

STYLE EXAMPLES
lemma "{goal}"
proof (induction xs)
  case Nil
  have f1: "…"
    using Nil.prems
    sorry
  show ?case
    using f1
    sorry
next
  case (Cons x xs)
  have f1: "…"
    using Cons.prems
    sorry
  have f2: "…"
    using Cons.IH f1
    sorry
  show ?case
    using f2
    sorry
qed

lemma "{goal}"
proof (cases b)
  case True
  have f1: "…"
    sorry
  show ?thesis
    using f1
    sorry
next
  case False
  have f2: "…"
    sorry
  show ?thesis
    using f2
    sorry
qed

lemma "{goal}"
proof -
  have f1: "A = B"  sorry
  have f2: "B = C"  using f1  sorry
  also have "... = D"  sorry
  finally show ?thesis  using f2  sorry
qed
"""
