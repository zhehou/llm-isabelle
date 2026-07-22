from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout
import requests
from typing import Callable
from planner.repair_inputs import _find_first_hole, _hole_line_bounds, _APPLY_OR_BY, _snippet_window, _clamp_line_index, _quick_state_and_errors, _extract_error_lines, _run_theory_with_timeout, _print_state_before_hole, _nearest_header, _recent_steps, _normalize_error_texts, _facts_from_state, get_counterexample_hints_for_repair, _earliest_failure_anchor
from planner.prompts import _LOCAL_SYSTEM, _LOCAL_USER, _BLOCK_SYSTEM, _BLOCK_USER
from prover.config import MODEL as DEFAULT_MODEL, OLLAMA_HOST, TIMEOUT_S as OLLAMA_TIMEOUT_S, OLLAMA_NUM_PREDICT, TEMP as OLLAMA_TEMP, TOP_P as OLLAMA_TOP_P
from prover.isabelle_api import (
    build_theory,
    run_theory,
    last_print_state_block,
    finished_ok,
    first_error_message,
)
from planner.goals import (
    _extract_local_assumptions,
    _print_state_before_hole as _context_state_before_hole,
)
from planner.frontier import frontier_advanced
from prover.strategies import (
    definition_guided_finishers,
    definition_guided_obligations,
)

# ========== Configuration ==========
_ISA_FAST_TIMEOUT_S = int(os.getenv("ISABELLE_FAST_TIMEOUT_S", "12"))
_ISA_VERIFY_TIMEOUT_S = int(os.getenv("ISABELLE_VERIFY_TIMEOUT_S", "30"))
_SESSION = requests.Session()
_REPAIR_RULES_JSON = os.getenv("REPAIR_RULES_JSON", "").strip()  # optional, declarative fallback rules


def _compact_diagnostic(text: str, *, limit: int = 600) -> str:
    compact = " | ".join(
        line.strip() for line in str(text or "").splitlines() if line.strip()
    )
    if len(compact) > limit:
        return compact[: max(0, limit - 1)].rstrip() + "…"
    return compact

# ========== Regex Patterns ==========
_CTX_HEAD = re.compile(r"^\s*(?:using|from|with|then|ultimately|finally|also|moreover)\b")
_HAS_BODY = re.compile(r"^\s*(?:by\b|apply\b|proof\b|sorry\b|done\b)")
_INLINE_BY_TAIL = re.compile(r"\s+by\s+.+$")
_TACTIC_LINE = re.compile(r"^\s*(?:apply|by)\b|(?:\s)by\s+\S")
_STRUCTURAL_LINE = re.compile(r"^\s*(?:lemma|theorem|qed|next|proof|case|have|show|assume|fix|from|using|thus|hence|ultimately|finally|also|moreover|let|where)\b")
_HEAD_CMD_RE = re.compile(r"^\s*(have|show|obtain|then\s+show|thus|hence)\b")
_PROOF_RE = re.compile(r"^\s*proof\b")
_QED_RE = re.compile(r"^\s*qed\b")
_CASE_LINE_RE = re.compile(r"^\s*case\b")
_NEXT_OR_QED_RE = re.compile(r"^\s*(?:next|qed)\b")
_WRAPPED_THEOREM_HEAD = re.compile(r"(?mx)\A(?:[ \t]*(?:\(\*.*?\*\)|\<comment\>.*?\<\/comment\>)[ \t]*\n|[ \t]*\n)*[ \t]*(?:lemma|theorem|corollary)\b")
# Outline-level strategies we want to ban on whole-proof regen
_OUTLINE_PROOF_LINE   = re.compile(r"(?m)^\s*proof(?:\s*\(([^)]*)\))?\s*$")
_OUTLINE_BARE         = re.compile(r"(?m)^\s*(?:induction|cases|coinduction)\b.*$")
_SORRY_RE             = re.compile(r"\bsorry\b")

# ========== Utility Functions ==========

def _log(prefix: str, label: str, content: str, trace: bool = True) -> None:
    if trace and content:
        print(f"[{prefix}] {label} (len={len(content)}):\n{content if content.strip() else '  (empty)'}")

def _sanitize_llm_block(text: str) -> str:
    if not text:
        return text
    patterns = [
        r"^\s*<<<BLOCK\s*$",
        r"^\s*BLOCK\s*$",
        r"^\s*<<<PROOF\s*$",
        r"^\s*PROOF\s*$",
        r"^\s*```\s*$",
        r"^\s*```isabelle\s*$",
        r"^\s*```isar\s*$",
        # strip stray fence markers sometimes emitted by LLMs
        r"^\s*<<<\s*$",
        r"^\s*>>>\s*$",
    ]
    # Also drop accidental headers LLMs sometimes leak mid-repair
    header_patterns = [
        r"^\s*lemma\b.*$",
        r"^\s*theorem\b.*$",
        r"^\s*corollary\b.*$",
        r"^\s*proposition\b.*$",
        r"^\s*---\s*$",
    ]
    compiled = [re.compile(p) for p in (patterns + header_patterns)]
    lines = [l for l in text.splitlines() if not any(p.match(l) for p in compiled)]

    # Balance 'proof'/'qed' and cut off any text after the final balanced 'qed'
    balance = 0
    last_closed_idx = -1
    for i, l in enumerate(lines):
        if re.match(r"^\s*proof\b", l):
            balance += 1
        elif re.match(r"^\s*qed\b", l):
            if balance > 0:
                balance -= 1
                if balance == 0:
                    last_closed_idx = i
    if last_closed_idx != -1 and last_closed_idx + 1 < len(lines):
        lines = lines[: last_closed_idx + 1]

    return "\n".join(lines).strip()

def _is_effective_block(text: str) -> bool:
    return bool(_sanitize_llm_block(text or "").strip())

def _fingerprint_block(text: str) -> str:
    """Canonicalize a block to detect duplicates across rounds."""
    if not text:
        return ""
    # Collapse whitespace, drop zero-width and backticks, normalize quotes.
    t = re.sub(r"\s+", " ", text.strip())
    t = t.replace("`", "").replace("“", '"').replace("”", '"').replace("’", "'")
    return t

def _trim_block_for_prompt(text: str, max_chars: int = 800) -> str:
    """Keep prompt sizes sane by trimming long blocks."""
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    head = t[: max_chars // 2].rstrip()
    tail = t[- max_chars // 2 :].lstrip()
    return head + "\n…\n" + tail

def _is_tactic_line(s: str) -> bool:
    return bool(_TACTIC_LINE.search(s)) and not bool(_STRUCTURAL_LINE.match(s))

def _extract_proof_context(full_text: str, block_start_line: int) -> str:
    """
    Extract theory declarations, the lemma header and proven prefix.

    Stage-3 repair must see user datatypes and recursive equations; otherwise
    it invents library names for constants whose definitions were simply
    hidden from the prompt.
    """
    lines = full_text.splitlines()
    
    # Find the lemma/theorem header
    lemma_line = -1
    for i in range(min(block_start_line, len(lines) - 1), -1, -1):
        if re.match(r"^\s*(?:lemma|theorem|corollary|proposition)\b", lines[i]):
            lemma_line = i
            break
    
    if lemma_line < 0:
        # No lemma found, return a small window before the block
        start = max(0, block_start_line - 10)
        return "\n".join(lines[start:block_start_line]).strip()
    
    theory_context = _trim_block_for_prompt(
        "\n".join(lines[:lemma_line]).strip(), max_chars=6000
    )
    context_lines = lines[lemma_line:block_start_line]
    proof_prefix = "\n".join(context_lines).strip()
    if theory_context:
        return (
            "THEORY DECLARATIONS BEFORE THE TARGET:\n"
            + theory_context
            + "\n\nTARGET PROOF PREFIX:\n"
            + proof_prefix
        ).strip()
    return proof_prefix

# ========== LLM Generation ==========
def _generate_simple(prompt: str, model: Optional[str] = None, *, timeout_s: Optional[int] = None) -> str:
    m = model or DEFAULT_MODEL
    timeout = timeout_s or OLLAMA_TIMEOUT_S
    
    if m.startswith("hf:"):
        return _hf_generate(prompt, m[3:], timeout)
    elif m.startswith("gemini:"):
        return _gemini_generate(prompt, m[7:], timeout)
    elif m.startswith("ollama:"):
        m = m[7:]
    return _ollama_generate(prompt, m, timeout)

def _ollama_generate(prompt: str, model: str, timeout_s: int) -> str:
    payload = {"model": model, "prompt": prompt, "options": {"temperature": OLLAMA_TEMP, "top_p": OLLAMA_TOP_P, "num_predict": OLLAMA_NUM_PREDICT}, "stream": False}
    timeout = (10.0, max(30.0, float(timeout_s)))
    resp = _SESSION.post(f"{OLLAMA_HOST.rstrip('/')}/api/generate", json=payload, timeout=timeout)
    resp.raise_for_status()
    return _sanitize_llm_block(resp.json().get("response", "").strip())

def _hf_generate(prompt: str, model_id: str, timeout_s: int) -> str:
    token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACE_API_TOKEN is not set")
    payload = {"inputs": prompt, "parameters": {"temperature": OLLAMA_TEMP, "top_p": OLLAMA_TOP_P, "max_new_tokens": OLLAMA_NUM_PREDICT, "return_full_text": False}, "options": {"wait_for_model": True}}
    resp = _SESSION.post(f"https://api-inference.huggingface.co/models/{model_id}", headers={"Authorization": f"Bearer {token}"}, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and data:
        result = data[0].get("generated_text", "")
    elif isinstance(data, dict):
        result = data.get("generated_text", "") or (data["choices"][0].get("text", "") if "choices" in data and data["choices"] else "")
    else:
        result = str(data)
    return _sanitize_llm_block(result.strip())

def _gemini_generate(prompt: str, model_id: str, timeout_s: int) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": {"maxOutputTokens": OLLAMA_NUM_PREDICT}}
    resp = _SESSION.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}", json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    result = ""
    try:
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                result = parts[0].get("text", "")
    except Exception:
        result = str(data)
    return _sanitize_llm_block(result.strip())

# ========== Repair Operations (Data Classes) ==========
@dataclass(frozen=True)
class InsertBeforeHole:
    line: str

@dataclass(frozen=True)
class ReplaceInSnippet:
    find: str
    replace: str

@dataclass(frozen=True)
class InsertHaveBlock:
    label: str
    statement: str
    after_line_matching: str
    body_hint: str

RepairOp = Tuple[str, object]

@dataclass
class _RepairMemory:
    rounds: int = 0
    # Keep full failed blocks (same block_type) we tried this session
    prev_blocks: List[str] = field(default_factory=list)
    # Fingerprints to dedup within a session
    prev_fps: Set[str] = field(default_factory=set)

# --- Prior-block store shared across repairs of the *same hole* ---------------
# Maps block_type -> list of failed blocks (latest first, length-capped)
_MAX_PREV_BLOCKS = int(os.getenv("REPAIR_MAX_PREV_BLOCKS", "4"))

# ========== Repair Operations (Parsing & Application) ==========

def _propose_block_repair(*, goal: str, errors: List[str], ce_hints: Dict[str, List[str]], 
                         proof_context: str,  block_type: str,
                         block_text: str, model: Optional[str], timeout_s: int,
                         why: str = "Previous attempt failed; propose a different block-level change.",
                         prior_failed_blocks: Optional[str] = None) -> str:
    ce = ce_hints.get("bindings", []) + ce_hints.get("def_hints", [])
    if block_type == "have-show":
        prompt = _LOCAL_SYSTEM + "\n\n" + _LOCAL_USER.format(
            goal=goal, errors="\n".join(f"- {e}" for e in errors) or "(none)",
            ce_hints="\n".join(ce) or "(none)", 
            proof_context=(proof_context or "").strip(),  # Changed from state_block
            block_text=block_text.rstrip(), why=why,
            prior_failed_blocks=(prior_failed_blocks or "(none)")
        )
    else:
        prompt = _BLOCK_SYSTEM + "\n\n" + _BLOCK_USER.format(
            goal=goal, errors="\n".join(f"- {e}" for e in errors) or "(none)",
            ce_hints="\n".join(ce) or "(none)", 
            proof_context=(proof_context or "").strip(),  # Changed from state_block
            block_text=block_text.rstrip(), why=why,
            prior_failed_blocks=(prior_failed_blocks or "(none)")
        )
    try:
        candidate = _sanitize_llm_block(_generate_simple(prompt, model=model, timeout_s=timeout_s))
        return "" if _SORRY_RE.search(candidate or "") else candidate
    except Exception:
        return ""

def propose_rule_based_repairs(goal_text: str, state_block: str, header: str, facts: List[str]) -> List[RepairOp]:
    """
    Declarative, data-driven fallback:
    - If REPAIR_RULES_JSON is set to a JSON file, load rules and emit ops that match.
    - Otherwise return [] (i.e., no ad-hoc heuristics).
    Rule schema (list):
      {
        "when": {
          "goal_contains_any": ["@", "map"],
          "goal_regex": "length\\s",
          "facts_contains_any": ["append_assoc"],
          "state_contains_any": ["Let "],
          "header_startswith": "proof (induction",
          "header_regex": "proof \\(induction.*\\)"
        },
        "op": { "insert_before_hole": "apply (simp add: append_assoc)" }
      }
      or
      {
        "when": { "header_startswith": "proof (induction", "not_header_contains": ["arbitrary:"] },
        "op": { "replace_in_snippet": { "find": "proof (induction xs)", "replace": "proof (induction xs arbitrary: ys)" } }
      }
    """
    path = _REPAIR_RULES_JSON
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            rules = json.load(f)
    except Exception:
        return []
    def _match(rule) -> Optional[RepairOp]:
        cond = rule.get("when", {}) or {}
        op   = rule.get("op", {}) or {}
        g, st, hd = goal_text or "", state_block or "", header or ""
        fs = facts or []
        import re as _re
        def contains_any(text, keys): return any(k in text for k in keys)
        def not_contains(text, keys): return not any(k in text for k in keys)
        # boolean guards (all must pass if present)
        checks = [
            ("goal_contains_any", lambda v: contains_any(g, v)),
            ("state_contains_any", lambda v: contains_any(st, v)),
            ("facts_contains_any", lambda v: any(x in fs for x in v)),
            ("goal_regex",        lambda v: bool(_re.search(v, g))),
            ("header_startswith", lambda v: hd.startswith(v)),
            ("header_regex",      lambda v: bool(_re.search(v, hd))),
            ("not_header_contains", lambda v: not_contains(hd, v)),
        ]
        for key, pred in checks:
            if key in cond:
                val = cond[key]
                if isinstance(val, list) and not val: 
                    continue
                if not pred(val):
                    return None
        # build op
        if "insert_before_hole" in op and isinstance(op["insert_before_hole"], str):
            return ("insert_before_hole", InsertBeforeHole(op["insert_before_hole"].strip()))
        if "replace_in_snippet" in op and isinstance(op["replace_in_snippet"], dict):
            fnd = (op["replace_in_snippet"].get("find") or "").strip()
            rep = (op["replace_in_snippet"].get("replace") or "").strip()
            if fnd and rep:
                return ("replace_in_snippet", ReplaceInSnippet(fnd, rep))
        if "insert_have_block" in op and isinstance(op["insert_have_block"], dict):
            v = op["insert_have_block"]; lab=v.get("label","H"); stmt=v.get("statement",""); aft=v.get("after_line_matching","then show ?thesis"); hint=v.get("body_hint","apply simp")
            if stmt.strip() and aft.strip():
                return ("insert_have_block", InsertHaveBlock(lab.strip(), stmt.strip(), aft.strip(), hint.strip()))
        return None
    out: List[RepairOp] = []
    for r in rules if isinstance(rules, list) else []:
        rop = _match(r)
        if rop: out.append(rop)
        if len(out) >= 3:
            break
    return out

# ========== Region Analysis ==========
def _enclosing_case_block(lines: List[str], hole_line: int) -> Tuple[int, int]:
    i = hole_line
    while i >= 0 and not _CASE_LINE_RE.match(lines[i]):
        i -= 1
    if i < 0:
        return (-1, -1)
    j = hole_line
    while j < len(lines) and not (_NEXT_OR_QED_RE.match(lines[j])):
        j += 1
    return (i, j)

def _enclosing_subproof(lines: List[str], hole_line: int) -> Tuple[int, int]:
    i = hole_line
    while i >= 0 and not _PROOF_RE.match(lines[i]):
        i -= 1
    if i < 0:
        return (-1, -1)
    depth, j = 1, i + 1
    while j < len(lines) and depth > 0:
        if _PROOF_RE.match(lines[j]):
            depth += 1
        elif _QED_RE.match(lines[j]):
            depth -= 1
        j += 1
    return (i, j if j > i else -1)

def _enclosing_have_show_block(lines: List[str], hole_line: int) -> Tuple[int, int]:
    if not lines:
        return (-1, -1)

    i = _clamp_line_index(lines, hole_line)

    head_re  = re.compile(r"^\s*(have|show|obtain)\b")
    # IMPORTANT: do NOT include `proof` here — subproofs belong to the block.
    fence_re = re.compile(
        r"^\s*(?:have|show|obtain|thus|hence|then|also|moreover|ultimately|finally|case\b|next\b|qed\b)\b"
    )

    # climb to the enclosing have/show head, but stop if we hit a block fence
    while i >= 0 and not head_re.match(lines[i]):
        if re.match(r"^\s*(?:case\b|next\b|qed\b)\b", lines[i]):
            return (-1, -1)
        i -= 1

    if i < 0 or not head_re.match(lines[i]):
        return (-1, -1)

    # NEW: if the head line itself has an inline "by …", keep only that line
    if _INLINE_BY_TAIL.search(lines[i] or ""):
        return (i, i + 1)

    # Track nested subproofs correctly from the head line outward.
    depth = 0
    j = i + 1
    while j < len(lines):
        L = lines[j]

        # Base-level one-liner endings
        if depth == 0:
            # stop immediately after a base-level "sorry"
            if (L or "").strip() == "sorry":
                j = j + 1
                break
            # do not include any fence token at base depth
            if fence_re.match(L or ""):
                break

        # Subproof bookkeeping
        if _PROOF_RE.match(L or ""):
            depth += 1
        elif _QED_RE.match(L or ""):
            depth = max(0, depth - 1)

        j += 1

    return (i, j if j > i else -1)

def _enclosing_whole_proof(lines: List[str]) -> Tuple[int, int]:
    """Return the outer proof block, not the nearest nested proof."""
    # A theory preamble may contain already-proved support lemmas.  Whole
    # repair targets the final theorem command assembled by the planner, not
    # the first declaration in the file.
    lemma = next(
        (
            i for i in range(len(lines) - 1, -1, -1)
            if re.match(r"^\s*(?:lemma|theorem|corollary)\b", lines[i])
        ),
        -1,
    )
    if lemma < 0:
        return (-1, -1)
    start = next((i for i in range(lemma + 1, len(lines)) if _PROOF_RE.match(lines[i])), -1)
    if start < 0:
        return (-1, -1)
    depth = 0
    for i in range(start, len(lines)):
        if _PROOF_RE.match(lines[i]):
            depth += 1
        elif _QED_RE.match(lines[i]):
            depth -= 1
            if depth == 0:
                return (start, i + 1)
    return (-1, -1)

# ========== Wrapper Stripping ==========
def _strip_wrapper_to_case_block(proposed: str, original_case_block: str) -> str:
    if not _WRAPPED_THEOREM_HEAD.match(proposed):
        return proposed
    case_name = None
    m = re.search(r"(?m)^\s*case\s*\((\w+)", original_case_block or "")
    if m:
        case_name = m.group(1)
    else:
        m = re.search(r"(?m)^\s*case\s+(\w+)", original_case_block or "")
        if m:
            case_name = m.group(1)
    lines = proposed.splitlines()
    start = None
    for i, L in enumerate(lines):
        if not _CASE_LINE_RE.match(L):
            continue
        if case_name is None or re.match(rf"^\s*case\s*\({re.escape(case_name)}\b", L) or re.match(rf"^\s*case\s+{re.escape(case_name)}\b", L):
            start = i
            break
    if start is None:
        return proposed
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if _NEXT_OR_QED_RE.match(lines[j]):
            end = j
            break
    return "\n".join(lines[start:end]).rstrip()

def _strip_wrapper_to_have_show(proposed: str, original_block: str) -> str:
    # Keep only the single have/show/obtain micro-block, including any nested
    # `proof … qed` it contains, but STOP at the first base-level `sorry`
    # or base-level inline `by …`.
    lines = proposed.splitlines()
    if not lines:
        return proposed

    head_re  = re.compile(r"^\s*(have|show|obtain)\b")
    fence_re = re.compile(
        r"^\s*(?:have|show|obtain|thus|hence|then|also|moreover|ultimately|finally|case\b|next\b|qed\b)\b"
    )

    # find the first have/show/obtain head
    head_idx = next((i for i, L in enumerate(lines) if head_re.match(L)), -1)
    if head_idx == -1:
        return proposed

    out: List[str] = [lines[head_idx]]
    depth = 0

    for L in lines[head_idx + 1:]:
        # Base-level one-line endings
        if depth == 0:
            # Keep a base-level "sorry", but stop right after it
            if (L or "").strip() == "sorry":
                out.append(L)
                break
            # Do not include any new head/fence (then/also/moreover/…/case/next/qed)
            if fence_re.match(L or ""):
                break

        # Track nested subproofs (kept inside the micro-block)
        if _PROOF_RE.match(L or ""):
            depth += 1
        elif _QED_RE.match(L or ""):
            depth = max(0, depth - 1)

        out.append(L)

    # Trim any trailing whitespace lines we may have kept
    while out and out[-1].strip() == "":
        out.pop()

    # Final guard: if the last kept line is an inline 'by …' on the head, trim to that line only
    if len(out) == 1 and _INLINE_BY_TAIL.search(out[0] or ""):
        return out[0].rstrip()
    return "\n".join(out).rstrip()

def _strip_wrapper_to_subproof(proposed: str) -> str:
    if not _WRAPPED_THEOREM_HEAD.match(proposed):
        return proposed
    lines = proposed.splitlines()
    start = None
    for i, L in enumerate(lines):
        if _PROOF_RE.match(L):
            start = i
            break
    if start is None:
        return proposed
    depth, j = 1, start + 1
    while j < len(lines) and depth > 0:
        if _PROOF_RE.match(lines[j]):
            depth += 1
        elif _QED_RE.match(lines[j]):
            depth -= 1
        j += 1
    return "\n".join(lines[start:j if depth == 0 else len(lines)]).rstrip()

# ========== Safe Sorry Insertion ==========
def _find_enclosing_head(block_lines: List[str], from_idx: int) -> Optional[int]:
    for i in range(from_idx, -1, -1):
        if _HEAD_CMD_RE.match(block_lines[i] or ""):
            return i
    return None

def _apply_sequence_bounds(block_lines: List[str], idx: int) -> Tuple[int, int]:
    s = idx
    while s > 0 and _is_tactic_line(block_lines[s-1]):
        s -= 1
    e = idx + 1
    while e < len(block_lines) and _is_tactic_line(block_lines[e]):
        e += 1
    return s, e

def _replace_failing_tactics_with_sorry(block_text: str, *, full_text_lines: List[str], start_line: int, 
                                       end_line: int, isabelle, session: str, trace: bool = False) -> str:
    block_lines = block_text.splitlines()
    if not block_lines:
        return block_text    
    def build_doc(with_block_lines: List[str]) -> str:
        s0, e0 = max(0, start_line - 1), max(max(0, start_line - 1), min(end_line - 1, len(full_text_lines)))
        return "\n".join(full_text_lines[:s0] + with_block_lines + full_text_lines[e0:])
    
    while True:
        doc = build_doc(block_lines)
        _, errs = _quick_state_and_errors(isabelle, session, doc)
        err_in_block = sorted(set(l for l in _extract_error_lines(errs) if start_line <= l < end_line))
        thy = build_theory(doc.splitlines(), add_print_state=False, end_with=None)
        ok, _ = finished_ok(_run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_VERIFY_TIMEOUT_S))
        
        if not err_in_block:
            break
        
        failing_idx = err_in_block[0] - start_line
        cand = None
        if 0 <= failing_idx < len(block_lines) and _is_tactic_line(block_lines[failing_idx]):
            cand = failing_idx
        else:
            for i in range(min(failing_idx, len(block_lines) - 1), -1, -1):
                if _is_tactic_line(block_lines[i]):
                    cand = i
                    break
            if cand is None:
                for i in range(max(0, failing_idx + 1), len(block_lines)):
                    if _is_tactic_line(block_lines[i]):
                        cand = i
                        break
        
        if cand is None:
            break

        # --- Diagnostics before modifying the block ---
        # Run Quickcheck/Nitpick on the exact failing tactic line, so we capture
        # a counterexample on the subgoal that is about to fail.
        # try:
        #     diag_txt = _run_nitpick_at_line(
        #         isabelle, session, full_text_lines,
        #         inject_before_1based=start_line + cand
        #     )
        #     if diag_txt:
        #         _log("repair", "nitpick (pre-sorry)", diag_txt, trace=trace)
        # except Exception:
        #     pass

        indent = block_lines[cand][:len(block_lines[cand]) - len(block_lines[cand].lstrip())]
        if block_lines[cand].lstrip().startswith("apply"):
            head_idx = _find_enclosing_head(block_lines, cand)
            if head_idx is not None:
                head_indent = block_lines[head_idx][:len(block_lines[head_idx]) - len(block_lines[head_idx].lstrip())]
                seq_s, seq_e = _apply_sequence_bounds(block_lines, cand)
                block_lines[seq_s:seq_e] = [f"{head_indent}proof -", f"{head_indent}  sorry", f"{head_indent}qed"]
            else:
                break
        else:
            block_lines[cand] = f"{indent}sorry"
    
    return "\n".join(block_lines)

def try_cegis_repairs(*, full_text: str, hole_span: Tuple[int, int], goal_text: str, model: Optional[str], 
                     isabelle, session: str, repair_budget_s: float = 15.0, max_ops_to_try: int = 3, 
                     beam_k: int = 1, allow_whole_fallback: bool = False, trace: bool = False, 
                     resume_stage: int = 0,
                     prior_store: Optional[Dict[str, List[str]]] = None,
                     failure_notes: Optional[List[str]] = None) -> Tuple[str, bool, str]:
    t0 = time.monotonic()
    left = lambda: max(0.0, repair_budget_s - (time.monotonic() - t0))
    current_text = full_text
    state0 = _context_state_before_hole(
        isabelle, session, current_text, hole_span, trace=trace
    )
    _log("repair", "State block", state0, trace=trace)
    
    if allow_whole_fallback and trace:
        print("[repair] (deprecated) allow_whole_fallback=True is ignored; driver handles regeneration.")        

    if prior_store is None:
        prior_store = {}
    max_candidates = max(1, int(max_ops_to_try)) * max(1, int(beam_k))

    # Stage 1: have/show/obtain micro-block
    hole_line, _, lines = _hole_line_bounds(current_text, hole_span)
    anchor_line, anchor_reason = _earliest_failure_anchor(isabelle, session, current_text, default_line_0=hole_line)
    focus_line = _clamp_line_index(lines, anchor_line)
    if trace and anchor_line != hole_line:
        print(f"[repair] Retargeting from hole line {hole_line + 1} to earliest-failure line {anchor_line + 1} ({anchor_reason})")
    
    hs_s, hs_e = _enclosing_have_show_block(lines, focus_line)
    if resume_stage <= 1 and hs_s >= 0 and left() > 5.0:
        if trace:
            print("[repair] Trying have/show block repair…")
        current_text = _repair_block(current_text, lines, hs_s, hs_e, goal_text, state0, 
                                     isabelle, session, model, left, trace, "have-show", 
                                     stage=1, prior_store=prior_store,
                                     max_candidates=max_candidates,
                                     failure_notes=failure_notes)
        if current_text != full_text:
            thy = build_theory(current_text.splitlines(), add_print_state=False, end_with=None)
            ok, _ = finished_ok(_run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_VERIFY_TIMEOUT_S))
            if ok:
                return current_text, True, "stage=1 block:have-show"
            # FIX: Return False for unverified changes
            return current_text, False, "stage=1 partial-progress"
        lines = current_text.splitlines()
        state0 = _context_state_before_hole(
            isabelle, session, current_text, hole_span, trace=trace
        )
    
    # Stage 2a: Case-block
    cs, ce = _enclosing_case_block(lines, focus_line)
    if resume_stage <= 2 and cs >= 0 and left() > 5.0:
        if trace:
            print("[repair] Trying case-block repair…")
        current_text = _repair_block(current_text, lines, cs, ce, goal_text, state0, isabelle, session, 
                                     model, left, trace, "case", stage=2, prior_store=prior_store,
                                     max_candidates=max_candidates,
                                     failure_notes=failure_notes)
        if current_text != full_text:
            thy = build_theory(current_text.splitlines(), add_print_state=False, end_with=None)
            ok, _ = finished_ok(_run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_VERIFY_TIMEOUT_S))
            if ok:
                return current_text, True, "stage=2 block:case"
            # FIX: Return False for unverified changes
            return current_text, False, "stage=2 partial-progress"
    
    # Stage 2b: Subproof
    ps, pe = _enclosing_subproof(lines, focus_line)
    if resume_stage <= 2 and ps >= 0 and left() > 3.0:
        if trace:
            print("[repair] Trying subproof repair…")
        current_text = _repair_block(current_text, lines, ps, pe, goal_text, state0, isabelle, session, 
                                     model, left, trace, "subproof", stage=2, prior_store=prior_store,
                                     max_candidates=max_candidates,
                                     failure_notes=failure_notes)
        if current_text != full_text:
            thy = build_theory(current_text.splitlines(), add_print_state=False, end_with=None)
            ok, _ = finished_ok(_run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_VERIFY_TIMEOUT_S))
            if ok:
                return current_text, True, "stage=2 block:subproof"
            # FIX: Return False for unverified changes
            return current_text, False, "stage=2 partial-progress"
    
    if current_text != full_text:
        return current_text, False, f"stage={resume_stage} partial-progress"
    return full_text, False, f"stage={resume_stage} cegis-nohelp"

def _repair_block(current_text: str, lines: List[str], start: int, end: int, goal_text: str, 
                 state0: str, isabelle, session: str, model: Optional[str], left, trace: bool, 
                 block_type: str, stage: int, *, prior_store: Optional[Dict[str, List[str]]] = None,
                 max_candidates: int = 3,
                 failure_notes: Optional[List[str]] = None) -> str:
    original_text = current_text
    original_lines = list(lines)
    original_holes = len(_SORRY_RE.findall(original_text))
    _, errs = _quick_state_and_errors(isabelle, session, current_text)
    err_texts = _normalize_error_texts(errs)
    original_had_error = bool(err_texts)
    for note in reversed(failure_notes or []):
        clean_note = re.sub(r"\s+", " ", str(note)).strip()
        if clean_note and clean_note not in err_texts:
            err_texts.insert(0, clean_note)
    ce = get_counterexample_hints_for_repair(isabelle, session, state0, timeout_s=10)
    block = "\n".join(lines[start:end])
    
    # NEW: Extract proof context instead of using state block
    proof_context = _extract_proof_context(current_text, start)
    local_assumptions = _extract_local_assumptions(state0)
    if local_assumptions:
        proof_context = (
            proof_context
            + "\n\nLOCAL ISABELLE ASSUMPTIONS:\n"
            + "\n".join(f"- {assumption}" for assumption in local_assumptions)
        ).strip()
    
    _log("repair", f"{block_type}-block (input)", block, trace=trace)
    _log("repair", "proof_context (LLM input)", proof_context, trace=trace)  # Changed log
    _log("repair", "errors (LLM input)", "\n".join(err_texts) or "(none)", trace=trace)
    ce_list = ce.get("bindings", []) + ce.get("def_hints", []) if isinstance(ce, dict) else []  
    _log("repair", "counterexamples (LLM input)", "\n".join(ce_list) or "(none)", trace=trace)
    time_rounds = 3 if left() >= 18.0 else 2 if left() >= 10.0 else 1
    rounds = max(1, min(int(max_candidates), time_rounds))
    mem = _RepairMemory()

    # Build proposals in a few rounds; track failures and surface them to the LLM
    for rr in range(rounds):
        if left() <= 3.0:
            break
        mem.rounds = rr + 1
        why = f"Previous {block_type}-block attempt did not solve the goal; try a different strategy."
        timeout = int(min(60, max(8, left() * (0.55 / max(1, rounds - rr)))))
        
        # Build prior failed blocks text (trim + separators)
        prior_blocks_for_type = list(prior_store.get(block_type, [])) if isinstance(prior_store, dict) else []
        seed_list = [block] + mem.prev_blocks + prior_blocks_for_type
        
        # De-dup while preserving order (by fingerprint)
        seen: Set[str] = set()
        uniq: List[str] = []
        for b in seed_list:
            fpb = _fingerprint_block(b)
            if fpb and fpb not in seen:
                seen.add(fpb); uniq.append(b)
        seed_list = uniq
        
        if seed_list:
            fails_txt = ("\n---\n".join(_trim_block_for_prompt(b) for b in seed_list[:_MAX_PREV_BLOCKS])) or "(none)"
            _log("repair", "prior_block_failures (LLM input)", fails_txt, trace=trace)
        else:
            fails_txt = "(none)"        
        
        try:
            blk = _propose_block_repair(
                goal=goal_text, errors=err_texts, ce_hints=ce, 
                proof_context=proof_context, block_type=block_type,
                block_text=block, model=model, timeout_s=timeout, why=why,
                prior_failed_blocks=fails_txt
            )
        except Exception:
            blk = ""
        
        if not _is_effective_block(blk):
            continue
        
        # STRICT DEDUP: If this block matches ANY prior failure, skip it immediately
        fp_new = _fingerprint_block(blk)
        all_prior_fps = set([_fingerprint_block(b) for b in (mem.prev_blocks + prior_blocks_for_type)])
        
        if fp_new in all_prior_fps:
            if trace:
                print(f"[repair] Skipping duplicate block (fingerprint: {fp_new[:8]}...)")
            continue  # Don't even try to verify, just skip
        
        if block_type == "case":
            blk = _strip_wrapper_to_case_block(blk, block)
        elif block_type == "have-show":
            blk = _strip_wrapper_to_have_show(blk, block)
        elif block_type == "subproof":
            blk = _strip_wrapper_to_subproof(blk)              
        if blk.strip() == block.strip():
            continue

        _log("repair", f"{block_type}-block (output)", blk, trace=trace)
        new_block_lines = blk.splitlines()
        patched_lines = original_lines[:start] + new_block_lines + original_lines[end:]
        patched = "\n".join(patched_lines)

        try:
            thy = build_theory(patched.splitlines(), add_print_state=False, end_with=None)
            ok, _ = finished_ok(
                _run_theory_with_timeout(
                    isabelle, session, thy, timeout_s=min(_ISA_VERIFY_TIMEOUT_S, max(1, int(left())))
                )
            )
        except Exception:
            ok = False
        patched_holes = len(_SORRY_RE.findall(patched))
        # A structural repair is allowed to split one opaque obligation into a
        # bounded number of later leaf obligations.  Isabelle must accept the
        # admitted document and its earliest source frontier must advance.
        bounded_decomposition = patched_holes <= original_holes + 6
        if ok and bounded_decomposition and (
            original_had_error or frontier_advanced(original_text, patched)
        ):
            return patched

        # Record the actual failed candidate and feed its Isabelle diagnostics
        # into the next proposal.  Do not compound unverified rewrites.
        fp = _fingerprint_block(blk)
        if fp and fp not in mem.prev_fps:
            mem.prev_fps.add(fp)
            mem.prev_blocks.insert(0, blk)
            mem.prev_blocks = mem.prev_blocks[:_MAX_PREV_BLOCKS]
            if isinstance(prior_store, dict):
                lst = prior_store.setdefault(block_type, [])
                if fp not in [_fingerprint_block(x) for x in lst]:
                    lst.insert(0, blk)
                    del lst[_MAX_PREV_BLOCKS:]
        _, candidate_errs = _quick_state_and_errors(isabelle, session, patched)
        candidate_error_texts = _normalize_error_texts(candidate_errs)
        if candidate_error_texts:
            err_texts = candidate_error_texts

    return original_text

# ---------- Public helper: whole-proof regeneration with prior-failure banlist ----------
def regenerate_whole_proof(*, full_text: str, goal_text: str, model: Optional[str],
                           isabelle, session: str, budget_s: float = 20.0,
                           trace: bool = False, prior_outline_text: Optional[str] = None
                          ) -> Tuple[str, bool, str]:
    """
    Re-generate the last proof..qed block (or from the lemma head to EOF if no qed yet),
    feeding decisive lines from `prior_outline_text` as a ban list so the LLM avoids
    repeating previously failed tactics. Only returns a patched text if it *verifies*.
    """
    lines = full_text.splitlines()
    ws, we = _enclosing_whole_proof(lines)
    lemma_index = next(
        (
            index for index in range(min(max(ws, 0), len(lines) - 1), -1, -1)
            if re.match(r"^\s*(?:lemma|theorem|corollary)\b", lines[index])
        ),
        -1,
    )
    if ws < 0 or we <= ws:
        # Fallback: from first lemma/theorem head to EOF
        start = None
        for i in range(len(lines) - 1, -1, -1):
            if re.match(r"^\s*(?:lemma|theorem|corollary)\b", lines[i]):
                start = i
                break
        if start is None:
            return full_text, False, "whole:region-not-found"
        ws, we = start, len(lines)
        lemma_index = start

    # Preserve the complete command header (including multi-line
    # assumes/shows/fixes clauses) when Stage 3 replaces only its proof body.
    # For proof..qed blocks ``ws`` is already the boundary.  A one-line
    # ``by``/``sorry`` fallback has no explicit proof block, so locate its
    # first body command instead.
    if ws > lemma_index:
        header_end = ws
    else:
        header_end = next(
            (
                index for index in range(lemma_index + 1, len(lines))
                if _HAS_BODY.match(lines[index])
            ),
            min(len(lines), lemma_index + 1),
        )
    header_lines = list(lines[lemma_index:header_end])
    if header_lines:
        # Also normalize compact commands such as ``lemma "P" by simp``.
        # Search only after the final quote so words inside the proposition do
        # not look like proof syntax.
        last = header_lines[-1]
        quote = last.rfind('"')
        if quote >= 0:
            inline = re.search(r"\s+(?:by\b.*|sorry\b.*)$", last[quote + 1:])
            if inline:
                header_lines[-1] = last[: quote + 1 + inline.start()].rstrip()

    # Simple local timer for the block repair
    t0 = time.monotonic()
    left = lambda: max(0.0, budget_s - (time.monotonic() - t0))

    # Stage 3a is deterministic strategy regeneration.  It uses the same
    # definition analysis as Fill but replaces the complete failed proof,
    # which also handles non-hole syntax/tactic failures.  LLM regeneration is
    # retained as Stage 3b when no verified structural portfolio succeeds.
    if lemma_index >= 0 and left() > 4.0:
        preamble = "\n".join(lines[:lemma_index]).strip()
        lemma_indent = lines[lemma_index][
            : len(lines[lemma_index]) - len(lines[lemma_index].lstrip())
        ]

        # Stage 3a.1: prove inferred auxiliary obligations independently, then
        # rebuild the target proof in the enriched theory.  This is a genuine
        # two-node proof DAG: a false shape/algebra conjecture cannot leak into
        # a repair candidate, and a successful invariant remains visible in
        # the returned Isar rather than being hidden inside an LLM rewrite.
        obligations = definition_guided_obligations(
            goal_text, preamble, limit=3
        )
        occupied = set(re.findall(
            r"\b(?:lemma|theorem|corollary)\s+([A-Za-z_][A-Za-z0-9_']*)",
            preamble,
        ))
        obligation_proof_cache: Dict[
            Tuple[str, str], Tuple[bool, str]
        ] = {}
        for obligation in obligations:
            if left() <= 5.0:
                break
            aux_name = obligation.name_hint
            suffix = 1
            while aux_name in occupied:
                suffix += 1
                aux_name = f"{obligation.name_hint}_{suffix}"
            if trace:
                print(
                    f'[repair] Stage 3 obligation candidate '
                    f'[{obligation.kind}] {aux_name}: '
                    f'"{obligation.top_level_statement}"'
                )
            checked_proof = None
            for proof_method in obligation.proof_methods[:2]:
                if left() <= 4.0:
                    break
                cache_key = (obligation.top_level_statement, proof_method)
                cached = cache_key in obligation_proof_cache
                if cached:
                    aux_ok, diagnostic = obligation_proof_cache[cache_key]
                else:
                    aux_lines = (
                        lines[:lemma_index]
                        + [
                            f'lemma {aux_name}: '
                            f'"{obligation.top_level_statement}"',
                            proof_method,
                        ]
                    )
                    try:
                        aux_theory = build_theory(
                            aux_lines, add_print_state=False, end_with=None
                        )
                        responses = _run_theory_with_timeout(
                            isabelle,
                            session,
                            aux_theory,
                            timeout_s=max(3, int(min(10.0, left()))),
                        )
                        aux_ok, _ = finished_ok(responses)
                        diagnostic = "" if aux_ok else first_error_message(responses)
                    except Exception as exc:
                        aux_ok = False
                        diagnostic = f"{type(exc).__name__}: {exc}"
                    obligation_proof_cache[cache_key] = (aux_ok, diagnostic)
                if trace:
                    mark = "✓" if aux_ok else "×"
                    cache_note = " [cached]" if cached else ""
                    print(
                        f"[repair] Stage 3 obligation {mark} "
                        f"[{obligation.kind}] {proof_method}{cache_note}"
                    )
                    rendered_diagnostic = _compact_diagnostic(diagnostic)
                    if not aux_ok and rendered_diagnostic:
                        print(f"[repair]   Isabelle: {rendered_diagnostic}")
                if aux_ok:
                    checked_proof = proof_method
                    break
            if checked_proof is None:
                continue
            occupied.add(aux_name)

            for main_template in obligation.main_methods[:3]:
                if left() <= 3.0:
                    break
                main_method = main_template.format(aux=aux_name)
                patched_lines = (
                    lines[:lemma_index]
                    + [
                        f'lemma {aux_name}: "{obligation.top_level_statement}"',
                        checked_proof,
                    ]
                    + header_lines
                    + [f"{lemma_indent}  {main_method}"]
                    + lines[we:]
                )
                candidate = "\n".join(patched_lines)
                try:
                    thy = build_theory(
                        candidate.splitlines(), add_print_state=False, end_with=None
                    )
                    ok, _ = finished_ok(_run_theory_with_timeout(
                        isabelle,
                        session,
                        thy,
                        timeout_s=max(3, int(min(12.0, left()))),
                    ))
                except Exception:
                    ok = False
                if not ok:
                    continue
                if trace:
                    print(
                        f"[repair] Stage 3 verified {obligation.kind} "
                        "obligation and rebuilt the whole proof."
                    )
                return candidate, True, "regen:verified-obligation"

        # Stage 3a.2: direct definition-guided whole-proof strategies.  These
        # remain the cheapest option when no auxiliary invariant is needed.
        strategies = definition_guided_finishers(
            goal_text, preamble, limit=6
        )
        for strategy in strategies:
            if left() <= 2.0:
                break
            patched_lines = (
                lines[:lemma_index]
                + header_lines
                + [f"{lemma_indent}  {strategy}"]
                + lines[we:]
            )
            candidate = "\n".join(patched_lines)
            try:
                thy = build_theory(candidate.splitlines(), add_print_state=False, end_with=None)
                ok, _ = finished_ok(_run_theory_with_timeout(
                    isabelle,
                    session,
                    thy,
                    timeout_s=max(2, int(min(10.0, left()))),
                ))
            except Exception:
                ok = False
            if ok:
                if trace:
                    print(f"[repair] Stage 3 definition strategy verified: {strategy}")
                return candidate, True, "regen:definition-strategy"
    # Use empty/quick state — the block prompt already carries enough context
    state0 = ""
    # Seed prior failed blocks with the previous outline (so the first round won't repeat it)
    prior_store: Dict[str, List[str]] = {}
    if prior_outline_text:
        prior_store["whole"] = [prior_outline_text]
    patched = _repair_block(full_text, lines, ws, we, goal_text, state0, isabelle, session,
                            model, left, trace, "whole", stage=3, prior_store=prior_store,
                            max_candidates=3,
                            failure_notes=["Earlier Fill and local/subproof repair were exhausted."])
    if patched != full_text:
        # _repair_block only returns a different text if it verified successfully
        return patched, True, "regen:whole-proof"
    return full_text, False, "regen:no-change"
