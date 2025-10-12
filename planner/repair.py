from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import requests

from planner.repair_inputs import (_find_first_hole, _hole_line_bounds, _snippet_window, 
    _clamp_line_index, _quick_state_and_errors, _extract_error_lines, _run_theory_with_timeout,
    _print_state_before_hole, _normalize_error_texts, _facts_from_state, 
    get_counterexample_hints_for_repair, _earliest_failure_anchor, _nearest_header, 
    _recent_steps, _APPLY_OR_BY)
from planner.prompts import _LOCAL_SYSTEM, _LOCAL_USER, _BLOCK_SYSTEM, _BLOCK_USER
from prover.config import (MODEL as DEFAULT_MODEL, OLLAMA_HOST, TIMEOUT_S as OLLAMA_TIMEOUT_S,
    OLLAMA_NUM_PREDICT, TEMP as OLLAMA_TEMP, TOP_P as OLLAMA_TOP_P)
from prover.isabelle_api import build_theory, finished_ok

# ========== Configuration ==========
_ISA_FAST_TIMEOUT_S = int(os.getenv("ISABELLE_FAST_TIMEOUT_S", "12"))
_ISA_VERIFY_TIMEOUT_S = int(os.getenv("ISABELLE_VERIFY_TIMEOUT_S", "30"))
_SESSION = requests.Session()
_REPAIR_RULES_JSON = os.getenv("REPAIR_RULES_JSON", "").strip()
_MAX_PREV_BLOCKS = int(os.getenv("REPAIR_MAX_PREV_BLOCKS", "4"))

# ========== Regex Patterns ==========
_HEAD_CMD_RE = re.compile(r"^\s*(have|show|obtain|then\s+show|thus|hence)\b")
_PROOF_RE = re.compile(r"^\s*proof\b")
_QED_RE = re.compile(r"^\s*qed\b")
_CASE_LINE_RE = re.compile(r"^\s*case\b")
_NEXT_OR_QED_RE = re.compile(r"^\s*(?:next|qed)\b")
_TACTIC_LINE = re.compile(r"^\s*(?:apply|by)\b|(?:\s)by\s+\S")
_STRUCTURAL_LINE = re.compile(r"^\s*(?:lemma|theorem|qed|next|proof|case|have|show|assume|fix|from|using|thus|hence|ultimately|finally|also|moreover|let|where)\b")
_INLINE_BY_TAIL = re.compile(r"\s+by\s+.+$")
_WRAPPED_THEOREM_HEAD = re.compile(r"(?mx)\A(?:[ \t]*(?:\(\*.*?\*\)|\<comment\>.*?\<\/comment\>)[ \t]*\n|[ \t]*\n)*[ \t]*(?:lemma|theorem|corollary)\b")

# ========== Utility Functions ==========
def _log(prefix: str, label: str, content: str, trace: bool = True) -> None:
    if trace and content:
        print(f"[{prefix}] {label} (len={len(content)}):\n{content if content.strip() else '  (empty)'}")

def _sanitize_llm_block(text: str) -> str:
    """Remove LLM artifacts and balance proof/qed."""
    if not text:
        return text
    
    patterns = [r"^\s*(?:<<<)?(?:BLOCK|PROOF)\s*$", r"^\s*```(?:isabelle|isar)?\s*$",
                r"^\s*(?:<<<|>>>)\s*$", r"^\s*(?:lemma|theorem|corollary|proposition)\b.*$", r"^\s*---\s*$"]
    compiled = [re.compile(p) for p in patterns]
    lines = [l for l in text.splitlines() if not any(p.match(l) for p in compiled)]
    
    # Balance proof/qed blocks
    balance, last_closed = 0, -1
    for i, l in enumerate(lines):
        if _PROOF_RE.match(l):
            balance += 1
        elif _QED_RE.match(l):
            if balance > 0:
                balance -= 1
                if balance == 0:
                    last_closed = i
    
    if last_closed != -1 and last_closed + 1 < len(lines):
        lines = lines[:last_closed + 1]
    
    return "\n".join(lines).strip()

def _fingerprint_block(text: str) -> str:
    """Canonicalize block for duplicate detection."""
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text.strip())
    return t.replace("`", "").replace(""", '"').replace(""", '"').replace("'", "'")

def _trim_block_for_prompt(text: str, max_chars: int = 800) -> str:
    """Trim long blocks for prompts."""
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    head, tail = t[:max_chars // 2].rstrip(), t[-max_chars // 2:].lstrip()
    return f"{head}\n…\n{tail}"

def _is_tactic_line(s: str) -> bool:
    return bool(_TACTIC_LINE.search(s)) and not bool(_STRUCTURAL_LINE.match(s))

def _extract_proof_context(full_text: str, block_start_line: int) -> str:
    """Extract lemma header and proof content before block."""
    lines = full_text.splitlines()
    lemma_line = next((i for i in range(min(block_start_line, len(lines) - 1), -1, -1)
                       if re.match(r"^\s*(?:lemma|theorem|corollary|proposition)\b", lines[i])), -1)
    
    if lemma_line < 0:
        start = max(0, block_start_line - 10)
        return "\n".join(lines[start:block_start_line]).strip()
    
    return "\n".join(lines[lemma_line:block_start_line]).strip()

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
    payload = {"model": model, "prompt": prompt, 
               "options": {"temperature": OLLAMA_TEMP, "top_p": OLLAMA_TOP_P, "num_predict": OLLAMA_NUM_PREDICT},
               "stream": False}
    resp = _SESSION.post(f"{OLLAMA_HOST.rstrip('/')}/api/generate", json=payload,
                        timeout=(10.0, max(30.0, float(timeout_s))))
    resp.raise_for_status()
    return _sanitize_llm_block(resp.json().get("response", "").strip())

def _hf_generate(prompt: str, model_id: str, timeout_s: int) -> str:
    token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACE_API_TOKEN is not set")
    
    payload = {"inputs": prompt,
               "parameters": {"temperature": OLLAMA_TEMP, "top_p": OLLAMA_TOP_P,
                            "max_new_tokens": OLLAMA_NUM_PREDICT, "return_full_text": False},
               "options": {"wait_for_model": True}}
    resp = _SESSION.post(f"https://api-inference.huggingface.co/models/{model_id}",
                        headers={"Authorization": f"Bearer {token}"}, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    
    data = resp.json()
    if isinstance(data, list) and data:
        result = data[0].get("generated_text", "")
    elif isinstance(data, dict):
        result = data.get("generated_text", "") or (data.get("choices", [{}])[0].get("text", ""))
    else:
        result = str(data)
    return _sanitize_llm_block(result.strip())

def _gemini_generate(prompt: str, model_id: str, timeout_s: int) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
               "generationConfig": {"maxOutputTokens": OLLAMA_NUM_PREDICT}}
    resp = _SESSION.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}",
                        json=payload, timeout=timeout_s)
    resp.raise_for_status()
    
    try:
        candidates = resp.json().get("candidates", [])
        result = candidates[0]["content"]["parts"][0]["text"] if candidates else ""
    except Exception:
        result = str(resp.json())
    return _sanitize_llm_block(result.strip())

# ========== Data Classes ==========
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
    prev_blocks: List[str] = field(default_factory=list)
    prev_fps: Set[str] = field(default_factory=set)

# ========== Repair Proposal ==========
def _propose_block_repair(*, goal: str, errors: List[str], ce_hints: Dict[str, List[str]],
                         proof_context: str, block_type: str, block_text: str, model: Optional[str],
                         timeout_s: int, why: str = "Previous attempt failed; propose a different block-level change.",
                         prior_failed_blocks: Optional[str] = None) -> str:
    ce = ce_hints.get("bindings", []) + ce_hints.get("def_hints", [])
    
    if block_type == "have-show":
        prompt = f"{_LOCAL_SYSTEM}\n\n{_LOCAL_USER}".format(
            goal=goal, errors="\n".join(f"- {e}" for e in errors) or "(none)",
            ce_hints="\n".join(ce) or "(none)", proof_context=proof_context.strip(),
            block_text=block_text.rstrip(), why=why, prior_failed_blocks=prior_failed_blocks or "(none)")
    else:
        prompt = f"{_BLOCK_SYSTEM}\n\n{_BLOCK_USER}".format(
            goal=goal, errors="\n".join(f"- {e}" for e in errors) or "(none)",
            ce_hints="\n".join(ce) or "(none)", proof_context=proof_context.strip(),
            block_text=block_text.rstrip(), why=why, prior_failed_blocks=prior_failed_blocks or "(none)")
    
    try:
        return _sanitize_llm_block(_generate_simple(prompt, model=model, timeout_s=timeout_s))
    except Exception:
        return ""

def propose_rule_based_repairs(goal_text: str, state_block: str, header: str, facts: List[str]) -> List[RepairOp]:
    """Declarative fallback using JSON rules."""
    if not _REPAIR_RULES_JSON:
        return []
    
    try:
        with open(_REPAIR_RULES_JSON, "r", encoding="utf-8") as f:
            rules = json.load(f)
    except Exception:
        return []
    
    def _match(rule) -> Optional[RepairOp]:
        cond, op = rule.get("when", {}), rule.get("op", {})
        g, st, hd, fs = goal_text or "", state_block or "", header or "", facts or []
        
        checks = [
            ("goal_contains_any", lambda v: any(k in g for k in v)),
            ("state_contains_any", lambda v: any(k in st for k in v)),
            ("facts_contains_any", lambda v: any(x in fs for x in v)),
            ("goal_regex", lambda v: bool(re.search(v, g))),
            ("header_startswith", lambda v: hd.startswith(v)),
            ("header_regex", lambda v: bool(re.search(v, hd))),
            ("not_header_contains", lambda v: not any(k in hd for k in v)),
        ]
        
        for key, pred in checks:
            if key in cond and cond[key] and not pred(cond[key]):
                return None
        
        if "insert_before_hole" in op and isinstance(op["insert_before_hole"], str):
            return ("insert_before_hole", InsertBeforeHole(op["insert_before_hole"].strip()))
        if "replace_in_snippet" in op and isinstance(op["replace_in_snippet"], dict):
            r = op["replace_in_snippet"]
            fnd, rep = (r.get("find") or "").strip(), (r.get("replace") or "").strip()
            if fnd and rep:
                return ("replace_in_snippet", ReplaceInSnippet(fnd, rep))
        if "insert_have_block" in op and isinstance(op["insert_have_block"], dict):
            v = op["insert_have_block"]
            lab, stmt = v.get("label", "H"), v.get("statement", "")
            aft, hint = v.get("after_line_matching", "then show ?thesis"), v.get("body_hint", "apply simp")
            if stmt.strip() and aft.strip():
                return ("insert_have_block", InsertHaveBlock(lab.strip(), stmt.strip(), aft.strip(), hint.strip()))
        return None
    
    return [rop for r in (rules if isinstance(rules, list) else []) if (rop := _match(r))][:3]

# ========== Region Analysis ==========
def _enclosing_case_block(lines: List[str], hole_line: int) -> Tuple[int, int]:
    i = next((idx for idx in range(hole_line, -1, -1) if _CASE_LINE_RE.match(lines[idx])), -1)
    if i < 0:
        return (-1, -1)
    j = next((idx for idx in range(hole_line, len(lines)) if _NEXT_OR_QED_RE.match(lines[idx])), len(lines))
    return (i, j)

def _enclosing_subproof(lines: List[str], hole_line: int) -> Tuple[int, int]:
    i = next((idx for idx in range(hole_line, -1, -1) if _PROOF_RE.match(lines[idx])), -1)
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
    head_re = re.compile(r"^\s*(have|show|obtain)\b")
    fence_re = re.compile(r"^\s*(?:have|show|obtain|thus|hence|then|also|moreover|ultimately|finally|case\b|next\b|qed\b)\b")
    
    while i >= 0 and not head_re.match(lines[i]):
        if re.match(r"^\s*(?:case\b|next\b|qed\b)\b", lines[i]):
            return (-1, -1)
        i -= 1
    
    if i < 0:
        return (-1, -1)
    
    if _INLINE_BY_TAIL.search(lines[i] or ""):
        return (i, i + 1)
    
    depth, j = 0, i + 1
    while j < len(lines):
        L = lines[j]
        if depth == 0:
            if (L or "").strip() == "sorry":
                return (i, j + 1)
            if fence_re.match(L or ""):
                break
        
        if _PROOF_RE.match(L or ""):
            depth += 1
        elif _QED_RE.match(L or ""):
            depth = max(0, depth - 1)
        j += 1
    
    return (i, j if j > i else -1)

def _enclosing_whole_proof(lines: List[str]) -> Tuple[int, int]:
    last_qed = next((i for i in range(len(lines) - 1, -1, -1) if _QED_RE.match(lines[i])), -1)
    if last_qed < 0:
        return (-1, -1)
    proof_start = next((i for i in range(last_qed, -1, -1) if _PROOF_RE.match(lines[i])), -1)
    return (proof_start, last_qed + 1) if proof_start >= 0 else (-1, -1)

# ========== Wrapper Stripping ==========
def _strip_wrapper_to_case_block(proposed: str, original_case_block: str) -> str:
    if not _WRAPPED_THEOREM_HEAD.match(proposed):
        return proposed
    
    case_name = None
    for pattern in [r"(?m)^\s*case\s*\((\w+)", r"(?m)^\s*case\s+(\w+)"]:
        if m := re.search(pattern, original_case_block or ""):
            case_name = m.group(1)
            break
    
    lines = proposed.splitlines()
    start = next((i for i, L in enumerate(lines) if _CASE_LINE_RE.match(L) and
                 (case_name is None or re.search(rf"\b{re.escape(case_name)}\b", L))), None)
    
    if start is None:
        return proposed
    
    end = next((j for j in range(start + 1, len(lines)) if _NEXT_OR_QED_RE.match(lines[j])), len(lines))
    return "\n".join(lines[start:end]).rstrip()

def _strip_wrapper_to_have_show(proposed: str, original_block: str) -> str:
    lines = proposed.splitlines()
    if not lines:
        return proposed
    
    head_re = re.compile(r"^\s*(have|show|obtain)\b")
    fence_re = re.compile(r"^\s*(?:have|show|obtain|thus|hence|then|also|moreover|ultimately|finally|case\b|next\b|qed\b)\b")
    
    head_idx = next((i for i, L in enumerate(lines) if head_re.match(L)), -1)
    if head_idx == -1:
        return proposed
    
    out, depth = [lines[head_idx]], 0
    for L in lines[head_idx + 1:]:
        if depth == 0:
            if (L or "").strip() == "sorry":
                out.append(L)
                break
            if fence_re.match(L or ""):
                break
        
        if _PROOF_RE.match(L or ""):
            depth += 1
        elif _QED_RE.match(L or ""):
            depth = max(0, depth - 1)
        out.append(L)
    
    while out and out[-1].strip() == "":
        out.pop()
    
    if len(out) == 1 and _INLINE_BY_TAIL.search(out[0] or ""):
        return out[0].rstrip()
    return "\n".join(out).rstrip()

def _strip_wrapper_to_subproof(proposed: str) -> str:
    if not _WRAPPED_THEOREM_HEAD.match(proposed):
        return proposed
    
    lines = proposed.splitlines()
    start = next((i for i, L in enumerate(lines) if _PROOF_RE.match(L)), None)
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
def _replace_failing_tactics_with_sorry(block_text: str, *, full_text_lines: List[str],
                                       start_line: int, end_line: int, isabelle, session: str,
                                       trace: bool = False) -> str:
    block_lines = block_text.splitlines()
    if not block_lines:
        return block_text
    
    def build_doc(with_block_lines: List[str]) -> str:
        s0 = max(0, start_line - 1)
        e0 = max(s0, min(end_line - 1, len(full_text_lines)))
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
        
        indent = block_lines[cand][:len(block_lines[cand]) - len(block_lines[cand].lstrip())]
        
        if block_lines[cand].lstrip().startswith("apply"):
            head_idx = next((i for i in range(cand, -1, -1) if _HEAD_CMD_RE.match(block_lines[i] or "")), None)
            if head_idx is not None:
                head_indent = block_lines[head_idx][:len(block_lines[head_idx]) - len(block_lines[head_idx].lstrip())]
                seq_s = next((i for i in range(cand, -1, -1) if not _is_tactic_line(block_lines[i])), 0) + 1
                seq_e = next((i for i in range(cand + 1, len(block_lines)) if not _is_tactic_line(block_lines[i])), len(block_lines))
                block_lines[seq_s:seq_e] = [f"{head_indent}proof -", f"{head_indent}  sorry", f"{head_indent}qed"]
            else:
                break
        else:
            block_lines[cand] = f"{indent}sorry"
    
    return "\n".join(block_lines)

# ========== Main Repair Entry Point ==========
def try_cegis_repairs(*, full_text: str, hole_span: Tuple[int, int], goal_text: str, model: Optional[str],
                     isabelle, session: str, repair_budget_s: float = 15.0, max_ops_to_try: int = 3,
                     beam_k: int = 1, allow_whole_fallback: bool = False, trace: bool = False,
                     resume_stage: int = 0) -> Tuple[str, bool, str]:
    t0 = time.monotonic()
    left = lambda: max(0.0, repair_budget_s - (time.monotonic() - t0))
    current_text = full_text
    state0 = _print_state_before_hole(isabelle, session, current_text, hole_span, trace=trace)
    _log("repair", "State block", state0, trace=trace)
    
    prior_store: Dict[str, List[str]] = {}
    
    hole_line, _, lines = _hole_line_bounds(current_text, hole_span)
    anchor_line, anchor_reason = _earliest_failure_anchor(isabelle, session, current_text, default_line_0=hole_line)
    focus_line = _clamp_line_index(lines, anchor_line)
    
    if trace and anchor_line != hole_line:
        print(f"[repair] Retargeting from hole line {hole_line + 1} to earliest-failure line {anchor_line + 1} ({anchor_reason})")
    
    # Stage 1: have/show block
    hs_s, hs_e = _enclosing_have_show_block(lines, focus_line)
    if resume_stage <= 1 and hs_s >= 0 and left() > 5.0:
        if trace:
            print("[repair] Trying have/show block repair…")
        patched = _repair_block(current_text, lines, hs_s, hs_e, goal_text, state0,
                               isabelle, session, model, left, trace, "have-show", 1, prior_store)
        if patched != current_text:
            thy = build_theory(patched.splitlines(), add_print_state=False, end_with=None)
            ok, _ = finished_ok(_run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_VERIFY_TIMEOUT_S))
            return patched, ok, f"stage=1 {'block:have-show' if ok else 'partial-progress'}"
    
    lines = current_text.splitlines()
    state0 = _print_state_before_hole(isabelle, session, current_text, hole_span, trace=trace)
    
    # Stage 2a: Case block
    cs, ce = _enclosing_case_block(lines, focus_line)
    if resume_stage <= 2 and cs >= 0 and left() > 5.0:
        if trace:
            print("[repair] Trying case-block repair…")
        patched = _repair_block(current_text, lines, cs, ce, goal_text, state0, isabelle, session,
                               model, left, trace, "case", 2, prior_store)
        if patched != current_text:
            thy = build_theory(patched.splitlines(), add_print_state=False, end_with=None)
            ok, _ = finished_ok(_run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_VERIFY_TIMEOUT_S))
            return patched, ok, f"stage=2 {'block:case' if ok else 'partial-progress'}"
    
    # Stage 2b: Subproof
    ps, pe = _enclosing_subproof(lines, focus_line)
    if resume_stage <= 2 and ps >= 0 and left() > 3.0:
        if trace:
            print("[repair] Trying subproof repair…")
        patched = _repair_block(current_text, lines, ps, pe, goal_text, state0, isabelle, session,
                               model, left, trace, "subproof", 2, prior_store)
        if patched != current_text:
            thy = build_theory(patched.splitlines(), add_print_state=False, end_with=None)
            ok, _ = finished_ok(_run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_VERIFY_TIMEOUT_S))
            return patched, ok, f"stage=2 {'block:subproof' if ok else 'partial-progress'}"
    
    return current_text, False, f"stage={resume_stage} cegis-nohelp"

def _repair_block(current_text: str, lines: List[str], start: int, end: int, goal_text: str,
                 state0: str, isabelle, session: str, model: Optional[str], left, trace: bool,
                 block_type: str, stage: int, prior_store: Dict[str, List[str]]) -> str:
    _, errs = _quick_state_and_errors(isabelle, session, current_text)
    err_texts = _normalize_error_texts(errs)
    ce = get_counterexample_hints_for_repair(isabelle, session, state0, timeout_s=10)
    block = "\n".join(lines[start:end])
    proof_context = _extract_proof_context(current_text, start)
    
    _log("repair", f"{block_type}-block (input)", block, trace=trace)
    _log("repair", "proof_context (LLM input)", proof_context, trace=trace)
    ce_list = ce.get("bindings", []) + ce.get("def_hints", []) if isinstance(ce, dict) else []
    _log("repair", "errors (LLM input)", "\n".join(err_texts) or "(none)", trace=trace)
    _log("repair", "counterexamples (LLM input)", "\n".join(ce_list) or "(none)", trace=trace)
    
    rounds = 3 if left() >= 18.0 else 2 if left() >= 10.0 else 1
    mem = _RepairMemory()
    last_attempt = None  # Track if we made any valid attempts
    
    for rr in range(rounds):
        if left() <= 3.0:
            break
        
        mem.rounds = rr + 1
        why = f"Previous {block_type}-block attempt did not solve the goal; try a different strategy."
        timeout = int(min(60, max(8, left() * (0.55 / max(1, rounds - rr)))))
        
        # Build prior failed blocks (deduplicated)
        prior_blocks_for_type = list(prior_store.get(block_type, []))
        seed_list = [block] + mem.prev_blocks + prior_blocks_for_type
        
        seen, uniq = set(), []
        for b in seed_list:
            if fpb := _fingerprint_block(b):
                if fpb not in seen:
                    seen.add(fpb)
                    uniq.append(b)
        
        fails_txt = "\n---\n".join(_trim_block_for_prompt(b) for b in uniq[:_MAX_PREV_BLOCKS]) or "(none)"
        _log("repair", "prior_block_failures (LLM input)", fails_txt, trace=trace)
        
        try:
            blk = _propose_block_repair(goal=goal_text, errors=err_texts, ce_hints=ce,
                                       proof_context=proof_context, block_type=block_type,
                                       block_text=block, model=model, timeout_s=timeout,
                                       why=why, prior_failed_blocks=fails_txt)
        except Exception:
            blk = ""
        
        if not blk or not _sanitize_llm_block(blk).strip():
            continue
        
        # Strict deduplication
        fp_new = _fingerprint_block(blk)
        all_prior_fps = {_fingerprint_block(b) for b in (mem.prev_blocks + prior_blocks_for_type)}
        
        if fp_new in all_prior_fps:
            if trace:
                print(f"[repair] Skipping duplicate block (fingerprint: {fp_new[:8]}...)")
            continue
        
        # Strip wrapper based on block type
        if block_type == "case":
            blk = _strip_wrapper_to_case_block(blk, block)
        elif block_type == "have-show":
            blk = _strip_wrapper_to_have_show(blk, block)
        elif block_type == "subproof":
            blk = _strip_wrapper_to_subproof(blk)
        
        if blk.strip() == block.strip():
            continue
        
        blk_with_sorry = _replace_failing_tactics_with_sorry(blk, full_text_lines=lines,
                                                             start_line=start + 1, end_line=end + 1,
                                                             isabelle=isabelle, session=session, trace=trace)
        _log("repair", f"{block_type}-block (output)", blk_with_sorry, trace=trace)
        
        # Record failure
        fp = _fingerprint_block(blk_with_sorry)
        if fp and fp not in mem.prev_fps:
            mem.prev_fps.add(fp)
            mem.prev_blocks.insert(0, blk_with_sorry)
            mem.prev_blocks = mem.prev_blocks[:_MAX_PREV_BLOCKS]
            
            lst = prior_store.setdefault(block_type, [])
            if fp not in {_fingerprint_block(x) for x in lst}:
                lst.insert(0, blk_with_sorry)
                del lst[_MAX_PREV_BLOCKS:]
        
        # Apply patch
        new_block_lines = blk_with_sorry.splitlines()
        patched_lines = lines[:start] + new_block_lines + lines[end:]
        patched = "\n".join(patched_lines)
        
        thy = build_theory(patched.splitlines(), add_print_state=False, end_with=None)
        ok, _ = finished_ok(_run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_VERIFY_TIMEOUT_S))
        
        if ok:
            return patched
        
        # Track that we made at least one valid attempt (even if unverified)
        last_attempt = patched
    
    # Return last attempt if we made any, otherwise return original
    # This ensures driver can detect we tried something
    return last_attempt if last_attempt is not None else current_text

def regenerate_whole_proof(*, full_text: str, goal_text: str, model: Optional[str],
                           isabelle, session: str, budget_s: float = 20.0,
                           trace: bool = False, prior_outline_text: Optional[str] = None
                          ) -> Tuple[str, bool, str]:
    """Regenerate last proof..qed block with prior-failure banlist."""
    lines = full_text.splitlines()
    ws, we = _enclosing_whole_proof(lines)
    
    if ws < 0 or we <= ws:
        # Fallback: from lemma to EOF
        start = next((i for i, L in enumerate(lines) if re.match(r"^\s*(?:lemma|theorem|corollary)\b", L)), None)
        if start is None:
            return full_text, False, "whole:region-not-found"
        ws, we = start, len(lines)
    
    t0 = time.monotonic()
    left = lambda: max(0.0, budget_s - (time.monotonic() - t0))
    
    prior_store: Dict[str, List[str]] = {}
    if prior_outline_text:
        prior_store["whole"] = [prior_outline_text]
    
    patched = _repair_block(full_text, lines, ws, we, goal_text, "", isabelle, session,
                            model, left, trace, "whole", 3, prior_store)
    # Decide result & verify before returning
    if not patched or patched == full_text:
        return full_text, False, "whole:no-change"
    try:
        thy = build_theory(patched.splitlines(), add_print_state=False, end_with=None)
        ok, _ = finished_ok(_run_theory_with_timeout(
            isabelle, session, thy, timeout_s=int(min(60, max(15, budget_s)))
        ))
    except Exception:
        ok = False
    return patched, ok, ("whole:verified" if ok else "whole:unverified")

# Add this comment at the end of the file to clarify the repair flow:
# Note: Driver's _handle_repair_result should return should_cont=False when stage cap is reached
# so that whole-proof regeneration can be triggered. The current implementation in driver.py
# has a bug where should_cont is always True, preventing escalation to whole-proof regen.