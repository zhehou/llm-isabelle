from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
import concurrent.futures
import requests

from prover.config import (
    MODEL as DEFAULT_MODEL, OLLAMA_HOST, TIMEOUT_S as OLLAMA_TIMEOUT_S,
    OLLAMA_NUM_PREDICT, TEMP as OLLAMA_TEMP, TOP_P as OLLAMA_TOP_P,
)
from prover.isabelle_api import build_theory, run_theory, last_print_state_block
from prover.utils import parse_subgoals

# Global session for connection reuse
_SESSION = requests.Session()

def _sanitize_llm_block(text: str) -> str:
    """Remove known fence lines, preserving Isabelle content."""
    if not isinstance(text, str) or not text:
        return text
    
    fence_patterns = [
        r"^\s*<<<BLOCK\s*$", r"^\s*BLOCK\s*$", r"^\s*<<<PROOF\s*$", r"^\s*PROOF\s*$",
        r"^\s*```\s*$", r"^\s*```isabelle\s*$", r"^\s*```isar\s*$"
    ]
    compiled_patterns = [re.compile(pattern) for pattern in fence_patterns]
    
    lines = [line for line in text.splitlines() 
             if not any(pattern.match(line) for pattern in compiled_patterns)]
    return "\n".join(lines).strip()

def _is_effective_block(text: str) -> bool:
    """True if block has non-empty, non-fence content."""
    return bool(_sanitize_llm_block(text or "").strip())

def _extract_print_state_from_responses(resps: List) -> str:
    """Extract print_state output from Isabelle responses."""
    standard_result = last_print_state_block(resps) or ""
    llm_lines = []
    
    for resp in (resps or []):
        if str(getattr(resp, "response_type", "")).upper() != "FINISHED":
            continue
            
        body = getattr(resp, "response_body", None)
        if isinstance(body, bytes):
            body = body.decode(errors="replace")
            
        try:
            data = json.loads(body) if isinstance(body, str) and body.strip().startswith("{") else body
            if not isinstance(data, dict):
                continue
                
            for node in data.get("nodes", []):
                for msg in node.get("messages", []):
                    if msg.get("kind") != "writeln":
                        continue
                    text = msg.get("message", "")
                    if text.startswith(("[LLM_SUBGOAL]", "[LLM_VARS]")):
                        llm_lines.append(text)
                    elif "goal" in text and "subgoal" in text and not standard_result:
                        standard_result = text
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
    
    if llm_lines and standard_result:
        return standard_result + "\n" + "\n".join(llm_lines)
    return standard_result or "\n".join(llm_lines)

# =========================
# Generation backends
# =========================

def _ollama_generate(prompt: str, model: str, timeout_s: int) -> str:
    """Generate using Ollama API."""
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {"temperature": OLLAMA_TEMP, "top_p": OLLAMA_TOP_P, "num_predict": OLLAMA_NUM_PREDICT},
        "stream": False,
    }
    
    timeout = (10.0, max(30.0, float(timeout_s)))  # (connect, read)
    resp = _SESSION.post(f"{OLLAMA_HOST.rstrip('/')}/api/generate", 
                        json=payload, timeout=timeout)
    resp.raise_for_status()
    return _sanitize_llm_block(resp.json().get("response", "").strip())

def _hf_generate(prompt: str, model_id: str, timeout_s: int) -> str:
    """Generate using HuggingFace API."""
    token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACE_API_TOKEN is not set")
        
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": OLLAMA_TEMP, "top_p": OLLAMA_TOP_P,
            "max_new_tokens": OLLAMA_NUM_PREDICT, "return_full_text": False,
        },
        "options": {"wait_for_model": True},
    }
    
    resp = _SESSION.post(f"https://api-inference.huggingface.co/models/{model_id}",
                        headers={"Authorization": f"Bearer {token}"},
                        json=payload, timeout=timeout_s)
    resp.raise_for_status()
    
    data = resp.json()
    if isinstance(data, list) and data:
        result = data[0].get("generated_text", "")
    elif isinstance(data, dict):
        result = data.get("generated_text", "")
        if not result and "choices" in data:
            choices = data["choices"]
            result = choices[0].get("text", "") if choices else ""
    else:
        result = str(data)
    
    return _sanitize_llm_block(result.strip())

def _gemini_generate(prompt: str, model_id: str, timeout_s: int) -> str:
    """Generate using Gemini API."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": OLLAMA_NUM_PREDICT}
    }
    
    resp = _SESSION.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}",
        json=payload, timeout=timeout_s)
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

def _generate_simple(prompt: str, model: Optional[str] = None, *, timeout_s: Optional[int] = None) -> str:
    """Route to appropriate generation backend."""
    m = model or DEFAULT_MODEL
    timeout = timeout_s or OLLAMA_TIMEOUT_S
    
    if m.startswith("hf:"):
        return _hf_generate(prompt, m[3:], timeout)
    elif m.startswith("gemini:"):
        return _gemini_generate(prompt, m[7:], timeout)
    elif m.startswith("ollama:"):
        m = m[7:]
    
    return _ollama_generate(prompt, m, timeout)

# =========================
# Repair operations
# =========================

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

# Regex patterns
_CTX_HEAD = re.compile(r"^\s*(?:using|from|with|then|ultimately|finally|also|moreover)\b")
_HAS_BODY = re.compile(r"^\s*(?:by\b|apply\b|proof\b|sorry\b|done\b)")
_APPLY_OR_BY = re.compile(r"^\s*(apply\b|by\b)")
_INLINE_BY_TAIL = re.compile(r"\s+by\s+.+$")
_HEADER_RE = re.compile(r"^\s*(proof\s*\(|proof\b|case\s+|then\s+show\b)")

def _extract_json_array(text: str) -> Optional[list]:
    """Extract JSON array from text."""
    try:
        return json.loads(text)
    except Exception:
        i, j = text.find("["), text.rfind("]")
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(text[i:j+1])
            except Exception:
                pass
    return None

def _parse_repair_ops(text: str) -> List[RepairOp]:
    """Parse repair operations from LLM output."""
    data = _extract_json_array(text.strip())
    if not isinstance(data, list):
        return []
    
    ops = []
    for item in data:
        if not isinstance(item, dict) or len(item) != 1:
            continue
        
        k, v = next(iter(item.items()))
        
        if k == "insert_before_hole" and isinstance(v, str) and v.strip():
            ops.append(("insert_before_hole", InsertBeforeHole(v.strip())))
        elif k == "replace_in_snippet" and isinstance(v, dict):
            f, r = v.get("find", ""), v.get("replace", "")
            if isinstance(f, str) and isinstance(r, str) and f.strip() and r.strip():
                ops.append(("replace_in_snippet", ReplaceInSnippet(f.strip(), r.strip())))
        elif k == "insert_have_block" and isinstance(v, dict):
            lab = v.get("label", "H")
            stmt = v.get("statement", "")
            after = v.get("after_line_matching", "then show ?thesis")
            hint = v.get("body_hint", "apply simp")
            if all(isinstance(x, str) for x in (lab, stmt, after, hint)) and stmt.strip() and after.strip():
                ops.append(("insert_have_block", 
                           InsertHaveBlock(lab.strip(), stmt.strip(), after.strip(), hint.strip())))
    
    return ops[:3]

# =========================
# Context analysis helpers
# =========================

def _find_first_hole(lines: List[str]) -> Optional[int]:
    """Find first line containing 'sorry'."""
    for i, line in enumerate(lines):
        if "sorry" in line:
            return i
    return None

def _hole_line_bounds(full_text: str, hole_span: Tuple[int, int]) -> Tuple[int, int, List[str]]:
    """Get hole line number, indentation, and text lines."""
    lines = full_text.splitlines()
    hole_line = full_text[:hole_span[0]].count("\n")
    line_text = lines[hole_line] if 0 <= hole_line < len(lines) else ""
    indent = len(line_text) - len(line_text.lstrip(" "))
    return hole_line, indent, lines

def _snippet_window(lines: List[str], hole_line: int, radius: int = 12) -> Tuple[int, int]:
    """Get window around hole line."""
    return max(0, hole_line - radius), min(len(lines), hole_line + radius + 1)

def _facts_from_state(state_block: str, limit: int = 16) -> List[str]:
    """Extract facts from state block."""
    if not state_block:
        return []
    
    defs = re.findall(r"\b([A-Za-z_][A-Za-z0-9_']*)_def\b", state_block)
    tokens = re.findall(r"\b([A-Za-z_][A-Za-z0-9_']*)\b", state_block)
    
    seen, facts = set(), []
    for x in defs + tokens:
        if x and x not in seen:
            seen.add(x)
            facts.append(x)
            if len(facts) >= limit:
                break
    return facts

def _nearest_header(lines: List[str], hole_line: int) -> str:
    """Find nearest header line above hole."""
    for i in range(hole_line, -1, -1):
        if _HEADER_RE.match(lines[i].strip()):
            return lines[i].strip()
    return ""

def _recent_steps(lines: List[str], hole_line: int, max_lines: int = 5) -> List[str]:
    """Get recent proof steps above hole."""
    steps = []
    for i in range(hole_line - 1, -1, -1):
        line = lines[i]
        if _APPLY_OR_BY.match(line):
            steps.append(line.strip())
            if len(steps) >= max_lines:
                break
        if line.strip().startswith(("case ", "proof", "qed", "lemma ")):
            break
    return list(reversed(steps))

# =========================
# Patch application
# =========================

def _block_has_body_already(lines: List[str]) -> bool:
    """Check if block above hole already has body tactic."""
    idx = _find_first_hole(lines)
    if idx is None:
        return False
    
    k = idx - 1
    while k >= 0 and (lines[k].strip() == "" or _CTX_HEAD.match(lines[k])):
        k -= 1
    
    return k >= 0 and (_HAS_BODY.match(lines[k]) or _INLINE_BY_TAIL.search(lines[k]))

def _insert_before_hole_ctxaware(lines: List[str], payload_line: str) -> List[str]:
    """Insert after contiguous context lines before hole."""
    idx = _find_first_hole(lines)
    if idx is None:
        return lines
    
    k = idx - 1
    while k >= 0 and (lines[k].strip() == "" or _CTX_HEAD.match(lines[k])):
        k -= 1
    
    insert_at = k + 1
    indent = lines[idx][:len(lines[idx]) - len(lines[idx].lstrip(" "))]
    return lines[:insert_at] + [f"{indent}{payload_line}"] + lines[insert_at:]

def _apply_insert_before_hole(full_text: str, hole_span: Tuple[int, int], line: str) -> str:
    """Apply insert_before_hole operation."""
    hole_line, _, lines = _hole_line_bounds(full_text, hole_span)
    
    # Handle finalizers
    if _APPLY_OR_BY.match(line) or line.strip() in ("done", "."):
        if hole_line is not None:
            indent = lines[hole_line][:len(lines[hole_line]) - len(lines[hole_line].lstrip(" "))]
            lines[hole_line] = f"{indent}{line.strip()}"
            return "\n".join(lines) + ("\n" if full_text.endswith("\n") else "")
    
    # Skip if body already exists or line is duplicate
    if _block_has_body_already(lines):
        return full_text
    
    win_s, win_e = max(0, hole_line - 4), hole_line + 1
    if any(L.strip() == line.strip() for L in lines[win_s:win_e]):
        return full_text
    
    new_lines = _insert_before_hole_ctxaware(lines, line)
    return "\n".join(new_lines) + ("\n" if full_text.endswith("\n") else "") if new_lines != lines else full_text

def _apply_replace_in_snippet(full_text: str, hole_span: Tuple[int, int], find: str, replace: str) -> str:
    """Apply replace_in_snippet operation."""
    hole_line, _, lines = _hole_line_bounds(full_text, hole_span)
    s, e = _snippet_window(lines, hole_line)
    snippet = lines[s:e]
    
    # Try exact match first
    try:
        idx = snippet.index(find)
        if snippet[idx].strip() == replace.strip():
            return full_text
        snippet[idx] = replace
    except ValueError:
        # Try stripped match
        stripped = [L.strip() for L in snippet]
        try:
            idx = stripped.index(find.strip())
            orig = snippet[idx]
            leading = orig[:len(orig) - len(orig.lstrip(" "))]
            if orig.strip() == replace.strip():
                return full_text
            snippet[idx] = leading + replace.lstrip(" ")
        except ValueError:
            return full_text
    
    new_lines = lines[:s] + snippet + lines[e:]
    return "\n".join(new_lines) + ("\n" if full_text.endswith("\n") else "")

def _apply_insert_have_block(full_text: str, hole_span: Tuple[int, int], 
                           label: str, statement: str, after_line_matching: str, body_hint: str) -> str:
    """Apply insert_have_block operation."""
    hole_line, indent, lines = _hole_line_bounds(full_text, hole_span)
    s, e = _snippet_window(lines, hole_line)
    
    anchor_idx = hole_line
    for i in range(s, e):
        if lines[i].strip() == after_line_matching.strip():
            anchor_idx = i
            break
    
    pad = " " * max(2, indent)
    block = [f'{pad}have {label}: "{statement}"', f"{pad}  sorry"]
    new_lines = lines[:anchor_idx] + block + lines[anchor_idx:]
    return "\n".join(new_lines) + ("\n" if full_text.endswith("\n") else "")

# =========================
# Isabelle interaction
# =========================

def _quick_state_and_errors(isabelle, session: str, full_text: str) -> Tuple[str, List[str]]:
    """Get state and errors from Isabelle."""
    try:
        thy = build_theory(full_text.splitlines(), add_print_state=True, end_with="sorry")
        resps = run_theory(isabelle, session, thy)
        state_block = _extract_print_state_from_responses(resps)
        
        # Extract errors
        errors = []
        for r in resps or []:
            body = getattr(r, "response_body", None)
            if isinstance(body, bytes):
                body = body.decode(errors="replace")
            if not isinstance(body, str):
                continue
            try:
                data = json.loads(body)
                if isinstance(data, dict) and data.get("kind") == "error":
                    errors.append(str(data.get("message", "")))
            except json.JSONDecodeError:
                if any(err in body for err in ["*** Error:", "*** Outer syntax error", "*** Failed"]):
                    errors.append(body.strip().splitlines()[-1])
        
        return state_block, errors[:3]
    except Exception:
        return "", ["transport_or_build_error"]

def _quick_state_subgoals(isabelle, session: str, text: str) -> int:
    """Get subgoal count from Isabelle state."""
    try:
        thy = build_theory(text.splitlines(), add_print_state=True, end_with="sorry")
        resps = run_theory(isabelle, session, thy)
        block = _extract_print_state_from_responses(resps)
        if not block.strip():
            return 9999
        n = parse_subgoals(block)
        return int(n) if isinstance(n, int) else 9999
    except Exception:
        return 9999

# =========================
# Counterexample hints
# =========================

def _run_nitpick_at_hole(isabelle, session: str, full_text: str, hole_span: Tuple[int, int], timeout_s: int = 3) -> str:
    """Run Nitpick at hole location."""
    s, e = hole_span
    injected = f"  prefer 1\n  nitpick [timeout={max(1, timeout_s)}]\n  (* NITPICK-MARK *)\n  sorry\n"
    variant = full_text[:s] + injected + full_text[e:]
    
    try:
        thy = build_theory(variant.splitlines(), add_print_state=True, end_with="sorry")
        resps = run_theory(isabelle, session, thy)
        return "\n".join(getattr(r, "response_body", b"").decode(errors="replace") 
                        if isinstance(getattr(r, "response_body", None), bytes)
                        else str(getattr(r, "response_body", ""))
                        for r in resps or [])
    except Exception:
        return ""

def _nitpick_state_hints_from_text(text: str) -> Dict[str, List[str]]:
    """Extract hints from Nitpick output."""
    bindings = re.findall(r"\b([a-z][A-Za-z0-9_']*)\s*=\s*([^,\s][^,\n]*)", text or "")
    defs = list(dict.fromkeys(re.findall(r"\b([A-Za-z_][A-Za-z0-9_']*)_def\b", text or "")))
    
    bind_list = [f"{v} = {val.strip()}" for v, val in bindings]
    def_hints = []
    if defs:
        def_hints.extend(f"{d}_def" for d in defs)
        def_hints.extend(f"unfolding {d}_def" for d in defs)
        def_hints.extend(f"simp only: {d}_def" for d in defs)
    
    return {"bindings": bind_list[:8], "def_hints": def_hints[:12]}

def _counterexample_hints(isabelle, session: str, full_text: str, hole_span: Tuple[int, int]) -> Dict[str, List[str]]:
    """Get counterexample hints from Nitpick or fallback to state."""
    nit = _run_nitpick_at_hole(isabelle, session, full_text, hole_span, timeout_s=3)
    hints = _nitpick_state_hints_from_text(nit)
    
    if hints.get("bindings") or hints.get("def_hints"):
        return hints
    
    # Fallback to print_state
    state_only, _ = _quick_state_and_errors(isabelle, session, full_text)
    return _nitpick_state_hints_from_text(state_only)

# =========================
# Repair proposal
# =========================

_REPAIR_SYSTEM = """You patch an Isabelle/Isar proof LOCALLY. Do not regenerate the whole proof.
Return ONLY a JSON array of at most 3 patch operations from the allowed schema.

ALLOWED OPS:
1) {"insert_before_hole": "<ONE LINE>"}
2) {"replace_in_snippet": {"find": "<EXACT LINE>", "replace": "<NEW LINE>"}}
3) {"insert_have_block": {"label":"H", "statement":"<FORMULA>", "after_line_matching":"<LINE>", "body_hint":"<ONE LINE>"}}

Rules:
- Edit only inside the provided SNIPPET
- Prefer single-line hints like `apply (simp add: …)`, `using …`
- Output MUST be valid JSON"""

_REPAIR_USER = """GOAL: {goal}

STATE_BEFORE_HOLE: {state_block}

ISABELLE_ERRORS: {errors}

COUNTEREXAMPLE_HINTS: {ce_hints}

FACTS_CANDIDATES: {facts_list}

NEAREST_HEADER: {nearest_header}

RECENT_STEPS: {recent_steps}

SNIPPET:
<<<SNIPPET
{block_snippet}
SNIPPET

Output JSON array of patch ops:"""

def propose_local_repairs(*, goal: str, state_block: str, errors: List[str], 
                         ce_hints: Dict[str, List[str]], block_snippet: str,
                         nearest_header: str, recent_steps: List[str], facts: List[str],
                         model: Optional[str], timeout_s: int) -> List[RepairOp]:
    """Propose local repair operations."""
    ce_list = (ce_hints.get("bindings", []) + ce_hints.get("def_hints", []))
    
    prompt = _REPAIR_SYSTEM + "\n\n" + _REPAIR_USER.format(
        goal=goal,
        state_block=(state_block or "").strip(),
        errors="\n".join(f"- {e}" for e in errors) or "(none)",
        ce_hints="\n".join(ce_list) or "(none)",
        block_snippet=block_snippet.rstrip(),
        nearest_header=nearest_header.strip(),
        recent_steps="\n".join(recent_steps),
        facts_list=", ".join(facts),
    )
    
    try:
        raw = _generate_simple(prompt, model=model, timeout_s=timeout_s)
        return _parse_repair_ops(raw)
    except Exception:
        return []

def _heuristic_fallback_ops(goal_text: str, state_block: str, header: str, facts: List[str]) -> List[RepairOp]:
    """Generate heuristic fallback operations."""
    ops = []
    
    # Common patterns
    if "Let " in state_block or "Let_def" in facts:
        ops.append(("insert_before_hole", InsertBeforeHole("apply (unfolding Let_def)")))
    
    g = goal_text
    if ("map" in g and "@" in g) or ("map_append" in facts):
        ops.append(("insert_before_hole", InsertBeforeHole("apply (simp add: map_append)")))
    if ("length" in g and "@" in g) or ("length_append" in facts):
        ops.append(("insert_before_hole", InsertBeforeHole("apply (simp add: length_append)")))
    if "@" in g or "append_assoc" in facts:
        ops.append(("insert_before_hole", InsertBeforeHole("apply (simp add: append_assoc)")))
    
    # Induction header fix
    if header.startswith("proof (induction") and "arbitrary:" not in header:
        for v in ("ys", "zs"):
            if v in g:
                new_header = header.rstrip(")") + f" arbitrary: {v})"
                ops.append(("replace_in_snippet", ReplaceInSnippet(header, new_header)))
                break
    
    return ops[:3]

# =========================
# Main repair function
# =========================

def try_local_repairs(*, full_text: str, hole_span: Tuple[int, int], goal_text: str,
                     model: Optional[str], isabelle, session: str,
                     repair_budget_s: float = 12.0, max_ops_to_try: int = 2,
                     beam_k: int = 1, trace: bool = False) -> Tuple[str, bool, str]:
    """Try local repairs with budget management."""
    start = time.monotonic()
    left = lambda: max(0.0, repair_budget_s - (time.monotonic() - start))
    
    # Get baseline and context
    s0 = _quick_state_subgoals(isabelle, session, full_text)
    state0, errs0 = _quick_state_and_errors(isabelle, session, full_text)
    
    hole_line, _, lines = _hole_line_bounds(full_text, hole_span)
    s, e = _snippet_window(lines, hole_line, radius=12)
    snippet = "\n".join(lines[s:e])
    
    header = _nearest_header(lines, hole_line)
    rsteps = _recent_steps(lines, hole_line)
    
    # Get facts and CE hints if time allows
    facts, ce = [], {"bindings": [], "def_hints": []}
    if left() >= 8.0:
        facts = _facts_from_state(state0, limit=8)
    if left() >= 12.0:
        ce = _counterexample_hints(isabelle, session, full_text, hole_span)
        if ce.get("def_hints"):
            facts = list(dict.fromkeys(ce["def_hints"] + facts))[:12]
    
    # Propose operations
    ops = []
    remaining = left()
    HEADROOM_S = 3
    
    if remaining > HEADROOM_S + 2.0:
        propose_timeout = int(max(2, min(45, (remaining - HEADROOM_S) * 0.6)))
        if propose_timeout >= 3:
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(propose_local_repairs,
                                     goal=goal_text, state_block=state0, errors=errs0,
                                     ce_hints=ce, block_snippet=snippet, nearest_header=header,
                                     recent_steps=rsteps, facts=facts, model=model, timeout_s=propose_timeout)
                    ops = future.result(timeout=propose_timeout)
            except (concurrent.futures.TimeoutError, requests.RequestException):
                ops = []
    
    if not ops:
        ops = _heuristic_fallback_ops(goal_text, state0, header, facts)
    
    # Try operations
    best_text, best_score, best_kind = None, 9999, ""
    tried, any_changed = 0, False
    
    for kind, payload in ops:
        if left() <= 0 or tried >= max_ops_to_try:
            break
        tried += 1
        
        # Apply operation
        if kind == "insert_before_hole":
            cand = _apply_insert_before_hole(full_text, hole_span, payload.line)
        elif kind == "replace_in_snippet":
            cand = _apply_replace_in_snippet(full_text, hole_span, payload.find, payload.replace)
        elif kind == "insert_have_block":
            cand = _apply_insert_have_block(full_text, hole_span, payload.label, 
                                          payload.statement, payload.after_line_matching, payload.body_hint)
        else:
            continue
        
        if cand == full_text:
            continue
            
        any_changed = True
        s1 = _quick_state_subgoals(isabelle, session, cand)
        
        if trace:
            print(f"[repair] {kind} scored: {s0} -> {s1}")
        
        if s1 != 9999 and s1 < best_score:
            best_score, best_text, best_kind = s1, cand, kind
    
    # Accept non-regressive improvements
    if best_text is not None:
        non_regressive = (best_score != 9999) and (best_score <= s0 or best_kind in ("insert_before_hole", "replace_in_snippet"))
        if non_regressive:
            return best_text, True, f"beam:{best_kind}"
    
    # Fallback strategies
    if tried == 0 and ops:
        kind, payload = ops[0]
        if kind == "insert_before_hole":
            return _apply_insert_before_hole(full_text, hole_span, payload.line), True, "blind-insert0"
        elif kind == "replace_in_snippet":
            return _apply_replace_in_snippet(full_text, hole_span, payload.find, payload.replace), True, "blind-replace0"
    
    if any_changed and all(s == 9999 for s in [best_score]):
        kind, payload = ops[0]
        if kind == "insert_before_hole":
            return _apply_insert_before_hole(full_text, hole_span, payload.line), True, "blind-accept"
        elif kind == "replace_in_snippet":
            return _apply_replace_in_snippet(full_text, hole_span, payload.find, payload.replace), True, "blind-accept"
    
    return full_text, False, "repairs-did-not-help"

# =========================
# Block repair for larger regions
# =========================

_BLOCK_SYSTEM = """You repair an Isabelle/Isar PROOF BLOCK.
Edit ONLY the given BLOCK. Preserve the rest verbatim.
The repaired block MUST compile and be as small as possible.
Use 'sorry' as placeholder for steps that cannot be proven.
Output ONLY the new block (no fences, no comments)."""

_BLOCK_USER = """GOAL: {goal}
ISABELLE_ERRORS: {errors}
COUNTEREXAMPLE_HINTS: {ce_hints}
LOCAL_CONTEXT: {state_block}

ORIGINAL BLOCK:
<<<BLOCK
{block_text}
BLOCK
"""

def _propose_block_repair(*, goal: str, errors: List[str], ce_hints: Dict[str, List[str]],
                         state_block: str, block_text: str, model: Optional[str], timeout_s: int) -> str:
    """Propose block-level repair."""
    ce = ce_hints.get("bindings", []) + ce_hints.get("def_hints", [])
    
    prompt = _BLOCK_SYSTEM + "\n\n" + _BLOCK_USER.format(
        goal=goal,
        errors="\n".join(f"- {e}" for e in errors) or "(none)",
        ce_hints="\n".join(ce) or "(none)",
        state_block=(state_block or "").strip(),
        block_text=block_text.rstrip(),
    )
    
    try:
        out = _generate_simple(prompt, model=model, timeout_s=timeout_s)
        return _sanitize_llm_block(out)
    except Exception:
        return ""

# =========================
# Region analysis for block repairs
# =========================

def _enclosing_case_block(lines: List[str], hole_line: int) -> Tuple[int, int]:
    """Find enclosing case block."""
    case_re = re.compile(r"(?m)^\s*case\b")
    next_re = re.compile(r"(?m)^\s*next\b")
    qed_re = re.compile(r"(?m)^\s*qed\b")
    
    i = hole_line
    while i >= 0 and not case_re.match(lines[i]):
        i -= 1
    if i < 0:
        return (-1, -1)
    
    j = hole_line
    while j < len(lines) and not (next_re.match(lines[j]) or qed_re.match(lines[j])):
        j += 1
    return (i, j)

def _enclosing_subproof(lines: List[str], hole_line: int) -> Tuple[int, int]:
    """Find enclosing subproof block."""
    proof_re = re.compile(r"(?m)^\s*proof\b")
    qed_re = re.compile(r"(?m)^\s*qed\b")
    
    i = hole_line
    while i >= 0 and not proof_re.match(lines[i]):
        i -= 1
    if i < 0:
        return (-1, -1)
    
    depth, j = 1, i + 1
    while j < len(lines) and depth > 0:
        if proof_re.match(lines[j]):
            depth += 1
        elif qed_re.match(lines[j]):
            depth -= 1
        j += 1
    return (i, j if j > i else -1)

def _looks_like_reasonable_proof_attempt(block_text: str) -> bool:
    """Check if block looks like reasonable proof attempt."""
    lines = block_text.strip().split('\n')
    if len(lines) < 2:
        return False
    
    has_structure = any(keyword in line for line in lines 
                       for keyword in ['have ', 'show ', 'using '])
    non_sorry_lines = [line for line in lines if 'sorry' not in line.strip()]
    
    return has_structure and len(non_sorry_lines) >= len(lines) // 2

def _replace_failing_tactics_with_sorry(block_text: str) -> str:
    """Replace likely-failing tactics with sorry."""
    lines = block_text.split('\n')
    result_lines = []
    
    for line in lines:
        if any(keyword in line for keyword in ['have ', 'show ', 'case ', 'using ', 'then ']):
            result_lines.append(line)
        elif any(tactic in line.strip() for tactic in ['by simp', 'by auto', 'by blast', 'by clarsimp']):
            indent = line[:len(line) - len(line.lstrip())]
            result_lines.append(f"{indent}sorry")
        else:
            result_lines.append(line)
    
    return '\n'.join(result_lines)

# =========================
# CEGIS: Multi-stage repair with iteration
# =========================

def try_cegis_repairs(*, full_text: str, hole_span: Tuple[int, int], goal_text: str,
                     model: Optional[str], isabelle, session: str,
                     repair_budget_s: float = 15.0, max_ops_to_try: int = 3,
                     beam_k: int = 1, allow_whole_fallback: bool = False,
                     trace: bool = False) -> Tuple[str, bool, str]:
    """Try CEGIS repairs with multiple stages."""
    t0 = time.monotonic()
    left = lambda: max(0.0, repair_budget_s - (time.monotonic() - t0))
    
    base_s = _quick_state_subgoals(isabelle, session, full_text)
    current_text = full_text
    
    # Stage 1: Local repairs (iterate a few rounds)
    for round_i in range(3):
        if left() <= 5.0:
            break
        
        eff_k = 1 if left() < 15.0 else max(1, beam_k)
        patched, ok, tag = try_local_repairs(
            full_text=current_text, hole_span=hole_span, goal_text=goal_text,
            model=model, isabelle=isabelle, session=session,
            repair_budget_s=min(12.0, max(8.0, left() * 0.4)),
            max_ops_to_try=max_ops_to_try, beam_k=eff_k, trace=trace,
        )
        
        if ok and patched != current_text:
            s1 = _quick_state_subgoals(isabelle, session, patched)
            if trace:
                print(f"[cegis] local round {round_i}: {base_s} -> {s1}")
            
            if s1 != 9999 and s1 <= base_s:
                return patched, True, f"local:{tag}"
            
            if "blind" in (tag or ""):
                return patched, True, f"local:{tag}(neutral)"
            
            if s1 == 9999 and s1 == base_s:
                current_text = patched
                if trace:
                    print(f"[cegis] using changed text for next iteration")
    
    # Stage 2: Case-block rewrite
    hole_line, _, lines = _hole_line_bounds(current_text, hole_span)
    cs, ce = _enclosing_case_block(lines, hole_line)
    
    if cs >= 0 and left() > 5.0:
        state_block, errs = _quick_state_and_errors(isabelle, session, current_text)
        ceh = _counterexample_hints(isabelle, session, current_text, hole_span)
        block = "\n".join(lines[cs:ce])
        
        block_timeout = int(min(60, max(20, left() * 0.6)))
        
        try:
            blk = _propose_block_repair(
                goal=goal_text, errors=errs, ce_hints=ceh,
                state_block=state_block, block_text=block,
                model=model, timeout_s=block_timeout
            )
        except Exception:
            blk = ""
        
        if _is_effective_block(blk) and blk.strip() != block.strip():
            # Try raw suggestion
            patched_raw = "\n".join(lines[:cs] + [blk] + lines[ce:])
            s1_raw = _quick_state_subgoals(isabelle, session, patched_raw)
            if trace:
                print(f"[cegis] case-block (raw): {base_s} -> {s1_raw}")
            
            if s1_raw != 9999 and s1_raw <= base_s:
                return patched_raw, True, "block:case(raw)"
            
            # Aggressive acceptance for reasonable attempts
            if s1_raw == 9999 and _looks_like_reasonable_proof_attempt(blk):
                if trace:
                    print("[cegis] accepting reasonable case block despite scoring issues")
                blk_with_sorry = _replace_failing_tactics_with_sorry(blk)
                patched_sorry = "\n".join(lines[:cs] + [blk_with_sorry] + lines[ce:])
                return patched_sorry, True, "block:case(aggressive)"
            
            # Conservative fallback
            blk_sanit = re.sub(r"(?m)^\s*(?:by\b.*|done\s*)$", "  sorry", blk)
            patched = "\n".join(lines[:cs] + [blk_sanit] + lines[ce:])
            s1 = _quick_state_subgoals(isabelle, session, patched)
            if trace:
                print(f"[cegis] case-block (sanit): {base_s} -> {s1}")
            
            if s1 != 9999 and s1 <= base_s:
                return patched, True, "block:case(sanit)"
    
    # Stage 3: Subproof rewrite
    ps, pe = _enclosing_subproof(lines, hole_line)
    
    if ps >= 0 and left() > 3.0:
        state_block, errs = _quick_state_and_errors(isabelle, session, current_text)
        ceh = _counterexample_hints(isabelle, session, current_text, hole_span)
        block = "\n".join(lines[ps:pe])
        
        subproof_timeout = int(min(45, max(15, left() * 0.7)))
        
        try:
            blk = _propose_block_repair(
                goal=goal_text, errors=errs, ce_hints=ceh,
                state_block=state_block, block_text=block,
                model=model, timeout_s=subproof_timeout
            )
        except Exception:
            blk = ""
        
        if _is_effective_block(blk) and blk.strip() != block.strip():
            patched_raw = "\n".join(lines[:ps] + [blk] + lines[pe:])
            s1_raw = _quick_state_subgoals(isabelle, session, patched_raw)
            if trace:
                print(f"[cegis] subproof-block (raw): {base_s} -> {s1_raw}")
            
            if s1_raw != 9999 and s1_raw <= base_s:
                return patched_raw, True, "block:subproof(raw)"
            
            if s1_raw == 9999 and _looks_like_reasonable_proof_attempt(blk):
                if trace:
                    print("[cegis] accepting reasonable subproof despite scoring issues")
                blk_with_sorry = _replace_failing_tactics_with_sorry(blk)
                patched_sorry = "\n".join(lines[:ps] + [blk_with_sorry] + lines[pe:])
                return patched_sorry, True, "block:subproof(aggressive)"
    
    # Return modified text even without measurable improvement
    if current_text != full_text:
        if trace:
            print("[cegis] returning modified text despite no scoring improvement")
        return current_text, True, "partial-progress"
    
    return full_text, False, "cegis-nohelp"