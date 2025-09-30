from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
import requests

from prover.config import (
    MODEL as DEFAULT_MODEL, OLLAMA_HOST, TIMEOUT_S as OLLAMA_TIMEOUT_S,
    OLLAMA_NUM_PREDICT, TEMP as OLLAMA_TEMP, TOP_P as OLLAMA_TOP_P,
)
from prover.isabelle_api import build_theory, run_theory, last_print_state_block, finished_ok

from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout

# -------- Timeout config (env-tunable) ----------
# "fast" queries (print_state / quick parse)
_ISA_FAST_TIMEOUT_S  = int(os.getenv("ISABELLE_FAST_TIMEOUT_S", "12"))
# "verify" gates (post-patch full theory checks)
_ISA_VERIFY_TIMEOUT_S = int(os.getenv("ISABELLE_VERIFY_TIMEOUT_S", "30"))

def _run_theory_with_timeout(isabelle, session: str, thy: List[str], *, timeout_s: Optional[int]) -> List:
    """
    Run Isabelle with a hard wall-clock timeout. If the timeout elapses:
      - try to interrupt the client (best effort),
      - raise TimeoutError("isabelle_run_timeout").
    """
    if not timeout_s or timeout_s <= 0:
        return run_theory(isabelle, session, thy)
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(run_theory, isabelle, session, thy)
        try:
            return fut.result(timeout=timeout_s)
        except _FuturesTimeout:
            try:
                # Best-effort interrupt if supported by your client
                if hasattr(isabelle, "interrupt"):
                    isabelle.interrupt()
            except Exception:
                pass
            raise TimeoutError("isabelle_run_timeout")

# Global session for connection reuse
_SESSION = requests.Session()

def _log_state_block(prefix: str, block: str, trace: bool = True) -> None:
    """Pretty-print the exact proof state block we consumed."""
    if not trace:
        return
    b = block or ""
    print(f"[{prefix}] State block (length={len(b)}):")
    if b.strip():
        print(b)
    else:
        print("  (empty or whitespace only)")

def _log_block(prefix: str, label: str, block: str, trace: bool = True) -> None:
    """Pretty-print an LLM-proposed proof block (output)."""
    if not trace:
        return
    b = block or ""
    print(f"[{prefix}] Proposed {label} (length={len(b)}):")
    if b.strip():
        print(b)
    else:
        print("  (empty or whitespace only)")

def _log_input_block(prefix: str, label: str, block: str, trace: bool = True) -> None:
    """Pretty-print an input/original proof block (what we send to the LLM)."""
    if not trace:
        return
    b = block or ""
    print(f"[{prefix}] Input {label} (length={len(b)}):")
    if b.strip():
        print(b)
    else:
        print("  (empty or whitespace only)")        

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

def _clamp_line_index(lines: List[str], idx: int) -> int:
    """Return idx clamped into [0, len(lines)-1]. If the file is empty, return -1."""
    if not lines:
        return -1
    if idx < 0:
        return 0
    if idx >= len(lines):
        return len(lines) - 1
    return idx

def _nearest_structural_head_before(lines: List[str], idx: int) -> int:
    """
    Walk upward from idx to find a meaningful structural head to anchor:
    have/show/obtain | case | proof
    Returns a valid 0-based line index.
    """
    if not lines:
        return -1
    i = idx
    if i >= len(lines):
        i = len(lines) - 1
    if i < 0:
        i = 0
    head_re = re.compile(r"^\s*(?:have|show|obtain|case\b|proof\b)\b")
    for j in range(i, -1, -1):
        if head_re.match(lines[j]):
            return j
    return i

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
_APPLY_OR_BY_DECISIVE = re.compile(r"^\s*(apply|by)\b")
_INLINE_BY_TAIL = re.compile(r"\s+by\s+.+$")
_HEADER_RE = re.compile(r"^\s*(proof\s*\(|proof\b|case\s+|then\s+show\b)")

# Detect lines we should / should not replace with 'sorry'
_TACTIC_LINE = re.compile(r"^\s*(?:apply|by)\b|(?:\s)by\s+\S")
_STRUCTURAL_LINE = re.compile(
    r"^\s*(?:lemma|theorem|qed|next|proof|case|have|show|assume|fix|from|using|"
    r"thus|hence|ultimately|finally|also|moreover|let|where)\b"
)

def _is_tactic_line(s: str) -> bool:
    return bool(_TACTIC_LINE.search(s)) and not bool(_STRUCTURAL_LINE.match(s))

def _extract_error_lines(errs) -> list[int]:
    """Best-effort: pull 1-based line numbers from Isabelle errors."""
    out = []
    if not errs:
        return out
    for e in errs:
        # Support several possible shapes (dict / obj with .line)
        ln = None
        if isinstance(e, dict):
            ln = e.get("line") or e.get("start_line") or e.get("pos_line")
        else:
            ln = getattr(e, "line", None) or getattr(e, "start_line", None)
        if isinstance(ln, int):
            out.append(ln)
    return out

def _normalize_error_texts(errs) -> List[str]:
    texts: List[str] = []
    for ee in (errs or []):
        if isinstance(ee, dict):
            t = str(ee.get("text", "")).strip()
        else:
            t = str(ee).strip()
        if t:
            texts.append(t)
    return texts[:8]    

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
# Light-weight banlist + memory for iterative repair
# =========================

def _canon_line(s: str) -> str:
    """Canonicalize a single Isar/tactic line for comparison."""
    if not s:
        return ""
    t = s.strip()
    t = re.sub(r"\s+", " ", t)              # collapse whitespace
    t = re.sub(r";\s*$", "", t)             # drop trailing semicolons
    t = t.replace("`", "").replace("‹", "").replace("›", "")
    return t

def _extract_lines_from_op(op: Dict[str, Any]) -> List[str]:
    """Pull decisive line(s) from a JSON op to record as prior failures."""
    out: List[str] = []
    if "insert_before_hole" in op:
        out.append(op["insert_before_hole"])
    elif "replace_in_snippet" in op:
        r = op["replace_in_snippet"]
        out.append(r.get("replace", ""))
    elif "insert_have_block" in op:
        hb = op["insert_have_block"]
        # Record the one-liner hint if present, otherwise the have header.
        if isinstance(hb.get("body_hint"), str) and hb["body_hint"].strip():
            out.append(hb["body_hint"])
        stmt = hb.get("statement")
        if stmt:
            out.append(f'have "{stmt}"')
    return [l for l in out if isinstance(l, str) and l.strip()]

def _filter_ops_against_banlist(ops_json_text: str, ban: Set[str]) -> List[RepairOp]:
    """Parse ops JSON and drop ops whose decisive line is already banned."""
    ops = _parse_repair_ops(ops_json_text)
    keep: List[RepairOp] = []
    for kind, payload in ops:
        # Convert back to the JSON-shape dict for reuse of extractor
        op_dict: Dict[str, Any]
        if kind == "insert_before_hole":
            op_dict = {"insert_before_hole": payload.line}
        elif kind == "replace_in_snippet":
            op_dict = {"replace_in_snippet": {"find": payload.find, "replace": payload.replace}}
        elif kind == "insert_have_block":
            op_dict = {"insert_have_block": {
                "label": payload.label,
                "statement": payload.statement,
                "after_line_matching": payload.after_line_matching,
                "body_hint": payload.body_hint,
            }}
        else:
            keep.append((kind, payload))
            continue
        lines = _extract_lines_from_op(op_dict)
        key = _canon_line(" ".join(lines)) if lines else ""
        if key and key in ban:
            continue
        keep.append((kind, payload))
    return keep

@dataclass
class _RepairMemory:
    ban: Set[str] = field(default_factory=set)           # canonicalized lines to avoid
    rounds: int = 0

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
    """Extract *useful* facts from the printed state.
    Priority:
      (1) verbatim lines following `using this:` (indented), as propositions
      (2) any explicitly quoted facts (between double quotes) on their own lines
      (3) fall back to *_def names
    """
    if not state_block:
        return []

    facts: List[str] = []
    seen: set[str] = set()

    # (1) Grab the exact propositions under "using this:"
    m = re.search(r"using this:\n((?:[ \t].*\n)+)", state_block)
    if m:
        for L in m.group(1).splitlines():
            s = L.strip()
            if s and s not in seen:
                seen.add(s)
                facts.append(s)
                if len(facts) >= limit:
                    return facts

    # (2) Quoted propositions on their own lines (e.g., from subgoals or messages)
    for q in re.findall(r'(?m)^\s*"(.*?)"\s*$', state_block):
        s = q.strip()
        if s and s not in seen:
            seen.add(s)
            facts.append(s)
            if len(facts) >= limit:
                return facts

    # (3) *_def names remain handy for unfolding/simp
    for d in re.findall(r"\b([A-Za-z_][A-Za-z0-9_']*)_def\b", state_block):
        if d and d not in seen:
            seen.add(d)
            facts.append(f"{d}_def")
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
    block = [
        f'{pad}have {label}: "{statement}"',
        f"{pad}  proof -",
        f"{pad}    sorry",
        f"{pad}  qed",
    ]
    new_lines = lines[:anchor_idx] + block + lines[anchor_idx:]
    return "\n".join(new_lines) + ("\n" if full_text.endswith("\n") else "")

# =========================
# Isabelle interaction
# =========================

def _print_state_before_hole(isabelle, session: str, full_text: str, hole_span: Tuple[int, int], trace: bool = False) -> str:
    """
    Compile a variant that prints the goal state *at the hole*.
    We replace the hole's `sorry` with:
        prefer 1
        print_state
        (* REPAIR-PRINT-STATE *)
        sorry
    so Isabelle will always emit a state even if later text is malformed.
    """
    hole_line, indent, lines = _hole_line_bounds(full_text, hole_span)
    # If the span is stale (line no longer contains 'sorry'), fall back to the nearest actual hole
    if not (0 <= hole_line < len(lines) and "sorry" in lines[hole_line]):
        nearest = _find_first_hole(lines)
        if nearest is not None:
            hole_line = nearest
            indent = len(lines[hole_line]) - len(lines[hole_line].lstrip(" "))
    pad = " " * max(2, indent)
    # Purely diagnostic; no 'sorry' here
    injected_lines = [f"{pad}prefer 1", f"{pad}print_state", f"{pad}(* REPAIR-PRINT-STATE *)"]
    variant_lines = lines[:hole_line] + injected_lines + lines[hole_line:]
    variant = "\n".join(variant_lines) + ("\n" if full_text.endswith("\n") else "")
    try:
        thy = build_theory(variant.splitlines(), add_print_state=False, end_with=None)
        resps = _run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_FAST_TIMEOUT_S)
        return _extract_print_state_from_responses(resps)
    except Exception:
        return ""

def _quick_state_and_errors(isabelle, session: str, full_text: str) -> Tuple[str, List[dict]]:
    """Get state and errors from Isabelle (preserve line numbers when available)."""
    try:
        thy = build_theory(full_text.splitlines(), add_print_state=True, end_with=None)
        resps = _run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_FAST_TIMEOUT_S)
        state_block = _extract_print_state_from_responses(resps)
        
        # Extract errors (structured, with line numbers when possible)
        errors: List[dict] = []
        for r in resps or []:
            raw = getattr(r, "response_body", None)
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode(errors="replace")
            try:
                data = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                data = None

            # FINISHED JSON with nodes/messages
            if isinstance(data, dict) and data.get("nodes"):
                for node in (data.get("nodes") or []):
                    for msg in (node.get("messages") or []):
                        if str(msg.get("kind", "")).lower() != "error":
                            continue
                        txt = str(msg.get("message", "") or "")
                        pos = msg.get("pos") or {}
                        rng = msg.get("range") or {}
                        line = (
                            pos.get("line")
                            or (rng.get("start") or {}).get("line")
                            or msg.get("line")
                        )
                        if not isinstance(line, int):
                            m = re.search(r"\bline\s+(\d+)\b", txt)
                            if m:
                                try:
                                    line = int(m.group(1))
                                except Exception:
                                    line = None
                        err_obj = {"text": txt}
                        if isinstance(line, int):
                            err_obj["line"] = line
                        errors.append(err_obj)
                continue

            # Legacy string bodies — try best-effort line recovery
            if isinstance(raw, str):
                if any(k in raw for k in ["*** Error:", "*** Outer syntax error", "*** Failed"]):
                    txt = raw.strip().splitlines()[-1]
                    err_obj = {"text": txt}
                    m = re.search(r"\bline\s+(\d+)\b", raw)
                    if m:
                        try:
                            err_obj["line"] = int(m.group(1))
                        except Exception:
                            pass
                    errors.append(err_obj)

        return state_block, errors[:5]
    except Exception:
        return "", [{"text": "transport_or_build_error"}]

def _earliest_failure_anchor(isabelle, session: str, full_text: str, *, default_line_0: int) -> Tuple[int, str]:
    """
    Choose an anchor line (0-based) to target the smallest enclosing structure that actually fails first.
    Priority:
      1) earliest Isabelle error line (from _quick_state_and_errors)
      2) if no error lines but the theory doesn't finish OK, anchor to the first 'sorry' in the file
      3) fallback to the caller-provided default_line_0 (usually the current hole line)
    Returns (anchor_line_0, reason_tag).
    """
    try:
        lines = full_text.splitlines()
        _state, errs = _quick_state_and_errors(isabelle, session, full_text)
        err_lines = sorted(_extract_error_lines(errs))
        if err_lines:
            pos0 = (err_lines[0] - 1)  # Isabelle lines are 1-based
            if 0 <= pos0 < len(lines):
                return pos0, "error_line"
            # Out-of-range error line → prefer first sorry, else nearest head before EOF
            for i, L in enumerate(lines):
                if "sorry" in L:
                    return i, "first_sorry_from_error"
            return _nearest_structural_head_before(lines, len(lines) - 1), "error_line_out_of_range"
        # No error lines — check if the whole theory actually finishes OK.
        thy = build_theory(lines, add_print_state=False, end_with=None)
        ok, _ = finished_ok(_run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_VERIFY_TIMEOUT_S))
        if not ok:
            # Unsolved/failed but with no localized error spans: anchor to earliest 'sorry'
            for i, L in enumerate(lines):
                if "sorry" in L:
                    return i, "first_sorry"
        # Clean or no better info: keep the default
        return default_line_0, "default"
    except Exception:
        return default_line_0, "default"

# =========================
# Counterexample hints
# =========================

def _run_nitpick_at_hole(isabelle, session: str, full_text: str, hole_span: Tuple[int, int], timeout_s: int = 3) -> str:
    """Run Nitpick at hole location."""
    hole_line, indent, lines = _hole_line_bounds(full_text, hole_span)
    if not (0 <= hole_line < len(lines) and "sorry" in lines[hole_line]):
        nearest = _find_first_hole(lines)
        if nearest is not None:
            hole_line = nearest
            indent = len(lines[hole_line]) - len(lines[hole_line].lstrip(" "))
    pad = " " * max(2, indent)
    injected_lines = [
        f"{pad}prefer 1",
        f"{pad}nitpick [timeout={max(1, timeout_s)}]",
        f"{pad}(* REPAIR-NITPICK *)",
    ]
    variant_lines = lines[:hole_line] + injected_lines + lines[hole_line:]
    variant = "\n".join(variant_lines) + ("\n" if full_text.endswith("\n") else "")
    
    try:
        thy = build_theory(variant.splitlines(), add_print_state=True, end_with=None)
        resps = _run_theory_with_timeout(isabelle, session, thy, timeout_s=max(3, timeout_s + 2))
        return "\n".join(getattr(r, "response_body", b"").decode(errors="replace") 
                        if isinstance(getattr(r, "response_body", None), bytes)
                        else str(getattr(r, "response_body", ""))
                        for r in resps or [])
    except Exception:
        return ""

def _nitpick_state_hints_from_text(text: str) -> Dict[str, List[str]]:
    """Extract practical hints from Nitpick output."""
    t = text or ""
    t_lc = t.lower()

    # Only trust hints if Nitpick indicates a (potential) counterexample/model
    found_model = any(
        key in t_lc
        for key in [
            "nitpick found a counterexample",
            "nitpick found a potential counterexample",
            "model found",  # appears in some Nitpick prints
        ]
    )
    if not found_model:
        return {"bindings": [], "def_hints": []}

    # Variable bindings like x = ..., S = {...}, etc.
    bindings = re.findall(r"\b([a-z][A-Za-z0-9_']*)\s*=\s*([^,\n][^,\n]*)", t)
    bind_list = [f"{v} = {val.strip()}" for v, val in bindings]

    # *_def occurrences suggest unfolding opportunities
    defs = list(dict.fromkeys(re.findall(r"\b([A-Za-z_][A-Za-z0-9_']*)_def\b", t)))
    def_hints: List[str] = []
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

_REPAIR_SYSTEM = """You repair ONLY the local Isar snippet around a failing hole.
Do NOT regenerate the whole proof. Return a JSON array (≤3) of patch operations.

ALLOWED OPS:
1) {"insert_before_hole": "<ONE LINE>"}
2) {"replace_in_snippet": {"find": "<EXACT LINE>", "replace": "<NEW LINE>"}}
3) {"insert_have_block": {"label":"H", "statement":"<FORMULA>", "after_line_matching":"<LINE>", "body_hint":"<ONE LINE>"}}

Global rules:
- Edit only inside the provided SNIPPET; keep surrounding text verbatim.
- Do NOT repeat or reinsert any line that already appears in SNIPPET or RECENT_STEPS.
- Use ONLY identifiers that appear in STATE_BEFORE_HOLE or SNIPPET. Do NOT invent new fact names.
- In `using`, refer to named facts only; do NOT paste raw propositions or backticks/cartouches.
- Each suggested op must reflect a distinct strategy family (automation vs rule/intro/elim vs cases/induct vs small have..qed).
- Output MUST be valid JSON (no code fences, no comments)."""

_REPAIR_USER = """WHAT FAILED (concise):
{why}

GOAL: {goal}

STATE_BEFORE_HOLE: {state_block}

ISABELLE_ERRORS: {errors}

COUNTEREXAMPLE_HINTS: {ce_hints}

FACTS_CANDIDATES: {facts_list}

NEAREST_HEADER: {nearest_header}

RECENT_STEPS: {recent_steps}

BANLIST / PRIOR_FAILURES (avoid exact or near-duplicate lines):
{prior_failures}

SNIPPET:
<<<SNIPPET
{block_snippet}
SNIPPET

Output JSON array of patch ops:"""

def propose_local_repairs(*, goal: str, state_block: str, errors: List[str], 
                         ce_hints: Dict[str, List[str]], block_snippet: str,
                         nearest_header: str, recent_steps: List[str], facts: List[str],
                         model: Optional[str], timeout_s: int,
                         prior_failures: str = "(none)", why: str = "Previous attempt failed; propose a new approach.") -> str:
    """Propose local repair operations."""
    ce_list = (ce_hints.get("bindings", []) + ce_hints.get("def_hints", []))
    
    prompt = _REPAIR_SYSTEM + "\n\n" + _REPAIR_USER.format(
        goal=goal,
        state_block=(state_block or "").strip(),
        errors="\n".join(f"- {e}" for e in errors) or "(none)",
        ce_hints="\n".join(ce_list) or "(none)",
        block_snippet=block_snippet.rstrip(),
        nearest_header=nearest_header.strip(),
        recent_steps="\n".join(recent_steps) or "(none)",
        facts_list=", ".join(facts) or "(none)",
        prior_failures=prior_failures,
        why=why,
    )
    
    try:
        return _generate_simple(prompt, model=model, timeout_s=timeout_s)
    except Exception:
        return "[]"

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
    """Single-shot local repair (no iteration)."""
    start = time.monotonic()
    left = lambda: max(0.0, repair_budget_s - (time.monotonic() - start))
    
    # (scoring removed) Get context once
    state0 = _print_state_before_hole(isabelle, session, full_text, hole_span, trace=False)
    _,  errs0 = _quick_state_and_errors(isabelle, session, full_text)
    
    hole_line, _, lines = _hole_line_bounds(full_text, hole_span)
    s, e = _snippet_window(lines, hole_line, radius=12)
    snippet = "\n".join(lines[s:e])
    
    header = _nearest_header(lines, hole_line)
    rsteps = _recent_steps(lines, hole_line)
    
    # Always provide facts; try Nitpick (returns empty hints if no model found).
    facts = _facts_from_state(state0, limit=8)
    ce = _counterexample_hints(isabelle, session, full_text, hole_span)
    if ce.get("def_hints"):
        facts = list(dict.fromkeys(ce["def_hints"] + facts))[:12]
    
    # Single-pass local repair with a small banlist to avoid trivial repeats
    mem = _RepairMemory()
    # Seed banlist with tactic lines already present near the hole and recent steps
    seed_lines: List[str] = []
    for ln in (snippet.splitlines() + rsteps):
        if _APPLY_OR_BY.match(ln or ""):
            seed_lines.append(ln)
    for L in seed_lines:
        k = _canon_line(L)
        if k:
            mem.ban.add(k)

    remaining = left()
    HEADROOM_S = 3
    propose_timeout = int(max(2, min(45, (remaining - HEADROOM_S) * 0.6)))
    # Normalize Isabelle errors to plain strings for the prompt
    err_texts: List[str] = []
    for ee in (errs0 or []):
        t = str(ee.get("text", "") if isinstance(ee, dict) else ee).strip()
        if t:
            err_texts.append(t)
    err_texts = err_texts[:8]
    why_msg = "Previous attempt failed; propose a corrected, different strategy." if err_texts else \
              "Previous attempt did not close the subgoal; propose NEW strategies."
    prior_failures_txt = "\n".join(f"- {b}" for b in sorted(mem.ban)) or "(none)"

    raw = "[]"
    if propose_timeout >= 3:
        try:
            raw = propose_local_repairs(
                goal=goal_text, state_block=state0, errors=err_texts,
                ce_hints=ce, block_snippet=snippet, nearest_header=header,
                recent_steps=rsteps, facts=facts, model=model, timeout_s=propose_timeout,
                prior_failures=prior_failures_txt, why=why_msg
            )
        except Exception:
            raw = "[]"

    # Parse + filter against banlist
    ops = _filter_ops_against_banlist(raw, mem.ban)
    if not ops:
        # One-shot heuristic fallback if LLM gave nothing useful
        ops = _heuristic_fallback_ops(goal_text, state0, header, facts)

    tried = 0
    for kind, payload in ops:
        if left() <= 0 or tried >= max_ops_to_try:
            break
        tried += 1
        if kind == "insert_before_hole":
            cand = _apply_insert_before_hole(full_text, hole_span, payload.line)
            decisive = payload.line
        elif kind == "replace_in_snippet":
            cand = _apply_replace_in_snippet(full_text, hole_span, payload.find, payload.replace)
            decisive = payload.replace
        elif kind == "insert_have_block":
            cand = _apply_insert_have_block(full_text, hole_span, payload.label,
                                            payload.statement, payload.after_line_matching, payload.body_hint)
            decisive = payload.body_hint or f'have "{payload.statement}"'
        else:
            continue
        k = _canon_line(decisive)
        if k:
            mem.ban.add(k)
        if cand != full_text:
            return cand, False, f"beam:{kind or 'local'}(partial)"

    return full_text, False, "repairs-did-not-help"

# =========================
# Block repair for larger regions
# =========================

_BLOCK_SYSTEM = """You propose a replacement for the provided Isabelle/Isar BLOCK.
Preserve the surrounding text; return only the new BLOCK text (no JSON, no comments).

Global rules:
- Edit only inside this BLOCK. Keep lemma headers and surrounding text unchanged.
- Do NOT invent new fact names; only use identifiers that appear in LOCAL_CONTEXT or in this BLOCK.
- Do NOT paste raw propositions into 'using'; refer only to named facts or previously introduced labels.
- Each suggestion must be a distinct strategy family (automation vs rule/intro/elim vs cases/induct vs small have..qed).
"""

_BLOCK_USER = """WHAT FAILED (concise):
{why}

GOAL: {goal}

ISABELLE_ERRORS: {errors}

COUNTEREXAMPLE_HINTS: {ce_hints}

LOCAL_CONTEXT: {state_block}

ORIGINAL BLOCK:
<<<BLOCK
{block_text}
BLOCK

BANLIST / PRIOR_FAILURES (avoid exact/near-duplicate decisive lines):
{prior_failures}

Return ONLY the new BLOCK text.
"""

def _propose_block_repair(*, goal: str, errors: List[str], ce_hints: Dict[str, List[str]],
                         state_block: str, block_text: str, model: Optional[str], timeout_s: int,
                         prior_failures: str = "(none)", why: str = "Previous attempt failed; propose a different block-level change.") -> str:
    """Propose block-level repair."""
    ce = ce_hints.get("bindings", []) + ce_hints.get("def_hints", [])
    
    prompt = _BLOCK_SYSTEM + "\n\n" + _BLOCK_USER.format(
        goal=goal,
        errors="\n".join(f"- {e}" for e in errors) or "(none)",
        ce_hints="\n".join(ce) or "(none)",
        state_block=(state_block or "").strip(),
        block_text=block_text.rstrip(),
        prior_failures=prior_failures,
        why=why,        
    )
    
    try:
        out = _generate_simple(prompt, model=model, timeout_s=timeout_s)
        return _sanitize_llm_block(out)
    except Exception:
        return ""

# =========================
# Decisive-line extraction for BLOCK proposals
# =========================

_APPLY_OR_BY = re.compile(r"^\s*(apply|by)\b")
_PROOF_HDR   = re.compile(r"^\s*proof\b")

def _decisive_lines_from_block_text(block_text: str) -> List[str]:
    """
    Heuristic yet theory-agnostic: pick the first 'apply/by' or 'proof' line(s)
    as the decisive step to ban on subsequent rounds.
    """
    lines = (block_text or "").splitlines()
    out: List[str] = []
    for ln in lines:
        if _APPLY_OR_BY_DECISIVE.match(ln) or _PROOF_HDR.match(ln):
            out.append(ln)
            break
    # also catch a single 'show ?case by ...' if present
    for ln in lines:
        if re.search(r"\bshow\b", ln) and " by " in ln:
            out.append(ln)
            break
    return out[:2]

def _decisive_key_for_block(block_text: str) -> str:
    xs = _decisive_lines_from_block_text(block_text)
    if not xs:
        return ""
    return _canon_line(" ".join(xs))

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

def _enclosing_have_show_block(lines: List[str], hole_line: int) -> Tuple[int, int]:
    """
    Smallest micro-block headed by have/show/obtain at the current proof depth.
    Start = nearest head above the hole; End = next structural head at the same depth.
    """
    head_re = re.compile(r"(?m)^\s*(have|show|obtain)\b")
    stop_re = re.compile(r"(?m)^\s*(?:have|show|obtain|thus|hence|then|also|moreover|ultimately|finally|case\b|next\b|qed\b|proof\b)\b")
    proof_re = re.compile(r"(?m)^\s*proof\b")
    qed_re   = re.compile(r"(?m)^\s*qed\b")

    # Guard empty file and clamp starting index
    if not lines:
        return (-1, -1)
    i = hole_line
    if i < 0:
        i = 0
    if i >= len(lines):
        i = len(lines) - 1

    # Find the nearest head above
    while i >= 0 and not head_re.match(lines[i]):
        # Stop if we just crossed an enclosing boundary
        if re.match(r"(?m)^\s*(?:case\b|next\b|qed\b)\b", lines[i]):
            break
        i -= 1
    if i < 0 or not head_re.match(lines[i]):
        return (-1, -1)

    # Determine current proof depth at i
    depth = 0
    for k in range(0, i + 1):
        if proof_re.match(lines[k]): depth += 1
        elif qed_re.match(lines[k]): depth = max(0, depth - 1)
    base = depth

    # Scan forward to the next stop at the same depth
    j = i + 1
    while j < len(lines):
        if proof_re.match(lines[j]): depth += 1
        elif qed_re.match(lines[j]): depth = max(0, depth - 1)
        if depth == base and stop_re.match(lines[j]):
            break
        j += 1
    return (i, j)

def _enclosing_whole_proof(lines: List[str]) -> Tuple[int, int]:
    """
    Find the outermost top-level proof..qed block that encloses the hole’s proof.
    Very simple scan: last 'proof' before the last 'qed'.
    """
    proof_re = re.compile(r"(?m)^\s*proof\b")
    qed_re   = re.compile(r"(?m)^\s*qed\b")
    # find last 'qed'
    last_qed = -1
    for i, line in enumerate(lines):
        if qed_re.match(line):
            last_qed = i
    if last_qed < 0:
        return (-1, -1)
    # find the nearest 'proof' before that qed
    start = -1
    for i in range(last_qed, -1, -1):
        if proof_re.match(lines[i]):
            start = i
            break
    if start < 0:
        return (-1, -1)
    return (start, last_qed + 1)

# -------------------------
# Wrapper-stripping helpers
# -------------------------

_WRAPPED_THEOREM_HEAD = re.compile(
    r"""(?mx)
    \A
    (?:
        [ \t]* (?:\(\*.*?\*\)|\<comment\>.*?\<\/comment\>) [ \t]* \n   # skip ML/Isabelle comments
      | [ \t]* \n                                                      # skip blank lines
    )*
    [ \t]* (?:lemma|theorem|corollary)\b
    """
)
_CASE_LINE_RE         = re.compile(r"^\s*case\b")
_NEXT_OR_QED_RE       = re.compile(r"^\s*(?:next|qed)\b")
_PROOF_RE             = re.compile(r"^\s*proof\b")
_QED_RE               = re.compile(r"^\s*qed\b")

def _strip_wrapper_to_case_block(proposed: str, original_case_block: str) -> str:
    """
    If 'proposed' starts with lemma/theorem/corollary, extract only the target case-block.
    We try to match the case name from the original case header; otherwise fall back to the
    first 'case' block.
    """
    if not isinstance(proposed, str) or not _WRAPPED_THEOREM_HEAD.match(proposed):
        return proposed

    # Try to identify the target case name from the original block
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
    # Prefer matching the same case name
    for i, L in enumerate(lines):
        if not _CASE_LINE_RE.match(L):
            continue
        if case_name is None:
            start = i
            break
        if re.match(rf"^\s*case\s*\({re.escape(case_name)}\b", L) or \
           re.match(rf"^\s*case\s+{re.escape(case_name)}\b", L):
            start = i
            break

    # If we couldn't match by name, take the first case in the proposed block
    if start is None:
        for i, L in enumerate(lines):
            if _CASE_LINE_RE.match(L):
                start = i
                break
    if start is None:
        # As a last resort, keep the original proposed text
        return proposed

    # The case-block ends at the next 'next' or 'qed' (or EOF)
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if _NEXT_OR_QED_RE.match(lines[j]):
            end = j
            break
    return "\n".join(lines[start:end]).rstrip()

def _strip_wrapper_to_have_show(proposed: str, original_block: str) -> str:
    """
    If the LLM returns a wrapped lemma, keep only the target have/show/obtain block.
    Prefer matching the same head kind (have|show|obtain) from the original when present.
    """
    if not isinstance(proposed, str) or not _WRAPPED_THEOREM_HEAD.match(proposed):
        return proposed

    # Identify the head word from the original (have/show/obtain)
    m = re.search(r"(?m)^\s*(have|show|obtain)\b", original_block or "")
    prefer = m.group(1) if m else None

    lines = proposed.splitlines()
    head_idx = None
    for i, L in enumerate(lines):
        if prefer:
            if re.match(rf"^\s*{prefer}\b", L):
                head_idx = i
                break
        else:
            if re.match(r"^\s*(have|show|obtain)\b", L):
                head_idx = i
                break
    if head_idx is None:
        return proposed

    stop_re = re.compile(r"(?m)^\s*(?:have|show|obtain|thus|hence|then|also|moreover|ultimately|finally|case\b|next\b|qed\b|proof\b)\b")
    end = len(lines)
    for j in range(head_idx + 1, len(lines)):
        if stop_re.match(lines[j]):
            end = j
            break
    return "\n".join(lines[head_idx:end]).rstrip()

def _strip_wrapper_to_subproof(proposed: str) -> str:
    """
    If 'proposed' starts with lemma/theorem/corollary, extract only the inner
    'proof ... qed' region (balanced).
    """
    if not isinstance(proposed, str) or not _WRAPPED_THEOREM_HEAD.match(proposed):
        return proposed

    lines = proposed.splitlines()
    # Find first 'proof'
    start = None
    for i, L in enumerate(lines):
        if _PROOF_RE.match(L):
            start = i
            break
    if start is None:
        return proposed

    # Balance nested proof..qed
    depth, j = 1, start + 1
    while j < len(lines) and depth > 0:
        if _PROOF_RE.match(lines[j]):
            depth += 1
        elif _QED_RE.match(lines[j]):
            depth -= 1
        j += 1
    end = j if depth == 0 else len(lines)
    return "\n".join(lines[start:end]).rstrip()

# Helpers for safe 'sorry' insertion in tactic scripts
_HEAD_CMD_RE = re.compile(r"^\s*(have|show|obtain|then\s+show|thus|hence)\b")

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
    n = len(block_lines)
    while e < n and _is_tactic_line(block_lines[e]):
        e += 1
    return s, e

def _replace_failing_tactics_with_sorry(
    block_text: str,
    *,
    full_text_lines: List[str],
    start_line: int,   # 1-based in the assembled Isabelle doc
    end_line: int,     # exclusive
    isabelle,
    session: str,
    trace: bool = False,
) -> str:
    """
    Systematic version: compile the whole document, read Isabelle's error positions,
    and replace only those lines *inside [start_line, end_line)* that:
      (1) are tactic lines, and
      (2) are the earliest failing line (top-down) at that iteration.
    Repeat until no errors remain within the block or no tactic lines left to replace.
    """
    block_lines = block_text.splitlines()
    if not block_lines:
        # Nothing to do; avoid any indexing when the block is empty.
        if trace:
            print("[repair] Block is empty; skipping 'sorry' replacement.")
        return block_text

    def build_doc(with_block_lines: List[str]) -> str:
        # 1-based half-open [start_line, end_line) → 0-based [s0, e0)
        s0 = max(0, start_line - 1)
        # Clamp e0 so we never slice beyond the full document
        e0 = max(s0, min(end_line - 1, len(full_text_lines)))
        return "\n".join(full_text_lines[:s0] + with_block_lines + full_text_lines[e0:])

    # Iterate: each pass replaces the earliest failing tactic line (if any).
    while True:
        doc = build_doc(block_lines)
        # Get errors (with line numbers, if available)
        _state, errs = _quick_state_and_errors(isabelle, session, doc)
        err_lines = sorted(_extract_error_lines(errs))
        # Restrict to errors inside the block span
        err_in_block = sorted(set(l for l in err_lines if start_line <= l < end_line))

        # Also compute global "ok" (theory finished without errors), to catch cases
        # where Isabelle reports errors *without* precise line locations.
        thy = build_theory(doc.splitlines(), add_print_state=False, end_with=None)
        ok, _ = finished_ok(_run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_VERIFY_TIMEOUT_S))

        if not err_in_block:
            if trace:
                if ok:
                    print("[repair] Block has no Isabelle errors; no 'sorry' needed.")
                else:
                    print("[repair] Theory failed but without line-localized errors inside block; leaving block unchanged.")
            break

        failing_abs = err_in_block[0]             # earliest failing line in the block (1-based)
        failing_idx = failing_abs - start_line    # 0-based index into block_lines

        # If the exact line isn't a tactic, try to anchor to the *nearest preceding* tactic line;
        # if none, try the next following tactic line.
        cand = None
        if 0 <= failing_idx < len(block_lines) and _is_tactic_line(block_lines[failing_idx]):
            cand = failing_idx
        else:
            # search up (clamp start to last valid index)
            start_up = min(failing_idx, len(block_lines) - 1)
            for i in range(start_up, -1, -1):
                if _is_tactic_line(block_lines[i]):
                    cand = i
                    break
            # if none up, search down (clamp lower bound to [0, len))
            if cand is None:
                start_down = max(0, failing_idx + 1)
                for i in range(start_down, len(block_lines)):
                    if _is_tactic_line(block_lines[i]):
                        cand = i
                        break

        if cand is None:
            if trace:
                print(f"[repair] Earliest error at doc line {failing_abs}, but no tactic lines "
                      f"found in the block to replace. Leaving block as-is.")
            break

        indent = block_lines[cand][: len(block_lines[cand]) - len(block_lines[cand].lstrip())]
        line_stripped = block_lines[cand].lstrip()
        if trace:
            print(f"[repair] Marking failing tactic at doc line {start_line + cand} "
                  f"→ 'sorry' (context-aware).")
        # If this is an 'apply' step, prefer to wrap the enclosing head into a tiny proof.
        if line_stripped.startswith("apply"):
            head_idx = _find_enclosing_head(block_lines, cand)
            if head_idx is not None:
                head_indent = block_lines[head_idx][: len(block_lines[head_idx]) - len(block_lines[head_idx].lstrip())]
                seq_s, seq_e = _apply_sequence_bounds(block_lines, cand)
                replacement = [f"{head_indent}proof -", f"{head_indent}  sorry", f"{head_indent}qed"]
                block_lines[seq_s:seq_e] = replacement
            else:
                if trace:
                    print("[repair] 'apply' step without an enclosing have/show; skipping risky 'sorry' insertion.")
                break
        else:
            # 'by ...' or other single-step tactic → safe to replace with 'sorry'
            block_lines[cand] = f"{indent}sorry"

    return "\n".join(block_lines)

# =========================
# CEGIS: Multi-stage repair with iteration
# =========================

def try_cegis_repairs(*, full_text: str, hole_span: Tuple[int, int], goal_text: str,
                     model: Optional[str], isabelle, session: str,
                     repair_budget_s: float = 15.0, max_ops_to_try: int = 3,
                     beam_k: int = 1, allow_whole_fallback: bool = False,
                     trace: bool = False,
                     resume_stage: int = 0) -> Tuple[str, bool, str]:
    """Try CEGIS repairs with multiple stages."""
    t0 = time.monotonic()
    left = lambda: max(0.0, repair_budget_s - (time.monotonic() - t0))
    
    # (scoring removed)
    current_text = full_text
    # compute the state at the hole once; reuse until text actually changes
    state0 = _print_state_before_hole(isabelle, session, current_text, hole_span, trace=trace)
    _log_state_block("repair", state0, trace=trace)
    
    # Stage 0: Local repair — one tactic only, exactly one attempt; if not successful, proceed to later stages.
    if resume_stage <= 0 and left() > 5.0:
        eff_k = 1 if left() < 15.0 else max(1, beam_k)
        if trace:
            print("[repair] Trying local proof step repair…")
        patched, ok, tag = try_local_repairs(
            full_text=current_text, hole_span=hole_span, goal_text=goal_text,
            model=model, isabelle=isabelle, session=session,
            repair_budget_s=min(12.0, max(8.0, left() * 0.4)),
            max_ops_to_try=max_ops_to_try, beam_k=eff_k, trace=trace,
        )
        if ok and patched != current_text:
            if trace:
                print(f"[cegis] local repair accepted")
            return patched, True, f"stage=0 local:{tag}"
        elif patched != current_text:
            # carry forward partial change (no misleading logs)
            current_text = patched
            state0 = _print_state_before_hole(isabelle, session, current_text, hole_span, trace=trace)
    
    # Stage 1: have/show/obtain micro-block rewrite (smallest structural unit)
    hole_line, _, lines = _hole_line_bounds(current_text, hole_span)
    # Retarget to earliest failure (error line or first 'sorry') if different from the hole
    anchor_line, anchor_reason = _earliest_failure_anchor(isabelle, session, current_text, default_line_0=hole_line)
    focus_line = _clamp_line_index(lines, anchor_line)
    if trace and anchor_line != hole_line:
        shown_anchor = "EOF" if (anchor_line >= len(lines) and len(lines) > 0) else f"{anchor_line + 1}"
        shown_focus  = "EOF" if (focus_line == len(lines) - 1 and anchor_line >= len(lines)) else f"{focus_line + 1}"
        print(f"[repair] Retargeting from hole line {hole_line + 1} to earliest-failure line {shown_anchor} ({anchor_reason}); focus={shown_focus}.")
    hs_s, hs_e = _enclosing_have_show_block(lines, focus_line)
    if resume_stage <= 1 and hs_s >= 0 and left() > 5.0:
        state_block = state0
        _, errs = _quick_state_and_errors(isabelle, session, current_text)
        err_texts = _normalize_error_texts(errs)
        if trace and not state_block.strip() and errs:
            _first = errs[0].get("text", str(errs[0])) if isinstance(errs[0], dict) else str(errs[0])
            print(f"[repair] Isabelle errors (first): {_first[:200]}…")
        ceh = _counterexample_hints(isabelle, session, current_text, hole_span)
        if trace:
            print("[repair] prompt/errors ⤵")
            for t in err_texts: print("  -", t[:200])
            print("[repair] prompt/nitpick-hints ⤵")
            for b in ceh.get("bindings", []): print("  binding:", b[:200])
            for d in ceh.get("def_hints", []): print("  def_hint:", d[:200])
        block = "\n".join(lines[hs_s:hs_e])
        _log_input_block("repair", "have-show-block", block, trace=trace)
        # Iterative have/show repair: 2 rounds (time permitting) with BANLIST
        rounds = 2 if left() >= 8.0 else 1
        mem = _RepairMemory()
        seed_key = _decisive_key_for_block(block)
        if seed_key:
            mem.ban.add(seed_key)
        if trace:
            print("[repair] Trying have/show block repair…")
        for rr in range(rounds):
            if left() <= 3.0:
                break
            mem.rounds = rr + 1
            prior_failures_txt = "\n".join(f"- {b}" for b in sorted(mem.ban)) or "(none)"
            why = "Previous attempt did not discharge the goal; propose a different have/show-level change."
            hs_timeout = int(min(45, max(6, left() * (0.55 / max(1, rounds - rr)))))
            try:
                blk = _propose_block_repair(
                    goal=goal_text, errors=err_texts, ce_hints=ceh,
                    state_block=state_block, block_text=block,
                    model=model, timeout_s=hs_timeout,
                    prior_failures=prior_failures_txt, why=why
                )
            except Exception:
                blk = ""
            if not _is_effective_block(blk):
                continue
            before = blk
            blk = _strip_wrapper_to_have_show(blk, block)
            key = _decisive_key_for_block(blk)
            if key and key in mem.ban:
                if trace: print("[repair] filtered have/show proposal by banlist; next round…")
                continue
            if trace and blk.strip() != (before or "").strip():
                print("[repair] normalized full-lemma → have/show")
            if blk.strip() == block.strip():
                if key: mem.ban.add(key)
                continue
            blk_with_sorry = _replace_failing_tactics_with_sorry(
                blk,
                full_text_lines=lines,
                start_line=hs_s + 1,
                end_line=hs_e + 1,
                isabelle=isabelle,
                session=session,
                trace=trace,
            )
            _log_block("repair", "have-show-block", blk_with_sorry, trace=trace)
            patched_sorry = "\n".join(lines[:hs_s] + [blk_with_sorry] + lines[hs_e:])
            thy = build_theory(patched_sorry.splitlines(), add_print_state=False, end_with=None)
            ok, _ = finished_ok(_run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_VERIFY_TIMEOUT_S))
            if ok:
                return patched_sorry, True, f"stage=1 block:have-show(round={mem.rounds})"
            if key:
                mem.ban.add(key)
            # carry forward best effort before escalating
            current_text = patched_sorry
            lines = current_text.splitlines()
            state0 = _print_state_before_hole(isabelle, session, current_text, hole_span, trace=trace)
        if trace:
            print("[repair] have/show patch did not solve; escalating to sub-proof…")  
    
    # Stage 2a: Case-block rewrite (moved under sub-proof stage)
    cs, ce = _enclosing_case_block(lines, _clamp_line_index(lines, focus_line))
    if resume_stage <= 2 and cs >= 0 and left() > 5.0:
        # Reuse the same state until text changes
        state_block = state0
        _, errs = _quick_state_and_errors(isabelle, session, current_text)
        err_texts = _normalize_error_texts(errs)
        # Avoid re-printing the exact same state block that was already logged
        # right before calling try_local_repairs.
        # (This reduces trace noise; no functional change.)
        # _log_state_block("repair", state_block, trace=trace)
        if trace and not state_block.strip() and errs:
            _first = errs[0].get("text", str(errs[0])) if isinstance(errs[0], dict) else str(errs[0])
            print(f"[repair] Isabelle errors (first): {_first[:200]}…")        
        ceh = _counterexample_hints(isabelle, session, current_text, hole_span)
        if trace:
            print("[repair] prompt/errors ⤵")
            for t in err_texts: print("  -", t[:200])
            print("[repair] prompt/nitpick-hints ⤵")
            for b in ceh.get("bindings", []): print("  binding:", b[:200])
            for d in ceh.get("def_hints", []): print("  def_hint:", d[:200])        
        block = "\n".join(lines[cs:ce])
        # log the input/original case block
        _log_input_block("repair", "case-block", block, trace=trace)     
        # Iterative case-block repair: 2 rounds (time permitting) with BANLIST
        rounds = 2 if left() >= 10.0 else 1
        mem = _RepairMemory()
        seed_key = _decisive_key_for_block(block)
        if seed_key:
            mem.ban.add(seed_key)
        if trace:
            print(f"[repair] Trying case-block repair…")
        for rr in range(rounds):
            if left() <= 3.0:
                break
            mem.rounds = rr + 1
            prior_failures_txt = "\n".join(f"- {b}" for b in sorted(mem.ban)) or "(none)"
            why = "Previous case-block attempt did not solve the goal; try a different strategy."
            block_timeout = int(min(60, max(8, left() * (0.55 / max(1, rounds - rr)))))
            try:
                blk = _propose_block_repair(
                    goal=goal_text, errors=err_texts, ce_hints=ceh,
                    state_block=state_block, block_text=block,
                    model=model, timeout_s=block_timeout,
                    prior_failures=prior_failures_txt, why=why
                )
            except Exception:
                blk = ""
            if not _is_effective_block(blk):
                continue
            before = blk
            blk = _strip_wrapper_to_case_block(blk, block)
            key = _decisive_key_for_block(blk)
            if key and key in mem.ban:
                if trace: print("[repair] filtered case proposal by banlist; next round…")
                continue
            if trace and blk.strip() != (before or "").strip():
                print("[repair] normalized full-lemma → case-block")
            if blk.strip() == block.strip():
                if key: mem.ban.add(key)
                continue
            blk_with_sorry = _replace_failing_tactics_with_sorry(
                blk,
                full_text_lines=lines,
                start_line=cs + 1,
                end_line=ce + 1,
                isabelle=isabelle,
                session=session,
                trace=trace,
            )
            _log_block("repair", "case-block", blk_with_sorry, trace=trace)
            patched_sorry = "\n".join(lines[:cs] + [blk_with_sorry] + lines[ce:])
            if trace:
                print("[repair] in-doc slice after case-block patch:")
                print("\n".join(patched_sorry.splitlines()[max(0, cs-2):min(len(lines), ce+2)]))
            thy = build_theory(patched_sorry.splitlines(), add_print_state=False, end_with=None)
            ok, _ = finished_ok(_run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_VERIFY_TIMEOUT_S))
            if ok:
                return patched_sorry, True, f"stage=2 block:case(round={mem.rounds})"
            if key:
                mem.ban.add(key)
        # fallthrough to sub-proof
    
    # Stage 2b: Subproof rewrite
    ps, pe = _enclosing_subproof(lines, _clamp_line_index(lines, focus_line))
    
    if resume_stage <= 2 and ps >= 0 and left() > 3.0:
        state_block = state0
        _, errs = _quick_state_and_errors(isabelle, session, current_text)
        err_texts = _normalize_error_texts(errs)
        _log_state_block("repair", state_block, trace=trace)
        if trace and not state_block.strip() and errs:
            _first = errs[0].get("text", str(errs[0])) if isinstance(errs[0], dict) else str(errs[0])
            print(f"[repair] Isabelle errors (first): {_first[:200]}…")       
        ceh = _counterexample_hints(isabelle, session, current_text, hole_span)
        if trace:
            print("[repair] prompt/errors ⤵")
            for t in err_texts: print("  -", t[:200])
            print("[repair] prompt/nitpick-hints ⤵")
            for b in ceh.get("bindings", []): print("  binding:", b[:200])
            for d in ceh.get("def_hints", []): print("  def_hint:", d[:200])        
        block = "\n".join(lines[ps:pe])
        _log_input_block("repair", "subproof-block", block, trace=trace)
        # Iterative subproof repair: 2–3 rounds (time permitting) with BANLIST
        rounds = 3 if left() >= 18.0 else 2
        mem = _RepairMemory()
        seed_key = _decisive_key_for_block(block)
        if seed_key:
            mem.ban.add(seed_key)
        if trace:
            print(f"[repair] Trying subproof repair…")
        for rr in range(rounds):
            if left() <= 3.0:
                break
            mem.rounds = rr + 1
            prior_failures_txt = "\n".join(f"- {b}" for b in sorted(mem.ban)) or "(none)"
            why = "Previous sub-proof attempt failed; try a different structured sub-proof."
            subproof_timeout = int(min(45, max(10, left() * (0.55 / max(1, rounds - rr)))))
            try:
                blk = _propose_block_repair(
                    goal=goal_text, errors=err_texts, ce_hints=ceh,
                    state_block=state_block, block_text=block,
                    model=model, timeout_s=subproof_timeout,
                    prior_failures=prior_failures_txt, why=why
                )
            except Exception:
                blk = ""
            if not _is_effective_block(blk):
                continue
            before = blk
            blk = _strip_wrapper_to_subproof(blk)
            key = _decisive_key_for_block(blk)
            if key and key in mem.ban:
                if trace: print("[repair] filtered subproof proposal by banlist; next round…")
                continue
            if trace and blk.strip() != (before or "").strip():
                print("[repair] normalized full-lemma → subproof")
            if blk.strip() == block.strip():
                if key: mem.ban.add(key)
                continue
            blk_with_sorry = _replace_failing_tactics_with_sorry(
                blk,
                full_text_lines=lines,
                start_line=ps + 1,
                end_line=pe + 1,
                isabelle=isabelle,
                session=session,
                trace=trace,
            )
            _log_block("repair", "subproof-block", blk_with_sorry, trace=trace)
            patched_sorry = "\n".join(lines[:ps] + [blk_with_sorry] + lines[pe:])
            thy = build_theory(patched_sorry.splitlines(), add_print_state=False, end_with=None)
            ok, _ = finished_ok(_run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_VERIFY_TIMEOUT_S))
            if ok:
                return patched_sorry, True, f"stage=2 block:subproof(round={mem.rounds})"
            if key:
                mem.ban.add(key)
            
    # Stage 3: Whole-proof rewrite (fallback)
    if allow_whole_fallback and resume_stage <= 3 and left() > 3.0:
        ws, we = _enclosing_whole_proof(lines)
        if ws >= 0 and we > ws:
            state_block = state0
            _, errs = _quick_state_and_errors(isabelle, session, current_text)
            err_texts = _normalize_error_texts(errs)
            _log_state_block("repair", state_block, trace=trace)
            if trace and not state_block.strip() and errs:
                _first = errs[0].get("text", str(errs[0])) if isinstance(errs[0], dict) else str(errs[0])
                print(f"[repair] Isabelle errors (first): {_first[:200]}…")
            ceh = _counterexample_hints(isabelle, session, current_text, hole_span)
            if trace:
                print("[repair] prompt/errors ⤵")
                for t in err_texts: print("  -", t[:200])
                print("[repair] prompt/nitpick-hints ⤵")
                for b in ceh.get("bindings", []): print("  binding:", b[:200])
                for d in ceh.get("def_hints", []): print("  def_hint:", d[:200])            
            block = "\n".join(lines[ws:we])
            _log_input_block("repair", "whole-proof", block, trace=trace)
            # Iterative whole-proof repair: 3–5 rounds (time permitting) with BANLIST
            rounds = 5 if left() >= 30.0 else (4 if left() >= 20.0 else 3)
            mem = _RepairMemory()
            seed_key = _decisive_key_for_block(block)
            if seed_key:
                mem.ban.add(seed_key)
            if trace:
                print(f"[repair] Trying whole-proof repair…")
            for rr in range(rounds):
                if left() <= 3.0:
                    break
                mem.rounds = rr + 1
                prior_failures_txt = "\n".join(f"- {b}" for b in sorted(mem.ban)) or "(none)"
                why = "Previous whole-proof attempt failed; propose a substantially different proof body."
                whole_timeout = int(min(60, max(12, left() * (0.55 / max(1, rounds - rr)))))
                try:
                    blk = _propose_block_repair(
                        goal=goal_text, errors=err_texts, ce_hints=ceh,
                        state_block=state_block, block_text=block,
                        model=model, timeout_s=whole_timeout,
                        prior_failures=prior_failures_txt, why=why
                    )
                except Exception:
                    blk = ""
                if not _is_effective_block(blk):
                    continue
                key = _decisive_key_for_block(blk)
                if key and key in mem.ban:
                    if trace: print("[repair] filtered whole-proof proposal by banlist; next round…")
                    continue
                if blk.strip() == block.strip():
                    if key: mem.ban.add(key)
                    continue
                blk_with_sorry = _replace_failing_tactics_with_sorry(
                    blk,
                    full_text_lines=lines,
                    start_line=ws + 1,
                    end_line=we + 1,
                    isabelle=isabelle,
                    session=session,
                    trace=trace,
                )
                patched_sorry = "\n".join(lines[:ws] + [blk_with_sorry] + lines[we:])
                _log_block("repair", "whole-proof", patched_sorry, trace=trace)
                thy = build_theory(patched_sorry.splitlines(), add_print_state=False, end_with=None)
                ok, _ = finished_ok(_run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_VERIFY_TIMEOUT_S))
                if ok:
                    return patched_sorry, True, f"stage=3 block:whole(round={mem.rounds})"
                if key:
                    mem.ban.add(key)
                                
    # If we made partial changes, return them quietly (not success).
    if current_text != full_text:
        return current_text, False, f"stage={resume_stage} partial-progress"
    
    return full_text, False, "cegis-nohelp"