from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout
import requests

from prover.config import MODEL as DEFAULT_MODEL, OLLAMA_HOST, TIMEOUT_S as OLLAMA_TIMEOUT_S, OLLAMA_NUM_PREDICT, TEMP as OLLAMA_TEMP, TOP_P as OLLAMA_TOP_P
from prover.isabelle_api import build_theory, run_theory, last_print_state_block, finished_ok

# ========== Configuration ==========
_ISA_FAST_TIMEOUT_S = int(os.getenv("ISABELLE_FAST_TIMEOUT_S", "12"))
_ISA_VERIFY_TIMEOUT_S = int(os.getenv("ISABELLE_VERIFY_TIMEOUT_S", "30"))
_SESSION = requests.Session()

# ========== Regex Patterns ==========
_CTX_HEAD = re.compile(r"^\s*(?:using|from|with|then|ultimately|finally|also|moreover)\b")
_HAS_BODY = re.compile(r"^\s*(?:by\b|apply\b|proof\b|sorry\b|done\b)")
_APPLY_OR_BY = re.compile(r"^\s*(apply|by)\b")
_APPLY_OR_BY_DECISIVE = re.compile(r"^\s*(apply|by)\b")
_INLINE_BY_TAIL = re.compile(r"\s+by\s+.+$")
_HEADER_RE = re.compile(r"^\s*(proof\s*\(|proof\b|case\s+|then\s+show\b)")
_TACTIC_LINE = re.compile(r"^\s*(?:apply|by)\b|(?:\s)by\s+\S")
_STRUCTURAL_LINE = re.compile(r"^\s*(?:lemma|theorem|qed|next|proof|case|have|show|assume|fix|from|using|thus|hence|ultimately|finally|also|moreover|let|where)\b")
_HEAD_CMD_RE = re.compile(r"^\s*(have|show|obtain|then\s+show|thus|hence)\b")
_PROOF_RE = re.compile(r"^\s*proof\b")
_QED_RE = re.compile(r"^\s*qed\b")
_CASE_LINE_RE = re.compile(r"^\s*case\b")
_NEXT_OR_QED_RE = re.compile(r"^\s*(?:next|qed)\b")
_WRAPPED_THEOREM_HEAD = re.compile(r"(?mx)\A(?:[ \t]*(?:\(\*.*?\*\)|\<comment\>.*?\<\/comment\>)[ \t]*\n|[ \t]*\n)*[ \t]*(?:lemma|theorem|corollary)\b")

# ========== Utility Functions ==========
def _run_theory_with_timeout(isabelle, session: str, thy: List[str], *, timeout_s: Optional[int]) -> List:
    if not timeout_s or timeout_s <= 0:
        return run_theory(isabelle, session, thy)
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(run_theory, isabelle, session, thy)
        try:
            return fut.result(timeout=timeout_s)
        except _FuturesTimeout:
            if hasattr(isabelle, "interrupt"):
                try:
                    isabelle.interrupt()
                except Exception:
                    pass
            raise TimeoutError("isabelle_run_timeout")

def _log(prefix: str, label: str, content: str, trace: bool = True) -> None:
    if trace and content:
        print(f"[{prefix}] {label} (len={len(content)}):\n{content if content.strip() else '  (empty)'}")

def _sanitize_llm_block(text: str) -> str:
    if not text:
        return text
    patterns = [r"^\s*<<<BLOCK\s*$", r"^\s*BLOCK\s*$", r"^\s*<<<PROOF\s*$", r"^\s*PROOF\s*$", r"^\s*```\s*$", r"^\s*```isabelle\s*$", r"^\s*```isar\s*$"]
    compiled = [re.compile(p) for p in patterns]
    lines = [l for l in text.splitlines() if not any(p.match(l) for p in compiled)]
    return "\n".join(lines).strip()

def _is_effective_block(text: str) -> bool:
    return bool(_sanitize_llm_block(text or "").strip())

def _clamp_line_index(lines: List[str], idx: int) -> int:
    if not lines:
        return -1
    return max(0, min(idx, len(lines) - 1))

def _canon_line(s: str) -> str:
    if not s:
        return ""
    t = re.sub(r"\s+", " ", s.strip())
    t = re.sub(r";\s*$", "", t)
    return t.replace("`", "").replace("​", "").replace("​", "")

def _is_tactic_line(s: str) -> bool:
    return bool(_TACTIC_LINE.search(s)) and not bool(_STRUCTURAL_LINE.match(s))

# ========== Isabelle Interaction ==========
def _extract_print_state_from_responses(resps: List) -> str:
    standard = last_print_state_block(resps) or ""
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
                    elif "goal" in text and "subgoal" in text and not standard:
                        standard = text
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
    if llm_lines and standard:
        return standard + "\n" + "\n".join(llm_lines)
    return standard or "\n".join(llm_lines)

def _quick_state_and_errors(isabelle, session: str, full_text: str) -> Tuple[str, List[dict]]:
    try:
        thy = build_theory(full_text.splitlines(), add_print_state=True, end_with=None)
        resps = _run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_FAST_TIMEOUT_S)
        state = _extract_print_state_from_responses(resps)
        errors = []
        for r in resps or []:
            raw = getattr(r, "response_body", None)
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode(errors="replace")
            try:
                data = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                data = None
            if isinstance(data, dict) and data.get("nodes"):
                for node in data.get("nodes", []):
                    for msg in node.get("messages", []):
                        if str(msg.get("kind", "")).lower() != "error":
                            continue
                        txt = str(msg.get("message", "") or "")
                        pos = msg.get("pos") or {}
                        rng = msg.get("range") or {}
                        line = pos.get("line") or (rng.get("start") or {}).get("line") or msg.get("line")
                        if not isinstance(line, int):
                            m = re.search(r"\bline\s+(\d+)\b", txt)
                            if m:
                                try:
                                    line = int(m.group(1))
                                except Exception:
                                    pass
                        err_obj = {"text": txt}
                        if isinstance(line, int):
                            err_obj["line"] = line
                        errors.append(err_obj)
            elif isinstance(raw, str) and any(k in raw for k in ["*** Error:", "*** Outer syntax error", "*** Failed"]):
                txt = raw.strip().splitlines()[-1]
                err_obj = {"text": txt}
                m = re.search(r"\bline\s+(\d+)\b", raw)
                if m:
                    try:
                        err_obj["line"] = int(m.group(1))
                    except Exception:
                        pass
                errors.append(err_obj)
        return state, errors[:5]
    except Exception:
        return "", [{"text": "transport_or_build_error"}]

def _print_state_before_hole(isabelle, session: str, full_text: str, hole_span: Tuple[int, int], trace: bool = False) -> str:
    hole_line, indent, lines = _hole_line_bounds(full_text, hole_span)
    if not (0 <= hole_line < len(lines) and "sorry" in lines[hole_line]):
        nearest = _find_first_hole(lines)
        if nearest is not None:
            hole_line = nearest
            indent = len(lines[hole_line]) - len(lines[hole_line].lstrip(" "))
    pad = " " * max(2, indent)
    injected = [f"{pad}prefer 1", f"{pad}print_state", f"{pad}(* REPAIR-PRINT-STATE *)"]
    variant_lines = lines[:hole_line] + injected + lines[hole_line:]
    variant = "\n".join(variant_lines) + ("\n" if full_text.endswith("\n") else "")
    try:
        thy = build_theory(variant.splitlines(), add_print_state=False, end_with=None)
        resps = _run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_FAST_TIMEOUT_S)
        return _extract_print_state_from_responses(resps)
    except Exception:
        return ""

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

# ========== Context Analysis ==========
def _hole_line_bounds(full_text: str, hole_span: Tuple[int, int]) -> Tuple[int, int, List[str]]:
    lines = full_text.splitlines()
    hole_line = full_text[:hole_span[0]].count("\n")
    line_text = lines[hole_line] if 0 <= hole_line < len(lines) else ""
    indent = len(line_text) - len(line_text.lstrip(" "))
    return hole_line, indent, lines

def _find_first_hole(lines: List[str]) -> Optional[int]:
    for i, line in enumerate(lines):
        if "sorry" in line:
            return i
    return None

def _snippet_window(lines: List[str], hole_line: int, radius: int = 12) -> Tuple[int, int]:
    return max(0, hole_line - radius), min(len(lines), hole_line + radius + 1)

def _facts_from_state(state_block: str, limit: int = 16) -> List[str]:
    if not state_block:
        return []
    facts, seen = [], set()
    # Priority 1: propositions under "using this:"
    m = re.search(r"using this:\n((?:[ \t].*\n)+)", state_block)
    if m:
        for L in m.group(1).splitlines():
            s = L.strip()
            if s and s not in seen:
                seen.add(s)
                facts.append(s)
                if len(facts) >= limit:
                    return facts
    # Priority 2: quoted propositions
    for q in re.findall(r'(?m)^\s*"(.*?)"\s*$', state_block):
        s = q.strip()
        if s and s not in seen:
            seen.add(s)
            facts.append(s)
            if len(facts) >= limit:
                return facts
    # Priority 3: *_def names
    for d in re.findall(r"\b([A-Za-z_][A-Za-z0-9_']*)_def\b", state_block):
        if d and d not in seen:
            seen.add(d)
            facts.append(f"{d}_def")
            if len(facts) >= limit:
                break
    return facts

def _nearest_header(lines: List[str], hole_line: int) -> str:
    for i in range(hole_line, -1, -1):
        if _HEADER_RE.match(lines[i].strip()):
            return lines[i].strip()
    return ""

def _recent_steps(lines: List[str], hole_line: int, max_lines: int = 5) -> List[str]:
    steps = []
    for i in range(hole_line - 1, -1, -1):
        if _APPLY_OR_BY.match(lines[i]):
            steps.append(lines[i].strip())
            if len(steps) >= max_lines:
                break
        if lines[i].strip().startswith(("case ", "proof", "qed", "lemma ")):
            break
    return list(reversed(steps))

def _extract_error_lines(errs) -> list[int]:
    out = []
    for e in errs:
        ln = e.get("line") if isinstance(e, dict) else getattr(e, "line", None)
        if isinstance(ln, int):
            out.append(ln)
    return out

def _normalize_error_texts(errs) -> List[str]:
    return [str(e.get("text", "") if isinstance(e, dict) else e).strip() for e in (errs or []) if str(e.get("text", "") if isinstance(e, dict) else e).strip()][:8]

def _earliest_failure_anchor(isabelle, session: str, full_text: str, *, default_line_0: int) -> Tuple[int, str]:
    try:
        lines = full_text.splitlines()
        _, errs = _quick_state_and_errors(isabelle, session, full_text)
        err_lines = sorted(_extract_error_lines(errs))
        if err_lines:
            pos0 = err_lines[0] - 1
            if 0 <= pos0 < len(lines):
                return pos0, "error_line"
            for i, L in enumerate(lines):
                if "sorry" in L:
                    return i, "first_sorry_from_error"
            return _nearest_structural_head_before(lines, len(lines) - 1), "error_line_out_of_range"
        thy = build_theory(lines, add_print_state=False, end_with=None)
        ok, _ = finished_ok(_run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_VERIFY_TIMEOUT_S))
        if not ok:
            for i, L in enumerate(lines):
                if "sorry" in L:
                    return i, "first_sorry"
        return default_line_0, "default"
    except Exception:
        return default_line_0, "default"

def _nearest_structural_head_before(lines: List[str], idx: int) -> int:
    if not lines:
        return -1
    i = _clamp_line_index(lines, idx)
    head_re = re.compile(r"^\s*(?:have|show|obtain|case\b|proof\b)\b")
    for j in range(i, -1, -1):
        if head_re.match(lines[j]):
            return j
    return i

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
    ban: Set[str] = field(default_factory=set)
    rounds: int = 0

# ========== Repair Operations (Parsing & Application) ==========
def _parse_repair_ops(text: str) -> List[RepairOp]:
    def extract_json(t):
        try:
            return json.loads(t)
        except Exception:
            i, j = t.find("["), t.rfind("]")
            if i != -1 and j != -1 and j > i:
                try:
                    return json.loads(t[i:j+1])
                except Exception:
                    pass
        return None
    
    data = extract_json(text.strip())
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
            if all(isinstance(x, str) and x.strip() for x in (f, r)):
                ops.append(("replace_in_snippet", ReplaceInSnippet(f.strip(), r.strip())))
        elif k == "insert_have_block" and isinstance(v, dict):
            lab, stmt, after, hint = v.get("label", "H"), v.get("statement", ""), v.get("after_line_matching", "then show ?thesis"), v.get("body_hint", "apply simp")
            if all(isinstance(x, str) for x in (lab, stmt, after, hint)) and stmt.strip() and after.strip():
                ops.append(("insert_have_block", InsertHaveBlock(lab.strip(), stmt.strip(), after.strip(), hint.strip())))
    return ops[:3]

def _block_has_body_already(lines: List[str]) -> bool:
    idx = _find_first_hole(lines)
    if idx is None:
        return False
    k = idx - 1
    while k >= 0 and (lines[k].strip() == "" or _CTX_HEAD.match(lines[k])):
        k -= 1
    return k >= 0 and (_HAS_BODY.match(lines[k]) or _INLINE_BY_TAIL.search(lines[k]))

def _insert_before_hole_ctxaware(lines: List[str], payload_line: str) -> List[str]:
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
    hole_line, _, lines = _hole_line_bounds(full_text, hole_span)
    if _APPLY_OR_BY.match(line) or line.strip() in ("done", "."):
        if hole_line is not None:
            indent = lines[hole_line][:len(lines[hole_line]) - len(lines[hole_line].lstrip(" "))]
            lines[hole_line] = f"{indent}{line.strip()}"
            return "\n".join(lines) + ("\n" if full_text.endswith("\n") else "")
    if _block_has_body_already(lines):
        return full_text
    win_s, win_e = max(0, hole_line - 4), hole_line + 1
    if any(L.strip() == line.strip() for L in lines[win_s:win_e]):
        return full_text
    new_lines = _insert_before_hole_ctxaware(lines, line)
    return "\n".join(new_lines) + ("\n" if full_text.endswith("\n") else "") if new_lines != lines else full_text

def _apply_replace_in_snippet(full_text: str, hole_span: Tuple[int, int], find: str, replace: str) -> str:
    hole_line, _, lines = _hole_line_bounds(full_text, hole_span)
    s, e = _snippet_window(lines, hole_line)
    snippet = lines[s:e]
    try:
        idx = snippet.index(find)
        if snippet[idx].strip() == replace.strip():
            return full_text
        snippet[idx] = replace
    except ValueError:
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

def _apply_insert_have_block(full_text: str, hole_span: Tuple[int, int], label: str, statement: str, after_line_matching: str, body_hint: str) -> str:
    hole_line, indent, lines = _hole_line_bounds(full_text, hole_span)
    s, e = _snippet_window(lines, hole_line)
    anchor_idx = hole_line
    for i in range(s, e):
        if lines[i].strip() == after_line_matching.strip():
            anchor_idx = i
            break
    pad = " " * max(2, indent)
    block = [f'{pad}have {label}: "{statement}"', f"{pad}  proof -", f"{pad}    sorry", f"{pad}  qed"]
    new_lines = lines[:anchor_idx] + block + lines[anchor_idx:]
    return "\n".join(new_lines) + ("\n" if full_text.endswith("\n") else "")

# ========== Counterexample Hints ==========
def _run_nitpick_at_hole(isabelle, session: str, full_text: str, hole_span: Tuple[int, int], timeout_s: int = 3) -> str:
    hole_line, indent, lines = _hole_line_bounds(full_text, hole_span)
    if not (0 <= hole_line < len(lines) and "sorry" in lines[hole_line]):
        nearest = _find_first_hole(lines)
        if nearest is not None:
            hole_line = nearest
            indent = len(lines[hole_line]) - len(lines[hole_line].lstrip(" "))
    pad = " " * max(2, indent)
    injected = [f"{pad}prefer 1", f"{pad}nitpick [timeout={max(1, timeout_s)}]", f"{pad}(* REPAIR-NITPICK *)"]
    variant_lines = lines[:hole_line] + injected + lines[hole_line:]
    variant = "\n".join(variant_lines) + ("\n" if full_text.endswith("\n") else "")
    try:
        thy = build_theory(variant.splitlines(), add_print_state=True, end_with=None)
        resps = _run_theory_with_timeout(isabelle, session, thy, timeout_s=max(3, timeout_s + 2))
        return "\n".join(getattr(r, "response_body", b"").decode(errors="replace") if isinstance(getattr(r, "response_body", None), bytes) else str(getattr(r, "response_body", "")) for r in resps or [])
    except Exception:
        return ""

def _nitpick_state_hints_from_text(text: str) -> Dict[str, List[str]]:
    t = (text or "").lower()
    if not any(k in t for k in ["nitpick found a counterexample", "nitpick found a potential counterexample", "model found"]):
        return {"bindings": [], "def_hints": []}
    bindings = [f"{v} = {val.strip()}" for v, val in re.findall(r"\b([a-z][A-Za-z0-9_']*)\s*=\s*([^,\n][^,\n]*)", text)]
    defs = list(dict.fromkeys(re.findall(r"\b([A-Za-z_][A-Za-z0-9_']*)_def\b", text)))
    def_hints = [hint for d in defs for hint in [f"{d}_def", f"unfolding {d}_def", f"simp only: {d}_def"]]
    return {"bindings": bindings[:8], "def_hints": def_hints[:12]}

def _counterexample_hints(isabelle, session: str, full_text: str, hole_span: Tuple[int, int]) -> Dict[str, List[str]]:
    nit = _run_nitpick_at_hole(isabelle, session, full_text, hole_span, timeout_s=3)
    hints = _nitpick_state_hints_from_text(nit)
    if hints.get("bindings") or hints.get("def_hints"):
        return hints
    state_only, _ = _quick_state_and_errors(isabelle, session, full_text)
    return _nitpick_state_hints_from_text(state_only)

# ========== Prompt Templates ==========
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

_BLOCK_SYSTEM = """You propose a replacement for the provided Isabelle/Isar BLOCK.
Preserve the surrounding text; return only the new BLOCK text (no JSON, no comments).

Global rules:
- Edit only inside this BLOCK. Keep lemma headers and surrounding text unchanged.
- Do NOT invent new fact names; only use identifiers that appear in LOCAL_CONTEXT or in this BLOCK.
- Do NOT paste raw propositions into 'using'; refer only to named facts or previously introduced labels.
- Each suggestion must be a distinct strategy family (automation vs rule/intro/elim vs cases/induct vs small have..qed)."""

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

Return ONLY the new BLOCK text."""

# ========== Repair Proposal Functions ==========
def propose_local_repairs(*, goal: str, state_block: str, errors: List[str], ce_hints: Dict[str, List[str]], 
                         block_snippet: str, nearest_header: str, recent_steps: List[str], facts: List[str],
                         model: Optional[str], timeout_s: int, prior_failures: str = "(none)", 
                         why: str = "Previous attempt failed; propose a new approach.") -> str:
    ce_list = ce_hints.get("bindings", []) + ce_hints.get("def_hints", [])
    prompt = _REPAIR_SYSTEM + "\n\n" + _REPAIR_USER.format(
        goal=goal, state_block=(state_block or "").strip(),
        errors="\n".join(f"- {e}" for e in errors) or "(none)",
        ce_hints="\n".join(ce_list) or "(none)", block_snippet=block_snippet.rstrip(),
        nearest_header=nearest_header.strip(), recent_steps="\n".join(recent_steps) or "(none)",
        facts_list=", ".join(facts) or "(none)", prior_failures=prior_failures, why=why
    )
    try:
        return _generate_simple(prompt, model=model, timeout_s=timeout_s)
    except Exception:
        return "[]"

def _propose_block_repair(*, goal: str, errors: List[str], ce_hints: Dict[str, List[str]], state_block: str, 
                         block_text: str, model: Optional[str], timeout_s: int, prior_failures: str = "(none)", 
                         why: str = "Previous attempt failed; propose a different block-level change.") -> str:
    ce = ce_hints.get("bindings", []) + ce_hints.get("def_hints", [])
    prompt = _BLOCK_SYSTEM + "\n\n" + _BLOCK_USER.format(
        goal=goal, errors="\n".join(f"- {e}" for e in errors) or "(none)",
        ce_hints="\n".join(ce) or "(none)", state_block=(state_block or "").strip(),
        block_text=block_text.rstrip(), prior_failures=prior_failures, why=why
    )
    try:
        return _sanitize_llm_block(_generate_simple(prompt, model=model, timeout_s=timeout_s))
    except Exception:
        return ""

def _filter_ops_against_banlist(ops_json_text: str, ban: Set[str]) -> List[RepairOp]:
    ops = _parse_repair_ops(ops_json_text)
    keep = []
    for kind, payload in ops:
        if kind == "insert_before_hole":
            key = _canon_line(payload.line)
        elif kind == "replace_in_snippet":
            key = _canon_line(payload.replace)
        elif kind == "insert_have_block":
            key = _canon_line(payload.body_hint or f'have "{payload.statement}"')
        else:
            keep.append((kind, payload))
            continue
        if not key or key not in ban:
            keep.append((kind, payload))
    return keep

def _heuristic_fallback_ops(goal_text: str, state_block: str, header: str, facts: List[str]) -> List[RepairOp]:
    ops = []
    if "Let " in state_block or "Let_def" in facts:
        ops.append(("insert_before_hole", InsertBeforeHole("apply (unfolding Let_def)")))
    g = goal_text
    if ("map" in g and "@" in g) or ("map_append" in facts):
        ops.append(("insert_before_hole", InsertBeforeHole("apply (simp add: map_append)")))
    if ("length" in g and "@" in g) or ("length_append" in facts):
        ops.append(("insert_before_hole", InsertBeforeHole("apply (simp add: length_append)")))
    if "@" in g or "append_assoc" in facts:
        ops.append(("insert_before_hole", InsertBeforeHole("apply (simp add: append_assoc)")))
    if header.startswith("proof (induction") and "arbitrary:" not in header:
        for v in ("ys", "zs"):
            if v in g:
                new_header = header.rstrip(")") + f" arbitrary: {v})"
                ops.append(("replace_in_snippet", ReplaceInSnippet(header, new_header)))
                break
    return ops[:3]

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
    head_re = re.compile(r"(?m)^\s*(have|show|obtain)\b")
    stop_re = re.compile(r"(?m)^\s*(?:have|show|obtain|thus|hence|then|also|moreover|ultimately|finally|case\b|next\b|qed\b|proof\b)\b")
    while i >= 0 and not head_re.match(lines[i]):
        if re.match(r"(?m)^\s*(?:case\b|next\b|qed\b)\b", lines[i]):
            break
        i -= 1
    if i < 0 or not head_re.match(lines[i]):
        return (-1, -1)
    depth = sum(1 if _PROOF_RE.match(lines[k]) else (-1 if _QED_RE.match(lines[k]) else 0) for k in range(i + 1))
    base, j = depth, i + 1
    while j < len(lines):
        if _PROOF_RE.match(lines[j]):
            depth += 1
        elif _QED_RE.match(lines[j]):
            depth = max(0, depth - 1)
        if depth == base and stop_re.match(lines[j]):
            break
        j += 1
    return (i, j)

def _enclosing_whole_proof(lines: List[str]) -> Tuple[int, int]:
    last_qed = -1
    for i, line in enumerate(lines):
        if _QED_RE.match(line):
            last_qed = i
    if last_qed < 0:
        return (-1, -1)
    for i in range(last_qed, -1, -1):
        if _PROOF_RE.match(lines[i]):
            return (i, last_qed + 1)
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
    if not _WRAPPED_THEOREM_HEAD.match(proposed):
        return proposed
    m = re.search(r"(?m)^\s*(have|show|obtain)\b", original_block or "")
    prefer = m.group(1) if m else None
    lines = proposed.splitlines()
    head_idx = None
    for i, L in enumerate(lines):
        if (prefer and re.match(rf"^\s*{prefer}\b", L)) or (not prefer and re.match(r"^\s*(have|show|obtain)\b", L)):
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

# ========== Main Repair Functions ==========
def try_local_repairs(*, full_text: str, hole_span: Tuple[int, int], goal_text: str, model: Optional[str], 
                     isabelle, session: str, repair_budget_s: float = 12.0, max_ops_to_try: int = 2, 
                     beam_k: int = 1, trace: bool = False) -> Tuple[str, bool, str]:
    start = time.monotonic()
    left = lambda: max(0.0, repair_budget_s - (time.monotonic() - start))
    
    state0 = _print_state_before_hole(isabelle, session, full_text, hole_span, trace=False)
    _, errs0 = _quick_state_and_errors(isabelle, session, full_text)
    hole_line, _, lines = _hole_line_bounds(full_text, hole_span)
    s, e = _snippet_window(lines, hole_line, radius=12)
    snippet = "\n".join(lines[s:e])
    header = _nearest_header(lines, hole_line)
    rsteps = _recent_steps(lines, hole_line)
    facts = _facts_from_state(state0, limit=8)
    ce = _counterexample_hints(isabelle, session, full_text, hole_span)
    if ce.get("def_hints"):
        facts = list(dict.fromkeys(ce["def_hints"] + facts))[:12]
    
    mem = _RepairMemory()
    for ln in (snippet.splitlines() + rsteps):
        if _APPLY_OR_BY.match(ln or ""):
            k = _canon_line(ln)
            if k:
                mem.ban.add(k)
    
    remaining = left()
    propose_timeout = int(max(2, min(45, (remaining - 3) * 0.6)))
    err_texts = _normalize_error_texts(errs0)
    why_msg = "Previous attempt failed; propose a corrected, different strategy." if err_texts else "Previous attempt did not close the subgoal; propose NEW strategies."
    prior_failures_txt = "\n".join(f"- {b}" for b in sorted(mem.ban)) or "(none)"
    
    raw = "[]"
    if propose_timeout >= 3:
        try:
            raw = propose_local_repairs(goal=goal_text, state_block=state0, errors=err_texts, ce_hints=ce, 
                                       block_snippet=snippet, nearest_header=header, recent_steps=rsteps, 
                                       facts=facts, model=model, timeout_s=propose_timeout, 
                                       prior_failures=prior_failures_txt, why=why_msg)
        except Exception:
            pass
    
    ops = _filter_ops_against_banlist(raw, mem.ban)
    if not ops:
        ops = _heuristic_fallback_ops(goal_text, state0, header, facts)
    
    for kind, payload in ops[:max_ops_to_try]:
        if left() <= 0:
            break
        if kind == "insert_before_hole":
            cand, decisive = _apply_insert_before_hole(full_text, hole_span, payload.line), payload.line
        elif kind == "replace_in_snippet":
            cand, decisive = _apply_replace_in_snippet(full_text, hole_span, payload.find, payload.replace), payload.replace
        elif kind == "insert_have_block":
            cand = _apply_insert_have_block(full_text, hole_span, payload.label, payload.statement, payload.after_line_matching, payload.body_hint)
            decisive = payload.body_hint or f'have "{payload.statement}"'
        else:
            continue
        k = _canon_line(decisive)
        if k:
            mem.ban.add(k)
        if cand != full_text:
            return cand, False, f"beam:{kind or 'local'}(partial)"
    
    return full_text, False, "repairs-did-not-help"

def try_cegis_repairs(*, full_text: str, hole_span: Tuple[int, int], goal_text: str, model: Optional[str], 
                     isabelle, session: str, repair_budget_s: float = 15.0, max_ops_to_try: int = 3, 
                     beam_k: int = 1, allow_whole_fallback: bool = False, trace: bool = False, 
                     resume_stage: int = 0) -> Tuple[str, bool, str]:
    t0 = time.monotonic()
    left = lambda: max(0.0, repair_budget_s - (time.monotonic() - t0))
    current_text = full_text
    state0 = _print_state_before_hole(isabelle, session, current_text, hole_span, trace=trace)
    _log("repair", "State block", state0, trace=trace)
    
    # Stage 0: Local repair
    if resume_stage <= 0 and left() > 5.0:
        eff_k = 1 if left() < 15.0 else max(1, beam_k)
        if trace:
            print("[repair] Trying local proof step repair…")
        patched, ok, tag = try_local_repairs(full_text=current_text, hole_span=hole_span, goal_text=goal_text, 
                                            model=model, isabelle=isabelle, session=session, 
                                            repair_budget_s=min(12.0, max(8.0, left() * 0.4)), 
                                            max_ops_to_try=max_ops_to_try, beam_k=eff_k, trace=trace)
        if ok and patched != current_text:
            if trace:
                print(f"[cegis] local repair accepted")
            return patched, True, f"stage=0 local:{tag}"
        elif patched != current_text:
            current_text = patched
            state0 = _print_state_before_hole(isabelle, session, current_text, hole_span, trace=trace)
    
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
        current_text = _repair_block(current_text, lines, hs_s, hs_e, goal_text, state0, isabelle, session, 
                                     model, left, trace, "have-show", stage=1)
        if current_text != full_text:
            thy = build_theory(current_text.splitlines(), add_print_state=False, end_with=None)
            ok, _ = finished_ok(_run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_VERIFY_TIMEOUT_S))
            if ok:
                return current_text, True, "stage=1 block:have-show"
        lines = current_text.splitlines()
        state0 = _print_state_before_hole(isabelle, session, current_text, hole_span, trace=trace)
    
    # Stage 2a: Case-block
    cs, ce = _enclosing_case_block(lines, focus_line)
    if resume_stage <= 2 and cs >= 0 and left() > 5.0:
        if trace:
            print("[repair] Trying case-block repair…")
        current_text = _repair_block(current_text, lines, cs, ce, goal_text, state0, isabelle, session, 
                                     model, left, trace, "case", stage=2)
        if current_text != full_text:
            thy = build_theory(current_text.splitlines(), add_print_state=False, end_with=None)
            ok, _ = finished_ok(_run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_VERIFY_TIMEOUT_S))
            if ok:
                return current_text, True, "stage=2 block:case"
    
    # Stage 2b: Subproof
    ps, pe = _enclosing_subproof(lines, focus_line)
    if resume_stage <= 2 and ps >= 0 and left() > 3.0:
        if trace:
            print("[repair] Trying subproof repair…")
        current_text = _repair_block(current_text, lines, ps, pe, goal_text, state0, isabelle, session, 
                                     model, left, trace, "subproof", stage=2)
        if current_text != full_text:
            thy = build_theory(current_text.splitlines(), add_print_state=False, end_with=None)
            ok, _ = finished_ok(_run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_VERIFY_TIMEOUT_S))
            if ok:
                return current_text, True, "stage=2 block:subproof"
    
    # Stage 3: Whole-proof
    if allow_whole_fallback and resume_stage <= 3 and left() > 3.0:
        ws, we = _enclosing_whole_proof(lines)
        if ws >= 0 and we > ws:
            if trace:
                print("[repair] Trying whole-proof repair…")
            current_text = _repair_block(current_text, lines, ws, we, goal_text, state0, isabelle, session, 
                                         model, left, trace, "whole", stage=3)
            if current_text != full_text:
                thy = build_theory(current_text.splitlines(), add_print_state=False, end_with=None)
                ok, _ = finished_ok(_run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_VERIFY_TIMEOUT_S))
                if ok:
                    return current_text, True, "stage=3 block:whole"
    
    return current_text if current_text != full_text else full_text, False, f"stage={resume_stage} " + ("partial-progress" if current_text != full_text else "cegis-nohelp")

def _repair_block(current_text: str, lines: List[str], start: int, end: int, goal_text: str, 
                 state0: str, isabelle, session: str, model: Optional[str], left, trace: bool, 
                 block_type: str, stage: int) -> str:
    _, errs = _quick_state_and_errors(isabelle, session, current_text)
    err_texts = _normalize_error_texts(errs)
    ce = _counterexample_hints(isabelle, session, current_text, (0, 0))
    block = "\n".join(lines[start:end])
    _log("repair", f"{block_type}-block (input)", block, trace=trace)
    
    rounds = 3 if left() >= 18.0 else 2 if left() >= 10.0 else 1
    mem = _RepairMemory()
    seed_key = _canon_line(" ".join(_APPLY_OR_BY_DECISIVE.findall(block)[:2]))
    if seed_key:
        mem.ban.add(seed_key)
    
    for rr in range(rounds):
        if left() <= 3.0:
            break
        mem.rounds = rr + 1
        prior_failures = "\n".join(f"- {b}" for b in sorted(mem.ban)) or "(none)"
        why = f"Previous {block_type}-block attempt did not solve the goal; try a different strategy."
        timeout = int(min(60, max(8, left() * (0.55 / max(1, rounds - rr)))))
        
        try:
            blk = _propose_block_repair(goal=goal_text, errors=err_texts, ce_hints=ce, state_block=state0, 
                                       block_text=block, model=model, timeout_s=timeout, 
                                       prior_failures=prior_failures, why=why)
        except Exception:
            blk = ""
        
        if not _is_effective_block(blk):
            continue
        
        before = blk
        if block_type == "case":
            blk = _strip_wrapper_to_case_block(blk, block)
        elif block_type == "have-show":
            blk = _strip_wrapper_to_have_show(blk, block)
        elif block_type == "subproof":
            blk = _strip_wrapper_to_subproof(blk)
        
        key = _canon_line(" ".join(_APPLY_OR_BY_DECISIVE.findall(blk)[:2]))
        if key and key in mem.ban:
            if trace:
                print(f"[repair] filtered {block_type} proposal by banlist")
            continue
        
        if blk.strip() == block.strip():
            if key:
                mem.ban.add(key)
            continue
        
        blk_with_sorry = _replace_failing_tactics_with_sorry(blk, full_text_lines=lines, start_line=start + 1, 
                                                             end_line=end + 1, isabelle=isabelle, 
                                                             session=session, trace=trace)
        _log("repair", f"{block_type}-block (output)", blk_with_sorry, trace=trace)
        patched = "\n".join(lines[:start] + [blk_with_sorry] + lines[end:])
        
        thy = build_theory(patched.splitlines(), add_print_state=False, end_with=None)
        ok, _ = finished_ok(_run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_VERIFY_TIMEOUT_S))
        
        if ok:
            return patched
        if key:
            mem.ban.add(key)
        current_text = patched
        lines = current_text.splitlines()
    
    return current_text