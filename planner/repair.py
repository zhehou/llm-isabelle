from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
import time
import concurrent.futures
from typing import List, Dict, Tuple, Optional
import requests

from prover.config import (
    MODEL as DEFAULT_MODEL,
    OLLAMA_HOST,
    TIMEOUT_S as OLLAMA_TIMEOUT_S,
    OLLAMA_NUM_PREDICT,
    TEMP as OLLAMA_TEMP,
    TOP_P as OLLAMA_TOP_P,
)
from prover.isabelle_api import build_theory, run_theory, last_print_state_block
from prover.utils import parse_subgoals

def _sanitize_llm_block(text: str) -> str:
    """
    Remove *only* known fence lines that some models echo, preserving all
    Isabelle content verbatim. Also drops stray triple backtick fences.
    Never collapses non-fence content.
    """
    if not isinstance(text, str) or not text:
        return text
    
    # Known fence patterns - be very specific to avoid false positives
    fence_patterns = [
        r"^\s*<<<BLOCK\s*$",
        r"^\s*BLOCK\s*$", 
        r"^\s*<<<PROOF\s*$",
        r"^\s*PROOF\s*$",
        r"^\s*```\s*$",
        r"^\s*```isabelle\s*$",
        r"^\s*```isar\s*$"
    ]
    
    compiled_patterns = [re.compile(pattern) for pattern in fence_patterns]
    
    out_lines = []
    for line in text.splitlines():
        # Check if this line matches any fence pattern
        is_fence = any(pattern.match(line) for pattern in compiled_patterns)
        if not is_fence:
            out_lines.append(line)
    
    # Preserve inner newlines but trim outer blank lines
    return "\n".join(out_lines).strip("\n")

def _is_effective_block(text: str) -> bool:
    """
    True iff the block has any non-empty, non-fence line after sanitisation.
    """
    if not text:
        return False
    t = _sanitize_llm_block(text)
    return bool(t.strip())

# requests is optional; we only reference its exception types if present
try:
    import requests  # type: ignore
    _REQ_TIMEOUTS = (requests.Timeout, requests.ReadTimeout, requests.ConnectionError)
except Exception:  # pragma: no cover
    _REQ_TIMEOUTS = tuple()

_SESSION = requests.Session()

def _extract_print_state_from_responses(resps: List) -> str:
    """Extract print_state output from Isabelle responses."""
    # Collect the standard print_state block (for numbered goal etc.)
    standard_result = last_print_state_block(resps) or ""

    # Also collect our ML 'writeln' markers (authoritative subgoal + var classes)
    llm_lines: List[str] = []
    for resp in (resps or []):
        resp_type = getattr(resp, "response_type", None)
        if str(resp_type).upper() == "FINISHED":
            body = getattr(resp, "response_body", None)
            if isinstance(body, bytes):
                body = body.decode(errors="replace")
            data = None
            try:
                if isinstance(body, str) and body.strip().startswith("{"):
                    data = json.loads(body)
                elif isinstance(body, dict):
                    data = body
            except (json.JSONDecodeError, TypeError):
                data = None
            if not data:
                continue
            nodes = data.get("nodes", [])
            for node in nodes:
                for msg in node.get("messages", []) or []:
                    if msg.get("kind") != "writeln":
                        continue
                    text = msg.get("message", "") or ""
                    # Keep our markers; keep extra goal text if present
                    if text.startswith("[LLM_SUBGOAL]") or text.startswith("[LLM_VARS]"):
                        llm_lines.append(text)
                    elif ("goal" in text and "subgoal" in text and not standard_result):
                        standard_result = text

    if llm_lines and standard_result:
        return standard_result + "\n" + "\n".join(llm_lines)
    return standard_result or ("\n".join(llm_lines) if llm_lines else "")

# =========================
# Backends: Ollama / HF / Gemini
# =========================

def _ollama_generate_simple(
    prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    num_predict: Optional[int] = None,
    timeout_s: Optional[int] = None,
) -> str:
    url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
    payload = {
        "model": model or DEFAULT_MODEL,
        "prompt": prompt,
        "options": {
            "temperature": OLLAMA_TEMP if temperature is None else temperature,
            "top_p": OLLAMA_TOP_P if top_p is None else top_p,
            "num_predict": OLLAMA_NUM_PREDICT if num_predict is None else num_predict,
        },
        "stream": False,
    }
    # For repair operations, we need much longer timeouts than normal
    # The issue in your trace shows 15s timeout, but large models need more time
    eff_timeout = timeout_s or OLLAMA_TIMEOUT_S
    # Ensure minimum timeout for large models (30B needs substantial time)
    eff_read = max(30.0, float(eff_timeout))  # Minimum 30s for read operations
    eff_conn = 10.0  # Conservative connection timeout
    
    try:
        resp = _SESSION.post(url, json=payload, timeout=(eff_conn, eff_read))
        resp.raise_for_status()
        data = resp.json()
        result = (data.get("response") or "").strip()
        return _sanitize_llm_block(result)
    except requests.exceptions.ReadTimeout as e:
        print(f"DEBUG: Ollama read timeout after {eff_read}s - model may need more time")
        raise
    except requests.exceptions.ConnectTimeout as e:
        print(f"DEBUG: Ollama connection timeout after {eff_conn}s")
        raise
    except Exception as e:
        print(f"DEBUG: Ollama request failed: {e}")
        raise

def _hf_generate_simple(
    prompt: str,
    model_id: str,
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    timeout_s: Optional[int] = None,
) -> str:
    token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACE_API_TOKEN is not set")
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}"}
    payload: Dict[str, object] = {
        "inputs": prompt,
        "parameters": {
            "temperature": OLLAMA_TEMP if temperature is None else temperature,
            "top_p": OLLAMA_TOP_P if top_p is None else top_p,
            "max_new_tokens": OLLAMA_NUM_PREDICT if max_new_tokens is None else max_new_tokens,
            "return_full_text": False,
        },
        "options": {"wait_for_model": True},
    }
    resp = _SESSION.post(url, headers=headers, json=payload, timeout=timeout_s or OLLAMA_TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()
    result = ""
    if isinstance(data, list) and data:
        result = (data[0].get("generated_text") or "").strip()
    elif isinstance(data, dict):
        if "generated_text" in data:
            result = (data["generated_text"] or "").strip()
        else:
            choices = data.get("choices") or []
            if choices:
                t = choices[0].get("text") or choices[0].get("generated_text") or ""
                result = str(t).strip()
    else:
        result = str(data).strip()
    
    return _sanitize_llm_block(result)

@lru_cache(maxsize=1)
def _gemini_list_models_cached(api_key: str) -> List[str]:
    if not api_key:
        return []
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        resp = _SESSION.get(url, timeout=OLLAMA_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
        out = []
        for m in data.get("models", []):
            name = m.get("name", "")
            short = name.split("/")[-1] if name else ""
            if short:
                out.append(short)
        return out
    except Exception:
        return []

def _gemini_resolve_model_id(model_id: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return model_id
    models = _gemini_list_models_cached(api_key)
    if model_id in models:
        return model_id
    cands = [m for m in models if m.startswith(model_id)]
    if cands:
        stable = [m for m in cands if ("preview" not in m and "exp" not in m)]
        return (stable or cands)[0]
    return model_id

def _gemini_cli_available() -> bool:
    from shutil import which
    return which("gemini") is not None

def _gemini_cli_generate_simple(prompt: str, model_id: str, *, timeout_s: Optional[int] = None) -> str:
    import subprocess
    cmd = ["gemini", "-m", model_id]
    proc = subprocess.run(
        cmd, input=prompt, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=timeout_s or OLLAMA_TIMEOUT_S, env=os.environ.copy()
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gemini CLI failed ({proc.returncode}): {(proc.stderr or proc.stdout).strip()}")
    return _sanitize_llm_block(proc.stdout.strip())

def _gemini_rest_generate_simple(prompt: str, model_id: str, *, timeout_s: Optional[int] = None) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set (needed for Gemini REST)")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": OLLAMA_NUM_PREDICT}}
    resp = _SESSION.post(url, json=body, timeout=timeout_s or OLLAMA_TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()
    result = ""
    try:
        cands = data.get("candidates") or []
        if cands:
            parts = ((cands[0].get("content") or {}).get("parts")) or []
            if parts:
                result = (parts[0].get("text") or "").strip()
    except Exception:
        pass
    if not result:
        result = str(data).strip()
    return _sanitize_llm_block(result)

def _gemini_generate_simple(prompt: str, model_id: str, *, timeout_s: Optional[int] = None) -> str:
    resolved = _gemini_resolve_model_id(model_id)
    if _gemini_cli_available():
        try:
            return _gemini_cli_generate_simple(prompt, resolved, timeout_s=timeout_s)
        except Exception:
            pass
    try:
        return _gemini_rest_generate_simple(prompt, resolved, timeout_s=timeout_s)
    except Exception:
        fallback = "gemini-2.5-pro"
        if _gemini_cli_available():
            try:
                return _gemini_cli_generate_simple(prompt, fallback, timeout_s=timeout_s)
            except Exception:
                pass
        return _gemini_rest_generate_simple(prompt, fallback, timeout_s=timeout_s)

def _generate_simple(
    prompt: str,
    model: Optional[str] = None,
    *,
    timeout_s: Optional[int] = None,
) -> str:
    m = (model or DEFAULT_MODEL) or ""
    if m.startswith("hf:"):
        return _hf_generate_simple(prompt, model_id=m[len("hf:"):], timeout_s=timeout_s)
    if m.startswith("gemini:"):
        return _gemini_generate_simple(prompt, model_id=m[len("gemini:"):], timeout_s=timeout_s)
    if m.startswith("ollama:"):
        m = m[len("ollama:"):]
    return _ollama_generate_simple(prompt, model=m, timeout_s=timeout_s)

# =========================
# Repair op schema
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

_CTX_HEAD = re.compile(r"^\s*(?:using|from|with|then|ultimately|finally|also|moreover)\b")
_HAS_BODY = re.compile(r"^\s*(?:by\b|apply\b|proof\b|sorry\b|done\b)")

def _extract_json_array(text: str) -> Optional[list]:
    """Try to salvage the first JSON array from free-form text."""
    try:
        return json.loads(text)
    except Exception:
        pass
    i = text.find("["); j = text.rfind("]")
    if i != -1 and j != -1 and j > i:
        frag = text[i:j+1]
        try:
            return json.loads(frag)
        except Exception:
            return None
    return None

def _parse_repair_ops(text: str) -> List[RepairOp]:
    data = _extract_json_array(text.strip())
    if not isinstance(data, list):
        return []
    out: List[RepairOp] = []
    for item in data:
        if not isinstance(item, dict) or len(item) != 1:
            continue
        k = next(iter(item.keys()))
        v = item[k]
        if k == "insert_before_hole" and isinstance(v, str) and v.strip():
            out.append(("insert_before_hole", InsertBeforeHole(v.strip())))
        elif k == "replace_in_snippet" and isinstance(v, dict):
            f = v.get("find", ""); r = v.get("replace", "")
            if isinstance(f, str) and isinstance(r, str) and f.strip() and r.strip():
                out.append(("replace_in_snippet", ReplaceInSnippet(f.strip(), r.strip())))
        elif k == "insert_have_block" and isinstance(v, dict):
            lab = v.get("label", "H"); stmt = v.get("statement", "")
            after = v.get("after_line_matching", "then show ?thesis"); hint = v.get("body_hint", "apply simp")
            if all(isinstance(x, str) for x in (lab, stmt, after, hint)) and stmt.strip() and after.strip():
                out.append(("insert_have_block", InsertHaveBlock(lab.strip(), stmt.strip(), after.strip(), hint.strip())))
    return out[:3]

# =========================
# Op application helpers (context-aware and non-destructive)
# =========================

def _find_first_hole(lines: List[str]) -> Optional[int]:
    for i, L in enumerate(lines):
        if "sorry" in L:
            return i
    return None

def _insert_before_hole_ctxaware(lines: List[str], payload_line: str) -> List[str]:
    """Insert *after* any contiguous context lines that immediately precede the hole."""
    idx = _find_first_hole(lines)
    if idx is None:
        return lines
    # walk upward while previous lines are context heads or blank
    k = idx - 1
    while k >= 0 and (lines[k].strip() == "" or _CTX_HEAD.match(lines[k])):
        k -= 1
    insert_at = k + 1  # right above the hole, after the context block
    indent = lines[idx][: len(lines[idx]) - len(lines[idx].lstrip(" "))]
    new = lines[:insert_at] + [f"{indent}{payload_line}"] + lines[insert_at:]
    return new

def _block_has_body_already(lines: List[str]) -> bool:
    """Check if the local block right above the hole already has a body tactic."""
    idx = _find_first_hole(lines)
    if idx is None:
        return False
    k = idx - 1
    while k >= 0 and (lines[k].strip() == "" or _CTX_HEAD.match(lines[k])):
        k -= 1
    # next non-blank, non-context line above, if any, is where a body would be
    if k >= 0 and (_HAS_BODY.match(lines[k]) or _INLINE_BY_TAIL.search(lines[k])):
        return True
    return False

# =========================
# Local context mining
# =========================

_ID = r"[A-Za-z_][A-Za-z0-9_']*"
_DEF_RE = re.compile(rf"\b({_ID})_def\b")
_FACT_RE = re.compile(rf"\b({_ID})\b")
_APPLY_OR_BY = re.compile(r"^\s*(apply\b|by\b)")
_INLINE_BY_TAIL = re.compile(r"\s+by\s+.+$")  # mirror of driver's notion of inline 'by'
_HEADER_RE = re.compile(r"^\s*(proof\s*\(|proof\b|case\s+|then\s+show\b)")

def _hole_line_bounds(full_text: str, hole_span: Tuple[int, int]) -> Tuple[int, int, List[str]]:
    lines = full_text.splitlines()
    upto = full_text[:hole_span[0]]
    hole_line = upto.count("\n")
    line_text = lines[hole_line] if 0 <= hole_line < len(lines) else ""
    indent = len(line_text) - len(line_text.lstrip(" "))
    return hole_line, indent, lines

def _snippet_window(lines: List[str], hole_line: int, radius: int = 12) -> Tuple[int, int]:
    s = max(0, hole_line - radius)
    e = min(len(lines), hole_line + radius + 1)
    return s, e

def _facts_from_state(state_block: str, limit: int = 16) -> List[str]:
    defs = _DEF_RE.findall(state_block or "")
    tokens = _FACT_RE.findall(state_block or "")
    out: List[str] = []
    seen = set()
    for x in defs + tokens:
        if not x or x in seen:
            continue
        seen.add(x); out.append(x)
        if len(out) >= limit:
            break
    return out

def _nearest_header(lines: List[str], hole_line: int) -> str:
    for i in range(hole_line, -1, -1):
        L = lines[i].strip()
        if _HEADER_RE.match(L):
            return L
    return ""

def _recent_steps(lines: List[str], hole_line: int, max_lines: int = 5) -> List[str]:
    out: List[str] = []
    for i in range(hole_line - 1, -1, -1):
        L = lines[i]
        if _APPLY_OR_BY.match(L):
            out.append(L.strip())
            if len(out) >= max_lines:
                break
        if L.strip().startswith(("case ", "proof", "qed", "lemma ")):
            break
    return list(reversed(out))

# =========================
# Patch application helpers
# =========================

def _apply_insert_before_hole(full_text: str, hole_span: Tuple[int, int], line: str) -> str:
    """
    Context-aware insertion that:
      - respects preceding context heads (using/from/then/also/…),
      - NO-OPs if the local block already has a body tactic (by/apply/proof/sorry/done),
      - de-duplicates if the same tactic is already within a small window above the hole.
    """
    hole_line, _indent, lines = _hole_line_bounds(full_text, hole_span)
    # If payload is a *finalizer* ('by …' / 'done' / '.'), replace the hole instead of inserting above it.
    if _APPLY_OR_BY.match(line) or line.strip() in ("done", "."):
        if hole_line is not None:
            indent = lines[hole_line][: len(lines[hole_line]) - len(lines[hole_line].lstrip(" "))]
            lines[hole_line] = f"{indent}{line.strip()}"
            return "\n".join(lines) + ("\n" if full_text.endswith("\n") else "")    
    # 1) If there is already a body tactic for the local block, don't touch it.
    if _block_has_body_already(lines):
        return full_text
    # 2) De-duplicate if the same line is already present right above the hole.
    win_s = max(0, hole_line - 4)
    win_e = hole_line + 1
    if any(L.strip() == line.strip() for L in lines[win_s:win_e]):
        return full_text
    # If the immediate previous non-context line already has an inline '... by ...', do nothing.
    k = hole_line - 1
    while k >= 0 and (lines[k].strip() == "" or _CTX_HEAD.match(lines[k])):
        k -= 1
    if k >= 0 and _INLINE_BY_TAIL.search(lines[k] or ""):
        return full_text    
    # 3) Insert after contiguous context-heads (then/using/also/…),
    #    keeping indentation aligned with the hole line.
    new_lines = _insert_before_hole_ctxaware(lines, line)
    if new_lines == lines:
        return full_text
    return "\n".join(new_lines) + ("\n" if full_text.endswith("\n") else "")

def _apply_replace_in_snippet(full_text: str, hole_span: Tuple[int, int], find: str, replace: str) -> str:
    hole_line, _indent, lines = _hole_line_bounds(full_text, hole_span)
    s, e = _snippet_window(lines, hole_line)
    snippet = lines[s:e]
    try:
        idx = snippet.index(find)
        # Avoid no-op churn; if identical after strip, skip.
        if snippet[idx].strip() == replace.strip():
            return full_text
        snippet[idx] = replace
    except ValueError:
        stripped = [L.strip() for L in snippet]
        try:
            idx = stripped.index(find.strip())
            orig = snippet[idx]
            leading = orig[: len(orig) - len(orig.lstrip(" "))]
            # Avoid no-op churn here too.
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
    anchor_idx = None
    for i in range(s, e):
        if lines[i].strip() == after_line_matching.strip():
            anchor_idx = i; break
    if anchor_idx is None:
        anchor_idx = hole_line
    pad = " " * max(2, indent)
    block = [
        f'{pad}have {label}: "{statement}"',
        f"{pad}  sorry",
    ]
    new_lines = lines[:anchor_idx] + block + lines[anchor_idx:]
    return "\n".join(new_lines) + ("\n" if full_text.endswith("\n") else "")

# =========================
# LLM prompt for local repair
# =========================

_REPAIR_SYSTEM = """You patch an Isabelle/Isar proof LOCALLY. Do not regenerate the whole proof.
Return ONLY a JSON array of at most 3 patch operations from the allowed schema.

ALLOWED OPS (choose from):
1) {"insert_before_hole": "<ONE LINE, e.g., apply (simp add: …)>"}
2) {"replace_in_snippet": {"find": "<EXACT LINE>", "replace": "<NEW LINE>"}}
3) {"insert_have_block": {"label":"H", "statement":"<FORMULA>", "after_line_matching":"<LINE IN SNIPPET>", "body_hint":"<ONE LINE>"}}

Rules:
- Edit only inside the provided SNIPPET. Never change the lemma line or text outside the snippet.
- Prefer single-line hints like `apply (simp add: …)`, `apply (unfolding …_def)`, `apply (simp only: …_def)`, `using …`.
- If variables escape induction, adjust header via replace_in_snippet (e.g., add `arbitrary: ys`).
- If necessary, insert a small `have` block just above the failing `show`. Keep its body a single `sorry`.
- The output MUST be valid JSON with no comments.
"""

_REPAIR_USER = """GOAL:
{goal}

STATE_BEFORE_HOLE:
{state_block}

ISABELLE_ERRORS (recent):
{errors}

COUNTEREXAMPLE/MODEL HINTS:
{ce_hints}

FACTS/DEFS CANDIDATES:
{facts_list}

NEAREST_HEADER:
{nearest_header}

RECENT_STEPS:
{recent_steps}

SNIPPET (edit only inside this region):
<<<SNIPPET
{block_snippet}
SNIPPET

Output a JSON array of at most 3 patch ops according to the allowed schema.
"""

# Add debugging information to repair functions
def propose_local_repairs(
    *,
    goal: str,
    state_block: str,
    errors: List[str],
    ce_hints: Dict[str, List[str]],
    block_snippet: str,
    nearest_header: str,
    recent_steps: List[str],
    facts: List[str],
    model: Optional[str],
    timeout_s: int,
) -> List[RepairOp]:
    # Flatten CE hints for readability
    ce_list: List[str] = []
    if ce_hints.get("bindings"):
        ce_list += ce_hints["bindings"]
    if ce_hints.get("def_hints"):
        ce_list += ce_hints["def_hints"]

    prompt = _REPAIR_SYSTEM + "\n\n" + _REPAIR_USER.format(
        goal=goal,
        state_block=(state_block or "").strip(),
        errors="\n".join(f"- {e}" for e in (errors or [])) or "(none)",
        ce_hints="\n".join(ce_list) or "(none)",
        block_snippet=block_snippet.rstrip(),
        nearest_header=nearest_header.strip(),
        recent_steps="\n".join(recent_steps),
        facts_list=", ".join(facts),
    )
    try:
        raw = _generate_simple(prompt, model=model, timeout_s=timeout_s)
        print(f"DEBUG: Raw LLM output: {repr(raw)}")  # Add debug output
        ops = _parse_repair_ops(raw)
        print(f"DEBUG: Parsed ops: {ops}")  # Add debug output
        return ops
    except requests.exceptions.ReadTimeout:
        print("DEBUG: LLM timeout occurred")  # Add debug output
        # Treat as "no proposals" instead of crashing the whole planner.
        return []
    except Exception as e:
        print(f"DEBUG: LLM generation failed: {e}")  # Add debug output
        return []

# =========================
# Heuristic fallback ops (when LLM gives nothing usable)
# =========================

def _heuristic_fallback_ops(goal_text: str, state_block: str, header: str, facts: List[str]) -> List[RepairOp]:
    ops: List[RepairOp] = []

    # Unfolding (Let)
    if "Let " in state_block or "Let_def" in facts:
        ops.append(("insert_before_hole", InsertBeforeHole("apply (unfolding Let_def)")))

    # Classic list lemmas
    g = goal_text
    if ("map" in g and "@" in g) or ("map_append" in facts):
        ops.append(("insert_before_hole", InsertBeforeHole("apply (simp add: map_append)")))
    if ("length" in g and "@" in g) or ("length_append" in facts):
        ops.append(("insert_before_hole", InsertBeforeHole("apply (simp add: length_append)")))
    if ("@" in g) or ("append_assoc" in facts):
        ops.append(("insert_before_hole", InsertBeforeHole("apply (simp add: append_assoc)")))

    # Light induction header tweak: add arbitrary: ys if likely
    if header.startswith("proof (induction") and ("arbitrary:" not in header):
        for v in ("ys", "zs"):
            if v in g:
                ops.append(("replace_in_snippet",
                            ReplaceInSnippet(find=header, replace=header.rstrip(")") + f" arbitrary: {v})")))
                break

    return ops[:3]

# =========================
# Isabelle error harvesting (for CEGIS feedback)
# =========================
def _collect_isabelle_errors(resps) -> List[str]:
    errs: List[str] = []
    for r in (resps or []):
        body = getattr(r, "response_body", None)
        if isinstance(body, bytes):
            body = body.decode(errors="replace")
        if not isinstance(body, str):
            continue
        try:
            data = json.loads(body)
            if isinstance(data, dict) and data.get("kind") == "error" and "message" in data:
                errs.append(str(data["message"]))
        except json.JSONDecodeError:
            if "*** Error:" in body or "*** Outer syntax error" in body or "*** Failed" in body:
                errs.append(body.strip().splitlines()[-1])
    # Stable de-dup, keep short
    out: List[str] = []
    seen = set()
    for e in errs:
        if e not in seen:
            seen.add(e); out.append(e)
    return out[:3]

def _quick_state_and_errors(isabelle, session: str, full_text: str) -> Tuple[str, List[str]]:
    try:
        thy = build_theory(full_text.splitlines(), add_print_state=True, end_with="sorry")
        resps = run_theory(isabelle, session, thy)
        # Use the enhanced extraction instead of the broken last_print_state_block
        state_block = _extract_print_state_from_responses(resps) or ""
        return state_block, _collect_isabelle_errors(resps)
    except Exception:
        return "", ["transport_or_build_error"]

# =========================
# Nitpick counterexample harvesting (CE hints)
# =========================
_CE_BINDING_RE = re.compile(r"\b([a-z][A-Za-z0-9_']*)\s*=\s*([^,\s][^,\n]*)")
_DEF_HINT_RE   = re.compile(r"\b([A-Za-z_][A-Za-z0-9_']*)_def\b")

def _nitpick_state_hints_from_text(text: str) -> Dict[str, List[str]]:
    """
    Extracts lightweight CE hints from Nitpick/print_state output:
      - variable bindings x = …
      - *_def occurrences → suggest unfolding/simp only
    """
    binds: List[str] = []
    for m in _CE_BINDING_RE.finditer(text or ""):
        v, val = m.group(1), m.group(2)
        if v and val:
            binds.append(f"{v} = {val.strip()}")
    defs = list(dict.fromkeys(_DEF_HINT_RE.findall(text or "")))
    ce_facts: List[str] = []
    if defs:
        ce_facts += [f"{d}_def" for d in defs]
        ce_facts += [f"unfolding {d}_def" for d in defs]
        ce_facts += [f"simp only: {d}_def" for d in defs]
    return {"bindings": binds[:8], "def_hints": ce_facts[:12]}

def _run_nitpick_at_hole(isabelle, session: str, full_text: str, hole_span: Tuple[int,int], timeout_s: int = 3) -> str:
    """
    Build a minimal variant that invokes Nitpick at the hole subgoal.
    We insert a tiny command sequence right at the hole:
      prefer 1; nitpick [timeout=…]; (* marker *); sorry
    Return the concatenated Isabelle responses as text (for regex mining).
    """
    s, e = hole_span
    prefix = full_text[:s]
    suffix = full_text[e:]
    injected = (
        "  prefer 1\n"
        f"  nitpick [timeout={max(1,int(timeout_s))}]\n"
        "  (* CEGIS-NITPICK-MARK *)\n"
        "  sorry\n"
    )
    variant = prefix + injected + suffix
    try:
        thy = build_theory(variant.splitlines(), add_print_state=True, end_with="sorry")
        resps = run_theory(isabelle, session, thy)
        chunks: List[str] = []
        for r in (resps or []):
            body = getattr(r, "response_body", None)
            if isinstance(body, bytes):
                chunks.append(body.decode(errors="replace"))
            elif isinstance(body, str):
                chunks.append(body)
        return "\n".join(chunks)
    except Exception:
        return ""

def _counterexample_hints(isabelle, session: str, full_text: str, hole_span: Tuple[int,int]) -> Dict[str, List[str]]:
    """
    Prefer Nitpick-derived hints; fall back to print_state-derived ones.
    """
    nit = _run_nitpick_at_hole(isabelle, session, full_text, hole_span, timeout_s=3)
    hints = _nitpick_state_hints_from_text(nit)
    if hints.get("bindings") or hints.get("def_hints"):
        return hints
    # Fallback 1: Quickcheck (new)
    qc_text = _run_quickcheck_at_hole(isabelle, session, full_text, hole_span, timeout_s=3)
    qc_hints = _quickcheck_state_hints_from_text(qc_text)
    if qc_hints.get("bindings") or qc_hints.get("def_hints"):
        return qc_hints
    # Fallback 2: reuse last print_state if both Nitpick and Quickcheck gave nothing usable
    state_only, _errs = _quick_state_and_errors(isabelle, session, full_text)
    return _nitpick_state_hints_from_text(state_only)

# -------- NEW: Quickcheck CE fallback ----------
_QC_BINDING_RE = re.compile(r"\b([a-z][A-Za-z0-9_']*)\s*=\s*([^,\s][^,\n]*)")
_QC_FOUND_RE   = re.compile(r"Quickcheck found a counterexample", re.IGNORECASE)

def _run_quickcheck_at_hole(isabelle, session: str, full_text: str, hole_span: Tuple[int,int], timeout_s: int = 3) -> str:
    """
    Insert a lightweight quickcheck at the current subgoal.
      prefer 1; quickcheck[timeout=…]; (* mark *); sorry
    """
    s, e = hole_span
    prefix = full_text[:s]
    suffix = full_text[e:]
    injected = (
        "  prefer 1\n"
        f"  quickcheck[timeout={max(1,int(timeout_s))}]\n"
        "  (* CEGIS-QUICKCHECK-MARK *)\n"
        "  sorry\n"
    )
    variant = prefix + injected + suffix
    try:
        thy = build_theory(variant.splitlines(), add_print_state=True, end_with="sorry")
        resps = run_theory(isabelle, session, thy)
        chunks: List[str] = []
        for r in (resps or []):
            body = getattr(r, "response_body", None)
            if isinstance(body, bytes):
                chunks.append(body.decode(errors="replace"))
            elif isinstance(body, str):
                chunks.append(body)
        return "\n".join(chunks)
    except Exception:
        return ""

def _quickcheck_state_hints_from_text(text: str) -> Dict[str, List[str]]:
    """
    Very lightweight parsing of Quickcheck output into bindings and *_def nudges.
    If QC didn't run or found nothing, returns empty hints.
    """
    if not text or not _QC_FOUND_RE.search(text):
        return {"bindings": [], "def_hints": []}
    binds: List[str] = []
    for m in _QC_BINDING_RE.finditer(text):
        v, val = m.group(1), m.group(2)
        if v and val:
            binds.append(f"{v} = {val.strip()}")
    # Reuse _DEF_HINT_RE to nudge unfolding when QC prints defs in context.
    defs = list(dict.fromkeys(_DEF_HINT_RE.findall(text)))
    ce_facts: List[str] = []
    if defs:
        ce_facts += [f"{d}_def" for d in defs]
        ce_facts += [f"unfolding {d}_def" for d in defs]
        ce_facts += [f"simp only: {d}_def" for d in defs]
    return {"bindings": binds[:8], "def_hints": ce_facts[:12]}

# =========================
# Region growth helpers (case block / subproof)
# =========================
_CASE_RE  = re.compile(r"(?m)^\s*case\b")
_NEXT_RE  = re.compile(r"(?m)^\s*next\b")
_PROOF_RE = re.compile(r"(?m)^\s*proof\b")
_QED_RE   = re.compile(r"(?m)^\s*qed\b")

def _enclosing_case_block(lines: List[str], hole_line: int) -> Tuple[int, int]:
    i = hole_line
    while i >= 0 and not _CASE_RE.match(lines[i]): i -= 1
    if i < 0: return (-1, -1)
    j = hole_line
    while j < len(lines) and not (_NEXT_RE.match(lines[j]) or _QED_RE.match(lines[j])): j += 1
    return (i, j)

def _enclosing_subproof(lines: List[str], hole_line: int) -> Tuple[int, int]:
    i = hole_line
    while i >= 0 and not _PROOF_RE.match(lines[i]): i -= 1
    if i < 0: return (-1, -1)
    depth, j = 1, i + 1
    while j < len(lines) and depth > 0:
        if _PROOF_RE.match(lines[j]): depth += 1
        elif _QED_RE.match(lines[j]): depth -= 1
        j += 1
    return (i, j if j > i else -1)

# =========================
# Block repair (LLM rewrites just the selected region)
# =========================
_BLOCK_SYSTEM = """You repair an Isabelle/Isar PROOF BLOCK.
Edit ONLY the given BLOCK (between <<<BLOCK … BLOCK).
Preserve the rest of the proof verbatim.
The repaired block MUST compile in context, be as small as possible, and keep the intended strategy if reasonable.

Each proof step must be logically sound given the available facts.
If a step cannot be proven with the suggested tactic, use 'sorry' as a placeholder.

Output ONLY the new block (no fences, no comments)."""

_BLOCK_USER = """GOAL:
{goal}

ISABELLE_ERRORS (recent):
{errors}

COUNTEREXAMPLE/MODEL HINTS:
{ce_hints}

LOCAL CONTEXT (print_state excerpt):
{state_block}

ORIGINAL BLOCK (edit-only region):
<<<BLOCK
{block_text}
BLOCK
"""

def _propose_block_repair(
    *, goal: str, errors: List[str], ce_hints: Dict[str,List[str]],
    state_block: str, block_text: str, model: Optional[str], timeout_s: int
) -> str:
    ce = []
    if ce_hints.get("bindings"): ce += ce_hints["bindings"]
    if ce_hints.get("def_hints"): ce += ce_hints["def_hints"]
    prompt = _BLOCK_SYSTEM + "\n\n" + _BLOCK_USER.format(
        goal=goal,
        errors="\n".join(f"- {e}" for e in (errors or [])) or "(none)",
        ce_hints="\n".join(ce) or "(none)",
        state_block=(state_block or "").strip(),
        block_text=block_text.rstrip(),
    )    
    print(f"DEBUG: Block repair prompt length: {len(prompt)} chars")
    print(f"DEBUG: Block repair timeout: {timeout_s}s")
    
    try:
        out = _generate_simple(prompt, model=model, timeout_s=timeout_s)
        print(f"DEBUG: Block repair raw output: {repr(out)}")
        sanitized = _sanitize_llm_block(out)
        print(f"DEBUG: Block repair sanitized: {repr(sanitized)}")
        return sanitized
    except Exception as e:
        print(f"DEBUG: Block repair generation failed: {e}")
        return ""

# =========================
# Quick check & orchestrators (local + CEGIS wrapper)
# =========================

def _quick_state_subgoals(isabelle, session: str, text: str) -> int:
    """
    Run Isabelle on the CURRENT text (no offset surgery) and parse the last print_state.
    Returns a large sentinel on failure.
    """
    try:
        thy = build_theory(text.splitlines(), add_print_state=True, end_with="sorry")
        resps = run_theory(isabelle, session, thy)
        block = _extract_print_state_from_responses(resps) or ""  # Use enhanced extraction
        n = parse_subgoals(block)
        return int(n) if isinstance(n, int) else 9999
    except Exception:
        return 9999

def _quick_state_text(isabelle, session: str, full_text: str) -> str:
    try:
        thy = build_theory(full_text.splitlines(), add_print_state=True, end_with="sorry")
        resps = run_theory(isabelle, session, thy)
        return last_print_state_block(resps) or ""
    except Exception:
        return ""

def try_local_repairs(
    *,
    full_text: str,
    hole_span: Tuple[int, int],
    goal_text: str,
    model: Optional[str],
    isabelle,
    session: str,
    repair_budget_s: float = 12.0,  # Increased from 8.0 to give more time
    max_ops_to_try: int = 2,
    beam_k: int = 1,
    trace: bool = False,
) -> Tuple[str, bool, str]:
    start = time.monotonic()
    def _left() -> float:
        return max(0.0, repair_budget_s - (time.monotonic() - start))

    # ---- Gather only CHEAP signals first (so we don't burn the budget) ----
    # One quick run to get state + recent errors; do NOT run Nitpick/Quickcheck yet.
    s0 = _quick_state_subgoals(isabelle, session, full_text)  # baseline score
    state0, errs0 = _quick_state_and_errors(isabelle, session, full_text)

    hole_line, indent, lines = _hole_line_bounds(full_text, hole_span)
    s, e = _snippet_window(lines, hole_line, radius=12)
    snippet = "\n".join(lines[s:e])

    state_block = state0 or _quick_state_text(isabelle, session, full_text)
    header = _nearest_header(lines, hole_line)
    rsteps = _recent_steps(lines, hole_line)

    # Keep 'facts' and 'ce' EMPTY unless we have comfortable time left.
    # This removes the heavy Nitpick/Quickcheck calls from the hot path.
    facts: List[str] = []
    ce: Dict[str, List[str]] = {"bindings": [], "def_hints": []}
    if _left() >= 8.0:  # Increased threshold
        # Very light token harvesting only (no CE tools)
        facts = _facts_from_state(state_block, limit=8)
    if _left() >= 12.0:  # Increased threshold  
        # Only now run Nitpick/Quickcheck (expensive). If they add def_hints, merge them.
        ce = _counterexample_hints(isabelle, session, full_text, hole_span)
        if ce.get("def_hints"):
            facts = list(dict.fromkeys((ce["def_hints"] or []) + facts))[:12]

    # ---- Obtain ops, strictly respecting the remaining budget ----
    remaining_now = _left()
    # Always leave hard headroom to TRY at least one op (apply + quick check).
    # Reserve ~3s for apply/check even on slower boxes.
    HEADROOM_S = 3
    ops: List[RepairOp] = []
    if remaining_now > HEADROOM_S + 2.0:  # Increased minimum time requirement
        # Split remaining time between proposal and application (~60/40 instead of 70/30).
        propose_timeout = int(max(2, min(45, (remaining_now - HEADROOM_S) * 0.6)))
        if propose_timeout >= 5:  # Reduced from 8 to 5
            # Run the LLM proposal in a separate thread with a hard wall-clock timeout.
            def _call():
                return propose_local_repairs(
                    goal=goal_text,
                    state_block=state_block,
                    errors=errs0,
                    ce_hints=ce,
                    block_snippet=snippet,
                    nearest_header=header,
                    recent_steps=rsteps,
                    facts=facts,
                    model=model,
                    timeout_s=propose_timeout,  # still pass through for backends that honor it
                )
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(_call)
                    ops = fut.result(timeout=propose_timeout)
                if trace and ops:
                    print(f"[repair] LLM generated {len(ops)} repair operations")
            except concurrent.futures.TimeoutError:
                if trace:
                    print(f"[repair] LLM proposal timed out after {propose_timeout}s → using heuristics")
                ops = []
            except _REQ_TIMEOUTS as e:  # requests.* timeouts/connection errors
                if trace:
                    print(f"[repair] LLM HTTP error ({e.__class__.__name__}) → using heuristics")
                ops = []
            except Exception as e:
                if trace:
                    print(f"[repair] LLM proposal failed: {e!r} → using heuristics")
                ops = []
        else:
            if trace:
                print(f"[repair] Insufficient time for LLM proposal ({propose_timeout}s) → using heuristics")

    if not ops:
        ops = _heuristic_fallback_ops(goal_text, state_block, header, facts)
        if trace and ops:
            print(f"[repair] Using {len(ops)} heuristic fallback operations")

    if trace:
        printable = []
        for kind, payload in ops:
            if kind == "insert_before_hole":
                printable.append(f'insert_before_hole("{payload.line}")')
            elif kind == "replace_in_snippet":
                printable.append(f'replace_in_snippet(find="{payload.find}", replace="{payload.replace}")')
            else:
                printable.append(f'insert_have_block("{payload.label}", "...")')
        if printable:
            print("[repair] proposed:", "; ".join(printable))

    # Never bail out BEFORE trying at least one op.
    # If time is extremely tight, apply the simplest op deterministically and return.
    if _left() < 1.0:  # Increased from 0.5 to 1.0
        # Fall through to the tiny beam below; it will try one candidate quickly.
        pass

    # Tiny beam over ops (configurable via beam_k; default=1 preserves old behavior)
    # Score = resulting subgoal count (lower is better); tie-break by preferring simpler edits.
    tried = 0
    best_text = None
    best_score = 9999
    best_kind = ""
    scored: List[Tuple[int, str, str]] = []  # (s1, kind, summary)
    any_changed_text = False
    for (kind, payload) in ops:
        if _left() <= 0 or tried >= max_ops_to_try:
            break
        tried += 1

        if kind == "insert_before_hole":
            cand = _apply_insert_before_hole(full_text, hole_span, payload.line)
        elif kind == "replace_in_snippet":
            cand = _apply_replace_in_snippet(full_text, hole_span, payload.find, payload.replace)
        elif kind == "insert_have_block":
            cand = _apply_insert_have_block(full_text, hole_span, payload.label, payload.statement, payload.after_line_matching, payload.body_hint)
        else:
            continue

        if cand == full_text:
            if trace:
                print(f"[repair] Operation {kind} resulted in no changes")
            continue
        any_changed_text = True
        # Quick score; even if sentinel 9999, we may accept non-regressive simple edits later.
        s1 = _quick_state_subgoals(isabelle, session, cand)
        scored.append((s1, kind, getattr(payload, "label", "") or getattr(payload, "line", "") or getattr(payload, "find", "")))
        if trace:
            print(f"[repair] Operation {kind} scored: {s0} → {s1}")
        if s1 != 9999 and s1 < best_score:
            best_score, best_text, best_kind = s1, cand, kind

    # Greedy accept any non-regressive candidate; otherwise keep best if beam_k>1
    if best_text is not None:
        if trace:
            scored_str = ", ".join([f"{k}:{s}" for (s, k, _) in scored[:max(1, beam_k)]])
            print(f"[repair] scored candidates (top~{beam_k}): {scored_str} | base {s0} → best {best_score}")
        non_regressive = (best_score != 9999) and (best_score <= s0 or best_kind in ("insert_before_hole", "replace_in_snippet"))
        if non_regressive:
            return best_text, True, f"beam:{best_kind}"

    # If we get here without a measurable improvement:
    #  - If we tried nothing, but have ops, apply one blindly.
    #  - If everything scored as 9999 but text changed, accept the first changed candidate.
    if tried == 0 and ops:
        # Extremely defensive fallback; shouldn't happen with the guard above.
        kind, payload = ops[0]
        if kind == "insert_before_hole":
            if trace:
                print("[repair] Applying first operation blindly (no time to test)")
            return _apply_insert_before_hole(full_text, hole_span, payload.line), True, "blind-insert0"
        if kind == "replace_in_snippet":
            if trace:
                print("[repair] Applying first operation blindly (no time to test)")
            return _apply_replace_in_snippet(full_text, hole_span, payload.find, payload.replace), True, "blind-replace0"
    # Unknown scoring across the board — accept a changed candidate so the outer loop can re-evaluate.
    if any_changed_text and all(s == 9999 for (s, _, _) in scored):
        kind, payload = ops[0]
        if kind == "insert_before_hole":
            if trace:
                print("[repair] Accepting blind patch due to scoring issues")
            return _apply_insert_before_hole(full_text, hole_span, payload.line), True, "blind-accept"
        if kind == "replace_in_snippet":
            if trace:
                print("[repair] Accepting blind patch due to scoring issues")
            return _apply_replace_in_snippet(full_text, hole_span, payload.find, payload.replace), True, "blind-accept"        
    
    if trace:
        print(f"[repair] No effective repairs found (tried {tried} operations, changed_text={any_changed_text})")
    return full_text, False, "repairs-did-not-help"

# =========================
# CEGIS wrapper: iterate + region growth
# =========================

def try_cegis_repairs(
    *,
    full_text: str,
    hole_span: Tuple[int, int],
    goal_text: str,
    model: Optional[str],
    isabelle,
    session: str,
    repair_budget_s: float = 15.0,  # Increased budget for more iterations
    max_ops_to_try: int = 2,
    beam_k: int = 1,
    allow_whole_fallback: bool = False,    
    trace: bool = False,
) -> Tuple[str, bool, str]:
    t0 = time.monotonic()
    def left() -> float: return max(0.0, repair_budget_s - (time.monotonic() - t0))

    # ---- Stage 1: local (iterate a few rounds with fresh feedback) ----
    base_s = _quick_state_subgoals(isabelle, session, full_text)
    current_text = full_text
    
    for round_i in range(3):
        if left() <= 5.0: break  # Need at least 5s for meaningful work
        # Adaptive tiny-beam: when low on time, drop to beam_k=1; else use requested beam_k.
        eff_k = 1 if left() < 15.0 else max(1, beam_k)
        patched, ok, tag = try_local_repairs(
            full_text=current_text, hole_span=hole_span, goal_text=goal_text,
            model=model, isabelle=isabelle, session=session,
            # Give local-repair rounds a little more air so downstream LLM calls don't get <5s timeouts.
            repair_budget_s=min(12.0, max(8.0, left()*0.4)),  # Use less of total budget per round

            max_ops_to_try=max_ops_to_try, beam_k=eff_k, trace=trace,            
        )
        if ok and patched != current_text:
            s1 = _quick_state_subgoals(isabelle, session, patched)
            if trace: print(f"[cegis] local round {round_i}: {base_s} → {s1} (beam_k={eff_k})")
            # Normal acceptance: measurable, non-regressive improvement
            if s1 != 9999 and s1 <= base_s:
                return patched, True, f"local:{tag}"
            # NEW: accept neutral blind patches so outer loop can retry on the patched text.
            # This avoids discarding useful insertions when print_state parsing yields 9999.
            if "blind" in (tag or ""):
                if trace: print("[cegis] accepting neutral blind patch (retry hole on patched text)")
                return patched, True, f"local:{tag}(neutral)"
            # AGGRESSIVE: If we made textual changes, use them for next iteration
            if s1 == 9999 and s1 == base_s:  # Both failed, but text changed
                current_text = patched
                if trace: print(f"[cegis] using changed text for next iteration (both scored 9999)")
        # refresh signals for next round
        _ = _quick_state_and_errors(isabelle, session, current_text)

    # ---- Stage 2: case-block rewrite ----
    hole_line, _indent, lines = _hole_line_bounds(current_text, hole_span)
    cs, ce = _enclosing_case_block(lines, hole_line)
    if cs >= 0 and left() > 5.0:
        state_block, errs = _quick_state_and_errors(isabelle, session, current_text)
        ceh = _counterexample_hints(isabelle, session, current_text, hole_span)
        block = "\n".join(lines[cs:ce])
        
        # Calculate appropriate timeout for block repair
        block_timeout = int(min(60, max(20, left()*0.6)))  # Use more of remaining time
        if trace: print(f"[cegis] case-block repair with {block_timeout}s timeout")
        
        try:
            blk = _propose_block_repair(
                goal=goal_text, errors=errs, ce_hints=ceh,
                state_block=state_block, block_text=block,
                model=model, timeout_s=block_timeout
            )
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout):
            if trace: print("[cegis] case-block repair timed out, skipping")
            blk = ""
        except Exception as e:
            if trace: print(f"[cegis] case-block repair failed: {e}")
            blk = ""
            
        # Check if we got an effective block after sanitization
        if not _is_effective_block(blk):
            # No suggestion (or only fences) → no-op for this stage
            pass
        elif blk.strip() == block.strip():
            # Identical to current block → no change
            pass
        else:
            # Try the LLM suggestion AS-IS first
            patched_raw = "\n".join(lines[:cs] + [blk] + lines[ce:])
            s1_raw = _quick_state_subgoals(isabelle, session, patched_raw)
            if trace: print(f"[cegis] case-block (raw): {base_s} → {s1_raw}")
            if s1_raw != 9999 and s1_raw <= base_s:
                return patched_raw, True, "block:case(raw)"
                
            # AGGRESSIVE: Accept reasonable-looking changes even if they don't improve score
            # The key insight is that LLM-generated code with proper structure is often 
            # closer to correct than the original sorry placeholders
            if s1_raw == 9999 and _looks_like_reasonable_proof_attempt(blk):
                if trace: print("[cegis] accepting reasonable-looking case block despite scoring issues")
                # Apply sanitization to replace failed tactics with sorry
                blk_with_sorry = _replace_failing_tactics_with_sorry(blk)
                patched_sorry = "\n".join(lines[:cs] + [blk_with_sorry] + lines[ce:])
                return patched_sorry, True, "block:case(aggressive)"
            
            # Fallback: prevent brittle closes; rewrite lone finalizers to 'sorry' and re-check
            blk_sanit = re.sub(r"(?m)^\s*(?:by\b.*|done\s*)$", "  sorry", blk)
            patched = "\n".join(lines[:cs] + [blk_sanit] + lines[ce:])
            s1 = _quick_state_subgoals(isabelle, session, patched)
            if trace: print(f"[cegis] case-block (sanit): {base_s} → {s1}")
            if s1 != 9999 and s1 <= base_s:
                return patched, True, "block:case(sanit)"

    # ---- Stage 3: subproof rewrite ----
    ps, pe = _enclosing_subproof(lines, hole_line)
    if ps >= 0 and left() > 3.0:
        state_block, errs = _quick_state_and_errors(isabelle, session, current_text)
        ceh = _counterexample_hints(isabelle, session, current_text, hole_span)
        block = "\n".join(lines[ps:pe])
        
        # Calculate appropriate timeout for subproof repair
        subproof_timeout = int(min(45, max(15, left()*0.7)))
        if trace: print(f"[cegis] subproof repair with {subproof_timeout}s timeout")
        
        try:
            blk = _propose_block_repair(
                goal=goal_text, errors=errs, ce_hints=ceh,
                state_block=state_block, block_text=block,
                model=model, timeout_s=subproof_timeout
            )
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout):
            if trace: print("[cegis] subproof repair timed out, skipping")
            blk = ""
        except Exception as e:
            if trace: print(f"[cegis] subproof repair failed: {e}")
            blk = ""
            
        # Check if we got an effective block after sanitization
        if not _is_effective_block(blk):
            pass
        elif blk.strip() == block.strip():
            pass
        else:
            # Try raw first
            patched_raw = "\n".join(lines[:ps] + [blk] + lines[pe:])
            s1_raw = _quick_state_subgoals(isabelle, session, patched_raw)
            if trace: print(f"[cegis] subproof-block (raw): {base_s} → {s1_raw}")
            if s1_raw != 9999 and s1_raw <= base_s:
                return patched_raw, True, "block:subproof(raw)"
                
            # AGGRESSIVE: Accept reasonable-looking changes 
            if s1_raw == 9999 and _looks_like_reasonable_proof_attempt(blk):
                if trace: print("[cegis] accepting reasonable-looking subproof despite scoring issues")
                blk_with_sorry = _replace_failing_tactics_with_sorry(blk)
                patched_sorry = "\n".join(lines[:ps] + [blk_with_sorry] + lines[pe:])
                return patched_sorry, True, "block:subproof(aggressive)"
                
            # Fallback to conservative sanitization
            blk_sanit = re.sub(r"(?m)^\s*(?:by\b.*|done\s*)$", "  sorry", blk)
            patched = "\n".join(lines[:ps] + [blk_sanit] + lines[pe:])
            s1 = _quick_state_subgoals(isabelle, session, patched)
            if trace: print(f"[cegis] subproof-block (sanit): {base_s} → {s1}")
            if s1 != 9999 and s1 <= base_s:
                return patched, True, "block:subproof(sanit)"

    # ---- Stage 4: whole-proof fallback (optional) ----
    if allow_whole_fallback and left() > 0:
        state_block, errs = _quick_state_and_errors(isabelle, session, current_text)
        ceh = _counterexample_hints(isabelle, session, current_text, hole_span)
        
        whole_timeout = int(min(90, max(30, left())))
        if trace: print(f"[cegis] whole-proof repair with {whole_timeout}s timeout")
        
        try:
            whole = _propose_whole_repair(
                goal=goal_text, errors=errs, ce_hints=ceh,
                state_block=state_block, full_text=current_text,
                model=model, timeout_s=whole_timeout
            )
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout):
            if trace: print("[cegis] whole-proof repair timed out, skipping")
            whole = ""
        except Exception as e:
            if trace: print(f"[cegis] whole-proof repair failed: {e}")
            whole = ""
            
        if whole and whole.strip() != current_text.strip():
            s1 = _quick_state_subgoals(isabelle, session, whole)
            if trace: print(f"[cegis] whole-proof: {base_s} → {s1}")
            if s1 != 9999 and s1 <= base_s:
                return whole, True, "whole"

    # Return the current text even if we didn't achieve measurable improvement
    # The caller can decide whether to use it for further iterations
    if current_text != full_text:
        if trace: print("[cegis] returning modified text even though no scoring improvement")
        return current_text, True, "partial-progress"
        
    return full_text, False, "cegis-nohelp"

# Helper functions for the aggressive acceptance strategy
def _looks_like_reasonable_proof_attempt(block_text: str) -> bool:
    """
    Heuristic to determine if a block looks like a reasonable proof attempt
    that's worth accepting even if it doesn't parse correctly.
    """
    lines = block_text.strip().split('\n')
    if len(lines) < 2:
        return False
        
    # Should have some structure (have statements, show statements, etc.)
    has_have = any('have ' in line for line in lines)
    has_show = any('show ' in line for line in lines)
    has_using = any('using ' in line for line in lines)
    
    # Should not be just sorry statements
    non_sorry_lines = [line for line in lines if 'sorry' not in line.strip()]
    
    return (has_have or has_show or has_using) and len(non_sorry_lines) >= len(lines) // 2

def _replace_failing_tactics_with_sorry(block_text: str) -> str:
    """
    Replace likely-failing proof tactics with sorry, while preserving structure.
    This converts a potentially broken proof into a valid skeleton.
    """
    lines = block_text.split('\n')
    result_lines = []
    
    for line in lines:
        # Keep structural elements as-is
        if any(keyword in line for keyword in ['have ', 'show ', 'case ', 'using ', 'then ']):
            result_lines.append(line)
        # Replace proof tactics that are likely to fail
        elif any(tactic in line.strip() for tactic in ['by simp', 'by auto', 'by blast', 'by clarsimp']):
            # Extract indentation and replace with sorry
            indent = line[:len(line) - len(line.lstrip())]
            result_lines.append(f"{indent}sorry")
        else:
            result_lines.append(line)
            
    return '\n'.join(result_lines)

# -------- NEW: whole-proof fallback proposer ----------
_WHOLE_SYSTEM = """You repair an Isabelle/Isar PROOF.
You may rewrite the entire proof but keep the lemma statement and imports intact.
Prefer minimal edits; ensure the result compiles. Output ONLY the full repaired proof."""

_WHOLE_USER = """GOAL:
{goal}

ISABELLE_ERRORS (recent):
{errors}

COUNTEREXAMPLE/MODEL HINTS:
{ce_hints}

LOCAL CONTEXT (print_state excerpt):
{state_block}

ORIGINAL PROOF:
<<<PROOF
{full_text}
PROOF
"""

def _propose_whole_repair(
    *, goal: str, errors: List[str], ce_hints: Dict[str,List[str]],
    state_block: str, full_text: str, model: Optional[str], timeout_s: int
) -> str:
    ce = []
    if ce_hints.get("bindings"): ce += ce_hints["bindings"]
    if ce_hints.get("def_hints"): ce += ce_hints["def_hints"]
    prompt = _WHOLE_SYSTEM + "\n\n" + _WHOLE_USER.format(
        goal=goal,
        errors="\n".join(f"- {e}" for e in (errors or [])) or "(none)",
        ce_hints="\n".join(ce) or "(none)",
        state_block=(state_block or "").strip(),
        full_text=full_text.rstrip(),
    )
    out = _generate_simple(prompt, model=model, timeout_s=timeout_s)
    # Whole-proof proposer can also echo fences; sanitise but never empty it out.
    return _sanitize_llm_block(out or "")