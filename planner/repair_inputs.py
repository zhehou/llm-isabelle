import re
import os
import json
import requests
from typing import Dict, List, Optional, Tuple, Set, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout
from prover.isabelle_api import build_theory, run_theory, last_print_state_block, finished_ok

# ========== Configuration ==========
_ISA_FAST_TIMEOUT_S = int(os.getenv("ISABELLE_FAST_TIMEOUT_S", "12"))
_ISA_VERIFY_TIMEOUT_S = int(os.getenv("ISABELLE_VERIFY_TIMEOUT_S", "30"))
_SESSION = requests.Session()

# ========== Regex Patterns ==========
_HEADER_RE = re.compile(r"^\s*(proof\s*\(|proof\b|case\s+|then\s+show\b)")
_APPLY_OR_BY = re.compile(r"^\s*(apply|by)\b")

def _clamp_line_index(lines: List[str], idx: int) -> int:
    if not lines:
        return -1
    return max(0, min(idx, len(lines) - 1))

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

def _earliest_failing_anchor_or_hole(isabelle, session: str, full_text: str, hole_span):
    try:
        line_idx, err_excerpt = _earliest_failure_anchor(isabelle, session, full_text,
                                                         default_line_0=full_text.count("\n"))
        if isinstance(line_idx, int) and line_idx >= 0:
            return ("line", line_idx, err_excerpt)
    except Exception:
        pass
    return ("hole", hole_span, "")

def _counterexample_hints_precise(isabelle, session: str, full_text: str, hole_span):
    kind, at, _ = _earliest_failing_anchor_or_hole(isabelle, session, full_text, hole_span)
    if kind == "line":
        lines = full_text.splitlines()
        nit = _run_nitpick_at_line(isabelle, session, lines,
                                   inject_before_1based=at + 1,
                                   qc_timeout_s=3, nitpick_timeout_s=5)
    else:
        nit = _run_nitpick_at_hole(isabelle, session, full_text, hole_span, timeout_s=3)
    return _nitpick_state_hints_from_text(nit)

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

def _run_nitpick_at_line(
    isabelle,
    session: str,
    full_text_lines: List[str],
    inject_before_1based: int,
    qc_timeout_s: int = 3,
    nitpick_timeout_s: int = 5,
) -> str:
    """
    Build a transient doc variant that inserts Quickcheck/Nitpick *before* the
    earliest failing tactic line, so diagnostics run on the exact subgoal that
    was about to fail. Returns concatenated Isabelle response text.
    """
    i0 = max(0, inject_before_1based - 1)
    pad = ""
    if 0 <= i0 < len(full_text_lines):
        ln = full_text_lines[i0]
        pad = ln[: len(ln) - len(ln.lstrip(" "))]
    injected = [
        f"{pad}prefer 1",
        f"{pad}quickcheck[timeout={max(1, qc_timeout_s)}]",
        f"{pad}nitpick[timeout={max(1, nitpick_timeout_s)}]",
    ]
    variant = "\n".join(full_text_lines[:i0] + injected + full_text_lines[i0:])
    try:
        thy = build_theory(variant.splitlines(), add_print_state=True, end_with=None)
        resps = _run_theory_with_timeout(isabelle, session, thy, timeout_s=max(3, qc_timeout_s + nitpick_timeout_s + 2))
        return "\n".join(
            getattr(r, "response_body", b"").decode(errors="replace")
            if isinstance(getattr(r, "response_body", None), (bytes, bytearray))
            else str(getattr(r, "response_body", ""))
            for r in resps or []
        )
    except Exception:
        return ""
    
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
            
            # Try structured JSON first
            try:
                data = json.loads(raw) if isinstance(raw, str) and raw.strip() else None
                if isinstance(data, dict):
                    for node in data.get("nodes", []):
                        for msg in node.get("messages", []):
                            if str(msg.get("kind", "")).lower() == "error":
                                txt = str(msg.get("message", "") or "").strip()
                                if txt:  # Only add non-empty errors
                                    errors.append({"text": txt, "line": msg.get("line")})
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
            
            # Fallback: raw text parsing for error markers
            if isinstance(raw, str):
                for pattern in ["*** Error:", "*** Outer syntax error", "*** Failed"]:
                    if pattern in raw:
                        # Extract the actual error message
                        for line in raw.split('\n'):
                            if pattern in line:
                                errors.append({"text": line.strip()})
                                break
        
        # Deduplicate by text
        seen = set()
        deduped = []
        for e in errors:
            txt = e.get("text", "")
            if txt and txt not in seen:
                seen.add(txt)
                deduped.append(e)
        
        return state, deduped[:5]
    except Exception as e:
        return "", [{"text": f"extraction_error: {type(e).__name__}"}]

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
    if not text:
        return {"bindings": [], "def_hints": []}
    
    t_lower = text.lower()
    has_cex = any(marker in t_lower for marker in [
        "nitpick found a counterexample",
        "nitpick found a potential counterexample",
        "quickcheck found a counterexample",
        "model found"
    ])
    
    if not has_cex:
        return {"bindings": [], "def_hints": []}
    
    # Extract bindings more carefully
    bindings = []
    for line in text.split('\n'):
        # Match "var = value" patterns
        match = re.match(r'\s*([a-z][A-Za-z0-9_\']*)\s*=\s*(.+)', line)
        if match:
            var, val = match.groups()
            # Clean up value (remove trailing punctuation/whitespace)
            val = re.sub(r'[,;.\s]+$', '', val.strip())
            if val:
                bindings.append(f"{var} = {val}")
    
    # Extract definition names
    defs = list(dict.fromkeys(re.findall(r"\b([A-Za-z_]\w*'*)_def\b", text)))
    def_hints = [f"unfolding {d}_def" for d in defs]  # Use more specific syntax
    
    return {"bindings": bindings[:8], "def_hints": def_hints[:12]}

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