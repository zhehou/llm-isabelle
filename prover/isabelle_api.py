# prover/isabelle_api.py 
from __future__ import annotations

import os, json, tempfile, textwrap, re
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

# Re-export these (cli.py and experiments.py import them from here)
from isabelle_client import start_isabelle_server, get_isabelle_client, IsabelleResponse

# ------------------ Config (kept light and backwards-compatible) ------------------
try:
    from .config import EXTRA_IMPORTS  # list[str]
except Exception:
    EXTRA_IMPORTS: List[str] = []

# Optional per-call timeout (seconds) for Isabelle 'use_theories'
try:
    from .config import ISABELLE_USE_THEORIES_TIMEOUT_S  # int
except Exception:
    try:
        ISABELLE_USE_THEORIES_TIMEOUT_S = int(os.getenv("ISABELLE_USE_THEORIES_TIMEOUT_S", "").strip() or 0)
    except Exception:
        ISABELLE_USE_THEORIES_TIMEOUT_S = 60  # 0 = disabled

# ------------------ Small helpers & constants ------------------
FOOTER = "end\n"
_TIMEOUT_KWARGS = ("timeout", "timeout_s", "timeout_sec", "request_timeout")  # best-effort spellings
_SUBGOALS_RE = re.compile(r"(\d+)\s+subgoals?")

_use_calls = 0
_use_timeouts = 0


def _header(imports: Optional[List[str]] = None) -> str:
    imps = ["Main"] + list(imports or []) + list(EXTRA_IMPORTS or [])
    return f"theory Scratch\nimports {' '.join(imps)}\nbegin\n"


def _get_field(obj: Any, names: Tuple[str, ...]) -> Any:
    # dict-like
    if isinstance(obj, dict):
        for n in names:
            if n in obj:
                return obj[n]
    # attribute-style
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def _normalize_type(rt: Any) -> str:
    """Return a normalized uppercase type name ('FINISHED'/'NOTE'/...) across variants."""
    try:
        if hasattr(rt, "name"):               # Enum.name -> 'FINISHED'
            return str(rt.name).strip().upper()
        if hasattr(rt, "value"):              # Enum.value -> 'FINISHED'
            v = getattr(rt, "value")
            return (v if isinstance(v, str) else str(v)).strip().upper()
        s = str(rt).strip()
        su = s.upper()
        if "FINISHED" in su:
            return "FINISHED"
        if "NOTE" in su:
            return "NOTE"
        if su.endswith(".OK") or su == "OK" or "OK'" in su:
            return "OK"
        return su
    except Exception:
        return ""


def _decode_body_to_dict(body: Any) -> Optional[Dict[str, Any]]:
    """Body may be dict/JSON string/bytes; return dict or None."""
    if body is None:
        return None
    if isinstance(body, (bytes, bytearray)):
        try:
            body = body.decode("utf-8", "replace")
        except Exception:
            body = str(body)
    if isinstance(body, dict):
        return body
    try:
        return json.loads(body)
    except Exception:
        return None


# ------------------ Public utils ------------------
def _write_tmp_theory(theory_text: str) -> Tuple[str, str]:
    """
    Legacy helper (not used by this module). Creates a temp dir and writes Scratch.thy.
    Returns (tmpdir_path, file_path). Note: the temp directory lifetime is not managed here.
    """
    tmpdir = tempfile.TemporaryDirectory()
    p = os.path.join(tmpdir.name, "Scratch.thy")
    with open(p, "w", encoding="utf-8") as f:
        f.write(theory_text)
    # DO NOT change the return shape for compatibility
    return tmpdir.name, p


def parse_n_subgoals(msg: str) -> Optional[int]:
    """Heuristic extractor for the number of subgoals from a print_state block."""
    m = _SUBGOALS_RE.search(msg or "")
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def build_theory(steps: List[str], add_print_state: bool, end_with: Optional[str]) -> str:
    body = [steps[0]] + ["  " + s for s in steps[1:]]
    if add_print_state:
        body.append("  print_state")
    if end_with:
        body.append("  " + end_with)
    return textwrap.dedent(_header() + "\n".join(body) + "\n\n" + FOOTER)


def _use_theories_call(isabelle, *, session_id: str, master_dir: str) -> List[IsabelleResponse]:
    return list(isabelle.use_theories(theories=["Scratch"], session_id=session_id, master_dir=master_dir))


def run_theory(isabelle, session_id: str, theory_text: str) -> List[IsabelleResponse]:
    """Run a small throwaway theory through Isabelle with an optional wall-clock timeout.

    Timeout value defaults to ISABELLE_USE_THEORIES_TIMEOUT_S (0 = disabled).
    Tries native timeouts first (various kwarg spellings), else falls back to a thread + Future timeout.
    Returns the collected IsabelleResponse list; on timeout returns an empty list.
    """
    global _use_calls
    _use_calls += 1

    tmpdir = tempfile.TemporaryDirectory()
    try:
        p = os.path.join(tmpdir.name, "Scratch.thy")
        with open(p, "w", encoding="utf-8") as f:
            f.write(theory_text)

        timeout_s = int(ISABELLE_USE_THEORIES_TIMEOUT_S or 0)
        if timeout_s > 0:
            # Try native timeout kwarg spellings first (best-effort)
            for kw in _TIMEOUT_KWARGS:
                try:
                    return list(
                        isabelle.use_theories(
                            theories=["Scratch"], session_id=session_id, master_dir=tmpdir.name, **{kw: timeout_s}
                        )
                    )
                except TypeError:
                    continue
                except Exception:
                    return []

            # Fallback: thread with Future timeout
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_use_theories_call, isabelle, session_id=session_id, master_dir=tmpdir.name)
                try:
                    return fut.result(timeout=timeout_s)
                except FuturesTimeout:
                    global _use_timeouts
                    _use_timeouts += 1
                    return []

        # No timeout requested → direct call
        return list(isabelle.use_theories(theories=["Scratch"], session_id=session_id, master_dir=tmpdir.name))
    finally:
        tmpdir.cleanup()


def finished_ok(resps: List[IsabelleResponse]) -> Tuple[bool, Dict[str, Any]]:
    """
    Return success if **any** FINISHED block reports ok=true (or result='ok').
    Robust across client variants:
      - response type can be Enum or str
      - response body may be bytes, JSON string, or dict
      - dict-like or attribute-style access
    """
    any_ok = False
    last_obj: Dict[str, Any] = {}

    for r in (resps or []):
        if _normalize_type(_get_field(r, ("response_type", "type", "kind", "tag", "name"))) != "FINISHED":
            continue
        obj = _decode_body_to_dict(_get_field(r, ("response_body", "body", "message", "payload")))
        if not isinstance(obj, dict):
            continue
        last_obj = obj  # track last FINISHED
        if bool(obj.get("ok", False)) or str(obj.get("result", "")).lower() == "ok":
            any_ok = True

    return any_ok, (last_obj or {})


def last_print_state_block(resps: List[IsabelleResponse]) -> str:
    """Return the text of the last NOTE/writeln message that looks like a goal/subgoal block."""
    txt = ""
    for r in (resps or []):
        if _normalize_type(_get_field(r, ("response_type", "type", "kind", "tag", "name"))) != "NOTE":
            continue
        body = _get_field(r, ("response_body", "body", "message", "payload"))
        obj = _decode_body_to_dict(body)
        if not isinstance(obj, dict):
            continue
        if obj.get("kind") == "writeln":
            msg = str(obj.get("message", ""))
            if ("subgoal" in msg) or ("goal (" in msg) or ("goal\n" in msg):
                txt = msg
    return txt


def use_calls_count() -> int:
    return _use_calls


def use_timeouts_count() -> int:
    return int(_use_timeouts)


__all__ = [
    # re-exports
    "start_isabelle_server", "get_isabelle_client", "IsabelleResponse",
    # config-driven helpers
    "_header", "FOOTER", "parse_n_subgoals", "build_theory", "run_theory",
    "finished_ok", "last_print_state_block", "use_calls_count", "use_timeouts_count",
    "graceful_terminate",
]

# Cross-runtime shutdown helper (works for multiprocessing.Process and subprocess.Popen)
def graceful_terminate(proc, timeout_s: int = 3) -> None:
    """
    Terminate an Isabelle server process robustly across runtimes.
    Tries terminate→wait(timeout)→join(timeout)→kill, ignoring errors.
    """
    if proc is None:
        return
    try:
        if hasattr(proc, "terminate"):
            proc.terminate()
    except Exception:
        pass
    # Prefer Popen.wait(timeout) if available
    try:
        if hasattr(proc, "wait"):
            try:
                proc.wait(timeout=timeout_s)  # subprocess.Popen
                return
            except TypeError:
                proc.wait()  # older signature without timeout
                return
            except Exception:
                pass
    except Exception:
        pass
    # multiprocessing.Process
    try:
        if hasattr(proc, "join"):
            proc.join(timeout=timeout_s)
            return
    except Exception:
        pass
    # Last resort
    try:
        if hasattr(proc, "kill"):
            proc.kill()
    except Exception:
        pass