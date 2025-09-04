# prover/isabelle_api.py
import os, json, tempfile, textwrap
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from isabelle_client import start_isabelle_server, get_isabelle_client, IsabelleResponse

# Config (keep import light and backward compatible)
try:
    from .config import EXTRA_IMPORTS  # list[str]
except Exception:
    EXTRA_IMPORTS = []
# Optional per-call timeout (seconds) for Isabelle 'use_theories'
try:
    from .config import ISABELLE_USE_THEORIES_TIMEOUT_S  # int
except Exception:
    # allow env override without touching config.py
    try:
        ISABELLE_USE_THEORIES_TIMEOUT_S = int(os.getenv("ISABELLE_USE_THEORIES_TIMEOUT_S", "").strip() or 0)
    except Exception:
        ISABELLE_USE_THEORIES_TIMEOUT_S = 60  # 0 = disabled

def _header(imports=None):
    imps = ["Main"] + list(imports or []) + list(EXTRA_IMPORTS or [])
    return f"theory Scratch\nimports {' '.join(imps)}\nbegin\n"

FOOTER = "end\n"

_use_calls = 0
_use_timeouts = 0

def _write_tmp_theory(theory_text: str) -> Tuple[str, str]:
    tmpdir = tempfile.TemporaryDirectory()
    p = os.path.join(tmpdir.name, "Scratch.thy")
    with open(p, "w", encoding="utf-8") as f:
        f.write(theory_text)
    return tmpdir.name, p

def parse_n_subgoals(msg: str) -> Optional[int]:
    """Heuristic extractor for the number of subgoals from a print_state block."""
    import re
    for line in msg.splitlines():
        m = re.search(r"(\d+)\s+subgoals?", line)
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
    """Run a small throwaway theory through Isabelle.

    Adds a **wall-clock timeout** to avoid rare hangs inside tactics (simp/auto/etc.).
    Timeout value defaults to ISABELLE_USE_THEORIES_TIMEOUT_S (0 = disabled).
    We first try to pass a native timeout kwarg to isabelle_client if it exists;
    if not, we fall back to a thread with future.result(timeout=...).

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
            for kw in ("timeout", "timeout_s", "timeout_sec", "request_timeout"):
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
    Return success if **any** FINISHED block reports ok=true.

    Robust across client variants:
      - response type can be Enum (e.g., IsabelleResponseType.FINISHED) or str
      - response body may be bytes or already-dict
      - dict-like or attribute-style access
    """
    def _normalize_type(rt: Any) -> str:
        # Handle Enums (e.g., IsabelleResponseType.FINISHED) and plain strings.
        try:
            if hasattr(rt, "name"):  # Enum.name -> 'FINISHED'
                return str(rt.name).strip().upper()
            if hasattr(rt, "value"):  # Enum.value -> 'FINISHED'
                v = getattr(rt, "value")
                if isinstance(v, str):
                    return v.strip().upper()
                return str(v).strip().upper()
            s = str(rt)
            su = s.upper()
            # Cope with representations like "<IsabelleResponseType.OK: 'OK'>"
            if "FINISHED" in su:
                return "FINISHED"
            if su.endswith(".OK") or su == "OK" or "OK'" in su:
                return "OK"
            if "NOTE" in su:
                return "NOTE"
            return s.strip().upper()
        except Exception:
            return ""

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

    def _decode_body(body: Any) -> Optional[Dict[str, Any]]:
        if body is None:
            return None
        if isinstance(body, (bytes, bytearray)):
            try:
                body = body.decode("utf-8", "replace")
            except Exception:
                try:
                    body = body.decode(errors="replace")
                except Exception:
                    body = str(body)
        if isinstance(body, dict):
            return body
        try:
            return json.loads(body)
        except Exception:
            return None

    any_ok = False
    last_obj: Dict[str, Any] = {}

    for r in (resps or []):
        rt_raw = _get_field(r, ("response_type", "type", "kind", "tag", "name"))
        rt_norm = _normalize_type(rt_raw)
        if rt_norm != "FINISHED":
            continue

        body_raw = _get_field(r, ("response_body", "body", "message", "payload"))
        obj = _decode_body(body_raw)
        if not isinstance(obj, dict):
            continue

        last_obj = obj  # track last FINISHED

        if bool(obj.get("ok", False)):
            any_ok = True
        elif str(obj.get("result", "")).lower() == "ok":
            any_ok = True

    return any_ok, (last_obj or {})

def last_print_state_block(resps: List[IsabelleResponse]) -> str:
    def _get_field(obj: Any, names: Tuple[str, ...]) -> Any:
        if isinstance(obj, dict):
            for n in names:
                if n in obj:
                    return obj[n]
        for n in names:
            if hasattr(obj, n):
                return getattr(obj, n)
        return None

    def _normalize_type(rt: Any) -> str:
        try:
            if hasattr(rt, "name"):
                return str(rt.name).strip().upper()
            if hasattr(rt, "value"):
                v = getattr(rt, "value")
                if isinstance(v, str):
                    return v.strip().upper()
                return str(v).strip().upper()
            s = str(rt)
            su = s.upper()
            if "NOTE" in su:
                return "NOTE"
            if "FINISHED" in su:
                return "FINISHED"
            if "OK" in su:
                return "OK"
            return s.strip().upper()
        except Exception:
            return ""

    txt = ""
    for r in (resps or []):
        rt = _get_field(r, ("response_type", "type", "kind", "tag", "name"))
        if _normalize_type(rt) != "NOTE":
            continue
        body = _get_field(r, ("response_body", "body", "message", "payload"))
        if isinstance(body, (bytes, bytearray)):
            try:
                body = body.decode("utf-8", "replace")
            except Exception:
                body = str(body)
        try:
            obj = body if isinstance(body, dict) else json.loads(body)
        except Exception:
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
