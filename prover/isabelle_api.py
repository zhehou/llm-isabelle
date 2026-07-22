# prover/isabelle_api.py 
from __future__ import annotations

import os, json, tempfile, textwrap, re, subprocess, sys
from importlib.resources import files as resource_files
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

# Re-export the stable client types.  The upstream ``start_isabelle_server``
# currently creates an asyncio subprocess inside ``asyncio.run`` and returns
# that subprocess after its owning event loop has already been closed.  When
# Isabelle later exits, Python attempts to notify the dead loop and prints a
# teardown warning.  Start the same command with ``subprocess.Popen`` instead;
# it has a synchronous lifetime that callers can close deterministically.
from isabelle_client import get_isabelle_client, IsabelleResponse


def start_isabelle_server(
    log_file: Optional[str] = None,
    name: Optional[str] = None,
    port: Optional[int] = None,
):
    """Start Isabelle with a synchronously owned process handle.

    The return shape and arguments match ``isabelle_client`` for backwards
    compatibility.  Keeping process ownership synchronous also makes shutdown
    behaviour identical in the CLI, planner, experiments and UI server.
    """

    server_args = ["server"]
    if log_file is not None:
        server_args.extend(["-L", str(log_file)])
    if port is not None:
        server_args.extend(["-p", str(port)])
    if name is not None:
        server_args.extend(["-n", str(name)])

    if sys.platform == "win32":  # pragma: no cover - exercised on Windows
        launcher = resource_files("isabelle_client").joinpath(
            "resources", "Cygwin-Isabelle.bat"
        )
        command = [str(launcher), subprocess.list2cmdline(server_args)]
    else:
        command = ["isabelle", *server_args]

    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    try:
        server_info = proc.stdout.readline() if proc.stdout is not None else ""
    except Exception:
        graceful_terminate(proc)
        raise
    if not server_info:
        return_code = proc.poll()
        graceful_terminate(proc)
        suffix = "" if return_code is None else f" (exit status {return_code})"
        raise ValueError(f"No stdout while starting the Isabelle server{suffix}.")
    return server_info, proc

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
_IMPORT_DIRECTIVE_RE = re.compile(
    r"^\s*\(\*\s*LLM_IMPORTS\s*:\s*(.*?)\s*\*\)\s*$"
)

_use_calls = 0
_use_timeouts = 0
_last_call_timed_out = False
_timed_out_tmpdirs: List[tempfile.TemporaryDirectory] = []


def _header(imports: Optional[List[str]] = None) -> str:
    imps = list(dict.fromkeys(["Main"] + list(imports or []) + list(EXTRA_IMPORTS or [])))
    # Isabelle theory names containing punctuation (most notably AFP/library
    # session paths such as HOL-Library.Multiset) must be quoted.  Emitting
    # them as bare outer-syntax tokens makes the hyphen look like an operator
    # and can silently turn a valid benchmark into an import error.
    def render_import(name: str) -> str:
        token = str(name).strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_'.]*", token):
            return token
        escaped = token.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    return f"theory Scratch\nimports {' '.join(render_import(x) for x in imps)}\nbegin\n"


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
    imports: List[str] = []
    cleaned: List[str] = []
    for step in steps:
        match = _IMPORT_DIRECTIVE_RE.match(str(step))
        if match:
            imports.extend(token for token in match.group(1).replace(",", " ").split() if token)
        else:
            cleaned.append(str(step))
    if not cleaned:
        cleaned = [""]
    body = [cleaned[0]] + ["  " + s for s in cleaned[1:]]
    if add_print_state:
        body.append("  print_state")
    if end_with:
        body.append("  " + end_with)
    return textwrap.dedent(_header(imports) + "\n".join(body) + "\n\n" + FOOTER)


def _use_theories_call(isabelle, *, session_id: str, master_dir: str, timeout_s: Optional[int] = None) -> List[IsabelleResponse]:
    """Internal: best-effort pass through native timeout kwargs (if supported)."""
    if timeout_s is not None and int(timeout_s or 0) > 0:
        # Try native timeout kwarg spellings first (best-effort). Some clients ignore these,
        # so the caller still enforces a wall-clock timeout via Future.result(...).
        for kw in _TIMEOUT_KWARGS:
            try:
                return list(
                    isabelle.use_theories(
                        theories=["Scratch"], session_id=session_id, master_dir=master_dir, **{kw: int(timeout_s)}
                    )
                )
            except TypeError:
                continue
            except Exception:
                raise
    return list(isabelle.use_theories(theories=["Scratch"], session_id=session_id, master_dir=master_dir))


def run_theory(
    isabelle,
    session_id: str,
    theory_text: str,
    timeout_s: Optional[int] = None,
) -> List[IsabelleResponse]:
    """Run a small throwaway theory through Isabelle with a wall-clock timeout.

    IMPORTANT: Isabelle-side timeouts are not always authoritative for our pipeline.
    This wrapper enforces a wall-clock timeout (thread + Future timeout fallback)
    and records whether the *last* call timed out so callers can avoid false positives.

    - timeout_s: seconds; if None, defaults to ISABELLE_USE_THEORIES_TIMEOUT_S.
      Use 0/<=0 to disable.

    Returns the collected IsabelleResponse list; on timeout returns an empty list.
    """
    global _use_calls, _last_call_timed_out
    _use_calls += 1
    _last_call_timed_out = False

    tmpdir = tempfile.TemporaryDirectory()
    try:
        p = os.path.join(tmpdir.name, "Scratch.thy")
        with open(p, "w", encoding="utf-8") as f:
            f.write(theory_text)

        # Resolve wall-clock timeout (seconds)
        if timeout_s is None:
            timeout_s = int(ISABELLE_USE_THEORIES_TIMEOUT_S or 0)
        else:
            try:
                timeout_s = int(timeout_s)
            except Exception:
                timeout_s = 0
        if timeout_s > 0:
            # Do not use ThreadPoolExecutor as a context manager here: its
            # implicit shutdown waits for a timed-out worker and defeats the
            # wall-clock boundary.  Isabelle is interrupted and the temporary
            # directory is retained until process shutdown so a retiring
            # worker never observes a deleted master directory.
            ex = ThreadPoolExecutor(max_workers=1)
            fut = ex.submit(
                _use_theories_call,
                isabelle,
                session_id=session_id,
                master_dir=tmpdir.name,
                timeout_s=timeout_s,
            )
            try:
                result = fut.result(timeout=timeout_s)
                ex.shutdown(wait=True)
                return result
            except FuturesTimeout:
                global _use_timeouts
                _use_timeouts += 1
                _last_call_timed_out = True
                try:
                    getattr(isabelle, "interrupt", lambda: None)()
                except Exception:
                    pass
                ex.shutdown(wait=False, cancel_futures=True)
                _timed_out_tmpdirs.append(tmpdir)
                tmpdir = None
                return []
            except Exception:
                ex.shutdown(wait=True)
                raise

        # No timeout requested → direct call
        return list(isabelle.use_theories(theories=["Scratch"], session_id=session_id, master_dir=tmpdir.name))
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()


def last_call_timed_out() -> bool:
    """Whether the most recent run_theory() call hit the wall-clock timeout."""
    return bool(_last_call_timed_out)


def finished_ok(resps: List[IsabelleResponse]) -> Tuple[bool, Dict[str, Any]]:
    """
    Return success if **any** FINISHED block reports ok=true (or result='ok').
    Robust across client variants:
      - response type can be Enum or str
      - response body may be bytes, JSON string, or dict
      - dict-like or attribute-style access
    """
    # If our wall-clock timeout fired, treat as NOT proved (prevents false positives).
    if last_call_timed_out():
        return False, {"timeout": True}

    any_ok = False
    last_obj: Dict[str, Any] = {}

    for r in (resps or []):
        if _normalize_type(_get_field(r, ("response_type", "type", "kind", "tag", "name"))) != "FINISHED":
            continue
        obj = _decode_body_to_dict(_get_field(r, ("response_body", "body", "message", "payload")))
        if not isinstance(obj, dict):
            continue
        last_obj = obj  # track last FINISHED
        # Some client variants can produce partial/unfinished results; keep the check strict.
        if bool(obj.get("ok", False)) or str(obj.get("result", "")).lower() == "ok":
            # If the FINISHED payload itself indicates a timeout, do not accept.
            if str(obj.get("timeout", "")).lower() in ("1", "true", "yes"):
                continue
            any_ok = True

    return any_ok, (last_obj or {})


def _clean_diagnostic(value: Any, *, limit: int = 1200) -> str:
    """Turn one Isabelle message payload into bounded, readable text."""

    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        try:
            text = json.dumps(value, ensure_ascii=False)
        except Exception:
            text = str(value)
    else:
        text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) > limit:
        text = text[: max(0, limit - 1)].rstrip() + "…"
    return text


def first_error_message(resps: List[IsabelleResponse]) -> str:
    """Return Isabelle's earliest useful failure diagnostic.

    Client versions disagree on whether messages live in NOTE responses, the
    FINISHED payload, or ``nodes[*].messages[*]``.  Search all three shapes,
    preferring messages explicitly tagged as errors and falling back to text
    that contains a conventional failure marker.  This function is purely
    diagnostic; :func:`finished_ok` remains the authoritative success check.
    """

    if last_call_timed_out():
        return "Isabelle call timed out"

    tagged: List[str] = []
    fallback: List[str] = []
    failure_words = re.compile(
        r"\b(?:error|failed|failure|exception|bad|illegal|timeout|unbound|unknown)\b",
        re.I,
    )

    def inspect_message(message: Any) -> None:
        if not isinstance(message, dict):
            text = _clean_diagnostic(message)
            if text and failure_words.search(text):
                fallback.append(text)
            return
        kind = str(message.get("kind", message.get("type", ""))).lower()
        value = (
            message.get("message")
            if message.get("message") is not None
            else message.get("error", message.get("body", message.get("text")))
        )
        text = _clean_diagnostic(value)
        if not text:
            return
        if kind in {"error", "fatal", "failed", "exception"}:
            tagged.append(text)
        elif failure_words.search(text):
            fallback.append(text)

    for response in (resps or []):
        raw = _get_field(response, ("response_body", "body", "message", "payload"))
        obj = _decode_body_to_dict(raw)
        if isinstance(obj, dict):
            inspect_message(obj)
            if obj.get("error") is not None:
                text = _clean_diagnostic(obj.get("error"))
                if text:
                    tagged.append(text)
            for node in (obj.get("nodes") or []):
                if not isinstance(node, dict):
                    continue
                for message in (node.get("messages") or []):
                    inspect_message(message)
        else:
            inspect_message(raw)

    return (tagged or fallback or [""])[0]


def last_print_state_block(resps: List[IsabelleResponse]) -> str:
    """
    Return a print_state-like goal block as a plain string.
    Backwards-compatible:
      1) Prefer the last NOTE/writeln 'goal' block if present (legacy behaviour).
      2) Otherwise, fall back to FINISHED JSON 'nodes[*].messages[*]' where many
         Isabelle client builds now place the pretty-printed goal text.
    """
    txt = ""
    # 1) Legacy NOTE/writeln path
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
    if txt:
        return txt
    # 2) FINISHED JSON fallback (authoritative on some builds)
    for r in (resps or []):
        if _normalize_type(_get_field(r, ("response_type", "type", "kind", "tag", "name"))) != "FINISHED":
            continue
        obj = _decode_body_to_dict(_get_field(r, ("response_body", "body", "message", "payload")))
        if not isinstance(obj, dict):
            continue
        for node in (obj.get("nodes") or []):
            for m in (node.get("messages") or []):
                if str(m.get("kind", "")).lower() != "writeln":
                    continue
                msg = str(m.get("message", "") or "")
                # Prefer explicit goal/subgoal blocks; also accept our ML markers
                if ("subgoal" in msg) or ("goal (" in msg) or ("goal\n" in msg) \
                   or msg.startswith("[LLM_SUBGOAL]"):
                    txt = msg or txt
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
    "finished_ok", "first_error_message", "last_print_state_block", "use_calls_count", "use_timeouts_count",
    "last_call_timed_out",
    "graceful_terminate", "shutdown_isabelle",
]

# Cross-runtime shutdown helper (works for subprocess.Popen and test doubles).
def graceful_terminate(proc, timeout_s: int = 3) -> None:
    """Stop and reap an Isabelle server process without leaking transports."""
    if proc is None:
        return
    try:
        if hasattr(proc, "poll") and proc.poll() is not None:
            return
        if hasattr(proc, "terminate"):
            try:
                proc.terminate()
            except Exception:
                pass
        if hasattr(proc, "wait"):
            try:
                proc.wait(timeout=timeout_s)
                return
            except Exception:
                try:
                    if hasattr(proc, "kill"):
                        proc.kill()
                    proc.wait(timeout=timeout_s)
                    return
                except Exception:
                    pass
        elif hasattr(proc, "join"):
            try:
                proc.join(timeout=timeout_s)
                if hasattr(proc, "kill") and getattr(proc, "is_alive", lambda: False)():
                    proc.kill()
            except Exception:
                pass
    finally:
        for stream_name in ("stdout", "stderr", "stdin"):
            stream = getattr(proc, stream_name, None)
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass


def shutdown_isabelle(client, proc, timeout_s: int = 3) -> None:
    """Shut down the protocol server, then wait for its owned process.

    All entry points use this one ordering.  A cooperative protocol shutdown
    is preferred; process termination is only the bounded fallback.
    """

    if client is not None:
        try:
            client.shutdown()
        except Exception:
            pass
    if proc is None:
        return
    try:
        if hasattr(proc, "wait"):
            proc.wait(timeout=timeout_s)
        elif hasattr(proc, "join"):
            proc.join(timeout=timeout_s)
    except Exception:
        graceful_terminate(proc, timeout_s=timeout_s)
        return
    finally:
        for stream_name in ("stdout", "stderr", "stdin"):
            stream = getattr(proc, stream_name, None)
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
