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
        ISABELLE_USE_THEORIES_TIMEOUT_S = 0  # 0 = disabled

def _header(imports=None):
    imps = ["Main"] + list(imports or []) + list(EXTRA_IMPORTS or [])
    return f"theory Scratch\nimports {' '.join(imps)}\nbegin\n"

FOOTER = "end\n"

_use_calls = 0

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
                    return []

        # No timeout requested → direct call
        return list(isabelle.use_theories(theories=["Scratch"], session_id=session_id, master_dir=tmpdir.name))
    finally:
        tmpdir.cleanup()

def finished_ok(resps: List[IsabelleResponse]) -> Tuple[bool, Dict[str, Any]]:
    """
    Return success if **any** FINISHED block reports ok=true.
    (Previously we only checked the *last* FINISHED, which is order-fragile across platforms.)
    """
    any_ok = False
    last_obj: Dict[str, Any] = {}
    for r in (resps or []):
        if getattr(r, "response_type", "") != "FINISHED":
            continue
        try:
            obj = json.loads(r.response_body)
        except Exception:
            continue
        last_obj = obj  # keep the most recent FINISHED payload for diagnostics
        if bool(obj.get("ok", False)):
            any_ok = True
            # keep scanning to preserve 'last_obj' but success is already locked in
    return any_ok, (last_obj or {})

def last_print_state_block(resps: List[IsabelleResponse]) -> str:
    txt = ""
    for r in (resps or []):
        if getattr(r, "response_type", "") != "NOTE":
            continue
        try:
            body = json.loads(r.response_body)
        except Exception:
            continue
        if body.get("kind") == "writeln":
            msg = str(body.get("message", ""))
            if ("subgoal" in msg) or ("goal (" in msg) or ("goal\n" in msg):
                txt = msg
    return txt

def use_calls_count() -> int:
    return _use_calls