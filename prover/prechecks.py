# prover/prechecks.py
import json, re
from typing import List
from .isabelle_api import build_theory, run_theory

_QC_HIT = re.compile(r"(?i)(counterexample\s+found|Quickcheck\s+found\s+a\s+counterexample)")
_QC_TRYING = re.compile(r"(?i)Quickcheck")
_NP_HIT = re.compile(r"(?i)(Nitpick\s+found\s+a\s+counterexample|genuine\s+counterexample)")
_NP_TRYING = re.compile(r"(?i)Nitpick")

def precheck_quickcheck_refutes(isabelle, session_id: str, steps_with_candidate: List[str], timeout_s: int) -> bool:
    cmd = f"quickcheck[timeout = {int(timeout_s)}]"
    thy = build_theory(steps_with_candidate + [cmd], add_print_state=False, end_with="sorry")
    resps = run_theory(isabelle, session_id, thy)
    saw_qc = False
    for r in resps:
        if getattr(r, "response_type", "") != "NOTE": continue
        try:
            body = json.loads(r.response_body)
        except Exception:
            continue
        msg = str(body.get("message", "") or "")
        if _QC_TRYING.search(msg): saw_qc = True
        if _QC_HIT.search(msg): return True
    return False

def precheck_nitpick_refutes(isabelle, session_id: str, steps_with_candidate: List[str], timeout_s: int) -> bool:
    cmd = f"nitpick[timeout = {int(timeout_s)}]"
    thy = build_theory(steps_with_candidate + [cmd], add_print_state=False, end_with="sorry")
    resps = run_theory(isabelle, session_id, thy)
    saw_np = False
    for r in resps:
        if getattr(r, "response_type", "") != "NOTE": continue
        try:
            body = json.loads(r.response_body)
        except Exception:
            continue
        msg = str(body.get("message", "") or "")
        if _NP_TRYING.search(msg): saw_np = True
        if _NP_HIT.search(msg): return True
    return False
