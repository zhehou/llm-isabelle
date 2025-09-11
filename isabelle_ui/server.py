from __future__ import annotations

import os
import atexit
from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel

from prover.isabelle_api import start_isabelle_server, get_isabelle_client
from prover.prover import prove_goal
from prover.config import MODEL as DEFAULT_MODEL
from planner.driver import plan_and_fill
from prover.llm import detect_backend_for_model

app = FastAPI(title="LLM Isabelle Prover API")

# ------------------------------------------------------------
# Built-in defaults (can be overridden by env or request JSON)
# ------------------------------------------------------------
DEFAULT_PRIORS_PATH = "datasets/isar_priors.json"
DEFAULT_HINTLEX_PATH = "datasets/isar_hintlex.json"
DEFAULT_HINTLEX_TOP = 8
DEFAULT_CONTEXT_HINTS = True
DEFAULT_LIB_TEMPLATES = True
DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 0.6
DEFAULT_GAMMA = 0.25
DEFAULT_DIVERSE = True
DEFAULT_K = 3

print(
    "[server] defaults:",
    f"priors={DEFAULT_PRIORS_PATH}",
    f"hintlex={DEFAULT_HINTLEX_PATH}",
    f"hintlex_top={DEFAULT_HINTLEX_TOP}",
    f"context_hints={DEFAULT_CONTEXT_HINTS}",
    f"lib_templates={DEFAULT_LIB_TEMPLATES}",
    f"alpha/beta/gamma={DEFAULT_ALPHA}/{DEFAULT_BETA}/{DEFAULT_GAMMA}",
    f"diverse={DEFAULT_DIVERSE} k={DEFAULT_K}",
)

print(f"[server] LLM_DEBUG={'ON' if os.getenv('LLM_DEBUG') not in (None, '', '0', 'false', 'False', 'no', 'NO') else 'OFF'}")

# ----------------------------
# Isabelle: start once, reuse
# ----------------------------
server_info, proc = start_isabelle_server(name="isabelle", log_file="ui_server.log")
isabelle = get_isabelle_client(server_info)
SESSION = isabelle.session_start(session="HOL")

@atexit.register
def _shutdown():
    try:
        isabelle.shutdown()
    except Exception:
        pass
    try:
        proc.terminate(); proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill(); proc.wait(timeout=2)
        except Exception:
            pass

# ----------------------------
# Small helpers
# ----------------------------
def _env_flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1","true","yes","on")

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, f"{default}").strip())
    except Exception:
        return default

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, f"{default}").strip())
    except Exception:
        return default

def _env_str(name: str, default: Optional[str]) -> Optional[str]:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip()

# ----------------------------
# /prove
# ----------------------------
class ProveReq(BaseModel):
    goal: str
    model: Optional[str] = None
    timeout: int = 60
    beam: int = 4
    max_depth: int = 8
    facts_limit: int = 6
    sledge: bool = True
    sledge_timeout: int = 20
    sledge_every: int = 2
    quickcheck: bool = True
    qc_timeout: int = 2
    qc_every: int = 1
    nitpick: bool = True
    np_timeout: int = 5
    np_every: int = 2
    variants: bool = True
    minimize: bool = True

class ProveResp(BaseModel):
    success: bool
    depth: int
    timeout: bool
    model: str
    backend: str
    steps: list[str]

@app.post("/prove", response_model=ProveResp)
def prove(req: ProveReq):
    model = req.model or DEFAULT_MODEL
    backend = detect_backend_for_model(model)
    print(f"[prove]   model={model} backend={backend} goal={req.goal[:80]}")

    res = prove_goal(
        isabelle, SESSION, req.goal,
        model_name_or_ensemble=model,
        beam_w=req.beam, max_depth=req.max_depth, hint_lemmas=6, timeout=req.timeout,
        models=None, save_dir=None,
        use_sledge=req.sledge, sledge_timeout=req.sledge_timeout, sledge_every=req.sledge_every,
        trace=False, use_color=False,
        use_qc=req.quickcheck, qc_timeout=req.qc_timeout, qc_every=req.qc_every,
        use_np=req.nitpick, np_timeout=req.np_timeout, np_every=req.np_every,
        facts_limit=req.facts_limit,
        do_minimize=req.minimize, minimize_timeout=8,
        do_variants=req.variants, variant_timeout=6, variant_tries=24,
        enable_reranker=True,
    )
    raw_steps = [str(s) for s in res.get("steps", [])]
    tactic_steps = [s for s in raw_steps if s.strip().startswith(("apply", "by "))]
    return ProveResp(
        success=bool(res.get("success", False)),
        depth=int(res.get("depth", -1)),
        timeout=bool(res.get("timeout", False)),
        model=str(res.get("model", model)),
        backend=backend,
        steps=tactic_steps or raw_steps,
    )

# ----------------------------
# /plan_fill
# ----------------------------
class PlanFillReq(BaseModel):
    goal: str
    model: Optional[str] = None
    timeout: int = 100
    mode: str = "auto"              # "auto" or "outline"

    # Diverse outlines
    diverse: Optional[bool] = None  # None → env/default
    k: Optional[int] = None
    temps: Optional[List[float]] = None  # e.g., [0.35, 0.55, 0.85]

    # Repairs
    repairs: Optional[bool] = None
    max_repairs_per_hole: Optional[int] = None
    repair_trace: Optional[bool] = None

    # Context / micro-RAG / priors
    context_hints: Optional[bool] = None
    lib_templates: Optional[bool] = None
    priors: Optional[str] = None          # path to isar_priors.json
    hintlex: Optional[str] = None         # path to isar_hintlex.json
    hintlex_top: Optional[int] = None     # max hints per token

    # Scoring weights
    alpha: Optional[float] = None
    beta: Optional[float] = None
    gamma: Optional[float] = None

class PlanFillResp(BaseModel):
    success: bool
    outline: str
    fills: list[str]
    failed_holes: list[int]
    model: str
    backend: str

@app.post("/plan_fill", response_model=PlanFillResp)
def plan_fill(req: PlanFillReq):
    model = req.model or DEFAULT_MODEL
    backend = detect_backend_for_model(model)

    # Diversity
    diverse = req.diverse if req.diverse is not None else _env_flag("PLANNER_DIVERSE", DEFAULT_DIVERSE)
    k = req.k if req.k is not None else _env_int("PLANNER_K", DEFAULT_K)
    temps = req.temps or None
    legacy_single_outline = not diverse

    # Repairs (keep previous defaults; env override if you set them)
    repairs = req.repairs if req.repairs is not None else _env_flag("PLANNER_REPAIRS", True)
    max_repairs = req.max_repairs_per_hole if req.max_repairs_per_hole is not None else _env_int("PLANNER_MAX_REPAIRS_PER_HOLE", 2)
    repair_trace = req.repair_trace if req.repair_trace is not None else _env_flag("PLANNER_REPAIR_TRACE", False)

    # Context / priors / micro-RAG
    priors_path   = req.priors  if req.priors  is not None else _env_str("PLANNER_PRIORS",   DEFAULT_PRIORS_PATH)
    hintlex_path  = req.hintlex if req.hintlex is not None else _env_str("PLANNER_HINTLEX",  DEFAULT_HINTLEX_PATH)
    hintlex_top   = req.hintlex_top if req.hintlex_top is not None else _env_int("PLANNER_HINTLEX_TOP", DEFAULT_HINTLEX_TOP)
    context_hints = req.context_hints if req.context_hints is not None else _env_flag("PLANNER_CONTEXT_HINTS", DEFAULT_CONTEXT_HINTS)
    lib_templates = req.lib_templates if req.lib_templates is not None else _env_flag("PLANNER_LIB_TEMPLATES", DEFAULT_LIB_TEMPLATES)

    # Scoring weights
    alpha = req.alpha if req.alpha is not None else _env_float("PLANNER_ALPHA", DEFAULT_ALPHA)
    beta  = req.beta  if req.beta  is not None else _env_float("PLANNER_BETA",  DEFAULT_BETA)
    gamma = req.gamma if req.gamma is not None else _env_float("PLANNER_GAMMA", DEFAULT_GAMMA)

    print(
        "[plan_fill]",
        f"model={model} backend={backend} mode={req.mode} diverse={diverse} k={k} temps={temps or 'default'}",
        f"context_hints={context_hints} priors={'on' if priors_path else 'off'}",
        f"hintlex={'on' if hintlex_path else 'off'} lib_templates={lib_templates}",
        f"alpha/beta/gamma={alpha}/{beta}/{gamma}",
        f"goal={req.goal[:80]}",
    )

    r = plan_and_fill(
        goal=req.goal,
        model=model,
        timeout=req.timeout,
        mode=req.mode,
        # diversity
        outline_k=(k if diverse else 1),
        outline_temps=temps,
        legacy_single_outline=legacy_single_outline,
        # repairs
        repairs=repairs,
        max_repairs_per_hole=max_repairs,
        repair_trace=repair_trace,
        # priors & micro-RAG
        priors_path=priors_path,
        context_hints=context_hints,
        lib_templates=lib_templates,
        alpha=alpha, beta=beta, gamma=gamma,
        hintlex_path=hintlex_path,
        hintlex_top=hintlex_top,
    )

    return PlanFillResp(
        success=bool(r.success),
        outline=str(r.outline),
        fills=[str(x) for x in r.fills],
        failed_holes=[int(i) for i in r.failed_holes],
        model=model,
        backend=backend,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5005)
