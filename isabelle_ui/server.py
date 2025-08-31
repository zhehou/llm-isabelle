# prover/httpd.py
from __future__ import annotations

import atexit
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from prover.isabelle_api import start_isabelle_server, get_isabelle_client
from prover.prover import prove_goal
from prover.config import MODEL as DEFAULT_MODEL

from planner.driver import plan_and_fill

app = FastAPI(title="LLM Isabelle Prover API")

# Start Isabelle once and reuse the session
server_info, proc = start_isabelle_server(name="isabelle", log_file="ui_server.log")
isabelle = get_isabelle_client(server_info)
SESSION = isabelle.session_start(session="HOL")


@atexit.register
def _shutdown():
    try: isabelle.shutdown()
    except Exception: pass
    try:
        proc.terminate(); proc.wait(timeout=2)
    except Exception:
        try: proc.kill(); proc.wait(timeout=2)
        except Exception: pass


# ---------- /prove ----------

class ProveReq(BaseModel):
    goal: str
    model: Optional[str] = None
    budget_s: int = 8
    beam: int = 2
    max_depth: int = 8
    facts_limit: int = 6
    sledge: bool = True
    sledge_timeout: int = 5
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
    steps: list[str]

@app.post("/prove", response_model=ProveResp)
def prove(req: ProveReq):
    model = req.model or DEFAULT_MODEL
    res = prove_goal(
        isabelle, SESSION, req.goal,
        model_name_or_ensemble=model,
        beam_w=req.beam, max_depth=req.max_depth, hint_lemmas=6, budget_s=req.budget_s,
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
        steps=tactic_steps or raw_steps,
    )


# ---------- /plan_fill ----------

class PlanFillReq(BaseModel):
    goal: str
    model: Optional[str] = None
    budget_s: int = 10
    mode: str = "auto"    # "auto" (allow complete) or "outline" (force placeholders)

class PlanFillResp(BaseModel):
    success: bool
    outline: str
    fills: list[str]
    failed_holes: list[int]

@app.post("/plan_fill", response_model=PlanFillResp)
def plan_fill(req: PlanFillReq):
    model = req.model or DEFAULT_MODEL
    r = plan_and_fill(goal=req.goal, model=model, budget_s=req.budget_s, mode=req.mode)
    return PlanFillResp(
        success=bool(r.success),
        outline=str(r.outline),
        fills=[str(x) for x in r.fills],
        failed_holes=[int(i) for i in r.failed_holes],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5005)
