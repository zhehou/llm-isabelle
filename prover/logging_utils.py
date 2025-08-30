# prover/logging_utils.py
import os, json, time, uuid
from typing import List, Optional, Dict, Any
from .config import (
    ATTEMPTS_LOG, RUNS_LOG,
    BEAM_WIDTH, MAX_DEPTH, NUM_CANDIDATES, TEMP, TOP_P,
)

def write_jsonl(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

class RunLogger:
    def __init__(self, goal: str, model_name: str):
        self.run_id = str(uuid.uuid4())
        self.goal = goal
        self.model = model_name
        self.start_ts = time.time()
        self.use_calls = 0
        self.success: Optional[bool] = None
        self.final_steps: List[str] = []
        self.depth_reached = 0
        self.elapsed_s = 0.0

    def log_attempt(self, step_type: str, steps_prefix: List[str], candidate: str,
                    ok: bool, n_sub: Optional[int], cache_hit: bool,
                    elapsed_ms: float, depth: int):
        write_jsonl(ATTEMPTS_LOG, {
            "run_id": self.run_id, "ts": time.time(), "model": self.model,
            "goal": self.goal, "type": step_type, "depth": depth,
            "prefix_len": len(steps_prefix), "prefix": steps_prefix,
            "candidate": candidate, "ok": ok, "n_subgoals": n_sub,
            "cache_hit": cache_hit, "elapsed_ms": round(elapsed_ms, 1),
        })

    def finish(self, success: bool, final_steps: List[str], depth_reached: int, use_calls: int):
        self.success = success
        self.final_steps = final_steps
        self.depth_reached = depth_reached
        self.use_calls = use_calls
        self.elapsed_s = time.time() - self.start_ts
        write_jsonl(RUNS_LOG, {
            "run_id": self.run_id, "ts": time.time(), "model": self.model,
            "goal": self.goal, "success": self.success,
            "depth_reached": self.depth_reached,
            "elapsed_s": round(self.elapsed_s, 2),
            "final_steps_len": len(self.final_steps),
            "final_steps": self.final_steps,
            "use_theories_calls": self.use_calls,
            # Config snapshot for analysis/aggregation
            "beam_width": BEAM_WIDTH,
            "max_depth": MAX_DEPTH,
            "num_candidates": NUM_CANDIDATES,
            "temp": TEMP,
            "top_p": TOP_P,
        })
