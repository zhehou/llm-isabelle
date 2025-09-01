# prover/logging_utils.py
import os, json, time, uuid
from typing import List, Optional, Dict, Any
from .config import (
    ATTEMPTS_LOG, RUNS_LOG,
    BEAM_WIDTH, MAX_DEPTH, NUM_CANDIDATES, TEMP, TOP_P,
)

def write_jsonl(path: str, obj: dict) -> None:
    """
    Append a JSON object to a JSONL file, creating parent dirs if needed.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

class RunLogger:
    """
    Per-run logger that writes both attempt-level rows (expand/finish/etc.)
    and a final run summary row. Intentionally minimal to avoid coupling.
    """
    def __init__(self, goal: str, model_name: str):
        self.run_id = str(uuid.uuid4())
        self.goal = goal
        self.model = model_name
        self.start_ts = time.time()
        self.elapsed_s: float = 0.0
        self.success: Optional[bool] = None
        self.final_steps: List[str] = []
        self.depth_reached: int = 0
        self.use_calls: int = 0
        # Best-effort tracking to populate subgoals_before if caller doesn't pass it
        self._last_known_subgoals: Optional[int] = None

    def log_attempt(
        self,
        kind: str,
        prefix_steps: List[str],
        candidate: str,
        ok: bool,
        n_subgoals: Optional[int],
        cache_hit: bool,
        elapsed_ms: float,
        depth: int,
        subgoals_before: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a single attempt row to ATTEMPTS_LOG.
        - kind: "expand", "expand_macro", "finish", etc.
        - prefix_steps: steps before applying candidate (we store as 'prefix')
        - candidate: the step/finisher being tried
        - ok: whether applying candidate succeeded (reduced goals / closed)
        - n_subgoals: number of subgoals AFTER applying candidate (None if unknown or failure)
        - cache_hit: whether the evaluation came from cache
        - elapsed_ms: time to evaluate the candidate
        - depth: current search depth (1-based suggested by caller)
        - subgoals_before: number of subgoals BEFORE applying candidate (optional)
        - extra: optional dict to splice into the row (future-proof)
        """
        if subgoals_before is None:
            subgoals_before = self._last_known_subgoals
        row = {
            "run_id": self.run_id,
            "ts": time.time(),
            "model": self.model,
            "goal": self.goal,
            "type": kind,
            "depth": int(depth),
            "prefix_len": len(prefix_steps or []),
            "prefix": list(prefix_steps or []),
            "candidate": candidate,
            "ok": bool(ok),
            "n_subgoals": n_subgoals if (n_subgoals is None or isinstance(n_subgoals, int)) else None,
            "subgoals_before": subgoals_before if (subgoals_before is None or isinstance(subgoals_before, int)) else None,
            "subgoals_after": n_subgoals if (n_subgoals is None or isinstance(n_subgoals, int)) else None,
            "cache_hit": bool(cache_hit),
            "elapsed_ms": round(float(elapsed_ms or 0.0), 1),
        }
        if extra and isinstance(extra, dict):
            # merge without overwriting core keys
            for k, v in extra.items():
                if k not in row:
                    row[k] = v
        write_jsonl(ATTEMPTS_LOG, row)
        # Update our last-known subgoal count if the attempt produced a valid number
        if isinstance(n_subgoals, int):
            self._last_known_subgoals = n_subgoals

    def finish(self, success: bool, final_steps: List[str], depth_reached: int, use_calls: int) -> None:
        """
        Log end-of-run summary to RUNS_LOG.
        """
        self.success = bool(success)
        self.final_steps = list(final_steps or [])
        self.depth_reached = int(depth_reached or 0)
        self.use_calls = int(use_calls or 0)
        self.elapsed_s = time.time() - self.start_ts
        write_jsonl(RUNS_LOG, {
            "run_id": self.run_id,
            "ts": time.time(),
            "model": self.model,
            "goal": self.goal,
            "success": self.success,
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