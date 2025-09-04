# prover/utils.py
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from typing import Optional, List, Dict, Any

# ANSI helpers
ANSI = {
    "reset": "\x1b[0m",
    "bold": "\x1b[1m",
    "dim": "\x1b[2m",
    "green": "\x1b[32m",
    "red": "\x1b[31m",
    "yellow": "\x1b[33m",
    "blue": "\x1b[34m",
    "cyan": "\x1b[36m",
    "gray": "\x1b[90m",
}

def color(use_color: bool, key: str, s: str) -> str:
    return (ANSI.get(key, "") + s + ANSI["reset"]) if use_color else s  # :contentReference[oaicite:12]{index=12}

# subgoal parsing
SUBGOALS_PATTERNS = [
    re.compile(r"\b(\d+)\s+subgoals?\b", re.IGNORECASE),
    re.compile(r"(?i)\bgoal\s*\(\s*(\d+)\s+subgoals?\s*\)"),
]

def parse_subgoals(block: str) -> Optional[int]:
    for pat in SUBGOALS_PATTERNS:
        m = pat.search(block)
        if m:
            return int(m.group(1))
    return None  # :contentReference[oaicite:13]{index=13}

def state_fingerprint(s: str) -> str:
    """Hash a normalized print_state block to detect revisits."""
    s = " ".join(s.strip().split())
    return hashlib.sha1(s.encode("utf-8")).hexdigest()  # :contentReference[oaicite:14]{index=14}

def slugify_goal(goal: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_]+", "_", goal).strip("_")
    import hashlib
    h = hashlib.sha1(goal.encode("utf-8")).hexdigest()[:8]
    return f"{base[:50]}_{h}" if base else h

def write_theory_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# ---------------------------
# Logging (moved from logging_utils.py)
# ---------------------------
from .config import (
    ATTEMPTS_LOG, RUNS_LOG,
    BEAM_WIDTH, MAX_DEPTH, NUM_CANDIDATES,
    TEMP, TOP_P,
)

def write_jsonl(path: str, obj: dict) -> None:
    """Append a JSON object to a JSONL file, creating parent dirs if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")  # :contentReference[oaicite:15]{index=15}

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
        self._last_known_subgoals: Optional[int] = None  # best-effort cache  # :contentReference[oaicite:16]{index=16}

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
            for k, v in extra.items():
                if k not in row:
                    row[k] = v
        write_jsonl(ATTEMPTS_LOG, row)
        if isinstance(n_subgoals, int):
            self._last_known_subgoals = n_subgoals  # :contentReference[oaicite:17]{index=17}

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
        })  # :contentReference[oaicite:18]{index=18}
