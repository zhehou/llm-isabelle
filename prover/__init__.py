# prover/__init__.py

__version__ = "0.2.0"

# Public submodules (importable as `prover.experiments`, etc.)
__all__ = [
    "prover", "cli", "experiments",
    "tactics", "minimize", "premises", "ranker",
    "context", "config", "isabelle_api", "llm",
    # training utilities
    "train_premises", "train_reranker",
]

# Convenience re-exports (lightweight, common top-level uses)
from .prover import prove_goal                       # high-level API :contentReference[oaicite:1]{index=1}
from .isabelle_api import start_isabelle_server, get_isabelle_client  # infra helpers :contentReference[oaicite:2]{index=2}
