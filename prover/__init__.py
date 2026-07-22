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

# Lazy convenience re-exports.  Importing ``prover.config`` or
# ``prover.isabelle_api`` should not eagerly load the LLM, reranker and optional
# ML dependencies.  ``from prover import prove_goal`` remains compatible.
def __getattr__(name):
    if name == "prove_goal":
        from .prover import prove_goal
        return prove_goal
    if name in {"start_isabelle_server", "get_isabelle_client"}:
        from .isabelle_api import get_isabelle_client, start_isabelle_server
        return {
            "start_isabelle_server": start_isabelle_server,
            "get_isabelle_client": get_isabelle_client,
        }[name]
    raise AttributeError(name)
