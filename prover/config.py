# prover/config.py
import os

# Ollama
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-coder:30b")
NUM_CANDIDATES = int(os.environ.get("OLLAMA_NUM_CANDIDATES", "6"))
TEMP = float(os.environ.get("OLLAMA_TEMP", "0.2"))
TOP_P = float(os.environ.get("OLLAMA_TOP_P", "0.95"))
TIMEOUT_S = int(os.environ.get("OLLAMA_TIMEOUT_S", "60"))

# Search
BEAM_WIDTH = int(os.environ.get("BEAM_WIDTH", "3"))
MAX_DEPTH  = int(os.environ.get("MAX_DEPTH", "8"))

# Logging
LOG_DIR = os.environ.get("LOG_DIR", ".")
RUNS_LOG = os.path.join(LOG_DIR, "runs.log.jsonl")
ATTEMPTS_LOG = os.path.join(LOG_DIR, "attempts.log.jsonl")
BATCH_JSONL = os.path.join(LOG_DIR, "batch_results.jsonl")
BATCH_CSV   = os.path.join(LOG_DIR, "batch_results.csv")

# Limits
HINT_LEMMAS = int(os.environ.get("HINT_LEMMAS", "6"))
FACTS_LIMIT = int(os.environ.get("FACTS_LIMIT", "6"))

# Minimizer
MINIMIZE_DEFAULT = True
MINIMIZE_TIMEOUT = int(os.environ.get("MINIMIZE_TIMEOUT", "8"))
MINIMIZE_MAX_FACT_TRIES = int(os.environ.get("MINIMIZE_MAX_FACT_TRIES", "40"))

# Variants
VARIANTS_DEFAULT = True
VARIANT_TIMEOUT  = int(os.environ.get("VARIANT_TIMEOUT", "6"))
VARIANT_TRIES    = int(os.environ.get("VARIANT_TRIES", "24"))
