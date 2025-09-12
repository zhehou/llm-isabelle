# Isabellm: A playground for Free and Lightweight LLM-Guided Isabelle/HOL Provers

This repository implements an Isabelle/HOL theorem prover guided by Large Language Models (LLMs). It integrates Isabelle’s proof engine with modern LLMs (via **Ollama**, **Gemini CLI**, or **Hugging Face**).

Key features:
- Stepwise prover (in the prover folder)
  - LLM guesses tactics
  - Combined with nitpick, quickcheck, and sledgehammer
  - Beam search for suitable tactics
  - ML reranker for tactics trained by proof logs
- Isar-style proof outline generator (in the planner folder)
  - LLM guesses outline
  - Calls the stepwise prover to fill the details
  - Micro RAG extracted from AFP
---

## Table of Contents
- [1. Installation](#1-installation)
  - [1.1 Core prerequisites](#11-core-prerequisites)
  - [1.2 Python setup](#12-python-setup)
  - [1.3 Ollama (local LLMs)](#13-ollama-local-llms)
  - [1.4 Gemini CLI (hosted Gemini models)](#14-gemini-cli-hosted-gemini-models)
- [2. Configuration](#2-configuration)
  - [2.1 Model string prefixes](#21-model-string-prefixes)
  - [2.2 Environment variables](#22-environment-variables)
- [3. Usage](#3-usage)
  - [3.1 Quick demo](#31-quick-demo)
  - [3.2 Stepwise prover](#32-stepwise-prover)
  - [3.3 Training a reranker for the prover](#33-training-a-reranker-for-the-prover)
  - [3.4 Isar-style proof outline sketching](#34-isar-style-proof-outline-sketching)
  - [3.5 Planner data corpus and micro RAG](#35-planner-data-corpus-and-micro-rag)
  - [3.6 Isabelle/jEdit GUI integration](#36-isabellejedit-gui-integration)
  - [3.6 Evaluation using mini-F2F](#37-evaluation-using-mini-f2f)
- [4. Project Structure](#5-project-structure)
- [5. Notes & Tips](#6-notes--tips)
- [6. License](#7-license)

---

## 1. Installation

### 1.1 Core prerequisites
- **Python** 3.10+
- **Isabelle/HOL** (tested with Isabelle2025). Ensure `isabelle` is on your `$PATH`.
- **System packages**: GNU Make, `g++`, etc. (used by Isabelle and some Python libs).

### 1.2 Python setup
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
# If you plan to use CPU-only PyTorch wheels:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 1.3 Ollama (local LLMs)
Install and run Ollama: https://ollama.ai

```bash
# Start the server (if not already running)
ollama serve &
# Pull a couple of useful models
ollama pull qwen3-coder:30b
ollama pull gemma3:27b
ollama pull deepseek-r1:8b
```

### 1.4 Gemini CLI (hosted Gemini models)
We support **Gemini via the official `gemini` CLI**.

```bash
# Install Gemini CLI following the instructions here: https://www.geminicli.cc/docs/installation
# Then install the following (pick one)
pipx install gemini-cli    # preferred
# or
pip install gemini-cli

# Get a free Gemini API key here: https://aistudio.google.com/app/apikey
# One-time setup (stores your key locally) 
gemini setup
# Or export GEMINI_API_KEY
export GEMINI_API_KEY=xxxxxxxx
# Quick sanity check
gemini -m gemini-2.5-pro -p "hello"
```

**Recommended model IDs**: `gemini-2.5-pro` (reasoning) and `gemini-2.5-flash` (fast).  
Some experimental models (e.g. `gemini-2.0-pro-exp`) are **versioned** and may not be enabled on every account. If you see NOT_FOUND, try a stable model first.

---

## 2. Configuration

### 2.1 Model string prefixes
We route backends via **prefixes**:

- **Ollama** (local):  
  - `"qwen3-coder:30b"` (no prefix → treated as Ollama for back-compat), or  
  - `"ollama:qwen3-coder-30b"` (explicit prefix).
- **Gemini CLI**: `"gemini:gemini-2.5-pro"` (or another model available to your CLI).
- **Hugging Face**: `"hf:<repo-id>"`, e.g. `"hf:meta-llama/Llama-3.1-8B-Instruct"`.

> If a model call fails, the prover can still produce results by falling back to heuristics/ATP tools (e.g., Sledgehammer). Use the debug tools below to confirm which backend actually ran.

### 2.2 Environment variables
Core:
- `OLLAMA_MODEL` – default model used when requests don’t specify one (e.g. `qwen3-coder:30b`).
- `OLLAMA_HOST` – base URL for Ollama (default `http://127.0.0.1:11434`).

Sampling/timeouts (Ollama & shared defaults):
- `OLLAMA_TEMP`, `OLLAMA_TOP_P`, `OLLAMA_TIMEOUT_S`, `OLLAMA_NUM_PREDICT`.

Gemini:
- `GEMINI_API_KEY` (only needed for REST fallback; the **CLI** uses its own stored key or this env var).
- `GEMINI_CLI_BIN` – path to the `gemini` binary if not on `PATH`.

Hugging Face:
- `HF_API_TOKEN` or `HUGGINGFACEHUB_API_TOKEN` – enables hosted Inference API.
- `HF_API_BASE` – override the Inference API base URL.
- `HF_MODE=local` – forces local `transformers` instead of the hosted API.

Debugging:
- `LLM_DEBUG=1` – verbose backend routing + error logs from `prover/llm.py`.

---

## 3. Usage

### 3.1 Quick demo
```bash
python -m prover.cli
# Default goal: rev (rev xs) = xs
```

### 3.2 Stepwise prover
**Ollama (local):**
```bash
python -m prover.cli --goal "map f (xs @ ys) = map f xs @ map f ys" \
  --model "qwen3-coder:30b"
```

**Gemini CLI (hosted):**
```bash
python -m prover.cli --goal 'rev (rev xs) = xs' \
  --model 'gemini:gemini-2.5-pro'
```

**Hugging Face (hosted API):**
```bash
export HF_API_TOKEN=hf_xxx
python -m prover.cli --goal 'rev (rev xs) = xs' \
  --model 'hf:meta-llama/Llama-3.1-8B-Instruct'
```

**Hugging Face (local Transformers):**
```bash
pip install transformers accelerate
export HF_MODE=local
python -m prover.cli --goal 'rev (rev xs) = xs' \
  --model 'hf:meta-llama/Llama-3.1-8B-Instruct'
```

More knobs:
```bash
python -m prover.cli --goal 'rev (rev xs) = xs' \
  --model 'qwen3-coder:30b' --beam 3 --max-depth 8 --timeout 20 \
  --sledge --quickcheck --nitpick --facts-limit 6 --variants
```

**Advanced features for the prover**
Prove multiple goals from a file
```bash
python -m prover.cli --goals-file datasets/lists.txt \
  --model "gemini:gemini-2.5-pro"
```

Benchmarking
```bash
python -m prover.experiments bench --suite lists
# Results → datasets/results/
```

Regression testing
Create baseline:
```bash
python -m prover.experiments regress --suite lists \
  --save-baseline datasets/baselines/lists.json
```
Compare to baseline:
```bash
python -m prover.experiments regress --suite lists \
  --baseline datasets/baselines/lists.json \
  --model "hf:meta-llama/Llama-3.1-8B-Instruct"
```

Aggregating results
```bash
python -m prover.experiments aggregate --best-only --top-k 2
```

### 3.3 Training a reranker for the prover
Supervised (sklearn / XGBoost):
```bash
python -m prover.train_reranker --algo sklearn-logreg --target bandit
python -m prover.train_reranker --algo xgb-classifier --target bandit
```
RL-style Q-estimation:
```bash
python -m prover.train_reranker --algo xgb-regressor --target q --attempts logs/attempts.log.jsonl --runs logs/runs.log.jsonl
```

AWR++ (with teacher & listwise):
```bash
# Train a teacher model first
python -m prover.train_reranker --algo xgb-ranker
# Then train an AWR++ with knowledge distilled from the teacher
python -m prover.train_reranker --algo awr --tau 0.6 --epochs 8 --batch 1024 --listwise_norm --teacher_w 0.3 --teacher auto
python -m prover.train_reranker --algo dqn --epochs 12 --batch 2048 --gamma 0.92 --target_update 500
```

Deep Q Network:
```bash
python -m prover.train_reranker --algo dqn --epochs 12 --batch 2048 --gamma 0.92 --target_update 500
```

Combining the above in curriculum training
```bash
# Create data for easy stage
python -m prover.experiments bench --file datasets/hol_main_easy_goals.txt --beam 3 --max-depth 6 --timeout 120 --facts-limit 6 --quickcheck --nitpick --reranker on --variants --no-minimize --model "qwen3-coder:30b" --shuffle
# Train a bandit classifier
python -m prover.train_reranker --algo xgb-classifier --target bandit

# Create data for mid stage
python -m prover.experiments bench --file datasets/hol_main_mid_goals.txt --beam 4 --max-depth 8 --timeout 120 --facts-limit 6 --quickcheck --nitpick --reranker on --sledge --variants --no-minimize --model "qwen3-coder:30b" --shuffle
# Re-train bandit classifier
python -m prover.train_reranker --algo xgb-classifier --target bandit
# And also train a Q-style regressor
python -m prover.train_reranker --algo xgb-regressor --target q

# Create data for hard stage
python -m prover.experiments bench --file datasets/hol_main_hard_goals.txt --beam 5 --max-depth 10 --timeout 200 --facts-limit 8 --quickcheck --nitpick --reranker on --sledge --variants --no-minimize --model "qwen3-coder:30b" --shuffle
# Retrain Q-style regressor
python -m prover.train_reranker --algo xgb-regressor --target q
# Train AWR++ with teacher knowledge distillation
python -m prover.train_reranker --algo awr --tau 0.6 --epochs 8 --batch 1024 --listwise_norm --teacher_w 0.3 --teacher auto
# Also train DQN
python -m prover.train_reranker --algo dqn --epochs 12 --batch 2048 --gamma 0.92 --target_update 500
# See which reranker works better.
```

### 3.4 Isar-style proof outline sketching
Run the planner to sketch a proof (fill the proof if possible). Internally, it proposes multiple outlines and picks the most suitable one to output.
```bash
python -m planner.cli --timeout 60 --mode auto "rev (rev xs) = xs"
```

Sketch an outline only
```bash
python -m planner.cli --timeout 60 --mode outline "map f (xs @ ys) = map f xs @ map f ys"
```

Controlling the divserity of multiple outlines
```bash
python -m planner.cli --timeout 60 --diverse-outlines --k 3 --temps "0.35,0.55,0.85" --mode auto "map f (xs @ ys) = map f xs @ map f ys"
```

Enable proof outline repair (by LLM)
```bash
python -m planner.cli --model "gemini:gemini-2.5-flash" \                   
  --timeout 120 --mode outline --repairs --repair-trace \
  "map f (xs @ ys) = map f xs @ map f ys"
```

Benchmarking a file of lemmas/proof goals
```bash
python -m planner.experiments bench \  
  --file datasets/lists.txt \                                 
  --mode auto --diverse --k 3 --temps "0.35,0.55,0.85" \
  --timeout 120 --strict-no-sorry --verify \
  --model "qwen3-coder:30b" --shuffle --seed 42
```

Regression testing and save a baseline
```bash
python -m planner.experiments regress \
  --file datasets/lists.txt \                                 
  --mode auto --diverse --k 3 --timeout 120 \
  --strict-no-sorry --verify \
  --save-baseline datasets/baselines/planner_lists.json
```

Regression testing against previously saved baseline
```bash
python -m planner.experiments regress \
  --file datasets/lists.txt --model "gemini:gemini-2.5-flash"\
  --mode auto --diverse --k 3 --timeout 120 \
  --strict-no-sorry --verify \
  --baseline datasets/baselines/planner_lists.json \
  --tol-rate 0.00 --tol-time 2.0
```

python -m planner.experiments regress \
  --file datasets/lists.txt --model "gemini:gemini-2.5-flash"\
  --mode auto --diverse --k 3 --timeout 120 \
  --strict-no-sorry --verify \
  --baseline datasets/baselines/planner_lists.json \
  --tol-rate 0.00 --tol-time 2.0

### 3.5 Planner data corpus and micro RAG

Extract a data corpus for the planner from AFP (replace the path to afp thys with a valid path)
```bash
python - <<'PY'
from planner.extract import mine_afp_corpus_rich
mine_afp_corpus_rich(src_dir="/path/to/afp/thys", out_jsonl="datasets/isar_pairs_rich.jsonl")
PY
```

Aggregate priors, generate a micro RAG (hint lexicon) from AFP.
```bash
python -m planner.priors \
  --input datasets/isar_pairs_rich.jsonl \
  --priors data/isar_priors.json \
  --hintlex data/isar_hintlex.json \
  --min-count 3 --topk 8
```

Run the planner with the new knowledge (alpha (default 1.0): weight on subgoals (keep dominant), beta (default 0.5): weight on pattern penalty, gamma (default 0.2): reward for using recommended hints)
```bash
python -m planner.cli --goal 'map f (xs @ ys) = map f xs @ map f ys' \
  --context-hints \
  --priors data/isar_priors.json \
  --hintlex data/isar_hintlex.json \
  --alpha 1.0 --beta 0.6 --gamma 0.25 \
  --lib-templates
```

Benchmarking a file of lemmas/proof goals with the micro RAG
```bash
python -m planner.experiments bench \  
  --file datasets/lists.txt \                                 
  --mode auto --diverse --k 3 --temps "0.35,0.55,0.85" \
  --timeout 120 --strict-no-sorry --verify \
  --context-hints --hintlex datasets/isar_hintlex.json --priors datasets/isar_priors.json \
  --model "qwen3-coder:30b" --shuffle --seed 42
```

### 3.6 Isabelle/jEdit GUI integration
Run the HTTP server:
```bash
python3 -m isabelle_ui.server
```
Copy the `.bsh` macros from `isabelle_ui/` to your jEdit macros folder, e.g.
- macOS/Linux: `~/.isabelle/Isabelle2025/jedit/macros/LLM_Prover`
- Windows: `C:\Users\<You>\.isabelle\Isabelle2025\jedit\macros\LLM_Prover`

Then in jEdit, use **Macros → LLM Prover** at a proof state.

### 3.7 Evaluation using mini-F2F
Download the dataset
```bash
git clone --depth=1 https://github.com/facebookresearch/miniF2F.git external/miniF2F  
```

Process the dataste for Isabelle/HOL
```bash
python datasets/prep_minif2f_isabelle.py \
  --repo external/miniF2F \
  --outdir datasets/mini_f2f
# mini_f2f_validation.txt and mini_f2f_test.txt are the ones to use.
```

Build Isabelle session to include necessary imports (need ROOT and MiniF2F_Base.thy in datasets/mini_f2f)
```bash
# Make sure you have already registered AFP entries to Isabelle/HOL
isabelle build -d datasets/mini_f2f -v MiniF2F_Base
export ISABELLE_LOGIC=MiniF2F_Base
```

Run the prover on the validation datasets
```bash
# Validation 
ISABELLE_LOGIC=MiniF2F_Base python -m prover.experiments bench --file datasets/mini_f2f/mini_f2f_validation.txt --beam 5 --max-depth 10 --timeout 200 --facts-limit 8 --quickcheck --nitpick --reranker on --sledge --variants --no-minimize --model "qwen3-coder:30b" --shuffle

# Testing
ISABELLE_LOGIC=MiniF2F_Base python -m prover.experiments bench --file datasets/mini_f2f/mini_f2f_test.txt --beam 5 --max-depth 10 --timeout 200 --facts-limit 8 --quickcheck --nitpick --reranker on --sledge --variants --no-minimize --model "qwen3-coder:30b" --shuffle
```

Maybe train rerankers using on the logs from the validation set, and then run the test set to see results.

---

## 4. Project Structure
```
datasets/          # Datasets and results
isabelle_ui/       # Isabelle/jEdit integration (HTTP server + macros)
logs/              # Logs data for training rerankers
planner/           # Proof outline planner (supports Ollama, Gemini CLI, HF)
prover/            # Step prover (supports Ollama, Gemini CLI, HF) + reranker
```

---

## 5. Notes & Tips
- Isabelle server is started once and reused; scripts manage lifecycle.
- Proof minimization is on by default; disable with `--no-minimize` when debugging.
- Ensembles (`--models`) improve robustness but cost more RAM/time.
- Retrain the reranker periodically; RL modes benefit from richer logs.
- If a run “works” without an LLM (because of fallbacks), **`LLM_DEBUG=1`** will make that visible.

---

## 6. License
MIT License.
