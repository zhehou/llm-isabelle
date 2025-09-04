LLM-Guided Isabelle/HOL Prover

This repository implements a stepwise Isabelle/HOL theorem prover guided by Large Language Models (LLMs). It integrates Isabelle’s proof engine with modern LLMs (via Ollama) and an optional machine-learned reranker.

Core Features

Stepwise proof search with beam search (prover/prover.py).

LLM proposal generation (prover/llm.py) with fact mining, heuristic scoring, and reranker integration.

Optional ML reranker (prover/ranker.py) trained from logs of past proof attempts.

Auxiliary tactics (prover/tactics.py):

Quickcheck/Nitpick pre-filters

Sledgehammer finisher suggestions

Lemma/fact mining from Isabelle state

Structured proof variants (induction/cases)

Auto-mined macros for frequent step continuations

Experiment harness (prover/experiments.py):

Benchmarking (bench)

Regression testing (regress)

Result aggregation (aggregate)

Command-line interface (prover/cli.py) for single goals, goal files, or quick demos.

1. Installation

Prerequisites

Python 3.10+

Isabelle/HOL (tested with Isabelle2024+). Ensure isabelle is on your $PATH.

Ollama running locally for LLM inference:
https://ollama.ai

Example model pulls:

ollama pull qwen3-coder:30b

ollama pull gemma3:27b

ollama pull gpt-oss:20b

ollama pull deepseek-r1:8b


System packages: GNU Make, g++, etc. (needed by Isabelle and Python libs)

Python dependencies

python3 -m venv .venv

source .venv/bin/activate

pip install -U pip

pip install -r requirements.txt

pip install torch --index-url https://download.pytorch.org/whl/cpu

2. Configuration

Defaults are in prover/config.py.
Runtime overrides via environment variables:

OLLAMA_MODEL – default LLM model (e.g. qwen3-coder:30b)

OLLAMA_TEMP, OLLAMA_TOP_P – sampling hyperparameters

OLLAMA_TIMEOUT_S – per-request timeout

RERANKER_OFF=1 – disable reranker even if a model exists

RERANKER_DIR – directory to load/save reranker model (default .models/)

3. Usage

3.1 Quick demo

python -m prover.cli

(Default goal: rev (rev xs) = xs)

3.2 Prove a single goal

python -m prover.cli --goal "map f (xs @ ys) = map f xs @ map f ys" --model "qwen3-coder:30b"

With more options:

python -m prover.cli --goal 'rev (rev xs) = xs' \
  --model 'qwen3-coder:30b' --beam 3 --max-depth 8 --budget-s 20 \
  --sledge --quickcheck --nitpick --facts-limit 6 --variants

3.3 Multiple goals from a file

python -m prover.cli --goals-file benchmarks/lists.txt \
  --model "qwen3-coder:30b"

3.4 Benchmarking

python -m prover.experiments bench --suite lists

All results are saved as CSV under benchmarks/results/.

3.5 Regression testing

Test and save baseline

python -m prover.experiments regress --suite lists \
  --save-baseline benchmarks/baselines/lists.json

Test and compare baseline

python -m prover.experiments regress --suite lists \
  --baseline benchmarks/baselines/lists.json

3.6 Aggregating results

python -m prover.experiments aggregate --best-only --top-k 2

3.7 Training a reranker

Supervised rerankers (sklearn / XGBoost)

These models treat reranking as binary classification:
given features of a candidate step (depth, subgoal count, step type, etc.), predict probability of success.

Examples:

Logistic regression

python -m prover.train_reranker --algo sklearn-logreg --target bandit

XGBoost classifier

python -m prover.train_reranker --algo xgb-classifier --target bandit

RL-style reranker (Q-estimation)

This variant uses reinforcement learning signals:
step/state pairs are labeled with discounted success values, and an XGBoost regressor is trained to approximate Q-values. At runtime, it exposes a predict_proba interface just like the classifiers.

Example:

python -m prover.train_reranker --algo xgb-regressor --target q \
  --attempts logs/attempts.log.jsonl --runs logs/runs.log.jsonl

To train a heavier Deep RL model, run

python -m prover.train_deeprl --algo awr \
  --attempts logs/attempts.log.jsonl \
  --runs     logs/runs.log.jsonl \
  --epochs 8 --batch 1024 --lr 1e-3 --val_split 0.1 --tau 0.6

τ (tau): start 0.5–0.8. Lower = peakier weights (trust top actions more); higher = smoother.

Batch: 512–2048 is fine on CPU.

Val split: keep 0.1 so you can spot overfit; stop if val loss turns up.

At runtime, prover/ranker.py just loads whatever model is present and feeds it into the search. Prefers Deep RL model over sklearn and xgboost models.

Advanced training:

3.7.1 AWR++ (with teacher & listwise):

Train teacher first

python -m prover.train_reranker --algo xgb-ranker

Then train AWR++ using knowledge distilled from teacher

python -m prover.train_deeprl --algo awr --tau 0.6 --epochs 8 --batch 1024 --listwise_norm --teacher_w 0.3

3.7.2 DQN (safe, offline, uses your logged next states)

python -m prover.train_deeprl --algo dqn --epochs 12 --batch 2048 --gamma 0.92 --target_update 500

3.8 Integration with Isabelle/HOL Jedit GUI

Keep an HTTP server running in a terminal window.

python3 -m isabelle_ui.server

Linux and Mac users copy the .bsh files in

llm-isabelle/isabelle_ui/

to (Create the folder if it doesn't exist)

~/.isabelle/Isabelle2025/jedit/macros/LLM_Prover

Windows users may need to put them under the user profile directory, e.g.,

C:\Users<YourName>.isabelle\Isabelle2025\jedit\macros\LLM_Prover

Open Isabelle/HOL jEdit GUI, and run the tools via Macros -> LLM Prover at a proof state.

4. Project Structure

benchmarks/        # Goal suites and CSV results

isabelle_ui/       # Isabelle/jEdit integration macros

planner/           # Proof outline planner 

prover/            # Step prover package

5. Notes & Tips

Isabelle server is started once per benchmark/regression run; scripts manage lifecycle automatically.

Macros are auto-mined from past runs to accelerate recurring proof patterns.

Proof minimization is on by default; disable with --no-minimize when debugging.

Use ensemble models (--models) for robustness, but they consume more RAM.

Retrain the reranker periodically; the RL mode benefits from longer runs and richer logs.

6. License

MIT License.