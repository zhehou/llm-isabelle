LLM-Guided Isabelle/HOL Prover

This repository implements a stepwise Isabelle/HOL theorem prover guided by Large Language Models (LLMs).
It integrates Isabelle’s proof engine with modern LLMs (via Ollama) and optional ML rerankers.

Core features include:

Stepwise proof search with beam search (prover/prover.py).

LLM proposal generation (prover/llm.py) with fact augmentation and heuristic reranking.

Optional ML reranker (prover/ranker.py) trained from proof attempt logs.

Auxiliary tools: Sledgehammer integration, lemma mining, quickcheck/nitpick prefilters, proof minimization, structured variants.

Experiment harnesses: benchmarking (prover/bench.py), regression testing (prover/regress.py), result aggregation (prover/aggregate.py).

Command-line interface (prover/cli.py) for proving single goals, goal files, or quick demos.

1. Installation

Prerequisites

Python 3.10+

Isabelle/HOL (tested with Isabelle2024+).
Ensure the isabelle binary is on your $PATH.

Ollama running locally for LLM inference.
Install from ollama.ai
 and pull desired models, e.g.:

ollama pull qwen3-coder:30b

ollama pull gemma3:27b

ollama pull gpt-oss:20b

ollama pull deepseek-r1:8b


System packages: GNU Make, g++, etc. (for Isabelle and Python libs to build).

Python dependencies

From the repo root, run:

python3 -m venv .venv

source .venv/bin/activate

pip install -U pip

pip install -r requirements.txt


2. Configuration

Configuration defaults are defined in prover/config.py
.

You can override runtime settings with environment variables:

OLLAMA_MODEL – default model if not passed on CLI (e.g. qwen3-coder:30b).

OLLAMA_TEMP – sampling temperature.

OLLAMA_TOP_P – nucleus sampling probability.

OLLAMA_TIMEOUT_S – per-request timeout.

RERANKER_OFF=1 – disable reranker (even if model file exists).

3. Usage

3.1 Quick demo

Prove the standard reverse lemma:

python -m prover.cli


This will default to proving rev (rev xs) = xs using the configured model.

3.2 Prove a single goal

python -m prover.cli --goal "map f (xs @ ys) = map f xs @ map f ys" --model "qwen3-coder:30b"

With more options

python -m prover.cli --goal 'rev (rev xs) = xs' --model 'qwen3-coder:30b' --beam 3 --max-depth 8 --budget-s 20 --sledge --quickcheck --nitpick --facts-limit 6 --variants

3.3 Prove multiple goals from a file

Goals should be listed in a text file (benchmarks/*.txt format). For example:

python -m prover.cli --goals-file benchmarks/lists.txt --models "qwen3-coder:30b,llama3.1:8b-instruct"

3.4 Benchmarking

Run a full benchmark suite:

python -m prover.bench --suite lists


Compare sledge on/off:

python -m prover.bench --suite sets --sledge both --budget-s 20


All results are written as CSV under benchmarks/results/.

3.5 Regression testing

Save a baseline on the lists suite:

python -m prover.regress --suite lists --save-baseline benchmarks/baselines/lists.json


Later runs compare to the baseline and exit nonzero on regression:

python -m prover.regress --suite lists --baseline benchmarks/baselines/lists.json

3.6 Aggregating results

Summarize all results under benchmarks/results/:

python -m prover.aggregate


With top-k configs only:

python -m prover.aggregate --best-only --top-k 2

3.7 Training a reranker

From proof attempt logs train a regression model reranker

python -m prover.train_reranker_sklearn

python -m prover.train_reranker_xgb

This produces a saved model under models/, automatically loaded by the runtime reranker.

From proof attempts and general logs train a reinforcement learning state-action agent that uses the reranker

python -m prover.train_qranker_xgb --mode bandit \
  --attempts logs/attempts.log.jsonl \
  --runs logs/runs.log.jsonl \
  --min_rows 200

3.8 Integration with Isabelle/HOL Jedit GUI

Keep an HTTP server running in a terminal window.

python3 -m isabelle_ui.serve

Linux and Mac users copy the .bsh files in 

llm-isabelle/isabelle_ui/

to (Create the folder if it doesn't exist)

~/.isabelle/Isabelle2025/jedit/macros/LLM_Prover

Windows users may need to put them under the user profile directory, e.g.,

C:\Users\<YourName>\.isabelle\Isabelle2025\jedit\macros\LLM_Prover

Open Isabelle/HOL jEdit GUI, and run the tools via Macros -> LLM Prover at a proof state.

3.9 Training dataset generation

As a curriculum, first generate easy datasets from built-in HOL library.

python datasets/hol_extract_goals.py \
  --isabelle-hol /Applications/Isabelle2025.app/src/HOL \
  --out datasets

Partition the datasets based on topic.

python datasets/hol_route_by_imports.py \
  --in datasets/hol_goals.jsonl \
  --out datasets

Run the prover per topic to collect data.

python -m prover.regress --file datasets/hol_main.txt --beam 3 --max-depth 2 --budget-s 30 --facts-limit 6 --quickcheck --sledge --no-minimize

python -m prover.regress --file datasets/hol_sets_lists.txt --beam 3 --max-depth 2 --budget-s 30 --facts-limit 6 --quickcheck --sledge --no-minimize

EXTRA_IMPORTS="Number_Theory" \
python -m prover.regress --file datasets/hol_number_theory.txt --beam 3 --max-depth 2 --budget-s 30 --facts-limit 6 --quickcheck --sledge --no-minimize

EXTRA_IMPORTS="Complex_Main" \
python -m prover.regress --file datasets/hol_complex.txt --beam 3 --max-depth 2 --budget-s 30 --facts-limit 6 --quickcheck --sledge --no-minimize

EXTRA_IMPORTS="Groups Rings Fields Vector_Spaces" \
python -m prover.regress --file datasets/hol_algebra.txt --beam 3 --max-depth 2 --budget-s 30 --facts-limit 6 --quickcheck --sledge --no-minimize

4. Project Structure
README.md
benchmarks/              # Benchmark goal suites and results
isabelle_ui/             # Isabelle/HOL jEdit integration
planner/                 # Proof outline planner
prover/                  # Stepwise prover

5. Notes & Tips

Start Isabelle/HOL once per run; the harness scripts (bench.py, regress.py) manage this automatically.

Macros are automatically mined from past runs to speed up common proof patterns.

Proof minimization is on by default; disable with --no-minimize to debug raw proof search.

Use ensembles (--models) for better robustness across goal suites, but it requires very large RAM.

The reranker improves success rates modestly; retrain it regularly with new logs.

6. License

MIT License.
