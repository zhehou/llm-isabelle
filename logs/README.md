The logs below

runs.log-hol-all.jsonl
attempts.log-hol-all.jsonl

are generated from synthetic datasets that only require HOL built-in lirabry for proving. They can be used to train baseline rerankers. Recommended training commands:

Train a basic reranker using XGBoost

python -m prover.train_reranker \
  --algo xgb-classifier --target bandit \
  --n_estimators 600 --max_depth 7 --eta 0.07 \
  --min_rows 500 \
  --attempts logs/attempts.log-hol-all.jsonl --runs logs/runs.log-hol-all.jsonl 

Train an AWR++ model with the basic reranker as teacher

python -m prover.train_reranker \
  --algo awr \
  --epochs 12 --batch 1024 --lr 8e-4 \
  --tau 0.55 --listwise_norm \
  --teacher auto --teacher_w 0.25 \
  --val_split 0.1 --min_rows 1000 \
  --attempts logs/attempts.log-hol-all.jsonl --runs logs/runs.log-hol-all.jsonl