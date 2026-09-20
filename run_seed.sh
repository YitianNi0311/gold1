#!/bin/bash
# usage: run_seed.sh SEED  -- runs all 8 models for one seed
cd ~/gold_nyt
PY=~/venvs/gold_nyt/bin/python
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 LOKY_MAX_CPU_COUNT=6
S=$1
mkdir -p seeded_runs
for m in random_walk random_forest xgboost lightgbm; do [ -f seeded_runs/seed$S/$m/run_manifest.json ] && grep -q completed seeded_runs/seed$S/$m/run_manifest.json && continue
  $PY run_seeded_baselines.py --model $m --seed $S --output seeded_runs/seed$S/$m >> seeded_runs/${m}_seed${S}_console.log 2>&1 || echo "FAILED $m $S" >> seeded_runs/failures.txt
done
for m in full no_fcnn without_fe; do
  $PY run_seeded_ensembles.py --rounds 1 --model $m --seed $S --output seeded_runs/seed$S/$m >> seeded_runs/${m}_seed${S}_console.log 2>&1 || echo "FAILED $m $S" >> seeded_runs/failures.txt
done
$PY reconstructed_cnn_lstm.py --seed $S --task both --output seeded_runs/seed$S/cnn_lstm >> seeded_runs/cnn_lstm_seed${S}_console.log 2>&1 || echo "FAILED cnn_lstm $S" >> seeded_runs/failures.txt
echo "SEED $S DONE $(date)" >> seeded_runs/progress.txt
