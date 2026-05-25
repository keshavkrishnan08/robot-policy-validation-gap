#!/usr/bin/env bash
# Kaggle Session 1 (≤12h): setup + LIBERO training + LIBERO metrics/rollouts.
# Runs the 4 LIBERO tasks (12 of the 18 runs) on dual T4, then evaluates them.
set -euo pipefail
cd "$(dirname "$0")/.."

LIBERO_TASKS=(LIBERO-Spatial-1 LIBERO-Object-1 LIBERO-Goal-1 LIBERO-Long-1)

echo "[S1] SA-1 setup + download"
python scripts/01_setup.py --download
python scripts/make_eval_conditions.py

# Per-task interleave: train -> metrics -> rollouts -> PRUNE checkpoints.
# Keeps peak storage to one task (~5GB) instead of all 144 LIBERO checkpoints (~14GB),
# staying well under Kaggle's 20GB output cap alongside the ~5GB of demos.
for task in "${LIBERO_TASKS[@]}"; do
  echo "[S1] === task $task: train both architectures ==="
  for arch in diffusion act; do
    for seed in 0 1 2; do
      gpu=$(( seed % 2 ))
      CUDA_VISIBLE_DEVICES=$gpu python scripts/02_train.py --task "$task" --seed "$seed" \
        --arch "$arch" --device cuda &
      if (( $(jobs -r | wc -l) >= 2 )); then wait -n; fi
    done
  done
  wait
  echo "[S1] === task $task: metrics + rollouts, then prune ==="
  python scripts/03_metrics.py  --tasks "$task" --all_archs --device cpu
  python scripts/04_rollouts.py --tasks "$task" --all_archs --device cpu
  rm -rf "checkpoints/$task"   # metric/rollout JSONs are kept; raw weights no longer needed
done
echo "[S1] done — results/metrics.csv + results/rollouts.csv hold all LIBERO rows"
