#!/usr/bin/env bash
# Kaggle Session 1 (≤12h): setup + LIBERO training + LIBERO metrics/rollouts.
# Runs the 4 LIBERO tasks (12 of the 18 runs) on dual T4, then evaluates them.
set -euo pipefail
cd "$(dirname "$0")/.."

LIBERO_TASKS=(LIBERO-Spatial-1 LIBERO-Object-1 LIBERO-Goal-1 LIBERO-Long-1)

echo "[S1] SA-1 setup + download"
python scripts/01_setup.py --download
python scripts/make_eval_conditions.py

echo "[S1] SA-2 training (LIBERO, both architectures) — two jobs in parallel per GPU"
for arch in diffusion act; do
  for task in "${LIBERO_TASKS[@]}"; do
    for seed in 0 1 2; do
      gpu=$(( seed % 2 ))
      CUDA_VISIBLE_DEVICES=$gpu python scripts/02_train.py --task "$task" --seed "$seed" \
        --arch "$arch" --device cuda &
      if (( $(jobs -r | wc -l) >= 2 )); then wait -n; fi
    done
  done
done
wait

echo "[S1] SA-3 metrics + SA-4 rollouts (LIBERO checkpoints, both archs, CPU)"
python scripts/03_metrics.py --all_archs --device cpu
python scripts/04_rollouts.py --all_archs --device cpu
echo "[S1] done"
