#!/usr/bin/env bash
# Kaggle Session 2 (≤12h): Robomimic training + metrics/rollouts + analysis.
set -euo pipefail
cd "$(dirname "$0")/.."

RM_TASKS=(Robomimic-Lift-PH Robomimic-Can-PH Robomimic-Square-PH Robomimic-Transport-PH)

echo "[S2] SA-2 training (Robomimic, both architectures)"
for arch in diffusion act; do
  for task in "${RM_TASKS[@]}"; do
    for seed in 0 1 2; do
      gpu=$(( seed % 2 ))
      CUDA_VISIBLE_DEVICES=$gpu python scripts/02_train.py --task "$task" --seed "$seed" \
        --arch "$arch" --device cuda &
      if (( $(jobs -r | wc -l) >= 2 )); then wait -n; fi
    done
  done
done
wait

echo "[S2] SA-3 metrics + SA-4 rollouts (all checkpoints, both archs, CPU)"
python scripts/03_metrics.py --all_archs --device cpu
python scripts/04_rollouts.py --all_archs --device cpu

echo "[S2] SA-5 analysis + figures"
python scripts/05_analysis.py
echo "[S2] done — see results/results.json"
