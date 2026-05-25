#!/usr/bin/env bash
# Kaggle Session 2 (≤12h): Robomimic training + metrics/rollouts + analysis.
set -euo pipefail
cd "$(dirname "$0")/.."

RM_TASKS=(Robomimic-Lift-PH Robomimic-Can-PH Robomimic-Square-PH Robomimic-Transport-PH)

# Per-task interleave with pruning (same storage discipline as Session 1).
for task in "${RM_TASKS[@]}"; do
  echo "[S2] === task $task: train both architectures ==="
  for arch in diffusion act; do
    for seed in 0 1 2; do
      gpu=$(( seed % 2 ))
      CUDA_VISIBLE_DEVICES=$gpu python scripts/02_train.py --task "$task" --seed "$seed" \
        --arch "$arch" --device cuda &
      if (( $(jobs -r | wc -l) >= 2 )); then wait -n; fi
    done
  done
  wait
  echo "[S2] === task $task: metrics + rollouts, then prune ==="
  python scripts/03_metrics.py  --tasks "$task" --all_archs --device cpu
  python scripts/04_rollouts.py --tasks "$task" --all_archs --device cpu
  rm -rf "checkpoints/$task"
done

echo "[S2] SA-5 analysis + figures"
python scripts/05_analysis.py
echo "[S2] done — see results/results.json"
