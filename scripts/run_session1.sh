#!/usr/bin/env bash
# Kaggle Session 1 (≤12h): setup + LIBERO training + LIBERO metrics/rollouts.
# Runs the 4 LIBERO tasks (12 of the 18 runs) on dual T4, then evaluates them.
set -euo pipefail
cd "$(dirname "$0")/.."

LIBERO_TASKS=(LIBERO-Spatial-1 LIBERO-Object-1 LIBERO-Goal-1 LIBERO-Long-1)
# 3 seeds (n=48) if both T4s activate; if you must run on a single T4, set SEEDS="0 1" (n=32).
SEEDS="${SEEDS:-0 1 2}"

# --- GPU detection: a single Python process only uses cuda:0. To use BOTH T4s we launch one
# --- process per GPU pinned via CUDA_VISIBLE_DEVICES. Confirm the count below is 2 for T4 x2;
# --- if it prints 1 you selected the wrong accelerator (Settings -> Accelerator -> GPU T4 x2).
NGPU=$(python3 -c "import torch; print(max(torch.cuda.device_count(),1))")
echo "[S1] torch sees $NGPU GPU(s) -- if you chose T4 x2 and this says 1, the 2nd did not activate"

gpu=0
launch() {  # pin each job to the next GPU; cap concurrency at NGPU
  CUDA_VISIBLE_DEVICES=$gpu python scripts/02_train.py "$@" --device cuda &
  gpu=$(( (gpu + 1) % NGPU ))
  while (( $(jobs -r | wc -l) >= NGPU )); do wait -n; done
}

echo "[S1] SA-1 setup + download"
python scripts/01_setup.py --download
python scripts/make_eval_conditions.py

# Per-task interleave with checkpoint pruning keeps peak storage to one task (~5GB) << 20GB cap.
for task in "${LIBERO_TASKS[@]}"; do
  echo "[S1] === task $task: train (across $NGPU GPU) ==="
  for arch in diffusion act; do
    for seed in $SEEDS; do
      launch --task "$task" --seed "$seed" --arch "$arch"
    done
  done
  wait   # finish this task's training before evaluating
  echo "[S1] === task $task: metrics + rollouts (on T4), then prune ==="
  python scripts/03_metrics.py  --tasks "$task" --all_archs --device cuda
  python scripts/04_rollouts.py --tasks "$task" --all_archs --device cuda
  rm -rf "checkpoints/$task"
done
echo "[S1] done — results/metrics.csv + results/rollouts.csv hold all LIBERO rows"
