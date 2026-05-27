#!/usr/bin/env bash
# Kaggle Session 1 (≤12h): setup + LIBERO training + LIBERO metrics/rollouts.
# Runs the 4 LIBERO tasks (12 of the 18 runs) on dual T4, then evaluates them.
set -euo pipefail
cd "$(dirname "$0")/.."

# Render MuJoCo offscreen on the GPU (EGL). Without this MuJoCo can fall back to CPU osmesa,
# which makes the 20-rollout-per-checkpoint eval crawl. Confirmed working in check_install.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

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

echo "[S1] SA-1 prelock check (data is downloaded per task below, NOT in bulk)"
python scripts/01_setup.py    # verify pre-lock only; NO --download (that pulls the whole benchmark)

# Per-task interleave: download THIS task's data -> train -> eval -> delete data + checkpoints.
# Bounds peak disk to one task (a few GB) instead of the whole benchmark (100s of GB).
for task in "${LIBERO_TASKS[@]}"; do
  echo "[S1] === task $task: download data ==="
  python scripts/download_task.py "$task"
  python scripts/make_eval_conditions.py --tasks "$task"
  echo "[S1] === task $task: train (across $NGPU GPU) ==="
  for arch in diffusion act; do
    for seed in $SEEDS; do
      launch --task "$task" --seed "$seed" --arch "$arch"
    done
  done
  wait   # finish this task's training before evaluating
  echo "[S1] === task $task: metrics + rollouts (on T4), then free disk ==="
  python scripts/03_metrics.py  --tasks "$task" --all_archs --device cuda
  python scripts/04_rollouts.py --tasks "$task" --all_archs --device cuda
  rm -rf "checkpoints/$task"
  python scripts/download_task.py "$task" --clean   # delete this task's demos
done
echo "[S1] done — results/metrics.csv + results/rollouts.csv hold all LIBERO rows"
