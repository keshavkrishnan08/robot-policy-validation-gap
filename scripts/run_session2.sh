#!/usr/bin/env bash
# Kaggle Session 2 (≤12h): Robomimic training + metrics/rollouts + analysis.
set -euo pipefail
cd "$(dirname "$0")/.."

# GPU (EGL) offscreen rendering; without it MuJoCo may fall back to slow CPU osmesa.
export MUJOCO_GL=egl

RM_TASKS=(Robomimic-Lift-PH Robomimic-Can-PH Robomimic-Square-PH Robomimic-Transport-PH)
SEEDS="${SEEDS:-0 1 2}"   # 3 seeds (n=48) on dual T4; set "0 1" (n=32) for single T4

NGPU=$(python3 -c "import torch; print(max(torch.cuda.device_count(),1))")
echo "[S2] torch sees $NGPU GPU(s) -- expect 2 for T4 x2"

gpu=0
launch() {
  CUDA_VISIBLE_DEVICES=$gpu python scripts/02_train.py "$@" --device cuda &
  gpu=$(( (gpu + 1) % NGPU ))
  while (( $(jobs -r | wc -l) >= NGPU )); do wait -n; done
}

# Per-task interleave: download THIS task's data -> train -> eval -> delete data + checkpoints.
# Bounds peak disk to one task instead of the whole benchmark.
for task in "${RM_TASKS[@]}"; do
  echo "[S2] === task $task: download data ==="
  python scripts/download_task.py "$task"
  python scripts/make_eval_conditions.py --tasks "$task"
  echo "[S2] === task $task: train (across $NGPU GPU) ==="
  for arch in diffusion act; do
    for seed in $SEEDS; do
      launch --task "$task" --seed "$seed" --arch "$arch"
    done
  done
  wait
  echo "[S2] === task $task: metrics + rollouts (on T4), then free disk ==="
  python scripts/03_metrics.py  --tasks "$task" --all_archs --device cuda
  python scripts/04_rollouts.py --tasks "$task" --all_archs --device cuda
  rm -rf "checkpoints/$task"
  python scripts/download_task.py "$task" --clean
done

echo "[S2] SA-5 analysis + figures"
python scripts/05_analysis.py
echo "[S2] done — see results/results.json"
