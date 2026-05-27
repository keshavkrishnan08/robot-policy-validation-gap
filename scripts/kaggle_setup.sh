#!/usr/bin/env bash
# Install the LIBERO + Robomimic simulation stack on Kaggle.
#
# IMPORTANT: LIBERO is NOT a usable PyPI package ("pip install libero" gives an unrelated 0.1.x).
# It installs from GitHub source. LIBERO historically targets Python 3.8 / torch 1.11 / numpy 1.x,
# while current Kaggle is Python 3.12 / torch 2.10 / numpy 2 -- so this is the fragile step.
#
# RUN THIS ALONE FIRST (it ends with an import check) before committing a training session.
# If the import check fails, do NOT proceed to training -- iterate the recipe (see notes at bottom).
set -e

echo "[setup] base: $(python -c 'import sys;print(sys.version.split()[0])'), torch $(python -c 'import torch;print(torch.__version__)' 2>/dev/null || echo none)"

# robosuite / mujoco / LIBERO need numpy 1.x; pin it before anything else pulls numpy 2.
pip install -q "numpy<2"

# --- LIBERO from source (NOT pip) ---
cd /kaggle/working
[ -d LIBERO ] || git clone -q https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
# install LIBERO's deps but do NOT let it pin torch back to 1.11 (keep Kaggle's CUDA torch)
grep -v -i '^torch' requirements.txt > /tmp/libero_reqs.txt || cp requirements.txt /tmp/libero_reqs.txt
pip install -q -r /tmp/libero_reqs.txt || echo "[setup] WARN: some LIBERO reqs failed; continuing"
pip install -q -e .
cd /kaggle/working

# --- Robomimic + Robosuite + MuJoCo (modern line: robomimic 0.4 + robosuite 1.5) ---
pip install -q "mujoco>=3.1" "robosuite>=1.4" robomimic h5py opencv-python-headless || \
  echo "[setup] WARN: robomimic/robosuite install hit an issue"

# re-pin numpy<2 in case a dep bumped it
pip install -q "numpy<2"

# LIBERO prompts interactively for dataset paths on first use ("specify a custom path? (Y/N)"),
# which hangs in a notebook. Answer "N" (use defaults) non-interactively to write ~/.libero/config.yaml.
echo "[setup] initializing LIBERO path config non-interactively"
yes "N" | python -c "from libero.libero import get_libero_path; get_libero_path('bddl_files')" >/dev/null 2>&1 || true
python -c "import os;p=os.path.expanduser('~/.libero/config.yaml');print('[setup] LIBERO config '+('created' if os.path.exists(p) else 'MISSING (will still prompt)'))"

echo "[setup] === import check (must all succeed) ==="
python - <<'PY'
import importlib, sys
ok = True
for m in ["numpy", "torch", "mujoco", "robosuite", "robomimic", "libero"]:
    try:
        mod = importlib.import_module(m)
        print(f"  OK  {m:10s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        ok = False; print(f"  FAIL {m:10s} {type(e).__name__}: {e}")
import numpy as np, torch
print(f"  numpy {np.__version__} (want <2), torch cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}")
sys.exit(0 if ok else 1)
PY
echo "[setup] import check passed -- safe to proceed to setup/training"

# --- If the import check FAILS, the likely fixes (iterate ONE at a time, cheap):
#   1) headless GL for robosuite:   apt-get install -y libgl1-mesa-glx libosmesa6  (Kaggle: !apt-get)
#      and  export MUJOCO_GL=egl  (or osmesa)
#   2) numpy conflicts: a dep may force numpy 2 -> rerun  pip install "numpy<2"  last.
#   3) Python 3.12 too new for LIBERO: fall back to the LeRobot LIBERO integration
#      (pip install lerobot ; huggingface/lerobot-libero) which targets modern Python, OR
#      install miniconda + a python=3.10 env and run everything inside it.
