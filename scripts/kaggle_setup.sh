#!/usr/bin/env bash
# Install the LIBERO + Robomimic simulation stack on Kaggle (Python 3.12 base).
#
# Key compatibility facts (learned the hard way):
#  - LIBERO is NOT a usable PyPI package; install from GitHub source.
#  - LIBERO needs robosuite==1.4.0 (1.5.x removed robosuite.environments.manipulation.single_arm_env).
#  - LIBERO's requirements.txt pins bddl==1.0.1 which fails to build on Python 3.12, and over-pins
#    numpy/opencv/etc. -> we do NOT use it; we install LIBERO --no-deps and control the deps here.
#  - The sim stack needs numpy<2 (re-pinned last, since deps try to pull numpy 2).
#
# RUN THIS ALONE FIRST; it ends with an import check. Do not train until it passes.
set -e
REPO_ROOT="$(pwd)"   # capture before any cd (CORL repo root)

echo "[setup] base: $(python -c 'import sys;print(sys.version.split()[0])'), torch $(python -c 'import torch;print(torch.__version__)' 2>/dev/null || echo none)"
pip install -q "numpy<2"

# --- LIBERO from source, package only (we provide deps ourselves) ---
cd /kaggle/working
[ -d LIBERO ] || git clone -q https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -q -e LIBERO --no-deps
cd "$REPO_ROOT"

# --- LIBERO-compatible sim deps (pinned to what robosuite 1.4.0 / LIBERO actually need) ---
#  mujoco 3.1.6 has py3.12 wheels and works with robosuite 1.4.0 (which requires mujoco>=2.3).
#  bddl: latest (1.0.1 fails to build on 3.12); LIBERO's bundled task files parse with it.
pip install -q \
  "robosuite==1.4.0" \
  "mujoco==3.1.6" \
  bddl \
  robomimic \
  h5py \
  opencv-python-headless \
  easydict thop einops "gym==0.25.2" future cloudpickle "hydra-core>=1.2" || \
  echo "[setup] WARN: a sim dep failed; see above"

pip install -q "numpy<2"   # re-pin last (robosuite/robomimic may bump numpy)

# --- write ~/.libero/config.yaml directly so LIBERO never enters its interactive setup ---
echo "[setup] writing LIBERO config directly (no prompt)"
python -c "import sys; sys.path.insert(0, '$REPO_ROOT'); from drc.envs import _ensure_libero_config; _ensure_libero_config()"
python -c "import os;p=os.path.expanduser('~/.libero/config.yaml');print('[setup] LIBERO config '+('written' if os.path.exists(p) else 'MISSING'))"

echo "[setup] === import + version check ==="
python - <<'PY'
import importlib, sys
ok = True
want = {"robosuite": "1.4.0"}   # LIBERO needs exactly 1.4.0
for m in ["numpy", "torch", "mujoco", "robosuite", "robomimic", "libero", "bddl"]:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        warn = ""
        if m in want and v != want[m]:
            warn = f"  <-- WARNING: LIBERO needs {want[m]}"
            ok = False
        print(f"  OK  {m:10s} {v}{warn}")
    except Exception as e:
        ok = False; print(f"  FAIL {m:10s} {type(e).__name__}: {e}")
import numpy as np, torch
if int(np.__version__.split('.')[0]) >= 2:
    ok = False; print(f"  numpy {np.__version__} >=2 -> rerun: pip install 'numpy<2'")
print(f"  numpy {np.__version__}, torch cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}")
sys.exit(0 if ok else 1)
PY
echo "[setup] import check passed. Now run scripts/check_install.py (builds a real LIBERO env)."

# --- If the LIBERO env build still fails after this:
#   * robosuite API error  -> robosuite is not 1.4.0; force it: pip install --force-reinstall --no-deps robosuite==1.4.0
#   * mujoco/rendering error-> export MUJOCO_GL=osmesa ; apt-get install -y libgl1-mesa-glx libosmesa6
#   * persistent py3.12 conflicts -> fall back to the LeRobot LIBERO integration (modern Python).
