#!/usr/bin/env python3
"""Validate the simulation stack AFTER kaggle_setup.sh, BEFORE training.

Two levels:
  1. imports + versions (numpy<2, torch CUDA, mujoco/robosuite/robomimic/libero)
  2. functional: build ONE real LIBERO env (catches the common headless-GL / MUJOCO_GL failure)

Exit 0 == safe to train. Exit 1 == fix the install first (do NOT spend a training session).
"""
import importlib
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Headless rendering for MuJoCo on Kaggle (no display). egl is usual; osmesa is the fallback.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

problems = []

print("=== 1. imports + versions ===")
for m in ["numpy", "torch", "mujoco", "robosuite", "robomimic", "libero"]:
    try:
        mod = importlib.import_module(m)
        print(f"  OK   {m:10s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        problems.append(f"import {m}: {type(e).__name__}: {e}")
        print(f"  FAIL {m:10s} {type(e).__name__}: {e}")

try:
    import numpy as np
    if int(np.__version__.split(".")[0]) >= 2:
        problems.append(f"numpy is {np.__version__} (>=2); the sim stack needs numpy<2 -> "
                        "rerun: pip install 'numpy<2'")
    import torch
    print(f"  numpy {np.__version__}; torch {torch.__version__} cuda={torch.cuda.is_available()} "
          f"gpus={torch.cuda.device_count()}")
except Exception as e:
    problems.append(f"numpy/torch: {e}")

print("=== 2. functional: build one real LIBERO env (tests headless rendering) ===")
# LIBERO prompts for paths on first use; create the config non-interactively first so the
# env build below cannot hang waiting on stdin.
import subprocess
subprocess.run(
    "yes N | python -c 'from libero.libero import get_libero_path; get_libero_path(\"bddl_files\")'",
    shell=True, capture_output=True, timeout=180,
)
try:
    from drc import config
    from drc.envs import make_env
    env = make_env("LIBERO-Spatial-1", config.load_tasks()["LIBERO-Spatial-1"], synthetic=False)
    print("  OK   LIBERO OffScreenRenderEnv built")
except Exception as e:
    traceback.print_exc()
    problems.append(f"LIBERO env build: {type(e).__name__}: {e}  "
                    "(try MUJOCO_GL=osmesa, or apt-get install libgl1-mesa-glx libosmesa6)")

print("\n=== result ===")
if problems:
    for p in problems:
        print("  FAIL:", p)
    print("INSTALL CHECK FAILED — fix the install before training (see scripts/kaggle_setup.sh notes).")
    sys.exit(1)
print("INSTALL CHECK PASSED — safe to proceed to download + training.")
