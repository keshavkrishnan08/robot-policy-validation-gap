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
# Write ~/.libero/config.yaml directly so LIBERO never enters its interactive setup.
try:
    from drc.envs import _ensure_libero_config
    _ensure_libero_config()
    print("  LIBERO config ensured (no interactive prompt)")
except Exception as e:
    problems.append(f"libero config: {e}")
try:
    import signal

    from drc import config
    from drc.envs import make_env

    # A misconfigured GL backend doesn't error — it HANGS on the first offscreen render. Bound the
    # whole build+render with SIGALRM so this check fails loudly in seconds instead of blocking.
    def _timeout(_sig, _frm):
        raise TimeoutError("env build/render exceeded 120s — almost certainly a headless-GL hang "
                           "(MUJOCO_GL). The real run would hang here too.")
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(120)

    env = make_env("LIBERO-Spatial-1", config.load_tasks()["LIBERO-Spatial-1"], synthetic=False)
    print(f"  OK   LIBERO OffScreenRenderEnv built (MUJOCO_GL={os.environ.get('MUJOCO_GL')})")
    # THE load-bearing check: actually render offscreen (this is what hung in the metrics step).
    raw = env.env.reset()
    img = raw.get("agentview_image")
    if img is None:
        problems.append("env.reset() returned no agentview_image — rendering misconfigured")
    else:
        print(f"  OK   offscreen render works: agentview_image {tuple(img.shape)}")
    signal.alarm(0)
except Exception as e:
    traceback.print_exc()
    problems.append(f"LIBERO env build/render: {type(e).__name__}: {e}  "
                    "(try MUJOCO_GL=osmesa, or apt-get install libgl1-mesa-glx libosmesa6)")

print("\n=== result ===")
if problems:
    for p in problems:
        print("  FAIL:", p)
    print("INSTALL CHECK FAILED — fix the install before training (see scripts/kaggle_setup.sh notes).")
    sys.exit(1)
print("INSTALL CHECK PASSED — safe to proceed to download + training.")
