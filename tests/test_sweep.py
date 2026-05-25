import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from drc.dynamics_sweep import SweepEnv, make_sweep_dataset, sweep_eval_conditions, DIM, TOL
from drc.lyapunov import amplification


def test_amplification_formula():
    assert amplification(1.0, 5) == 5.0            # L=1 limit -> H
    assert abs(amplification(2.0, 3) - 7.0) < 1e-9  # (8-1)/(2-1)
    # contractive: bounded, increasing in L
    assert amplification(0.5, 100) < amplification(0.9, 100) < amplification(1.5, 100)


def test_sweep_dataset_shapes():
    ds, info = make_sweep_dataset(1.1, n_demos=4, length=20, seed=0)
    assert info["action_dim"] == DIM and info["L"] == 1.1
    assert len(ds) > 0


def test_sweep_env_perfect_tracking():
    # Outputting the expert action a*=L*s should track the reference (success at horizon).
    env = SweepEnv(L=0.9, max_steps=20)
    obs = env.reset_to({"state": np.array([0.2, 0.2, 0, 0], dtype=np.float32)})
    success_seen = False
    for _ in range(20):
        s = env._s.copy()
        _, _, done, info = env.step(0.9 * s)   # perfect expert action
        success_seen = success_seen or info["success"]
        if done:
            break
    assert success_seen  # tracked to the horizon


def test_sweep_env_expansive_diverges_under_error():
    # With L>1 and a wrong action, tracking error compounds and the episode fails.
    env = SweepEnv(L=1.5, max_steps=20)
    env.reset_to({"state": np.array([0.1, 0.1, 0, 0], dtype=np.float32)})
    failed = False
    for _ in range(20):
        s = env._s.copy()
        _, _, done, info = env.step(1.5 * s + 0.3)  # action error 0.3 each step
        if done and not info["success"]:
            failed = True
            break
    assert failed


def test_sweep_eval_conditions():
    conds = sweep_eval_conditions(5, seed=1)
    assert len(conds) == 5 and len(conds[0]["state"]) == DIM
