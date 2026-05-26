import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from drc.bound_validation import deviation_persistent, run_p1, run_p2_identifiability
from drc.lyapunov import amplification


def test_persistent_deviation_equals_bound():
    # delta_H under persistent error eps must equal eps * (L^H-1)/(L-1) exactly.
    for L in [0.7, 0.9, 1.0, 1.2]:
        for eps in [0.01, 0.05]:
            for H in [5, 12]:
                d = deviation_persistent(L, eps, H)
                assert abs(d - eps * amplification(L, H)) < 1e-9 * max(1, d)


def test_p1_is_tight():
    rows = run_p1([0.6, 0.9, 1.0, 1.2], [0.01, 0.05], H=10)
    meas = np.array([r["measured"] for r in rows])
    pred = np.array([r["predicted"] for r in rows])
    assert np.allclose(meas, pred, rtol=1e-9)  # R^2 = 1


def test_p2_replay_beats_validation_loss():
    p2 = run_p2_identifiability(n_policies=600, H=12, tol=1.0, seed=0)
    # environment-querying replay must rank success better than rollout-free validation loss
    assert p2["replay_corr_pooled"] > p2["val_loss_corr_pooled"] + 0.2
    assert 0.0 <= p2["val_loss_corr_pooled"] <= 1.0
    assert p2["replay_corr_pooled"] > 0.6


def test_selection_regret_thm1_thm2():
    from drc.bound_validation import selection_regret
    sr = selection_regret(n_populations=4000, K=6, H=18, tol=1.0,
                          eps_range=(0.02, 0.03), L_range=(0.7, 1.35), seed=0)
    # Thm 2: env-querying replay achieves near-oracle selection (small regret)
    assert sr["regret_replay"] < 0.05
    # Thm 1: rollout-free validation loss is near-random (much larger regret)
    assert sr["regret_val_loss"] > 0.2
    assert sr["regret_val_loss"] > 5 * sr["regret_replay"]


def test_horizon_wall_prediction_matches():
    from drc.bound_validation import horizon_wall
    hw = horizon_wall(L=1.15, eps=0.03, tol=1.0)
    # predicted critical horizon must match the empirical collapse within one step
    assert abs(hw["H_star_predicted"] - hw["H_star_empirical"]) <= 1.0
