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
