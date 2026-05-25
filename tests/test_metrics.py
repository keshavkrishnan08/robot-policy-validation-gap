import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from torch.utils.data import DataLoader

from drc import metrics as M
from drc.data.dataset import collate
from drc.data.synthetic import make_synthetic_dataset
from drc.policy.diffusion_policy import DiffusionUnetImagePolicy


def setup_policy_and_loader():
    ds, info = make_synthetic_dataset(n_demos=3, length=24, seed=0)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate)
    p = DiffusionUnetImagePolicy(
        action_dim=info["action_dim"], image_shape=info["image_shape"], proprio_dim=info["proprio_dim"],
        horizon=16, n_action_steps=8, n_obs_steps=2, num_train_timesteps=20, num_inference_steps=2,
        crop_shape=info["crop_shape"], backbone="smallcnn",
    )
    p.normalizer.fit(ds.all_actions_flat())
    return p, loader


def test_all_per_checkpoint_metrics_finite():
    p, loader = setup_policy_and_loader()
    stats = M.compute_train_latent_stats(p, loader, "cpu", n_components=8)
    assert np.all(np.isfinite(stats["cov_inv"]))
    for fn, args in [
        (M.compute_m1, (p, loader, "cpu", 1)),
        (M.compute_m2, (p, loader, "cpu", 1)),
        (M.compute_m3, (p, loader, "cpu", 2)),
        (M.compute_m4, (p, loader, "cpu", stats)),
        (M.compute_m8, (p, loader, "cpu", 2)),
    ]:
        v = fn(*args)
        assert np.isfinite(v), fn.__name__


def test_m6_zero_for_identical_policies():
    p, loader = setup_policy_and_loader()
    # Same policy twice -> deterministic predictions identical -> zero disagreement.
    d = M.compute_m6([p, p], loader, "cpu", K=1)
    assert d == 0.0 or d < 1e-6


def test_integrated_jerk_short_traj():
    assert M._integrated_jerk(np.zeros((2, 3))) == 0.0
    straight = np.stack([np.linspace(0, 1, 10)] * 3, axis=1)  # constant velocity -> ~0 jerk
    assert M._integrated_jerk(straight) < 1e-6
