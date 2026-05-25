import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from drc.model.conditional_unet1d import ConditionalUnet1D, _safe_groups
from drc.model.scheduler import DiffusionScheduler
from drc.policy.diffusion_policy import DiffusionUnetImagePolicy


def tiny_policy(action_dim=4, proprio_dim=4):
    return DiffusionUnetImagePolicy(
        action_dim=action_dim, image_shape=(3, 12, 12), proprio_dim=proprio_dim,
        horizon=16, n_action_steps=8, n_obs_steps=2,
        num_train_timesteps=20, num_inference_steps=2, crop_shape=(10, 10), backbone="smallcnn",
    )


def tiny_obs(b=2):
    return {"image": torch.rand(b, 2, 3, 12, 12), "proprio": torch.rand(b, 2, 4)}


def test_safe_groups():
    assert _safe_groups(8, 8) == 8
    assert _safe_groups(14, 8) == 7   # largest divisor of 14 <= 8
    assert _safe_groups(7, 8) == 7
    assert _safe_groups(5, 8) == 5
    assert _safe_groups(1, 8) == 1


def test_unet_shape():
    net = ConditionalUnet1D(input_dim=4, global_cond_dim=16, down_dims=(32, 64, 128))
    x = torch.randn(2, 16, 4)
    t = torch.randint(0, 20, (2,))
    cond = torch.randn(2, 16)
    assert net(x, t, cond).shape == x.shape


def test_unet_handles_14dim_action():
    net = ConditionalUnet1D(input_dim=14, global_cond_dim=8, down_dims=(32, 64))
    x = torch.randn(1, 16, 14)
    out = net(x, torch.zeros(1, dtype=torch.long), torch.randn(1, 8))
    assert out.shape == x.shape


def test_scheduler_add_noise_and_sample():
    sch = DiffusionScheduler(num_train_timesteps=20, num_inference_steps=2)
    x0 = torch.randn(2, 16, 4)
    noise = torch.randn_like(x0)
    t = torch.tensor([0, 19])
    assert sch.add_noise(x0, noise, t).shape == x0.shape


def test_policy_loss_and_inference():
    p = tiny_policy()
    p.normalizer.fit(torch.randn(50, 4))
    batch = {"obs": tiny_obs(), "action": torch.randn(2, 16, 4)}
    loss = p.compute_loss(batch)
    assert loss.dim() == 0 and torch.isfinite(loss)

    det = p.predict_deterministic(tiny_obs(), K=2)
    assert det.shape == (2, 16, 4)
    chunk = p.predict_action_chunk(tiny_obs(), K=1)
    assert chunk.shape == (2, 8, 4)
    lat = p.encode_latent(tiny_obs())
    assert lat.shape[0] == 2 and lat.dim() == 2


def test_deterministic_noise_seed_repeatable():
    p = tiny_policy()
    p.normalizer.fit(torch.randn(50, 4))
    obs = tiny_obs(1)
    a1 = p.predict_action_chunk(obs, K=1, noise_seed=42)
    a2 = p.predict_action_chunk(obs, K=1, noise_seed=42)
    assert torch.allclose(a1, a2)
