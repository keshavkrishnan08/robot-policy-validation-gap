"""DiffusionUnetImagePolicy — the single locked architecture (PRD 8.2).

Wraps the obs encoder, the 1D conditional U-net denoiser, and the DDPM/DDIM
scheduler. Exposes the inference surface the rest of the pipeline depends on:

  compute_loss(batch)             -> training MSE on predicted noise
  predict_deterministic(obs, K)   -> mean of K DDIM samples (M1, M2, rollouts)
  sample(obs, generator)          -> a single stochastic action prediction (M3, M8)
  encode_latent(obs)              -> conditioning latent for M4

Actions are normalized to ~unit scale via stored buffers fit on the train set.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from drc.model.conditional_unet1d import ConditionalUnet1D
from drc.model.obs_encoder import MultiImageObsEncoder
from drc.model.scheduler import DiffusionScheduler


class Normalizer(nn.Module):
    """Affine action normalizer using stored mean/std buffers."""

    def __init__(self, action_dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(action_dim))
        self.register_buffer("std", torch.ones(action_dim))

    def fit(self, actions: torch.Tensor):  # (N, action_dim)
        self.mean = actions.mean(0)
        self.std = actions.std(0).clamp(min=1e-4)

    def normalize(self, a):
        return (a - self.mean) / self.std

    def unnormalize(self, a):
        return a * self.std + self.mean


class DiffusionUnetImagePolicy(nn.Module):
    def __init__(
        self,
        action_dim: int,
        image_shape=(3, 84, 84),
        proprio_dim: int = 9,
        horizon: int = 16,
        n_action_steps: int = 8,
        n_obs_steps: int = 2,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 16,
        crop_shape=(76, 76),
        crop_pad: int = 4,
        backbone: str = "smallcnn",
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps

        self.obs_encoder = MultiImageObsEncoder(
            image_shape=image_shape,
            proprio_dim=proprio_dim,
            n_obs_steps=n_obs_steps,
            crop_shape=crop_shape,
            crop_pad=crop_pad,
            backbone=backbone,
        )
        self.unet = ConditionalUnet1D(input_dim=action_dim, global_cond_dim=self.obs_encoder.out_dim)
        self.scheduler = DiffusionScheduler(
            num_train_timesteps=num_train_timesteps, num_inference_steps=num_inference_steps
        )
        self.normalizer = Normalizer(action_dim)

    # -- properties ---------------------------------------------------------
    @property
    def device(self):
        return next(self.parameters()).device

    def _sync_scheduler(self):
        self.scheduler.to(self.device)

    @contextlib.contextmanager
    def _inference_mode(self):
        """Force eval (disables random-crop augmentation) and restore prior mode.

        Inference must be deterministic given a fixed noise seed; the train-time
        RandomCrop would otherwise inject uncontrolled randomness.
        """
        was_training = self.training
        self.eval()
        try:
            yield
        finally:
            if was_training:
                self.train()

    # -- training -----------------------------------------------------------
    def compute_loss(self, batch: dict) -> torch.Tensor:
        self._sync_scheduler()
        actions = self.normalizer.normalize(batch["action"])  # (B, horizon, A)
        cond = self.obs_encoder(batch["obs"])                  # (B, cond_dim)
        noise = torch.randn_like(actions)
        t = self.scheduler.sample_timesteps(actions.shape[0])
        noisy = self.scheduler.add_noise(actions, noise, t)
        pred = self.unet(noisy, t, cond)
        return F.mse_loss(pred, noise)

    # -- inference ----------------------------------------------------------
    @torch.no_grad()
    def _sample_once(self, obs: dict, generator: torch.Generator | None = None) -> torch.Tensor:
        self._sync_scheduler()
        cond = self.obs_encoder(obs)
        b = cond.shape[0]
        x = self.scheduler.ddim_sample(
            self.unet, (b, self.horizon, self.action_dim), cond, generator=generator
        )
        return self.normalizer.unnormalize(x)  # (B, horizon, A)

    @torch.no_grad()
    def sample(self, obs: dict, generator: torch.Generator | None = None) -> torch.Tensor:
        """A single stochastic action-horizon prediction."""
        with self._inference_mode():
            return self._sample_once(obs, generator)

    @torch.no_grad()
    def predict_deterministic(self, obs: dict, K: int = 10, noise_seed: int | None = None) -> torch.Tensor:
        """Mean over K DDIM samples — the deterministic prediction (PRD 8.1).

        When noise_seed is given (rollouts, PRD 8.3) the generator is fixed so the
        prediction is reproducible across checkpoints.
        """
        with self._inference_mode():
            gen = None
            if noise_seed is not None:
                gen = torch.Generator(device=self.device).manual_seed(noise_seed)
            samples = torch.stack([self._sample_once(obs, generator=gen) for _ in range(K)], dim=0)
            return samples.mean(0)  # (B, horizon, A)

    @torch.no_grad()
    def encode_latent(self, obs: dict) -> torch.Tensor:
        with self._inference_mode():
            return self.obs_encoder.encode_latent(obs)

    @torch.no_grad()
    def predict_action_chunk(self, obs: dict, K: int = 10, noise_seed: int | None = None) -> torch.Tensor:
        """First n_action_steps of the deterministic prediction, for rollouts."""
        full = self.predict_deterministic(obs, K=K, noise_seed=noise_seed)
        return full[:, : self.n_action_steps]
