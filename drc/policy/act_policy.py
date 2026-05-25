"""ACT — Action Chunking with Transformers (Zhao et al. 2023), as the second
architecture for the validation-gap study.

A compact CVAE-transformer that satisfies the same inference interface as
DiffusionUnetImagePolicy, so metrics.py, rollouts.py, and train.py work
unchanged via the policy factory:

  compute_loss(batch)            -> L1 reconstruction + KL (CVAE objective)
  predict_deterministic(obs, K)  -> decode with style latent z = 0 (deterministic)
  sample(obs, generator)         -> decode with z ~ N(0, I) (stochastic; drives M3/M8)
  encode_latent(obs)             -> pooled transformer memory (for M4)

Stochasticity in ACT comes from the CVAE latent z, not diffusion noise; the
deterministic prediction fixes z = 0, the prior mean.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from drc.model.obs_encoder import _resnet18, _small_cnn, RandomCrop
from drc.policy.diffusion_policy import Normalizer


class ACTPolicy(nn.Module):
    def __init__(
        self,
        action_dim: int,
        image_shape=(3, 84, 84),
        proprio_dim: int = 9,
        horizon: int = 16,
        n_action_steps: int = 8,
        n_obs_steps: int = 2,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        z_dim: int = 32,
        kl_weight: float = 10.0,
        crop_shape=(76, 76),
        crop_pad: int = 4,
        backbone: str = "smallcnn",
        # accepted-and-ignored kwargs so the factory can pass diffusion-only args
        num_train_timesteps: int | None = None,
        num_inference_steps: int | None = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.z_dim = z_dim
        self.kl_weight = kl_weight

        self.crop = RandomCrop(crop_shape, crop_pad)
        self.image_backbone = _resnet18(d_model) if backbone == "resnet18" else _small_cnn(d_model)
        self.proprio_proj = nn.Linear(proprio_dim, d_model) if proprio_dim > 0 else None

        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model, batch_first=True)
        self.obs_transformer = nn.TransformerEncoder(enc_layer, num_layers)

        # CVAE encoder: [CLS] + action tokens -> z params (training only).
        self.action_embed = nn.Linear(action_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        cvae_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model, batch_first=True)
        self.cvae_encoder = nn.TransformerEncoder(cvae_layer, 2)
        self.z_head = nn.Linear(d_model, 2 * z_dim)
        self.z_proj = nn.Linear(z_dim, d_model)

        # Decoder: learned per-timestep queries cross-attend to obs+z memory.
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=4 * d_model, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.query_embed = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)
        self.action_head = nn.Linear(d_model, action_dim)

        self.normalizer = Normalizer(action_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    @contextlib.contextmanager
    def _inference_mode(self):
        was_training = self.training
        self.eval()
        try:
            yield
        finally:
            if was_training:
                self.train()

    def _obs_tokens(self, obs: dict) -> torch.Tensor:
        img = obs["image"]                       # (B, n_obs, C, H, W)
        b, n = img.shape[:2]
        img = self.crop(img)
        feats = self.image_backbone(img.reshape(b * n, *img.shape[2:])).reshape(b, n, -1)
        if self.proprio_proj is not None and "proprio" in obs:
            feats = feats + self.proprio_proj(obs["proprio"])
        return self.obs_transformer(feats)        # (B, n_obs, d_model)

    def _decode(self, memory: torch.Tensor) -> torch.Tensor:
        b = memory.shape[0]
        queries = self.query_embed.expand(b, -1, -1)
        dec = self.decoder(queries, memory)       # (B, horizon, d_model)
        return self.action_head(dec)              # (B, horizon, action_dim) [normalized]

    def compute_loss(self, batch: dict) -> torch.Tensor:
        actions = self.normalizer.normalize(batch["action"])      # (B, H, A)
        obs_tok = self._obs_tokens(batch["obs"])                  # (B, n_obs, d)
        b = actions.shape[0]

        # CVAE encoder -> z params from the action sequence + a CLS token.
        a_tok = self.action_embed(actions)                        # (B, H, d)
        cls = self.cls_token.expand(b, -1, -1)
        cvae_in = torch.cat([cls, a_tok], dim=1)
        cvae_out = self.cvae_encoder(cvae_in)[:, 0]               # CLS output (B, d)
        mu, logvar = self.z_head(cvae_out).chunk(2, dim=-1)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()

        memory = torch.cat([obs_tok, self.z_proj(z).unsqueeze(1)], dim=1)
        pred = self._decode(memory)
        recon = F.l1_loss(pred, actions)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + self.kl_weight * kl

    @torch.no_grad()
    def _decode_with_z(self, obs: dict, z: torch.Tensor) -> torch.Tensor:
        obs_tok = self._obs_tokens(obs)
        memory = torch.cat([obs_tok, self.z_proj(z).unsqueeze(1)], dim=1)
        return self.normalizer.unnormalize(self._decode(memory))

    @torch.no_grad()
    def predict_deterministic(self, obs: dict, K: int = 10, noise_seed: int | None = None) -> torch.Tensor:
        # Deterministic: style latent fixed at the prior mean z = 0. K/noise_seed
        # are irrelevant (no stochasticity), kept for interface compatibility.
        with self._inference_mode():
            b = obs["image"].shape[0]
            z = torch.zeros(b, self.z_dim, device=self.device)
            return self._decode_with_z(obs, z)

    @torch.no_grad()
    def sample(self, obs: dict, generator: torch.Generator | None = None) -> torch.Tensor:
        with self._inference_mode():
            b = obs["image"].shape[0]
            z = torch.randn(b, self.z_dim, device=self.device, generator=generator)
            return self._decode_with_z(obs, z)

    @torch.no_grad()
    def encode_latent(self, obs: dict) -> torch.Tensor:
        with self._inference_mode():
            return self._obs_tokens(obs).mean(dim=1)   # (B, d_model)

    @torch.no_grad()
    def predict_action_chunk(self, obs: dict, K: int = 10, noise_seed: int | None = None) -> torch.Tensor:
        return self.predict_deterministic(obs, K=K, noise_seed=noise_seed)[:, : self.n_action_steps]
