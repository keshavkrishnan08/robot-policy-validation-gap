"""Observation encoder: image stack + proprioception -> global cond vector.

Two backbones:
  - "resnet18": torchvision ResNet-18 (PRD default for the Kaggle run).
  - "smallcnn": a light 4-layer CNN, no torchvision, for CPU/synthetic runs.

Random-crop augmentation (PRD 8.4) is applied at training time only.
The encoder also exposes `encode_latent` returning the pre-fusion conditioning
latent used by metric M4 (latent Mahalanobis distance).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomCrop(nn.Module):
    """Pad-and-random-crop, plus a fixed center crop at eval."""

    def __init__(self, crop_shape, pad: int = 4):
        super().__init__()
        self.h, self.w = crop_shape
        self.pad = pad

    def forward(self, x):  # x: (..., C, H, W)
        *lead, c, h, w = x.shape
        x = x.reshape(-1, c, h, w)
        if self.training:
            x = F.pad(x, (self.pad,) * 4, mode="replicate")
            ph, pw = x.shape[-2:]
            top = torch.randint(0, ph - self.h + 1, (1,)).item()
            left = torch.randint(0, pw - self.w + 1, (1,)).item()
            x = x[..., top : top + self.h, left : left + self.w]
        else:
            top = (h - self.h) // 2
            left = (w - self.w) // 2
            x = x[..., top : top + self.h, left : left + self.w]
        return x.reshape(*lead, c, self.h, self.w)


def _small_cnn(out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, out_dim),
    )


def _resnet18(out_dim: int) -> nn.Module:  # pragma: no cover - needs torchvision
    from torchvision.models import resnet18

    net = resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, out_dim)
    return net


class MultiImageObsEncoder(nn.Module):
    def __init__(
        self,
        image_shape=(3, 84, 84),
        proprio_dim: int = 9,
        n_obs_steps: int = 2,
        feat_dim: int = 256,
        crop_shape=(76, 76),
        crop_pad: int = 4,
        backbone: str = "smallcnn",
    ):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.proprio_dim = proprio_dim
        self.feat_dim = feat_dim
        self.crop = RandomCrop(crop_shape, crop_pad)
        self.backbone = _resnet18(feat_dim) if backbone == "resnet18" else _small_cnn(feat_dim)
        self.proprio_proj = nn.Linear(proprio_dim, feat_dim) if proprio_dim > 0 else None
        per_step = feat_dim + (feat_dim if proprio_dim > 0 else 0)
        self.out_dim = per_step * n_obs_steps

    def encode_latent(self, obs: dict) -> torch.Tensor:
        """Per-(B) flattened conditioning latent before any diffusion fusion.

        Returns (B, out_dim). Used by M4. No augmentation (eval mode crop).
        """
        img = obs["image"]  # (B, n_obs, C, H, W)
        b, n = img.shape[:2]
        img = self.crop(img)
        feats = self.backbone(img.reshape(b * n, *img.shape[2:])).reshape(b, n, -1)
        parts = [feats]
        if self.proprio_proj is not None and "proprio" in obs:
            p = self.proprio_proj(obs["proprio"])  # (B, n_obs, feat)
            parts.append(p)
        cat = torch.cat(parts, dim=-1)  # (B, n_obs, per_step)
        return cat.reshape(b, -1)  # (B, out_dim)

    def forward(self, obs: dict) -> torch.Tensor:
        return self.encode_latent(obs)
