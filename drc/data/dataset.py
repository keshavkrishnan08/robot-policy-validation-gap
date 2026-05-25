"""In-memory sequence dataset shared by synthetic and real adapters.

Holds arrays:
  images  : (N, L, C, H, W) float32 in [0, 1]
  proprio : (N, L, P) float32
  actions : (N, L, A) float32

A sample at window start `s` yields observations over [s, s+n_obs) and the
action chunk over [s, s+horizon). Only fully-contained windows are indexed,
which is the simple, padding-free convention used throughout this project.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, images, proprio, actions, n_obs_steps: int, horizon: int):
        self.images = images.astype(np.float32)
        self.proprio = proprio.astype(np.float32)
        self.actions = actions.astype(np.float32)
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.window = max(n_obs_steps, horizon)

        self.index = []  # (demo, start)
        n, L = self.images.shape[:2]
        for d in range(n):
            for s in range(0, L - self.window + 1):
                self.index.append((d, s))

    def action_dim(self) -> int:
        return self.actions.shape[-1]

    def proprio_dim(self) -> int:
        return self.proprio.shape[-1]

    def image_shape(self):
        return tuple(self.images.shape[2:])

    def all_actions_flat(self) -> torch.Tensor:
        return torch.from_numpy(self.actions.reshape(-1, self.actions.shape[-1]))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        d, s = self.index[i]
        obs_img = self.images[d, s : s + self.n_obs_steps]
        obs_pro = self.proprio[d, s : s + self.n_obs_steps]
        act = self.actions[d, s : s + self.horizon]
        return {
            "obs": {
                "image": torch.from_numpy(obs_img),
                "proprio": torch.from_numpy(obs_pro),
            },
            "action": torch.from_numpy(act),
        }


def collate(batch):
    return {
        "obs": {
            "image": torch.stack([b["obs"]["image"] for b in batch]),
            "proprio": torch.stack([b["obs"]["proprio"] for b in batch]),
        },
        "action": torch.stack([b["action"] for b in batch]),
    }
