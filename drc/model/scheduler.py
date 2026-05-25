"""Minimal DDPM training + deterministic DDIM sampling scheduler.

Self-contained so the synthetic CPU path needs no `diffusers`. On Kaggle the
same math matches diffusers' DDPMScheduler(beta_schedule="linear") and the
DDIM deterministic sampler (eta=0). PRD section 8.4 locks: 100 train steps,
16 inference steps, linear beta.
"""
from __future__ import annotations

import torch


class DiffusionScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 16,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str | torch.device = "cpu",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.device = torch.device(device)

        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        self.betas = betas.to(self.device)
        self.alphas = alphas.to(self.device)
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(self.device)

    def to(self, device):
        self.device = torch.device(device)
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward diffusion q(x_t | x_0). t shape (B,)."""
        ac = self.alphas_cumprod[t].view(-1, *([1] * (x0.dim() - 1)))
        return ac.sqrt() * x0 + (1.0 - ac).sqrt() * noise

    def sample_timesteps(self, batch_size: int, generator: torch.Generator | None = None) -> torch.Tensor:
        return torch.randint(
            0, self.num_train_timesteps, (batch_size,), device=self.device, generator=generator
        )

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        shape: tuple[int, ...],
        global_cond: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Deterministic DDIM (eta=0) reverse process. Returns x0 estimate.

        `model(x_t, t, global_cond)` predicts the noise epsilon.
        """
        device = self.device
        x = torch.randn(shape, device=device, generator=generator)
        # Evenly spaced timestep subset, descending.
        step = max(self.num_train_timesteps // self.num_inference_steps, 1)
        timesteps = list(range(0, self.num_train_timesteps, step))[::-1]
        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            eps = model(x, t_batch, global_cond)
            ac_t = self.alphas_cumprod[t]
            x0_pred = (x - (1.0 - ac_t).sqrt() * eps) / ac_t.sqrt().clamp(min=1e-8)
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            ac_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
            # eta = 0 -> deterministic.
            x = ac_prev.sqrt() * x0_pred + (1.0 - ac_prev).sqrt() * eps
        return x
