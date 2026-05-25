"""1D conditional U-net denoiser (Chi et al. 2023, Diffusion Policy).

Compact but faithful: sinusoidal timestep embedding fused with the global
observation conditioning vector, FiLM-modulated Conv1d residual blocks over the
action-time axis, symmetric down/up path with skip connections.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))
        return emb


def _safe_groups(channels: int, n_groups: int) -> int:
    """Largest divisor of `channels` not exceeding n_groups (GroupNorm needs
    channels % groups == 0). Falls back to 1 for awkward dims like 14."""
    g = min(n_groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return g


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish."""

    def __init__(self, inp: int, out: int, kernel_size: int = 3, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp, out, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(_safe_groups(out, n_groups), out),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1d(nn.Module):
    def __init__(self, inp: int, out: int, cond_dim: int, kernel_size: int = 3, n_groups: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Conv1dBlock(inp, out, kernel_size, n_groups), Conv1dBlock(out, out, kernel_size, n_groups)]
        )
        # FiLM: produce per-channel scale and bias from the conditioning vector.
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out * 2))
        self.out_channels = out
        self.residual_conv = nn.Conv1d(inp, out, 1) if inp != out else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).view(x.shape[0], 2, self.out_channels, 1)
        scale, bias = embed[:, 0], embed[:, 1]
        out = scale * out + bias
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 128,
        down_dims: tuple[int, ...] = (128, 256, 512),
        kernel_size: int = 3,
        n_groups: int = 8,
    ):
        super().__init__()
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed), nn.Linear(dsed, dsed * 4), nn.Mish(), nn.Linear(dsed * 4, dsed)
        )
        cond_dim = dsed + global_cond_dim

        all_dims = [input_dim, *down_dims]
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        self.down_modules = nn.ModuleList()
        for ind, (din, dout) in enumerate(in_out):
            is_last = ind >= len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(din, dout, cond_dim, kernel_size, n_groups),
                        ConditionalResidualBlock1d(dout, dout, cond_dim, kernel_size, n_groups),
                        nn.Conv1d(dout, dout, 3, stride=2, padding=1) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1d(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1d(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
            ]
        )

        self.up_modules = nn.ModuleList()
        for ind, (din, dout) in enumerate(reversed(in_out)):
            is_last = ind >= len(in_out) - 1
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(dout * 2, din, cond_dim, kernel_size, n_groups),
                        ConditionalResidualBlock1d(din, din, cond_dim, kernel_size, n_groups),
                        nn.ConvTranspose1d(din, din, 4, stride=2, padding=1)
                        if not is_last
                        else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(input_dim, input_dim, kernel_size, n_groups), nn.Conv1d(input_dim, input_dim, 1)
        )

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        # sample: (B, T, input_dim) -> (B, input_dim, T)
        x = sample.transpose(1, 2)
        temb = self.diffusion_step_encoder(timestep)
        cond = torch.cat([temb, global_cond], dim=-1)

        skips = []
        for res1, res2, downsample in self.down_modules:
            x = res1(x, cond)
            x = res2(x, cond)
            skips.append(x)
            x = downsample(x)

        for mid in self.mid_modules:
            x = mid(x, cond)

        for res1, res2, upsample in self.up_modules:
            x = torch.cat([x, skips.pop()], dim=1)
            x = res1(x, cond)
            x = res2(x, cond)
            x = upsample(x)

        x = self.final_conv(x)
        return x.transpose(1, 2)  # (B, T, input_dim)
