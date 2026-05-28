from __future__ import annotations

import torch
from torch import nn


class LatentDynamicsHead(nn.Module):
    """Predict compact future dynamics from predicted latent rollout states."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128, output_dim: int = 3) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, future_latents: torch.Tensor) -> torch.Tensor:
        if future_latents.ndim != 5:
            raise ValueError(f"Expected latents [B, T, C, H, W], got {tuple(future_latents.shape)}")
        pooled = future_latents.mean(dim=(-2, -1))
        return self.net(pooled)
