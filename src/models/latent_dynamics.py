from __future__ import annotations

import torch
from torch import nn


class ActionLatentResidualDynamics(nn.Module):
    """Lightweight action-conditioned residual branch over SimVP latents.

    This branch preserves SimVP's parallel future prediction. It predicts a
    latent residual for the whole latent sequence at once, conditioned on the
    signed control representation [longitudinal, steer, speed].
    """

    def __init__(
        self,
        latent_dim: int,
        conditioning_dim: int = 3,
        hidden_dim: int = 128,
        gated: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.conditioning_dim = conditioning_dim
        self.hidden_dim = hidden_dim
        self.gated = gated

        self.condition_encoder = nn.Sequential(
            nn.Linear(conditioning_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.temporal_mixer = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.latent_projection = nn.Conv2d(latent_dim, hidden_dim, kernel_size=1)
        self.spatial_mixer = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
        )
        self.delta_head = nn.Conv2d(hidden_dim, latent_dim, kernel_size=1)
        self.gate_head = nn.Linear(hidden_dim, latent_dim) if gated else None

    def forward(self, latents: torch.Tensor, conditioning: torch.Tensor) -> dict[str, torch.Tensor | None]:
        if latents.ndim != 5:
            raise ValueError(f"Expected latents [B,T,C,H,W], got {tuple(latents.shape)}")
        if conditioning.ndim != 3:
            raise ValueError(f"Expected conditioning [B,T,D], got {tuple(conditioning.shape)}")
        b, t, c, h, w = latents.shape
        if c != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got {c}")
        if conditioning.shape[:2] != (b, t):
            raise ValueError("Latent and conditioning batch/time dimensions must match")
        if conditioning.shape[-1] != self.conditioning_dim:
            raise ValueError(f"Expected conditioning_dim={self.conditioning_dim}, got {conditioning.shape[-1]}")

        cond = self.condition_encoder(conditioning)
        cond = self.temporal_mixer(cond.transpose(1, 2)).transpose(1, 2)
        cond_map = cond[:, :, :, None, None].expand(-1, -1, -1, h, w)

        latent_2d = latents.reshape(b * t, c, h, w)
        latent_features = self.latent_projection(latent_2d).reshape(b, t, self.hidden_dim, h, w)
        mixed = latent_features + cond_map
        mixed_2d = self.spatial_mixer(mixed.reshape(b * t, self.hidden_dim, h, w))
        delta = self.delta_head(mixed_2d).reshape(b, t, c, h, w)

        gate = None
        if self.gate_head is not None:
            gate = torch.sigmoid(self.gate_head(cond)).view(b, t, c, 1, 1)

        return {"delta": delta, "gate": gate}


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ActionConditionedLatentDynamics(nn.Module):
    """Predict a full future latent sequence from past latents and controls.

    This is the dual WM branch used by ``av_wm_dual``. It keeps spatial latent
    structure by processing the stacked past latent sequence with lightweight
    Conv2d blocks, while past [longitudinal, steer, speed] controls modulate the
    future latent sequence through a small temporal conditioning encoder.
    """

    def __init__(
        self,
        latent_dim: int,
        past_len: int = 9,
        future_len: int = 8,
        conditioning_dim: int = 3,
        hidden_dim: int = 128,
        gated: bool = True,
        num_blocks: int = 3,
        conditioning_mode: str = "add",
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.past_len = past_len
        self.future_len = future_len
        self.conditioning_dim = conditioning_dim
        self.hidden_dim = hidden_dim
        self.gated = gated
        if conditioning_mode not in {"add", "concat", "film"}:
            raise ValueError(f"conditioning_mode must be add, concat, or film; got {conditioning_mode!r}")
        self.conditioning_mode = conditioning_mode

        self.condition_encoder = nn.Sequential(
            nn.Linear(conditioning_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.condition_temporal_mixer = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.future_queries = nn.Parameter(torch.zeros(future_len, hidden_dim))
        nn.init.normal_(self.future_queries, std=0.02)
        self.future_condition_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.past_projection = nn.Conv2d(past_len * latent_dim, hidden_dim, kernel_size=1)
        self.spatial_blocks = nn.Sequential(*[_ResidualConvBlock(hidden_dim) for _ in range(num_blocks)])
        self.concat_projection = (
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1)
            if conditioning_mode == "concat"
            else None
        )
        self.film = nn.Linear(hidden_dim, 2 * hidden_dim) if conditioning_mode == "film" else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        self.future_mixer = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
        )
        self.latent_head = nn.Conv2d(hidden_dim, latent_dim, kernel_size=1)
        self.gate_head = nn.Linear(hidden_dim, latent_dim) if gated else None

    def forward(self, latents: torch.Tensor, conditioning: torch.Tensor) -> dict[str, torch.Tensor | None]:
        if latents.ndim != 5:
            raise ValueError(f"Expected latents [B,T,C,H,W], got {tuple(latents.shape)}")
        if conditioning.ndim != 3:
            raise ValueError(f"Expected conditioning [B,T,D], got {tuple(conditioning.shape)}")
        b, t, c, h, w = latents.shape
        if t != self.past_len:
            raise ValueError(f"Expected past_len={self.past_len}, got {t}")
        if c != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got {c}")
        if conditioning.shape[:2] != (b, t):
            raise ValueError("Latent and conditioning batch/time dimensions must match")
        if conditioning.shape[-1] != self.conditioning_dim:
            raise ValueError(f"Expected conditioning_dim={self.conditioning_dim}, got {conditioning.shape[-1]}")

        cond = self.condition_encoder(conditioning)
        cond = self.condition_temporal_mixer(cond.transpose(1, 2)).transpose(1, 2)
        cond_context = cond.mean(dim=1)
        future_cond = cond_context[:, None, :] + self.future_queries[None, :, :]
        future_cond = self.future_condition_mlp(future_cond)

        stacked = latents.reshape(b, t * c, h, w)
        spatial = self.spatial_blocks(self.past_projection(stacked))
        future_features = spatial[:, None].expand(-1, self.future_len, -1, -1, -1)
        future_features = self._condition_future_features(future_features, future_cond)
        mixed = self.future_mixer(future_features.reshape(b * self.future_len, self.hidden_dim, h, w))
        z_wm = self.latent_head(mixed).reshape(b, self.future_len, c, h, w)

        gate = None
        gate_logits = None
        if self.gate_head is not None:
            gate_logits = self.gate_head(future_cond).view(b, self.future_len, c, 1, 1)
            gate = torch.sigmoid(gate_logits)
        return {"latents": z_wm, "gate": gate, "gate_logits": gate_logits}

    def _condition_future_features(self, spatial_t: torch.Tensor, future_cond: torch.Tensor) -> torch.Tensor:
        b, t, hidden_dim, h, w = spatial_t.shape
        cond_map = future_cond[:, :, :, None, None]
        if self.conditioning_mode == "add":
            return spatial_t + cond_map
        if self.conditioning_mode == "concat":
            if self.concat_projection is None:
                raise RuntimeError("Concat conditioning projection is not initialized.")
            cond_map = cond_map.expand(-1, -1, -1, h, w)
            combined = torch.cat([spatial_t, cond_map], dim=2)
            combined = combined.reshape(b * t, hidden_dim * 2, h, w)
            return self.concat_projection(combined).reshape(b, t, hidden_dim, h, w)
        if self.conditioning_mode == "film":
            if self.film is None:
                raise RuntimeError("FiLM conditioning layer is not initialized.")
            gamma_delta, beta = self.film(future_cond).chunk(2, dim=-1)
            gamma = 1.0 + gamma_delta
            return spatial_t * gamma[:, :, :, None, None] + beta[:, :, :, None, None]
        raise ValueError(f"Unknown conditioning_mode {self.conditioning_mode!r}")
