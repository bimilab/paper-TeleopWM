from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def normalize_spatial_grid(spatial_grid: tuple[int, int] | str) -> tuple[int, int]:
    if isinstance(spatial_grid, str):
        parts = spatial_grid.lower().split("x")
        if len(parts) != 2:
            raise ValueError(f"spatial_grid must use HxW format, got {spatial_grid!r}")
        grid = (int(parts[0]), int(parts[1]))
    else:
        grid = (int(spatial_grid[0]), int(spatial_grid[1]))
    if grid[0] <= 0 or grid[1] <= 0:
        raise ValueError(f"spatial_grid dimensions must be positive, got {grid}")
    return grid


class FutureActionPredictionHead(nn.Module):
    """Predict future [longitudinal, steer] controls from future latents.

    The head pools spatial latent maps into tokens, summarizes past
    [longitudinal, steer, speed] controls with a small GRU, then decodes a
    future control sequence with a lightweight temporal GRU.
    """

    def __init__(
        self,
        latent_dim: int,
        past_control_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        future_len: int = 8,
        classification: bool = False,
        future_motion_scale: float = 1.0,
        spatial_pooling: str = "global",
        spatial_grid: tuple[int, int] | str = (1, 1),
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.past_control_dim = past_control_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.future_len = future_len
        self.classification = bool(classification)
        self.future_motion_scale = float(future_motion_scale)

        self.latent_projection = nn.Linear(latent_dim, hidden_dim)
        self.control_context = nn.GRU(
            input_size=past_control_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.temporal_decoder = nn.GRU(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.longitudinal_cls_head = nn.Linear(hidden_dim, 3) if self.classification else None
        self.steer_cls_head = nn.Linear(hidden_dim, 3) if self.classification else None

    def forward(self, future_latents: torch.Tensor, past_controls: torch.Tensor) -> torch.Tensor:
        decoded = self.encode(future_latents, past_controls)
        return self.head(decoded)

    def forward_with_logits(self, future_latents: torch.Tensor, past_controls: torch.Tensor) -> dict[str, torch.Tensor]:
        decoded = self.encode(future_latents, past_controls)
        output = {"actions": self.head(decoded)}
        if self.longitudinal_cls_head is not None and self.steer_cls_head is not None:
            output["longitudinal_logits"] = self.longitudinal_cls_head(decoded)
            output["steer_logits"] = self.steer_cls_head(decoded)
        return output

    def encode(self, future_latents: torch.Tensor, past_controls: torch.Tensor) -> torch.Tensor:
        if future_latents.ndim != 5:
            raise ValueError(f"Expected future_latents [B,T,C,H,W], got {tuple(future_latents.shape)}")
        if past_controls.ndim != 3:
            raise ValueError(f"Expected past_controls [B,T,D], got {tuple(past_controls.shape)}")
        b, t, c, _, _ = future_latents.shape
        if t != self.future_len:
            raise ValueError(f"Expected future_len={self.future_len}, got {t}")
        if c != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got {c}")
        if past_controls.shape[0] != b or past_controls.shape[-1] != self.past_control_dim:
            raise ValueError(
                f"Expected past_controls [B,T,{self.past_control_dim}], got {tuple(past_controls.shape)}"
            )

        latent_tokens = future_latents.mean(dim=(-1, -2))
        latent_tokens = self.latent_projection(latent_tokens)
        _, context = self.control_context(past_controls)
        context = context[-1].unsqueeze(1).expand(-1, t, -1)
        decoded, _ = self.temporal_decoder(torch.cat([latent_tokens, context], dim=-1))
        return decoded


class MotionContextFutureActionPredictionHead(nn.Module):
    """Future action head with explicit latent-motion and control context.

    This variant is intentionally lightweight: it globally pools latent maps,
    summarizes observed latent deltas with a GRU, summarizes past controls with
    another GRU, then decodes the future action sequence from the predicted
    future latent tokens plus both contexts. It is isolated to the future-action
    pathway and does not alter image decoding or WM fusion.
    """

    def __init__(
        self,
        latent_dim: int,
        past_control_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        future_len: int = 8,
        classification: bool = False,
        future_motion_scale: float = 1.0,
        spatial_pooling: str = "global",
        spatial_grid: tuple[int, int] | str = (1, 1),
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.past_control_dim = past_control_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.future_len = future_len
        self.classification = bool(classification)
        self.future_motion_scale = float(future_motion_scale)

        self.future_latent_projection = nn.Linear(latent_dim, hidden_dim)
        self.motion_projection = nn.Linear(latent_dim + 1, hidden_dim)
        self.motion_context = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.control_context = nn.GRU(
            input_size=past_control_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.temporal_decoder = nn.GRU(
            input_size=hidden_dim * 3,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.longitudinal_cls_head = nn.Linear(hidden_dim, 3) if self.classification else None
        self.steer_cls_head = nn.Linear(hidden_dim, 3) if self.classification else None

    def forward(
        self,
        future_latents: torch.Tensor,
        past_controls: torch.Tensor,
        past_latents: torch.Tensor,
    ) -> torch.Tensor:
        decoded, _ = self.encode(future_latents, past_controls, past_latents)
        return self.head(decoded)

    def forward_with_logits(
        self,
        future_latents: torch.Tensor,
        past_controls: torch.Tensor,
        past_latents: torch.Tensor,
        return_debug: bool = False,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        decoded, debug = self.encode(future_latents, past_controls, past_latents)
        output: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {"actions": self.head(decoded)}
        if self.longitudinal_cls_head is not None and self.steer_cls_head is not None:
            output["longitudinal_logits"] = self.longitudinal_cls_head(decoded)
            output["steer_logits"] = self.steer_cls_head(decoded)
        if return_debug:
            output["debug"] = debug
        return output

    def encode(
        self,
        future_latents: torch.Tensor,
        past_controls: torch.Tensor,
        past_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if future_latents.ndim != 5:
            raise ValueError(f"Expected future_latents [B,T,C,H,W], got {tuple(future_latents.shape)}")
        if past_latents.ndim != 5:
            raise ValueError(f"Expected past_latents [B,T,C,H,W], got {tuple(past_latents.shape)}")
        if past_controls.ndim != 3:
            raise ValueError(f"Expected past_controls [B,T,D], got {tuple(past_controls.shape)}")
        b, t, c, _, _ = future_latents.shape
        if t != self.future_len:
            raise ValueError(f"Expected future_len={self.future_len}, got {t}")
        if c != self.latent_dim or past_latents.shape[2] != self.latent_dim:
            raise ValueError(
                f"Expected latent_dim={self.latent_dim}, got future={c}, past={past_latents.shape[2]}"
            )
        if past_controls.shape[0] != b or past_controls.shape[-1] != self.past_control_dim:
            raise ValueError(
                f"Expected past_controls [B,T,{self.past_control_dim}], got {tuple(past_controls.shape)}"
            )
        if past_latents.shape[0] != b or past_latents.shape[1] < 2:
            raise ValueError(f"Expected past_latents [B,T>=2,C,H,W], got {tuple(past_latents.shape)}")

        future_tokens = future_latents.mean(dim=(-1, -2))
        future_tokens = self.future_latent_projection(future_tokens)

        motion = past_latents[:, 1:] - past_latents[:, :-1]
        motion_tokens = motion.mean(dim=(-1, -2))
        motion_energy = motion.abs().mean(dim=(-1, -2, -3), keepdim=False).unsqueeze(-1)
        motion_tokens = self.motion_projection(torch.cat([motion_tokens, motion_energy], dim=-1))
        _, motion_context = self.motion_context(motion_tokens)
        motion_context = motion_context[-1]

        _, control_context = self.control_context(past_controls)
        control_context = control_context[-1]

        motion_repeated = motion_context.unsqueeze(1).expand(-1, t, -1)
        control_repeated = control_context.unsqueeze(1).expand(-1, t, -1)
        decoder_input = torch.cat([future_tokens, motion_repeated, control_repeated], dim=-1)
        decoded, _ = self.temporal_decoder(decoder_input)
        debug = {
            "future_latent_feature_norm": future_tokens.detach().norm(dim=-1).mean(),
            "motion_feature_norm": motion_context.detach().norm(dim=-1).mean(),
            "control_feature_norm": control_context.detach().norm(dim=-1).mean(),
            "motion_energy_mean": motion_energy.detach().mean(),
        }
        return decoded, debug


class MotionContextV2FutureActionPredictionHead(nn.Module):
    """Predict future actions from latent motion rather than absolute latents.

    This variant avoids feeding appearance-heavy absolute future latent tokens
    directly into the decoder.  It summarizes past latent motion, future latent
    motion, and past control context, then predicts absolute future
    [longitudinal, steer] targets from those temporal cues.
    """

    def __init__(
        self,
        latent_dim: int,
        past_control_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        future_len: int = 8,
        classification: bool = False,
        future_motion_scale: float = 1.0,
        spatial_pooling: str = "global",
        spatial_grid: tuple[int, int] | str = (1, 1),
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.past_control_dim = past_control_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.future_len = future_len
        self.classification = bool(classification)
        self.future_motion_scale = float(future_motion_scale)
        if spatial_pooling not in {"global", "grid"}:
            raise ValueError(f"spatial_pooling must be global or grid, got {spatial_pooling!r}")
        self.spatial_pooling = spatial_pooling
        self.spatial_grid = normalize_spatial_grid(spatial_grid)
        self.action_token_dim = latent_dim if spatial_pooling == "global" else latent_dim * self.spatial_grid[0] * self.spatial_grid[1]

        self.past_motion_projection = nn.Sequential(
            nn.Linear(self.action_token_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.future_motion_projection = nn.Sequential(
            nn.Linear(self.action_token_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.past_motion_context = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.future_motion_context = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.control_context = nn.GRU(
            input_size=past_control_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.temporal_decoder = nn.GRU(
            input_size=hidden_dim * 3,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.longitudinal_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.steer_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.longitudinal_cls_head = nn.Linear(hidden_dim, 3) if self.classification else None
        self.steer_cls_head = nn.Linear(hidden_dim, 3) if self.classification else None

    def forward(
        self,
        future_latents: torch.Tensor,
        past_controls: torch.Tensor,
        past_latents: torch.Tensor,
    ) -> torch.Tensor:
        decoded, _ = self.encode(future_latents, past_controls, past_latents)
        return self.regression(decoded)

    def forward_with_logits(
        self,
        future_latents: torch.Tensor,
        past_controls: torch.Tensor,
        past_latents: torch.Tensor,
        return_debug: bool = False,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        decoded, debug = self.encode(future_latents, past_controls, past_latents)
        output: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {"actions": self.regression(decoded)}
        if self.longitudinal_cls_head is not None and self.steer_cls_head is not None:
            output["longitudinal_logits"] = self.longitudinal_cls_head(decoded)
            output["steer_logits"] = self.steer_cls_head(decoded)
        if return_debug:
            output["debug"] = debug
        return output

    def regression(self, decoded: torch.Tensor) -> torch.Tensor:
        longitudinal = self.longitudinal_head(decoded)
        steer = self.steer_head(decoded)
        return torch.cat([longitudinal, steer], dim=-1)

    def _motion_tokens(self, latents: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        if latents.shape[1] < 2:
            raise ValueError(f"Expected {name} [B,T>=2,C,H,W], got {tuple(latents.shape)}")
        motion = latents[:, 1:] - latents[:, :-1]
        if self.spatial_pooling == "global":
            tokens = motion.mean(dim=(-1, -2))
        else:
            b, t, c, h, w = motion.shape
            grid = F.adaptive_avg_pool2d(motion.reshape(b * t, c, h, w), self.spatial_grid)
            tokens = grid.reshape(b, t, c * self.spatial_grid[0] * self.spatial_grid[1])
        energy = motion.abs().mean(dim=(-1, -2, -3), keepdim=False).unsqueeze(-1)
        return tokens, energy

    def encode(
        self,
        future_latents: torch.Tensor,
        past_controls: torch.Tensor,
        past_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if future_latents.ndim != 5:
            raise ValueError(f"Expected future_latents [B,T,C,H,W], got {tuple(future_latents.shape)}")
        if past_latents.ndim != 5:
            raise ValueError(f"Expected past_latents [B,T,C,H,W], got {tuple(past_latents.shape)}")
        if past_controls.ndim != 3:
            raise ValueError(f"Expected past_controls [B,T,D], got {tuple(past_controls.shape)}")
        b, t, c, _, _ = future_latents.shape
        if t != self.future_len:
            raise ValueError(f"Expected future_len={self.future_len}, got {t}")
        if c != self.latent_dim or past_latents.shape[2] != self.latent_dim:
            raise ValueError(
                f"Expected latent_dim={self.latent_dim}, got future={c}, past={past_latents.shape[2]}"
            )
        if past_controls.shape[0] != b or past_controls.shape[-1] != self.past_control_dim:
            raise ValueError(
                f"Expected past_controls [B,T,{self.past_control_dim}], got {tuple(past_controls.shape)}"
            )

        past_motion_tokens, past_motion_energy = self._motion_tokens(past_latents, "past_latents")
        past_motion_tokens = self.past_motion_projection(
            torch.cat([past_motion_tokens, past_motion_energy], dim=-1)
        )
        _, past_motion_context = self.past_motion_context(past_motion_tokens)
        past_motion_context = past_motion_context[-1]

        future_motion_tokens, future_motion_energy = self._motion_tokens(future_latents, "future_latents")
        future_motion_tokens = self.future_motion_projection(
            torch.cat([future_motion_tokens, future_motion_energy], dim=-1)
        )
        # future motion has T_future - 1 transitions. Repeat the last projected
        # transition to align one motion token with each predicted action step.
        future_motion_tokens = torch.cat([future_motion_tokens, future_motion_tokens[:, -1:]], dim=1)
        future_motion_encoded, _ = self.future_motion_context(future_motion_tokens)
        future_motion_encoded_pre_scale = future_motion_encoded
        future_motion_encoded = future_motion_encoded * self.future_motion_scale

        _, control_context = self.control_context(past_controls)
        control_context = control_context[-1]

        past_motion_repeated = past_motion_context.unsqueeze(1).expand(-1, t, -1)
        control_repeated = control_context.unsqueeze(1).expand(-1, t, -1)
        decoder_input = torch.cat([future_motion_encoded, past_motion_repeated, control_repeated], dim=-1)
        decoded, _ = self.temporal_decoder(decoder_input)
        debug = {
            "past_motion_feature_norm": past_motion_context.detach().norm(dim=-1).mean(),
            "future_motion_feature_norm": future_motion_encoded.detach().norm(dim=-1).mean(),
            "control_feature_norm": control_context.detach().norm(dim=-1).mean(),
            "past_motion_energy_mean": past_motion_energy.detach().mean(),
            "future_motion_energy_mean": future_motion_energy.detach().mean(),
            "decoder_feature_norm": decoded.detach().norm(dim=-1).mean(),
            "future_motion_scale": future_motion_encoded.detach().new_tensor(self.future_motion_scale),
            "spatial_pooling": self.spatial_pooling,
            "spatial_grid_h": future_motion_encoded.detach().new_tensor(self.spatial_grid[0]),
            "spatial_grid_w": future_motion_encoded.detach().new_tensor(self.spatial_grid[1]),
            "action_token_dim": future_motion_encoded.detach().new_tensor(self.action_token_dim),
            "future_latent_feature_norm_pre_scale": future_motion_encoded_pre_scale.detach().norm(dim=-1).mean(),
            "future_latent_feature_norm_post_scale": future_motion_encoded.detach().norm(dim=-1).mean(),
            # Compatibility aliases consumed by existing trainer logging.
            "motion_feature_norm": past_motion_context.detach().norm(dim=-1).mean(),
            "future_latent_feature_norm": future_motion_encoded.detach().norm(dim=-1).mean(),
            "motion_energy_mean": past_motion_energy.detach().mean(),
        }
        return decoded, debug
