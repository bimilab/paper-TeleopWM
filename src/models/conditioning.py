from __future__ import annotations

import torch
from torch import nn


CONDITIONING_REPRESENTATION = "longitudinal_steer_speed"


def controls_to_longitudinal_steer_speed(
    actions: torch.Tensor,
    speed: torch.Tensor,
    steer_input_scale: float = 1.0,
) -> torch.Tensor:
    """Convert raw MILE controls to signed conditioning controls.

    Raw dataset actions are [throttle, steer, brake].  Conditioning uses
    [longitudinal, steer, speed], where longitudinal = throttle - brake.
    ``steer_input_scale`` optionally scales only the model-conditioning steering
    channel as clamp(steer / scale, -1, 1).  Future action targets use their own
    target scaling and should call this with the default scale unless they
    explicitly want input-conditioning units.
    """

    if actions.ndim != 3:
        raise ValueError(f"Expected actions [B, T, 3], got {tuple(actions.shape)}")
    if speed.ndim != 3:
        raise ValueError(f"Expected speed [B, T, S], got {tuple(speed.shape)}")
    if actions.shape[-1] < 3:
        raise ValueError(f"Expected action channels [throttle, steer, brake], got {actions.shape[-1]}")
    if speed.shape[-1] != 1:
        raise ValueError(f"Expected speed shape [B, T, 1], got {tuple(speed.shape)}")
    if actions.shape[:2] != speed.shape[:2]:
        raise ValueError("Action and speed batch/time dimensions must match")

    if steer_input_scale <= 0:
        raise ValueError(f"steer_input_scale must be positive, got {steer_input_scale}")

    longitudinal = actions[..., 0:1] - actions[..., 2:3]
    steer = actions[..., 1:2]
    if steer_input_scale != 1.0:
        steer = torch.clamp(steer / float(steer_input_scale), -1.0, 1.0)
    return torch.cat([longitudinal, steer, speed], dim=-1)


class FiLMConditioning(nn.Module):
    """Feature-wise linear modulation for 4D or 5D feature tensors.

    The final projection is initialized to zero, making gamma = 1 and beta = 0
    at initialization.  SiLU keeps the conditioning path smooth for signed
    longitudinal controls.
    """

    def __init__(self, cond_dim: int, channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, channels * 2),
        )
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, features: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        if features.ndim == 5:
            if conditioning.ndim != 3:
                raise ValueError(
                    f"Expected conditioning [B, T, D] for 5D features, got {tuple(conditioning.shape)}"
                )
            b, t, c, _, _ = features.shape
            if conditioning.shape[:2] != (b, t):
                raise ValueError("FiLM conditioning batch/time dimensions must match features")
            params = self.net(conditioning).view(b, t, 2, c, 1, 1)
            gamma_delta = params[:, :, 0]
            beta = params[:, :, 1]
        elif features.ndim == 4:
            if conditioning.ndim != 2:
                raise ValueError(
                    f"Expected conditioning [B, D] for 4D features, got {tuple(conditioning.shape)}"
                )
            b, c, _, _ = features.shape
            if conditioning.shape[0] != b:
                raise ValueError("FiLM conditioning batch dimension must match features")
            params = self.net(conditioning).view(b, 2, c, 1, 1)
            gamma_delta = params[:, 0]
            beta = params[:, 1]
        else:
            raise ValueError(f"Expected feature tensor [B,C,H,W] or [B,T,C,H,W], got {tuple(features.shape)}")

        gamma = 1.0 + gamma_delta
        return gamma * features + beta
