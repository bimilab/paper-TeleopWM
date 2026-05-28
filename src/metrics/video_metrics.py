from __future__ import annotations

import torch
import torch.nn.functional as F


def mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(prediction - target))


def _gaussian_kernel(window_size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Compute mean SSIM over a rollout tensor shaped [B, T, C, H, W]."""

    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}")
    b, t, c, h, w = prediction.shape
    pred = prediction.reshape(b * t, c, h, w)
    true = target.reshape(b * t, c, h, w)

    kernel = _gaussian_kernel(window_size, sigma, c, pred.device, pred.dtype)
    padding = window_size // 2
    mu_x = F.conv2d(pred, kernel, padding=padding, groups=c)
    mu_y = F.conv2d(true, kernel, padding=padding, groups=c)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, kernel, padding=padding, groups=c) - mu_x2
    sigma_y2 = F.conv2d(true * true, kernel, padding=padding, groups=c) - mu_y2
    sigma_xy = F.conv2d(pred * true, kernel, padding=padding, groups=c) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    value = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return value.mean()


@torch.no_grad()
def video_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        "mae": float(mae(prediction, target).detach().cpu()),
        "ssim": float(ssim(prediction, target).detach().cpu()),
    }
