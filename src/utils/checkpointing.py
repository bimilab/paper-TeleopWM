from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    metrics: dict[str, float],
    config: dict,
    best_val_loss: float | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "metrics": metrics,
        "config": config,
        "best_val_loss": best_val_loss,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, map_location="cpu") -> dict:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint
