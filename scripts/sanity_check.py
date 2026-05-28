#!/usr/bin/env python3
"""Minimal release sanity check for TeleopWM imports and model construction."""
from __future__ import annotations

from src.models.teleopwm import TeleopWMConfig, build_teleopwm_model


def main() -> int:
    cfg = TeleopWMConfig(image_size=(64, 96), hid_s=8, hid_t=16, n_s=2, n_t=2)
    model = build_teleopwm_model(cfg)
    params = sum(parameter.numel() for parameter in model.parameters())
    print(f"TeleopWM sanity check passed: {params:,} parameters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
