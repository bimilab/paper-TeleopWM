from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
SIMVP_ROOT = REPO_ROOT / "external" / "SimVPv2"


def _load_module(module_name: str, path: Path, *, is_package: bool = False):
    kwargs = {}
    if is_package:
        kwargs["submodule_search_locations"] = [str(path.parent)]
    spec = importlib.util.spec_from_file_location(module_name, path, **kwargs)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_official_simvp_model():
    """Load the official SimVP model without importing unrelated OpenSTL models."""

    modules_dir = SIMVP_ROOT / "openstl" / "modules"
    models_dir = SIMVP_ROOT / "openstl" / "models"

    if not SIMVP_ROOT.exists():
        raise FileNotFoundError(f"Missing official SimVPv2 checkout: {SIMVP_ROOT}")

    openstl_pkg = sys.modules.setdefault("openstl", types.ModuleType("openstl"))
    openstl_pkg.__path__ = [str(SIMVP_ROOT / "openstl")]

    modules_pkg = types.ModuleType("openstl.modules")
    modules_pkg.__path__ = [str(modules_dir)]
    sys.modules["openstl.modules"] = modules_pkg
    setattr(openstl_pkg, "modules", modules_pkg)

    _load_module(
        "openstl.modules.layers",
        modules_dir / "layers" / "__init__.py",
        is_package=True,
    )
    simvp_modules = _load_module(
        "openstl.modules.simvp_modules", modules_dir / "simvp_modules.py"
    )

    required_symbols = [
        "ConvSC",
        "ConvNeXtSubBlock",
        "ConvMixerSubBlock",
        "GASubBlock",
        "gInception_ST",
        "HorNetSubBlock",
        "MLPMixerSubBlock",
        "MogaSubBlock",
        "PoolFormerSubBlock",
        "SwinSubBlock",
        "UniformerSubBlock",
        "VANSubBlock",
        "ViTSubBlock",
        "TAUSubBlock",
    ]
    for symbol in required_symbols:
        setattr(modules_pkg, symbol, getattr(simvp_modules, symbol))

    simvp_model = _load_module("simvp_model_baseline", models_dir / "simvp_model.py")
    return simvp_model.SimVP_Model


class SimVPPredictor(nn.Module):
    """Thin RGB-only compatibility wrapper around the official SimVP model.

    Official SimVP predicts an output block with the same length as the input
    block. For this wrapper we use the first `future_len` frames of that block
    for the 9 -> 8 rollout target, leaving the official model untouched.
    """

    requires_conditioning = False

    def __init__(
        self,
        past_len: int = 9,
        future_len: int = 8,
        channels: int = 3,
        image_size: tuple[int, int] = (160, 256),
        hid_s: int = 32,
        hid_t: int = 256,
        n_s: int = 4,
        n_t: int = 4,
        model_type: str = "gSTA",
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.past_len = past_len
        self.future_len = future_len
        self.channels = channels
        self.image_size = image_size

        height, width = image_size
        SimVPModel = load_official_simvp_model()
        self.model = SimVPModel(
            in_shape=(past_len, channels, height, width),
            hid_S=hid_s,
            hid_T=hid_t,
            N_S=n_s,
            N_T=n_t,
            model_type=model_type,
            drop_path=drop_path,
        )

    def forward(self, past_frames: torch.Tensor) -> torch.Tensor:
        if past_frames.ndim != 5:
            raise ValueError(f"Expected [B, T, C, H, W], got {tuple(past_frames.shape)}")
        if past_frames.shape[1] != self.past_len:
            raise ValueError(f"Expected {self.past_len} input frames, got {past_frames.shape[1]}")

        predicted_block = self.model(past_frames)
        return predicted_block[:, : self.future_len]
