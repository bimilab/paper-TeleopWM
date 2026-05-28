from __future__ import annotations

from dataclasses import dataclass

from .teleopwm_predictor import TeleopWMPredictor


@dataclass(frozen=True)
class TeleopWMConfig:
    """Configuration for the release TeleopWM model.

    The defaults match the final paper configuration. Legacy-compatible
    variant names are still passed internally so existing checkpoints remain
    loadable.
    """

    past_len: int = 9
    future_len: int = 8
    channels: int = 3
    image_size: tuple[int, int] = (320, 512)
    hid_s: int = 32
    hid_t: int = 256
    n_s: int = 4
    n_t: int = 4
    model_type: str = "gSTA"
    drop_path: float = 0.0
    dual_fusion: str = "conv1x1"
    dual_wm_hidden_dim: int = 512
    dual_wm_num_layers: int = 3
    dual_wm_conditioning: str = "film"
    future_action_hidden_dim: int = 256
    future_action_source: str = "wm"
    future_action_head_variant: str = "motion_context_v2"
    future_action_future_motion_scale: float = 3.0
    future_action_spatial_pooling: str = "grid"
    future_action_spatial_grid: str = "2x4"
    future_action_classification: bool = True
    future_action_detach_latents: bool = False
    control_steer_input_scale: float = 0.30


class TeleopWM(TeleopWMPredictor):
    """Release-facing TeleopWM model class."""

    def __init__(self, config: TeleopWMConfig | None = None, **overrides) -> None:
        cfg = config or TeleopWMConfig()
        values = {**cfg.__dict__, **overrides}
        super().__init__(
            past_len=values["past_len"],
            future_len=values["future_len"],
            channels=values["channels"],
            image_size=values["image_size"],
            hid_s=values["hid_s"],
            hid_t=values["hid_t"],
            n_s=values["n_s"],
            n_t=values["n_t"],
            model_type=values["model_type"],
            drop_path=values["drop_path"],
            model_variant="av_wm_dual_bigwm",
            simvp_conditioning="none",
            dual_fusion=values["dual_fusion"],
            dual_wm_hidden_dim=values["dual_wm_hidden_dim"],
            dual_wm_num_layers=values["dual_wm_num_layers"],
            dual_wm_conditioning=values["dual_wm_conditioning"],
            future_action_prediction=True,
            future_action_hidden_dim=values["future_action_hidden_dim"],
            future_action_source=values["future_action_source"],
            future_action_classification=values["future_action_classification"],
            future_action_head_variant=values["future_action_head_variant"],
            future_action_detach_latents=values["future_action_detach_latents"],
            future_action_future_motion_scale=values["future_action_future_motion_scale"],
            future_action_spatial_pooling=values["future_action_spatial_pooling"],
            future_action_spatial_grid=values["future_action_spatial_grid"],
            control_steer_input_scale=values["control_steer_input_scale"],
        )


def build_teleopwm_model(config: TeleopWMConfig | None = None, **overrides) -> TeleopWM:
    return TeleopWM(config=config, **overrides)
