from __future__ import annotations

import torch
from torch import nn

from .auxiliary_heads import LatentDynamicsHead
from .conditioning import (
    CONDITIONING_REPRESENTATION,
    FiLMConditioning,
    controls_to_longitudinal_steer_speed,
)
from .future_action_head import (
    FutureActionPredictionHead,
    MotionContextFutureActionPredictionHead,
    MotionContextV2FutureActionPredictionHead,
    normalize_spatial_grid,
)
from .latent_dynamics import ActionConditionedLatentDynamics, ActionLatentResidualDynamics
from .simvp_predictor import load_official_simvp_model


def normalize_model_variant(value: str) -> str:
    if value == "av":
        return "av_simvp"
    if value in {"rgb", "av_simvp", "av_wm", "av_wm_dual", "av_wm_dual_bigwm"}:
        return value
    raise ValueError(f"Unknown model_variant {value!r}")


def normalize_conditioning_stage(value: str) -> str:
    if value == "single":
        return "input"
    if value in {"input", "multipoint"}:
        return value
    raise ValueError(f"Unknown SimVP-backbone conditioning stage {value!r}")


class TeleopWMPredictor(nn.Module):
    """Final TeleopWM predictor built on an official SimVP backbone.

    Conditioning is intentionally small. Raw actions are converted from
    [throttle, steer, brake] to [longitudinal, steer, speed] where
    longitudinal = throttle - brake. This signed representation is used by all
    fusion modes, including FiLM.
    """

    requires_conditioning = True

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
        action_dim: int = 3,
        speed_dim: int = 1,
        conditioning_dim: int = 32,
        conditioning_fusion: str | None = None,
        conditioning_injection: str | None = None,
        model_variant: str = "av_simvp",
        simvp_conditioning: str | None = None,
        simvp_conditioning_stage: str | None = None,
        wm_latent_residual: bool = False,
        wm_residual_hidden_dim: int = 128,
        wm_residual_scale: float = 0.1,
        wm_residual_gated: bool = True,
        dual_fusion: str = "gated_add",
        dual_wm_scale: float = 1.0,
        dual_wm_hidden_dim: int = 128,
        dual_wm_num_layers: int = 3,
        dual_wm_conditioning: str = "add",
        dual_wm_gated: bool = True,
        aux_dynamics_hidden_dim: int | None = None,
        future_action_prediction: bool = False,
        future_action_hidden_dim: int = 128,
        future_action_num_layers: int = 1,
        future_action_dropout: float = 0.0,
        future_action_source: str = "final",
        future_action_classification: bool = False,
        future_action_head_variant: str = "default",
        future_action_detach_latents: bool = True,
        future_action_future_motion_scale: float = 1.0,
        future_action_spatial_pooling: str = "global",
        future_action_spatial_grid: tuple[int, int] | str = (1, 1),
        control_steer_input_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.model_variant = normalize_model_variant(model_variant)
        self.past_len = past_len
        self.future_len = future_len
        self.channels = channels
        self.image_size = image_size
        self.hid_s = hid_s
        self.action_dim = action_dim
        self.speed_dim = speed_dim
        self.conditioning_dim = conditioning_dim
        if simvp_conditioning is None:
            simvp_conditioning = conditioning_fusion if conditioning_fusion is not None else "concat"
        if simvp_conditioning_stage is None:
            simvp_conditioning_stage = conditioning_injection if conditioning_injection is not None else "input"
        simvp_conditioning_stage = normalize_conditioning_stage(simvp_conditioning_stage)
        self.simvp_conditioning = simvp_conditioning
        self.simvp_conditioning_stage = simvp_conditioning_stage
        self.conditioning_fusion = simvp_conditioning
        self.conditioning_injection = "multipoint" if simvp_conditioning_stage == "multipoint" else "single"
        self.conditioning_representation = CONDITIONING_REPRESENTATION
        if control_steer_input_scale <= 0:
            raise ValueError(f"control_steer_input_scale must be positive, got {control_steer_input_scale}")
        self.control_steer_input_scale = float(control_steer_input_scale)
        self.wm_latent_residual = bool(wm_latent_residual)
        self.wm_residual_scale = float(wm_residual_scale)
        self.wm_residual_gated = bool(wm_residual_gated)
        self.dual_fusion = dual_fusion
        self.dual_wm_scale = float(dual_wm_scale)
        self.dual_wm_hidden_dim = int(dual_wm_hidden_dim)
        self.dual_wm_num_layers = int(dual_wm_num_layers)
        if dual_wm_conditioning not in {"add", "concat", "film"}:
            raise ValueError(f"dual_wm_conditioning must be add, concat, or film; got {dual_wm_conditioning!r}")
        self.dual_wm_conditioning = dual_wm_conditioning
        self.dual_wm_gated = bool(dual_wm_gated)
        self.aux_dynamics_hidden_dim = aux_dynamics_hidden_dim
        self.future_action_prediction = bool(future_action_prediction)
        self.future_action_hidden_dim = int(future_action_hidden_dim)
        self.future_action_num_layers = int(future_action_num_layers)
        self.future_action_dropout = float(future_action_dropout)
        self.future_action_classification = bool(future_action_classification)
        self.future_action_future_motion_scale = float(future_action_future_motion_scale)
        if future_action_spatial_pooling not in {"global", "grid"}:
            raise ValueError(
                f"future_action_spatial_pooling must be global or grid; got {future_action_spatial_pooling!r}"
            )
        self.future_action_spatial_pooling = future_action_spatial_pooling
        self.future_action_spatial_grid = normalize_spatial_grid(future_action_spatial_grid)
        self.future_action_token_dim = (
            hid_s
            if self.future_action_spatial_pooling == "global"
            else hid_s * self.future_action_spatial_grid[0] * self.future_action_spatial_grid[1]
        )
        if future_action_head_variant not in {"default", "motion_context", "motion_context_v2"}:
            raise ValueError(
                "future_action_head_variant must be default, motion_context, or "
                f"motion_context_v2; got {future_action_head_variant!r}"
            )
        self.future_action_head_variant = future_action_head_variant
        self.future_action_detach_latents = bool(future_action_detach_latents)
        if future_action_source not in {"final", "wm", "simvp"}:
            raise ValueError(f"future_action_source must be one of final, wm, simvp; got {future_action_source!r}")
        self.future_action_source = future_action_source
        if simvp_conditioning not in {"none", "add", "concat", "film"}:
            raise ValueError(f"Unknown simvp_conditioning {simvp_conditioning!r}")
        if self.model_variant == "av_wm" and not self.wm_latent_residual:
            raise ValueError("model_variant='av_wm' currently requires wm_latent_residual=True")
        if dual_fusion not in {"add", "gated_add", "convex", "wm_only", "simvp_only", "conv1x1"}:
            raise ValueError(f"Unknown dual_fusion {dual_fusion!r}")

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
        self.conditioning_input_dim = 2 + speed_dim
        if action_dim != 3:
            raise ValueError(
                "TeleopWMPredictor expects raw MILE actions [throttle, steer, brake] "
                f"so action_dim must be 3, got {action_dim}"
            )
        if speed_dim != 1:
            raise ValueError(f"TeleopWMPredictor expects scalar speed so speed_dim must be 1, got {speed_dim}")

        self.film_pre_temporal = None
        self.film_pre_decoder = None
        if simvp_conditioning in {"none", "add"}:
            output_dim = hid_s
            self.fusion_projection_pre_temporal = None
            self.fusion_projection_pre_decoder = None
        elif simvp_conditioning == "film":
            output_dim = conditioning_dim
            self.fusion_projection_pre_temporal = None
            self.fusion_projection_pre_decoder = None
            self.film_pre_temporal = FiLMConditioning(
                cond_dim=conditioning_dim,
                channels=hid_s,
                hidden_dim=conditioning_dim,
            )
            self.film_pre_decoder = FiLMConditioning(
                cond_dim=conditioning_dim,
                channels=hid_s,
                hidden_dim=conditioning_dim,
            )
        else:
            output_dim = conditioning_dim
            self.fusion_projection_pre_temporal = nn.Conv2d(
                hid_s + conditioning_dim, hid_s, kernel_size=1
            )
            self.fusion_projection_pre_decoder = nn.Conv2d(
                hid_s + conditioning_dim, hid_s, kernel_size=1
            )
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.conditioning_input_dim, conditioning_dim),
            # SiLU preserves signed longitudinal/braking information better than
            # a hard ReLU-only conditioning path.
            nn.SiLU(),
            nn.Linear(conditioning_dim, output_dim),
        )
        self.wm_residual_dynamics = (
            ActionLatentResidualDynamics(
                latent_dim=hid_s,
                conditioning_dim=self.conditioning_input_dim,
                hidden_dim=wm_residual_hidden_dim,
                gated=wm_residual_gated,
            )
            if self.wm_latent_residual
            else None
        )
        self.dual_wm_dynamics = (
            ActionConditionedLatentDynamics(
                latent_dim=hid_s,
                past_len=past_len,
                future_len=future_len,
                conditioning_dim=self.conditioning_input_dim,
                hidden_dim=dual_wm_hidden_dim,
                gated=dual_wm_gated,
                num_blocks=dual_wm_num_layers,
                conditioning_mode=dual_wm_conditioning,
            )
            if self.model_variant in {"av_wm_dual", "av_wm_dual_bigwm"}
            else None
        )
        self.dual_conv1x1_fusion = (
            nn.Conv2d(2 * hid_s, hid_s, kernel_size=1)
            if self.model_variant in {"av_wm_dual", "av_wm_dual_bigwm"} and dual_fusion == "conv1x1"
            else None
        )
        if self.dual_conv1x1_fusion is not None:
            self._init_dual_conv1x1_fusion()
        self.aux_dynamics_head = (
            LatentDynamicsHead(latent_dim=hid_s, hidden_dim=aux_dynamics_hidden_dim, output_dim=3)
            if aux_dynamics_hidden_dim is not None
            else None
        )
        future_action_head_classes = {
            "default": FutureActionPredictionHead,
            "motion_context": MotionContextFutureActionPredictionHead,
            "motion_context_v2": MotionContextV2FutureActionPredictionHead,
        }
        future_action_head_cls = future_action_head_classes[self.future_action_head_variant]
        self.future_action_head = (
            future_action_head_cls(
                latent_dim=hid_s,
                past_control_dim=self.conditioning_input_dim,
                hidden_dim=future_action_hidden_dim,
                num_layers=future_action_num_layers,
                dropout=future_action_dropout,
                future_len=future_len,
                classification=future_action_classification,
                future_motion_scale=future_action_future_motion_scale,
                spatial_pooling=self.future_action_spatial_pooling,
                spatial_grid=self.future_action_spatial_grid,
            )
            if self.future_action_prediction
            else None
        )

    def _init_dual_conv1x1_fusion(self) -> None:
        """Initialize conv1x1 fusion as approximately SimVP-only.

        The first C input channels are z_simvp and the second C channels are
        dual_wm_scale * z_wm.  Identity on the SimVP half plus zeros on the WM
        half keeps initial behavior stable while still allowing the optimizer to
        learn direct latent-channel mixing without a saturating sigmoid gate.
        """

        if self.dual_conv1x1_fusion is None:
            return
        with torch.no_grad():
            self.dual_conv1x1_fusion.weight.zero_()
            self.dual_conv1x1_fusion.bias.zero_()
            channels = self.dual_conv1x1_fusion.out_channels
            for channel in range(channels):
                self.dual_conv1x1_fusion.weight[channel, channel, 0, 0] = 1.0

    def forward(
        self,
        past_frames: torch.Tensor,
        past_actions: torch.Tensor,
        past_speed: torch.Tensor,
        return_latents: bool = False,
        decode_wm_frames: bool = False,
        decode_simvp_frames: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if past_frames.ndim != 5:
            raise ValueError(f"Expected frames [B, T, C, H, W], got {tuple(past_frames.shape)}")
        if past_actions.ndim != 3:
            raise ValueError(f"Expected actions [B, T, A], got {tuple(past_actions.shape)}")
        if past_speed.ndim != 3:
            raise ValueError(f"Expected speed [B, T, S], got {tuple(past_speed.shape)}")
        if past_frames.shape[1] != self.past_len:
            raise ValueError(f"Expected {self.past_len} input frames, got {past_frames.shape[1]}")
        if past_actions.shape[:2] != past_frames.shape[:2] or past_speed.shape[:2] != past_frames.shape[:2]:
            raise ValueError("Frame/action/speed temporal dimensions must match")

        b, t, c, h, w = past_frames.shape
        embed, skip = self.encode_frames(past_frames, return_skip=True)
        _, latent_c, latent_h, latent_w = embed.shape

        z = embed.view(b, t, latent_c, latent_h, latent_w)
        cond_input = self.controls_to_conditioning(past_actions, past_speed)
        cond = self.condition_encoder(cond_input)
        if self.simvp_conditioning != "none":
            z = self._fuse_condition(
                z,
                cond,
                projection=self.fusion_projection_pre_temporal,
                film=self.film_pre_temporal,
            )

        simvp_hid = self.model.hid(z)
        hid = simvp_hid
        residual_info: dict[str, torch.Tensor | None] = {
            "delta": None,
            "gate": None,
            "applied": None,
        }
        dual_info: dict[str, torch.Tensor | str | None] = {
            "wm_latents": None,
            "gate": None,
            "gate_logits": None,
            "final_latents": None,
            "fusion": self.dual_fusion,
        }
        if self.dual_wm_dynamics is not None:
            dual_info = self._apply_dual_wm_fusion(z, simvp_hid, cond_input)
            hid = dual_info["final_latents"]

        if self.wm_residual_dynamics is not None:
            residual_info = self.wm_residual_dynamics(z, cond_input)
            delta = residual_info["delta"]
            gate = residual_info["gate"]
            if delta is None:
                raise RuntimeError("WM residual dynamics did not return delta.")
            applied = delta if gate is None else gate * delta
            applied = self.wm_residual_scale * applied
            residual_info["applied"] = applied
            hid = simvp_hid + applied

        if self.simvp_conditioning != "none" and self.simvp_conditioning_stage == "multipoint":
            hid = self._fuse_condition(
                hid,
                cond,
                projection=self.fusion_projection_pre_decoder,
                film=self.film_pre_decoder,
            )
        future_latents = hid[:, : self.future_len]
        prediction = self.decode_latents(hid, skip, output_shape=(b, t, c, h, w))
        frames = prediction[:, : self.future_len]
        if return_latents:
            output: dict[str, torch.Tensor | str | None] = {
                "frames": frames,
                "final_latents": future_latents,
                "future_latents": future_latents,
                "simvp_future_latents": simvp_hid[:, : self.future_len],
            }
            if dual_info["wm_latents"] is not None:
                output["wm_future_latents"] = dual_info["wm_latents"]
                output["dual_gate"] = dual_info["gate"]
                output["dual_gate_logits"] = dual_info.get("gate_logits")
                output["dual_fusion"] = self.dual_fusion
                if self.dual_conv1x1_fusion is not None:
                    output["dual_conv1x1_fusion_stats"] = self.dual_conv1x1_fusion_stats()
                if decode_wm_frames:
                    wm_block = self._future_latents_to_block(dual_info["wm_latents"], simvp_hid)
                    output["frames_wm"] = self.decode_latents(wm_block, skip, output_shape=(b, t, c, h, w))[
                        :, : self.future_len
                    ]
                if decode_simvp_frames:
                    output["frames_simvp"] = self.decode_latents(simvp_hid, skip, output_shape=(b, t, c, h, w))[
                        :, : self.future_len
                    ]
            if self.future_action_head is not None:
                action_latents = self._select_future_action_latents(
                    future_latents=future_latents,
                    wm_future_latents=output.get("wm_future_latents"),
                    simvp_future_latents=output.get("simvp_future_latents"),
                )
                action_past_latents = z
                if self.future_action_detach_latents:
                    action_latents = action_latents.detach()
                    action_past_latents = action_past_latents.detach()
                if self.future_action_head_variant in {"motion_context", "motion_context_v2"}:
                    action_output = self.future_action_head.forward_with_logits(
                        action_latents,
                        cond_input,
                        action_past_latents,
                        return_debug=True,
                    )
                    output["pred_future_actions"] = action_output["actions"]
                    output["future_action_feature_norms"] = action_output.get("debug")
                    if getattr(self.future_action_head, "classification", False):
                        output["pred_future_longitudinal_logits"] = action_output.get("longitudinal_logits")
                        output["pred_future_steer_logits"] = action_output.get("steer_logits")
                elif getattr(self.future_action_head, "classification", False):
                    action_output = self.future_action_head.forward_with_logits(action_latents, cond_input)
                    output["pred_future_actions"] = action_output["actions"]
                    output["pred_future_longitudinal_logits"] = action_output.get("longitudinal_logits")
                    output["pred_future_steer_logits"] = action_output.get("steer_logits")
                else:
                    output["pred_future_actions"] = self.future_action_head(action_latents, cond_input)
                output["future_action_source"] = self.future_action_source
                output["future_action_head_variant"] = self.future_action_head_variant
                output["future_action_detach_latents"] = self.future_action_detach_latents
                output["future_action_future_motion_scale"] = self.future_action_future_motion_scale
                output["future_action_spatial_pooling"] = self.future_action_spatial_pooling
                output["future_action_spatial_grid"] = self.future_action_spatial_grid
                output["future_action_token_dim"] = self.future_action_token_dim
                output["control_steer_input_scale"] = self.control_steer_input_scale
            if residual_info["delta"] is not None:
                output["wm_latent_residual"] = residual_info["delta"][:, : self.future_len]
                output["wm_latent_residual_gate"] = (
                    residual_info["gate"][:, : self.future_len] if residual_info["gate"] is not None else None
                )
                output["wm_latent_residual_applied"] = residual_info["applied"][:, : self.future_len]
            return output
        return frames

    def _select_future_action_latents(
        self,
        future_latents: torch.Tensor,
        wm_future_latents: torch.Tensor | None,
        simvp_future_latents: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.future_action_source == "final":
            return future_latents
        if self.future_action_source == "wm":
            if wm_future_latents is None:
                raise RuntimeError("future_action_source='wm' requires wm_future_latents; use model_variant='av_wm_dual' or 'av_wm_dual_bigwm'.")
            return wm_future_latents
        if self.future_action_source == "simvp":
            if simvp_future_latents is None:
                raise RuntimeError("future_action_source='simvp' requires simvp_future_latents.")
            return simvp_future_latents
        raise ValueError(f"Unknown future_action_source {self.future_action_source!r}")

    def controls_to_conditioning(self, actions: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        return controls_to_longitudinal_steer_speed(
            actions,
            speed,
            steer_input_scale=self.control_steer_input_scale,
        )

    def conditioning_input_stats(self, actions: torch.Tensor, speed: torch.Tensor) -> dict[str, float]:
        cond = self.controls_to_conditioning(actions, speed)
        steer = cond[..., 1].detach().float()
        raw_steer = actions[..., 1].detach().float()
        return {
            "scaled_steer_mean": float(steer.mean().cpu()),
            "scaled_steer_std": float(steer.std(unbiased=False).cpu()),
            "scaled_steer_min": float(steer.min().cpu()),
            "scaled_steer_max": float(steer.max().cpu()),
            "scaled_steer_saturation_fraction": float((raw_steer.abs() >= self.control_steer_input_scale).float().mean().cpu())
            if self.control_steer_input_scale != 1.0
            else 0.0,
        }

    def _future_latents_to_block(self, future_latents: torch.Tensor, fallback_block: torch.Tensor) -> torch.Tensor:
        if fallback_block.shape[1] == future_latents.shape[1]:
            return future_latents
        if fallback_block.shape[1] < future_latents.shape[1]:
            raise ValueError("Fallback latent block is shorter than future latents.")
        return torch.cat([future_latents, fallback_block[:, future_latents.shape[1] :]], dim=1)

    def _apply_dual_wm_fusion(
        self,
        z_past: torch.Tensor,
        z_simvp: torch.Tensor,
        cond_input: torch.Tensor,
    ) -> dict[str, torch.Tensor | str | None]:
        if self.dual_wm_dynamics is None:
            raise RuntimeError("Dual WM dynamics module is not initialized.")
        dual = self.dual_wm_dynamics(z_past, cond_input)
        z_wm = dual["latents"]
        gate = dual["gate"]
        gate_logits = dual.get("gate_logits")
        z_simvp_future = z_simvp[:, : self.future_len]
        if z_wm is None:
            raise RuntimeError("Dual WM dynamics did not return latents.")

        if self.dual_fusion == "simvp_only":
            z_final_future = z_simvp_future
        elif self.dual_fusion == "wm_only":
            z_final_future = z_wm
        elif self.dual_fusion == "convex":
            if gate is None:
                raise RuntimeError("convex dual fusion requires a gate.")
            z_final_future = (1.0 - gate) * z_simvp_future + gate * z_wm
        elif self.dual_fusion == "gated_add":
            if gate is None:
                raise RuntimeError("gated_add dual fusion requires a gate.")
            z_final_future = z_simvp_future + self.dual_wm_scale * gate * (z_wm - z_simvp_future)
        elif self.dual_fusion == "add":
            z_final_future = z_simvp_future + self.dual_wm_scale * (z_wm - z_simvp_future.detach())
        elif self.dual_fusion == "conv1x1":
            if self.dual_conv1x1_fusion is None:
                raise RuntimeError("conv1x1 dual fusion layer is not initialized.")
            z_final_future = self._apply_dual_conv1x1_fusion(z_simvp_future, z_wm)
        else:
            raise ValueError(f"Unknown dual_fusion {self.dual_fusion!r}")

        if z_simvp.shape[1] > self.future_len:
            z_final = torch.cat([z_final_future, z_simvp[:, self.future_len :]], dim=1)
        else:
            z_final = z_final_future
        return {
            "wm_latents": z_wm,
            "gate": gate,
            "gate_logits": gate_logits,
            "final_latents": z_final,
            "fusion": self.dual_fusion,
        }

    def _apply_dual_conv1x1_fusion(self, z_simvp: torch.Tensor, z_wm: torch.Tensor) -> torch.Tensor:
        if z_simvp.shape != z_wm.shape:
            raise ValueError(f"conv1x1 fusion requires matching latent shapes, got {tuple(z_simvp.shape)} and {tuple(z_wm.shape)}")
        if self.dual_conv1x1_fusion is None:
            raise RuntimeError("conv1x1 fusion layer is not initialized.")
        b, t, c, h, w = z_simvp.shape
        z_cat = torch.cat([z_simvp, self.dual_wm_scale * z_wm], dim=2)
        fused = self.dual_conv1x1_fusion(z_cat.reshape(b * t, 2 * c, h, w))
        return fused.reshape(b, t, c, h, w)

    def dual_conv1x1_fusion_stats(self) -> dict[str, float | int | bool]:
        if self.dual_conv1x1_fusion is None:
            return {
                "enabled": False,
                "param_count": 0,
                "weight_norm": 0.0,
                "bias_norm": 0.0,
                "simvp_half_weight_norm": 0.0,
                "wm_half_weight_norm": 0.0,
                "wm_to_simvp_weight_norm_ratio": 0.0,
            }
        weight = self.dual_conv1x1_fusion.weight.detach()
        bias = self.dual_conv1x1_fusion.bias.detach() if self.dual_conv1x1_fusion.bias is not None else None
        channels = weight.shape[0]
        simvp_half = weight[:, :channels]
        wm_half = weight[:, channels:]
        simvp_norm = float(simvp_half.double().norm().cpu())
        wm_norm = float(wm_half.double().norm().cpu())
        return {
            "enabled": True,
            "param_count": sum(parameter.numel() for parameter in self.dual_conv1x1_fusion.parameters()),
            "weight_norm": float(weight.double().norm().cpu()),
            "bias_norm": float(bias.double().norm().cpu()) if bias is not None else 0.0,
            "simvp_half_weight_norm": simvp_norm,
            "wm_half_weight_norm": wm_norm,
            "wm_to_simvp_weight_norm_ratio": wm_norm / simvp_norm if simvp_norm > 0.0 else 0.0,
        }

    def encode_frames(self, frames: torch.Tensor, return_skip: bool = False):
        if frames.ndim != 5:
            raise ValueError(f"Expected frames [B,T,C,H,W], got {tuple(frames.shape)}")
        b, t, c, h, w = frames.shape
        embed, skip = self.model.enc(frames.reshape(b * t, c, h, w))
        return (embed, skip) if return_skip else embed.view(b, t, embed.shape[1], embed.shape[2], embed.shape[3])

    def decode_latents(self, latents: torch.Tensor, skip: torch.Tensor, output_shape: tuple[int, int, int, int, int]) -> torch.Tensor:
        b, t, c, h, w = output_shape
        decoded = self.model.dec(latents.reshape(b * t, latents.shape[2], latents.shape[3], latents.shape[4]), skip)
        return decoded.reshape(b, t, c, h, w)

    def _fuse_condition(
        self,
        z: torch.Tensor,
        cond: torch.Tensor,
        projection: nn.Conv2d | None,
        film: FiLMConditioning | None,
    ) -> torch.Tensor:
        b, t, latent_c, latent_h, latent_w = z.shape
        if self.simvp_conditioning == "none":
            return z
        if self.simvp_conditioning == "add":
            if cond.shape[-1] != latent_c:
                raise ValueError(f"Add conditioning expected {latent_c} channels, got {cond.shape[-1]}")
            return z + cond[:, :, :, None, None]

        if self.simvp_conditioning == "film":
            if film is None:
                raise RuntimeError("FiLM conditioning module is not initialized.")
            if cond.shape[-1] != self.conditioning_dim:
                raise ValueError(f"FiLM conditioning expected {self.conditioning_dim} channels, got {cond.shape[-1]}")
            return film(z, cond)

        if projection is None:
            raise RuntimeError("Concat conditioning requires a fusion projection layer.")
        cond_map = cond[:, :, :, None, None].expand(-1, -1, -1, latent_h, latent_w)
        z_cat = torch.cat([z, cond_map], dim=2)
        z_cat = z_cat.reshape(b * t, latent_c + self.conditioning_dim, latent_h, latent_w)
        fused = projection(z_cat)
        return fused.reshape(b, t, latent_c, latent_h, latent_w)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        migrated_state = dict(state_dict)
        old_weight = "fusion_projection.weight"
        old_bias = "fusion_projection.bias"
        if old_weight in migrated_state:
            migrated_state.setdefault(
                "fusion_projection_pre_temporal.weight",
                migrated_state[old_weight],
            )
            migrated_state.setdefault(
                "fusion_projection_pre_decoder.weight",
                migrated_state[old_weight],
            )
            migrated_state.pop(old_weight)
        if old_bias in migrated_state:
            migrated_state.setdefault(
                "fusion_projection_pre_temporal.bias",
                migrated_state[old_bias],
            )
            migrated_state.setdefault(
                "fusion_projection_pre_decoder.bias",
                migrated_state[old_bias],
            )
            migrated_state.pop(old_bias)

        try:
            return super().load_state_dict(migrated_state, strict=strict, assign=assign)
        except RuntimeError as exc:
            message = str(exc)
            if "condition_encoder.0.weight" in message:
                raise RuntimeError(
                    "Checkpoint conditioning input dimension is incompatible with this TeleopWM model. "
                    "Current code uses conditioning representation [longitudinal, steer, speed] "
                    "instead of old [throttle, steer, brake, speed]. Retrain or load with a "
                    "matching pre-FiLM code revision."
                ) from exc
            raise
        except TypeError:
            try:
                return super().load_state_dict(migrated_state, strict=strict)
            except RuntimeError as exc:
                message = str(exc)
                if "condition_encoder.0.weight" in message:
                    raise RuntimeError(
                        "Checkpoint conditioning input dimension is incompatible with this TeleopWM model. "
                        "Current code uses conditioning representation [longitudinal, steer, speed] "
                        "instead of old [throttle, steer, brake, speed]. Retrain or load with a "
                        "matching pre-FiLM code revision."
                    ) from exc
                raise


# Backward-compatibility alias for older scripts/checkpoints that imported the
# development-era class name. New public code should use TeleopWMPredictor.
SimVPAVPredictor = TeleopWMPredictor
