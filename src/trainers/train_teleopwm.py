from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.losses import frame_l1_loss, frame_ssim_loss
from src.metrics import video_metrics
from src.models import controls_to_longitudinal_steer_speed
from src.utils import append_jsonl, create_run_dir, infer_run_dir_from_checkpoint, save_checkpoint, write_json


PROGRESS_KEYS = (
    "loss",
    "avg",
    "img_mae",
    "ssim",
    "act",
    "act_reg",
    "act_corr",
    "act_delta",
    "long_mae",
    "steer_mae",
    "act_cls",
    "long_acc",
    "steer_acc",
    "mot_norm",
    "ctl_norm",
    "lat_norm",
    "wm/simvp",
    "gate",
    "dual_gate",
    "grad",
    "gstep",
    "estep",
)


@dataclass
class TrainerConfig:
    output_dir: str = "outputs/teleopwm"
    run_tag: str = "teleopwm"
    epochs: int = 10
    batch_size: int = 2
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    device: str = "cuda"
    amp: bool = True
    log_interval: int = 20
    save_every: int = 1
    grad_clip_norm: float | None = 1.0
    max_train_steps: int | None = None
    max_val_batches: int | None = None
    max_interim_val_batches: int | None = None
    eval_every_steps: int | None = None
    stop_on_nan: bool = False
    log_grad_norms: bool = False
    debug_activation_stats: bool = False
    resume_checkpoint: str | None = None
    aux_dynamics_loss: bool = False
    aux_dynamics_weight: float = 0.05
    aux_dynamics_loss_type: str = "smooth_l1"
    progress_bar: bool = True
    wm_residual_loss: bool = False
    wm_residual_loss_weight: float = 0.05
    wm_residual_loss_type: str = "smooth_l1"
    ssim_loss_weight: float = 0.0
    dual_wm_image_loss_weight: float = 0.0
    dual_simvp_image_loss_weight: float = 0.0
    dual_align_loss_weight: float = 0.0
    dual_align_loss_type: str = "smooth_l1"
    dual_align_direction: str = "simvp_to_wm"
    dual_detach_simvp_target: bool = True
    future_action_loss: bool = False
    future_action_loss_weight: float = 0.1
    future_action_loss_type: str = "smooth_l1"
    future_action_source: str = "final"
    future_steer_target_scale: float = 1.0
    control_steer_input_scale: float = 1.0
    future_action_head_variant: str = "default"
    future_action_detach_latents: bool = True
    future_action_future_motion_scale: float = 1.0
    future_action_corr_loss_weight: float = 0.0
    future_action_delta_loss: bool = False
    future_action_delta_loss_weight: float = 0.0
    future_action_delta_loss_type: str = "smooth_l1"
    future_action_delta_longitudinal_weight: float = 1.0
    future_action_delta_steer_weight: float = 1.0
    future_action_cls_loss: bool = False
    future_action_cls_weight: float = 0.1
    future_action_longitudinal_cls_weight: float = 1.0
    future_action_steer_cls_weight: float = 1.0
    longitudinal_coast_threshold: float = 0.05
    steer_straight_threshold: float = 0.03
    debug_dual_gate: bool = False


class _ManualProgress:
    def __init__(self, total: int, desc: str, enabled: bool = True) -> None:
        self.total = max(int(total), 1)
        self.desc = desc
        self.enabled = enabled
        self.start_time = time.time()
        self.last_len = 0

    def update(self, step: int, metrics: dict[str, float] | None = None) -> None:
        if not self.enabled:
            return
        elapsed = max(time.time() - self.start_time, 1e-6)
        percent = 100.0 * step / self.total
        rate = step / elapsed
        remaining = max(self.total - step, 0)
        eta = remaining / rate if rate > 0 else 0.0
        filled = int(round(20 * step / self.total))
        bar = "█" * filled + "-" * (20 - filled)
        parts = [
            f"{self.desc} [{bar}]",
            f"{step}/{self.total}",
            f"{percent:5.1f}%",
        ]
        if metrics:
            parts.extend(f"{key}={value:.4f}" for key, value in _compact_progress_metrics(metrics).items())
        parts.append(f"eta={self._format_seconds(eta)}")
        text = " ".join(parts)
        padding = " " * max(self.last_len - len(text), 0)
        print("\r" + text + padding, end="", file=sys.stderr, flush=True)
        self.last_len = len(text)

    def close(self) -> None:
        if self.enabled:
            print(file=sys.stderr, flush=True)

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        seconds_i = int(max(seconds, 0))
        minutes, secs = divmod(seconds_i, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"


class _TqdmProgress:
    def __init__(self, total: int, desc: str, tqdm_cls) -> None:
        self.bar = tqdm_cls(total=total, desc=desc, dynamic_ncols=True, leave=False)

    def update(self, step: int, metrics: dict[str, float] | None = None) -> None:
        if metrics:
            compact = _compact_progress_metrics(metrics)
            self.bar.set_postfix({key: f"{value:.4f}" for key, value in compact.items()}, refresh=False)
        self.bar.update(1)

    def close(self) -> None:
        self.bar.close()


def _metric_or_na(metrics: dict[str, Any], key: str, precision: int = 6) -> str:
    value = metrics.get(key)
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return "n/a"


def _compact_progress_metrics(metrics: dict[str, float]) -> dict[str, float]:
    compact = {key: metrics[key] for key in PROGRESS_KEYS if key in metrics}
    if compact:
        return compact
    return metrics


class TeleopWMTrainer:
    """Small explicit PyTorch trainer used by the TeleopWM release pipeline."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainerConfig,
        run_config: dict,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.run_config = run_config
        self.resume_checkpoint = Path(config.resume_checkpoint) if config.resume_checkpoint else None
        self.run_dir = (
            infer_run_dir_from_checkpoint(self.resume_checkpoint)
            if self.resume_checkpoint is not None
            else create_run_dir(config.output_dir, config.run_tag)
        )
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.rollout_grid_dir = self.run_dir / "rollout_grids"
        self.benchmark_dir = self.run_dir / "benchmark"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=self.resume_checkpoint is not None)
        self.rollout_grid_dir.mkdir(parents=True, exist_ok=self.resume_checkpoint is not None)
        self.benchmark_dir.mkdir(parents=True, exist_ok=self.resume_checkpoint is not None)
        self.log_path = self.run_dir / "train.log"
        self.metrics_path = self.run_dir / "metrics.json"
        self.metrics_jsonl_path = self.run_dir / "metrics.jsonl"
        self.config_path = self.run_dir / "config.json"
        self.metrics_history: list[dict[str, Any]] = []

        requested_device = config.device
        if requested_device == "cuda" and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.use_amp = bool(config.amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.best_val_loss = float("inf")
        self.start_epoch = 1
        self.global_optimizer_steps = 0
        if self.resume_checkpoint is not None:
            self._resume_from_checkpoint(self.resume_checkpoint)
        self._write_config()

    @property
    def has_dual_losses(self) -> bool:
        return (
            self.config.dual_wm_image_loss_weight > 0
            or self.config.dual_simvp_image_loss_weight > 0
            or self.config.dual_align_loss_weight > 0
        )

    def _dual_fusion_metric_fields(self) -> dict[str, float]:
        if not hasattr(self.model, "dual_conv1x1_fusion_stats"):
            return {}
        stats = self.model.dual_conv1x1_fusion_stats()
        if not stats.get("enabled", False):
            return {}
        return {
            "dual_conv1x1_weight_norm": float(stats["weight_norm"]),
            "dual_conv1x1_simvp_half_weight_norm": float(stats["simvp_half_weight_norm"]),
            "dual_conv1x1_wm_half_weight_norm": float(stats["wm_half_weight_norm"]),
            "dual_conv1x1_wm_to_simvp_weight_norm_ratio": float(stats["wm_to_simvp_weight_norm_ratio"]),
        }

    def fit(self) -> None:
        self.log(f"device: {self.device}")
        self.log(f"amp: {self.use_amp}")
        self.log(f"max_train_steps: {self.config.max_train_steps} (global optimizer-step cap)")
        self.log(f"max_val_batches: {self.config.max_val_batches}")
        self.log(f"max_interim_val_batches: {self.config.max_interim_val_batches}")
        self.log(f"eval_every_steps: {self.config.eval_every_steps}")
        self.log(f"grad_clip_norm: {self.config.grad_clip_norm}")
        self.log(f"stop_on_nan: {self.config.stop_on_nan}")
        self.log(f"log_grad_norms: {self.config.log_grad_norms}")
        self.log(f"debug_dual_gate: {self.config.debug_dual_gate}")
        self.log(f"debug_activation_stats: {self.config.debug_activation_stats}")
        self.log(f"ssim_loss_weight: {self.config.ssim_loss_weight}")
        self.log(f"control_steer_input_scale: {self.config.control_steer_input_scale}")
        self.log("conditioning representation after scaling: [longitudinal, scaled_steer, speed]")
        dual_fusion = getattr(self.model, "dual_fusion", None)
        if dual_fusion is not None:
            self.log(f"dual_fusion: {dual_fusion}")
            if dual_fusion == "conv1x1" and hasattr(self.model, "dual_conv1x1_fusion_stats"):
                stats = self.model.dual_conv1x1_fusion_stats()
                self.log(
                    "dual conv1x1 fusion: "
                    f"enabled={stats['enabled']} "
                    f"params={stats['param_count']} "
                    f"simvp_half_norm={stats['simvp_half_weight_norm']:.6f} "
                    f"wm_half_norm={stats['wm_half_weight_norm']:.6f} "
                    f"wm/simvp={stats['wm_to_simvp_weight_norm_ratio']:.6f} "
                    "dual_wm_scale_applied_to_wm_before_concat=True"
                )
        if self.has_dual_losses:
            self.log(
                "dual losses: "
                f"wm_image={self.config.dual_wm_image_loss_weight} "
                f"simvp_image={self.config.dual_simvp_image_loss_weight} "
                f"align={self.config.dual_align_loss_weight}"
            )
        if self.config.future_action_loss:
            self.log(
                "future action loss: "
                f"weight={self.config.future_action_loss_weight} "
                f"type={self.config.future_action_loss_type} "
                f"source={self.config.future_action_source} "
                f"future_steer_target_scale={self.config.future_steer_target_scale} "
                f"head_variant={self.config.future_action_head_variant} "
                f"hidden_dim={getattr(self.model, 'future_action_hidden_dim', 'n/a')} "
                f"detach_latents={self.config.future_action_detach_latents} "
                f"future_motion_scale={self.config.future_action_future_motion_scale} "
                f"corr_loss_weight={self.config.future_action_corr_loss_weight} "
                f"spatial_pooling={getattr(self.model, 'future_action_spatial_pooling', 'global')} "
                f"spatial_grid={getattr(self.model, 'future_action_spatial_grid', (1, 1))} "
                f"action_token_dim={getattr(self.model, 'future_action_token_dim', 'n/a')} "
                "target=[longitudinal, scaled_steer]"
            )
            if self.config.future_action_delta_loss:
                self.log(
                    "future action delta loss: "
                    f"weight={self.config.future_action_delta_loss_weight} "
                    f"type={self.config.future_action_delta_loss_type} "
                    f"long_weight={self.config.future_action_delta_longitudinal_weight} "
                    f"steer_weight={self.config.future_action_delta_steer_weight} "
                    "target_space=[longitudinal, scaled_steer temporal differences]"
                )
            if self.config.future_action_cls_loss:
                self.log(
                    "future action classification loss: "
                    f"weight={self.config.future_action_cls_weight} "
                    f"long_weight={self.config.future_action_longitudinal_cls_weight} "
                    f"steer_weight={self.config.future_action_steer_cls_weight} "
                    f"longitudinal_coast_threshold={self.config.longitudinal_coast_threshold} "
                    f"steer_straight_threshold={self.config.steer_straight_threshold}"
                )
        self.log(f"run_dir: {self.run_dir}")
        if self.resume_checkpoint is not None:
            self.log(f"Resuming from checkpoint: {self.resume_checkpoint}")
            self.log(f"Starting at epoch: {self.start_epoch}")
            self.log(f"Best validation loss: {self.best_val_loss}")
        if self.start_epoch > self.config.epochs:
            self.log(
                f"Nothing to train: start_epoch={self.start_epoch} "
                f"is greater than requested epochs={self.config.epochs}"
            )
            self.generate_plots()
            return
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            if self._reached_max_train_steps():
                self.log(
                    f"Stopping before epoch {epoch:03d}: reached global max_train_steps="
                    f"{self.config.max_train_steps}"
                )
                break
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)

            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]
                self.save("best.pt", epoch, val_metrics)

            if epoch % self.config.save_every == 0:
                self.save(f"epoch_{epoch:04d}.pt", epoch, val_metrics)

            record = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "best_val_loss": self.best_val_loss,
            }
            self.metrics_history.append(record)
            append_jsonl(self.metrics_jsonl_path, record)
            write_json(
                self.metrics_path,
                {
                    "best_val_loss": self.best_val_loss,
                    "epochs": self.metrics_history,
                },
            )

            self.log(
                f"epoch {epoch:03d} "
                f"train_loss={train_metrics['loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"train_mae={_metric_or_na(train_metrics, 'mae')} "
                f"val_mae={val_metrics['mae']:.6f} "
                f"train_ssim={_metric_or_na(train_metrics, 'ssim', precision=4)} "
                f"val_ssim={val_metrics['ssim']:.4f}"
            )
            if self.config.aux_dynamics_loss:
                self.log(
                    f"epoch {epoch:03d} "
                    f"train_rgb_loss={train_metrics['rgb_loss']:.6f} "
                    f"train_aux_loss={train_metrics['aux_loss']:.6f} "
                    f"train_total_loss={train_metrics['total_loss']:.6f} "
                    f"val_rgb_loss={val_metrics['rgb_loss']:.6f} "
                    f"val_aux_loss={val_metrics['aux_loss']:.6f} "
                    f"val_total_loss={val_metrics['total_loss']:.6f}"
                )
            if self.config.wm_residual_loss or "wm_residual_gate_mean" in train_metrics:
                parts = [f"epoch {epoch:03d}"]
                if self.config.wm_residual_loss:
                    parts.extend(
                        [
                            f"train_wm_residual_loss={train_metrics['wm_residual_loss']:.6f}",
                            f"val_wm_residual_loss={val_metrics['wm_residual_loss']:.6f}",
                        ]
                    )
                if "wm_residual_gate_mean" in train_metrics:
                    parts.extend(
                        [
                            f"train_gate_mean={train_metrics['wm_residual_gate_mean']:.6f}",
                            f"train_gate_min={train_metrics['wm_residual_gate_min']:.6f}",
                            f"train_gate_max={train_metrics['wm_residual_gate_max']:.6f}",
                            f"val_gate_mean={val_metrics['wm_residual_gate_mean']:.6f}",
                            f"val_gate_min={val_metrics['wm_residual_gate_min']:.6f}",
                            f"val_gate_max={val_metrics['wm_residual_gate_max']:.6f}",
                        ]
                    )
                self.log(" ".join(parts))
            if self.config.ssim_loss_weight > 0:
                self.log(
                    f"epoch {epoch:03d} "
                    f"train_recon_l1_loss={train_metrics['recon_l1_loss']:.6f} "
                    f"train_ssim_loss={train_metrics['ssim_loss']:.6f} "
                    f"val_recon_l1_loss={val_metrics['recon_l1_loss']:.6f} "
                    f"val_ssim_loss={val_metrics['ssim_loss']:.6f} "
                    f"ssim_loss_weight={self.config.ssim_loss_weight:.4f}"
                )
            if self.has_dual_losses or "dual_gate_mean" in train_metrics or "dual_conv1x1_wm_to_simvp_weight_norm_ratio" in train_metrics:
                parts = [f"epoch {epoch:03d}"]
                for key in ("dual_wm_image_loss", "dual_simvp_image_loss", "dual_align_loss"):
                    if key in train_metrics:
                        parts.append(f"train_{key}={train_metrics[key]:.6f}")
                        parts.append(f"val_{key}={val_metrics[key]:.6f}")
                if "dual_conv1x1_wm_to_simvp_weight_norm_ratio" in train_metrics:
                    parts.extend(
                        [
                            f"fuse_w={train_metrics['dual_conv1x1_weight_norm']:.6f}",
                            f"simvp_w={train_metrics['dual_conv1x1_simvp_half_weight_norm']:.6f}",
                            f"wm_w={train_metrics['dual_conv1x1_wm_half_weight_norm']:.6f}",
                            f"wm_to_simvp_weight_norm_ratio={train_metrics['dual_conv1x1_wm_to_simvp_weight_norm_ratio']:.6f}",
                        ]
                    )
                elif "dual_gate_mean" in train_metrics:
                    parts.extend(
                        [
                            f"train_dual_gate_mean={train_metrics['dual_gate_mean']:.6f}",
                            f"train_dual_gate_min={train_metrics['dual_gate_min']:.6f}",
                            f"train_dual_gate_max={train_metrics['dual_gate_max']:.6f}",
                            f"val_dual_gate_mean={val_metrics['dual_gate_mean']:.6f}",
                            f"val_dual_gate_min={val_metrics['dual_gate_min']:.6f}",
                            f"val_dual_gate_max={val_metrics['dual_gate_max']:.6f}",
                        ]
                    )
                self.log(" ".join(parts))
            if self.config.future_action_loss:
                parts = [
                    f"epoch {epoch:03d}",
                    f"train_future_action_loss={train_metrics['future_action_loss']:.6f}",
                    f"val_future_action_loss={val_metrics['future_action_loss']:.6f}",
                    f"train_future_action_reg_loss={train_metrics['future_action_reg_loss']:.6f}",
                    f"val_future_action_reg_loss={val_metrics['future_action_reg_loss']:.6f}",
                    f"train_future_longitudinal_mae={train_metrics['future_longitudinal_mae']:.6f}",
                    f"val_future_longitudinal_mae={val_metrics['future_longitudinal_mae']:.6f}",
                    f"train_future_steer_mae={train_metrics['future_steer_mae']:.6f}",
                    f"val_future_steer_mae={val_metrics['future_steer_mae']:.6f}",
                ]
                if self.config.future_action_corr_loss_weight > 0 or "future_action_corr_loss" in train_metrics:
                    parts.extend(
                        [
                            f"train_future_action_corr_loss={_metric_or_na(train_metrics, 'future_action_corr_loss')}",
                            f"val_future_action_corr_loss={_metric_or_na(val_metrics, 'future_action_corr_loss')}",
                            f"train_future_action_corr_longitudinal={_metric_or_na(train_metrics, 'future_action_corr_longitudinal')}",
                            f"val_future_action_corr_longitudinal={_metric_or_na(val_metrics, 'future_action_corr_longitudinal')}",
                            f"train_future_action_corr_steer={_metric_or_na(train_metrics, 'future_action_corr_steer')}",
                            f"val_future_action_corr_steer={_metric_or_na(val_metrics, 'future_action_corr_steer')}",
                        ]
                    )
                if self.config.future_action_delta_loss or "future_action_delta_loss" in train_metrics:
                    parts.extend(
                        [
                            f"train_future_action_delta_loss={_metric_or_na(train_metrics, 'future_action_delta_loss')}",
                            f"val_future_action_delta_loss={_metric_or_na(val_metrics, 'future_action_delta_loss')}",
                            f"train_future_action_delta_longitudinal_loss={_metric_or_na(train_metrics, 'future_action_delta_longitudinal_loss')}",
                            f"val_future_action_delta_longitudinal_loss={_metric_or_na(val_metrics, 'future_action_delta_longitudinal_loss')}",
                            f"train_future_action_delta_steer_loss={_metric_or_na(train_metrics, 'future_action_delta_steer_loss')}",
                            f"val_future_action_delta_steer_loss={_metric_or_na(val_metrics, 'future_action_delta_steer_loss')}",
                        ]
                    )
                if "future_action_motion_feature_norm" in train_metrics:
                    parts.extend(
                        [
                            f"train_future_action_motion_feature_norm={train_metrics['future_action_motion_feature_norm']:.6f}",
                            f"val_future_action_motion_feature_norm={val_metrics['future_action_motion_feature_norm']:.6f}",
                            f"train_future_action_control_feature_norm={train_metrics['future_action_control_feature_norm']:.6f}",
                            f"val_future_action_control_feature_norm={val_metrics['future_action_control_feature_norm']:.6f}",
                            f"train_future_action_latent_feature_norm={train_metrics['future_action_latent_feature_norm']:.6f}",
                            f"val_future_action_latent_feature_norm={val_metrics['future_action_latent_feature_norm']:.6f}",
                        ]
                    )
                if self.config.future_action_cls_loss:
                    parts.extend(
                        [
                            f"train_future_action_cls_loss={train_metrics['future_action_cls_loss']:.6f}",
                            f"val_future_action_cls_loss={val_metrics['future_action_cls_loss']:.6f}",
                            f"train_future_longitudinal_cls_acc={train_metrics['future_longitudinal_cls_acc']:.4f}",
                            f"val_future_longitudinal_cls_acc={val_metrics['future_longitudinal_cls_acc']:.4f}",
                            f"train_future_steer_cls_acc={train_metrics['future_steer_cls_acc']:.4f}",
                            f"val_future_steer_cls_acc={val_metrics['future_steer_cls_acc']:.4f}",
                        ]
                    )
                self.log(" ".join(parts))
            if self._reached_max_train_steps():
                self.log(
                    f"Stopping after epoch {epoch:03d}: reached global max_train_steps="
                    f"{self.config.max_train_steps}"
                )
                break
        self.generate_plots()

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_rgb_loss = 0.0
        total_recon_l1_loss = 0.0
        total_ssim_loss = 0.0
        total_aux_loss = 0.0
        total_wm_loss = 0.0
        total_dual_wm_image_loss = 0.0
        total_dual_simvp_image_loss = 0.0
        total_dual_align_loss = 0.0
        total_train_mae = 0.0
        total_train_ssim = 0.0
        total_future_action_loss = 0.0
        total_future_action_reg_loss = 0.0
        total_future_action_corr_loss = 0.0
        total_future_action_corr_longitudinal = 0.0
        total_future_action_corr_steer = 0.0
        total_future_action_delta_loss = 0.0
        total_future_action_delta_longitudinal_loss = 0.0
        total_future_action_delta_steer_loss = 0.0
        total_future_action_cls_loss = 0.0
        total_future_longitudinal_cls_loss = 0.0
        total_future_steer_cls_loss = 0.0
        total_future_longitudinal_cls_acc = 0.0
        total_future_steer_cls_acc = 0.0
        total_future_latent_feature_norm = 0.0
        total_motion_feature_norm = 0.0
        total_control_feature_norm = 0.0
        total_motion_energy_mean = 0.0
        total_future_longitudinal_mae = 0.0
        total_future_steer_mae = 0.0
        dual_gate_sum = 0.0
        dual_gate_count = 0
        dual_gate_min = float("inf")
        dual_gate_max = float("-inf")
        dual_gate_history: list[tuple[int, dict[str, float]]] = []
        gate_sum = 0.0
        gate_count = 0
        gate_min = float("inf")
        gate_max = float("-inf")
        total_samples = 0
        epoch_steps = len(self.train_loader)
        remaining_steps = self._remaining_train_steps()
        if remaining_steps is not None:
            epoch_steps = min(epoch_steps, remaining_steps)
        progress = self._create_progress(epoch_steps, f"Epoch {epoch:03d}")

        try:
            for step, batch in enumerate(self.train_loader, start=1):
                if self._reached_max_train_steps():
                    break
                past_frames = batch["past_frames"].to(self.device, non_blocking=True)
                future_frames = batch["future_frames"].to(self.device, non_blocking=True)
                batch_size = past_frames.shape[0]

                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    prediction_output = self.forward_model(batch, past_frames)
                    prediction, aux_loss, wm_loss, gate_stats, dual_losses, dual_gate_stats, future_action_metrics = self.compute_losses_from_output(
                        prediction_output, batch, future_frames
                    )
                    recon_l1_loss = frame_l1_loss(prediction, future_frames)
                    ssim_loss = (
                        frame_ssim_loss(prediction.float(), future_frames.float())
                        if self.config.ssim_loss_weight > 0
                        else future_frames.new_tensor(0.0)
                    )
                    rgb_loss = recon_l1_loss + self.config.ssim_loss_weight * ssim_loss
                    loss = (
                        rgb_loss
                        + self.config.aux_dynamics_weight * aux_loss
                        + self.config.wm_residual_loss_weight * wm_loss
                        + self.config.dual_wm_image_loss_weight * dual_losses["wm_image"]
                        + self.config.dual_simvp_image_loss_weight * dual_losses["simvp_image"]
                        + self.config.dual_align_loss_weight * dual_losses["align"]
                        + self.config.future_action_loss_weight * future_action_metrics["loss"]
                        + self.config.future_action_delta_loss_weight * future_action_metrics["delta_loss"]
                    )

                loss_components = {
                    "total_loss": loss,
                    "rgb_loss": rgb_loss,
                    "recon_l1_loss": recon_l1_loss,
                    "ssim_loss": ssim_loss,
                    "aux_loss": aux_loss,
                    "wm_residual_loss": wm_loss,
                    "dual_wm_image_loss": dual_losses["wm_image"],
                    "dual_simvp_image_loss": dual_losses["simvp_image"],
                    "dual_align_loss": dual_losses["align"],
                    "future_action_loss": future_action_metrics["loss"],
                    "future_action_reg_loss": future_action_metrics["reg_loss"],
                    "future_action_corr_loss": future_action_metrics["corr_loss"],
                    "future_action_delta_loss": future_action_metrics["delta_loss"],
                    "future_action_cls_loss": future_action_metrics["cls_loss"],
                    "future_longitudinal_cls_loss": future_action_metrics["longitudinal_cls_loss"],
                    "future_steer_cls_loss": future_action_metrics["steer_cls_loss"],
                }
                self._check_finite_tensors(loss_components, context=f"epoch {epoch} step {step} losses")

                self.scaler.scale(loss).backward()

                needs_unscale = (
                    self.config.grad_clip_norm is not None
                    or self.config.log_grad_norms
                    or self.config.stop_on_nan
                )
                grad_norms = None
                if needs_unscale:
                    self.scaler.unscale_(self.optimizer)
                    grad_norms = self._gradient_diagnostics()
                    if self.config.stop_on_nan and not grad_norms["all_elements_finite"]:
                        raise FloatingPointError(
                            f"Non-finite gradients at epoch {epoch} step {step}: {grad_norms}"
                        )
                    if self.config.grad_clip_norm is not None:
                        pre_clip_nonfinite = grad_norms.get("first_nonfinite_grad_param")
                        clipped_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.grad_clip_norm,
                        )
                        grad_norms["clipped_total_norm"] = float(clipped_norm.detach().cpu())
                        post_clip = self._gradient_diagnostics()
                        grad_norms["post_clip_total"] = post_clip["total"]
                        grad_norms["post_clip_all_elements_finite"] = post_clip["all_elements_finite"]
                        grad_norms["post_clip_first_nonfinite_grad_param"] = post_clip["first_nonfinite_grad_param"]
                        if (
                            self.config.stop_on_nan
                            and pre_clip_nonfinite is None
                            and not post_clip["all_elements_finite"]
                        ):
                            raise FloatingPointError(
                                f"Gradients became non-finite after clipping at epoch {epoch} step {step}: {post_clip}"
                            )

                old_scale = float(self.scaler.get_scale()) if self.use_amp else 1.0
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.global_optimizer_steps += 1
                new_scale = float(self.scaler.get_scale()) if self.use_amp else 1.0

                loss_value = float(loss.detach().cpu())
                rgb_loss_value = float(rgb_loss.detach().cpu())
                recon_l1_loss_value = float(recon_l1_loss.detach().cpu())
                ssim_loss_value = float(ssim_loss.detach().cpu())
                train_mae_value = recon_l1_loss_value
                train_ssim_value = 1.0 - ssim_loss_value if self.config.ssim_loss_weight > 0 else None
                aux_loss_value = float(aux_loss.detach().cpu())
                wm_loss_value = float(wm_loss.detach().cpu())
                dual_wm_image_loss_value = float(dual_losses["wm_image"].detach().cpu())
                dual_simvp_image_loss_value = float(dual_losses["simvp_image"].detach().cpu())
                dual_align_loss_value = float(dual_losses["align"].detach().cpu())
                future_action_loss_value = float(future_action_metrics["loss"].detach().cpu())
                future_action_reg_loss_value = float(future_action_metrics["reg_loss"].detach().cpu())
                future_action_corr_loss_value = float(future_action_metrics["corr_loss"].detach().cpu())
                future_action_corr_longitudinal_value = float(future_action_metrics["corr_longitudinal"].detach().cpu())
                future_action_corr_steer_value = float(future_action_metrics["corr_steer"].detach().cpu())
                future_action_delta_loss_value = float(future_action_metrics["delta_loss"].detach().cpu())
                future_action_delta_longitudinal_loss_value = float(
                    future_action_metrics["delta_longitudinal_loss"].detach().cpu()
                )
                future_action_delta_steer_loss_value = float(future_action_metrics["delta_steer_loss"].detach().cpu())
                future_action_cls_loss_value = float(future_action_metrics["cls_loss"].detach().cpu())
                future_longitudinal_cls_loss_value = float(future_action_metrics["longitudinal_cls_loss"].detach().cpu())
                future_steer_cls_loss_value = float(future_action_metrics["steer_cls_loss"].detach().cpu())
                future_longitudinal_cls_acc_value = float(future_action_metrics["longitudinal_cls_acc"].detach().cpu())
                future_steer_cls_acc_value = float(future_action_metrics["steer_cls_acc"].detach().cpu())
                future_latent_feature_norm_value = float(future_action_metrics["future_latent_feature_norm"].detach().cpu())
                motion_feature_norm_value = float(future_action_metrics["motion_feature_norm"].detach().cpu())
                control_feature_norm_value = float(future_action_metrics["control_feature_norm"].detach().cpu())
                motion_energy_mean_value = float(future_action_metrics["motion_energy_mean"].detach().cpu())
                future_longitudinal_mae_value = float(future_action_metrics["longitudinal_mae"].detach().cpu())
                future_steer_mae_value = float(future_action_metrics["steer_mae"].detach().cpu())

                total_loss += loss_value * batch_size
                total_train_mae += train_mae_value * batch_size
                if train_ssim_value is not None:
                    total_train_ssim += train_ssim_value * batch_size
                total_samples += batch_size
                if self.config.ssim_loss_weight > 0:
                    total_recon_l1_loss += recon_l1_loss_value * batch_size
                    total_ssim_loss += ssim_loss_value * batch_size
                if self.config.aux_dynamics_loss:
                    total_rgb_loss += rgb_loss_value * batch_size
                    total_aux_loss += aux_loss_value * batch_size
                if self.config.wm_residual_loss:
                    total_wm_loss += wm_loss_value * batch_size
                if self.has_dual_losses:
                    total_dual_wm_image_loss += dual_wm_image_loss_value * batch_size
                    total_dual_simvp_image_loss += dual_simvp_image_loss_value * batch_size
                    total_dual_align_loss += dual_align_loss_value * batch_size
                if self.config.future_action_loss:
                    total_future_action_loss += future_action_loss_value * batch_size
                    total_future_action_reg_loss += future_action_reg_loss_value * batch_size
                    total_future_action_corr_loss += future_action_corr_loss_value * batch_size
                    total_future_action_corr_longitudinal += future_action_corr_longitudinal_value * batch_size
                    total_future_action_corr_steer += future_action_corr_steer_value * batch_size
                    total_future_action_delta_loss += future_action_delta_loss_value * batch_size
                    total_future_action_delta_longitudinal_loss += future_action_delta_longitudinal_loss_value * batch_size
                    total_future_action_delta_steer_loss += future_action_delta_steer_loss_value * batch_size
                    total_future_action_cls_loss += future_action_cls_loss_value * batch_size
                    total_future_longitudinal_cls_loss += future_longitudinal_cls_loss_value * batch_size
                    total_future_steer_cls_loss += future_steer_cls_loss_value * batch_size
                    total_future_longitudinal_cls_acc += future_longitudinal_cls_acc_value * batch_size
                    total_future_steer_cls_acc += future_steer_cls_acc_value * batch_size
                    total_future_latent_feature_norm += future_latent_feature_norm_value * batch_size
                    total_motion_feature_norm += motion_feature_norm_value * batch_size
                    total_control_feature_norm += control_feature_norm_value * batch_size
                    total_motion_energy_mean += motion_energy_mean_value * batch_size
                    total_future_longitudinal_mae += future_longitudinal_mae_value * batch_size
                    total_future_steer_mae += future_steer_mae_value * batch_size
                if gate_stats:
                    gate_sum += gate_stats["mean"] * batch_size
                    gate_count += batch_size
                    gate_min = min(gate_min, gate_stats["min"])
                    gate_max = max(gate_max, gate_stats["max"])
                if dual_gate_stats:
                    dual_gate_history.append((batch_size, dual_gate_stats))
                    dual_gate_sum += dual_gate_stats["mean"] * batch_size
                    dual_gate_count += batch_size
                    dual_gate_min = min(dual_gate_min, dual_gate_stats["min"])
                    dual_gate_max = max(dual_gate_max, dual_gate_stats["max"])

                running_avg = total_loss / max(total_samples, 1)
                progress_metrics = {"loss": loss_value, "avg": running_avg, "img_mae": train_mae_value}
                if self.config.ssim_loss_weight > 0:
                    progress_metrics.update({"ssim": train_ssim_value or 0.0})
                if self.config.aux_dynamics_loss:
                    progress_metrics.update({"rgb": rgb_loss_value, "aux": aux_loss_value})
                if self.config.wm_residual_loss:
                    progress_metrics["wm"] = wm_loss_value
                if self.has_dual_losses:
                    progress_metrics["dual"] = (
                        self.config.dual_wm_image_loss_weight * dual_wm_image_loss_value
                        + self.config.dual_simvp_image_loss_weight * dual_simvp_image_loss_value
                        + self.config.dual_align_loss_weight * dual_align_loss_value
                    )
                if self.config.future_action_loss:
                    progress_metrics["act"] = future_action_loss_value
                    progress_metrics["act_reg"] = future_action_reg_loss_value
                    if self.config.future_action_corr_loss_weight > 0:
                        progress_metrics["act_corr"] = future_action_corr_loss_value
                    if self.config.future_action_delta_loss and self.config.future_action_delta_loss_weight > 0:
                        progress_metrics["act_delta"] = future_action_delta_loss_value
                    progress_metrics["long_mae"] = future_longitudinal_mae_value
                    progress_metrics["steer_mae"] = future_steer_mae_value
                    if self.config.future_action_cls_loss:
                        progress_metrics["act_cls"] = future_action_cls_loss_value
                        progress_metrics["long_acc"] = future_longitudinal_cls_acc_value
                        progress_metrics["steer_acc"] = future_steer_cls_acc_value
                    if self.config.future_action_head_variant in {"motion_context", "motion_context_v2"}:
                        progress_metrics["mot_norm"] = motion_feature_norm_value
                        progress_metrics["ctl_norm"] = control_feature_norm_value
                        progress_metrics["lat_norm"] = future_latent_feature_norm_value
                fusion_stats = self._dual_fusion_metric_fields()
                if fusion_stats:
                    progress_metrics["wm/simvp"] = fusion_stats["dual_conv1x1_wm_to_simvp_weight_norm_ratio"]
                progress_metrics["gstep"] = float(self.global_optimizer_steps)
                progress_metrics["estep"] = float(step)
                if grad_norms is not None and self.config.log_grad_norms:
                    progress_metrics["grad"] = grad_norms["total"]
                if gate_stats:
                    progress_metrics["gate"] = gate_stats["mean"]
                if dual_gate_stats and not fusion_stats:
                    progress_metrics["dual_gate"] = dual_gate_stats["mean"]
                progress.update(step, progress_metrics)

                if step % self.config.log_interval == 0:
                    train_ssim_text = f"{train_ssim_value:.4f}" if train_ssim_value is not None else "n/a"
                    message = (
                        f"epoch {epoch:03d} step {step:05d}/{epoch_steps} "
                        f"loss={loss_value:.6f} "
                        f"train_mae={train_mae_value:.6f} "
                        f"train_ssim={train_ssim_text}"
                    )
                    if self.config.aux_dynamics_loss:
                        message += f" rgb_loss={rgb_loss_value:.6f} aux_loss={aux_loss_value:.6f}"
                    if self.config.ssim_loss_weight > 0:
                        message += (
                            f" recon_l1_loss={recon_l1_loss_value:.6f}"
                            f" ssim_loss={ssim_loss_value:.6f}"
                            f" ssim_weight={self.config.ssim_loss_weight:.4f}"
                        )
                    if self.config.wm_residual_loss:
                        message += f" wm_residual_loss={wm_loss_value:.6f}"
                    if self.has_dual_losses:
                        message += (
                            f" dual_wm_image_loss={dual_wm_image_loss_value:.6f}"
                            f" dual_simvp_image_loss={dual_simvp_image_loss_value:.6f}"
                            f" dual_align_loss={dual_align_loss_value:.6f}"
                        )
                    if self.config.future_action_loss:
                        message += (
                            f" future_action_loss={future_action_loss_value:.6f}"
                            f" future_action_reg_loss={future_action_reg_loss_value:.6f}"
                            f" future_longitudinal_mae={future_longitudinal_mae_value:.6f}"
                            f" future_steer_mae={future_steer_mae_value:.6f}"
                        )
                        if self.config.future_action_corr_loss_weight > 0:
                            message += (
                                f" future_action_corr_loss={future_action_corr_loss_value:.6f}"
                                f" future_action_corr_longitudinal={future_action_corr_longitudinal_value:.6f}"
                                f" future_action_corr_steer={future_action_corr_steer_value:.6f}"
                            )
                        if self.config.future_action_delta_loss and self.config.future_action_delta_loss_weight > 0:
                            message += (
                                f" future_action_delta_loss={future_action_delta_loss_value:.6f}"
                                f" future_action_delta_longitudinal_loss={future_action_delta_longitudinal_loss_value:.6f}"
                                f" future_action_delta_steer_loss={future_action_delta_steer_loss_value:.6f}"
                            )
                        if self.config.future_action_cls_loss:
                            message += (
                                f" future_action_cls_loss={future_action_cls_loss_value:.6f}"
                                f" future_longitudinal_cls_loss={future_longitudinal_cls_loss_value:.6f}"
                                f" future_steer_cls_loss={future_steer_cls_loss_value:.6f}"
                                f" future_longitudinal_cls_acc={future_longitudinal_cls_acc_value:.4f}"
                                f" future_steer_cls_acc={future_steer_cls_acc_value:.4f}"
                            )
                        if self.config.future_action_head_variant in {"motion_context", "motion_context_v2"}:
                            message += (
                                f" future_action_motion_feature_norm={motion_feature_norm_value:.6f}"
                                f" future_action_control_feature_norm={control_feature_norm_value:.6f}"
                                f" future_action_latent_feature_norm={future_latent_feature_norm_value:.6f}"
                                f" future_action_motion_energy_mean={motion_energy_mean_value:.6f}"
                            )
                    fusion_stats = self._dual_fusion_metric_fields()
                    if fusion_stats:
                        message += (
                            f" fuse_w={fusion_stats['dual_conv1x1_weight_norm']:.6f}"
                            f" simvp_w={fusion_stats['dual_conv1x1_simvp_half_weight_norm']:.6f}"
                            f" wm_w={fusion_stats['dual_conv1x1_wm_half_weight_norm']:.6f}"
                            f" wm_to_simvp_weight_norm_ratio={fusion_stats['dual_conv1x1_wm_to_simvp_weight_norm_ratio']:.6f}"
                        )
                    if grad_norms is not None:
                        if "clipped_total_norm" in grad_norms:
                            message += f" clipped_grad_norm={grad_norms['clipped_total_norm']:.6f}"
                        if self.config.log_grad_norms:
                            message += (
                                f" grad_total={grad_norms['total']:.6f}"
                                f" grad_simvp={grad_norms['simvp_backbone']:.6f}"
                                f" grad_wm={grad_norms['wm_branch']:.6f}"
                                f" grad_dual_gate={grad_norms['dual_gate']:.6f}"
                                f" grad_action_head={grad_norms['future_action_head']:.6f}"
                                f" dual_gate_param_weight_norm={grad_norms['dual_gate_param_weight_norm']:.6f}"
                                f" dual_gate_param_bias_mean={grad_norms['dual_gate_param_bias_mean']:.6f}"
                                f" dual_gate_param_bias_std={grad_norms['dual_gate_param_bias_std']:.6f}"
                                f" dual_gate_param_bias_min={grad_norms['dual_gate_param_bias_min']:.6f}"
                                f" dual_gate_param_bias_max={grad_norms['dual_gate_param_bias_max']:.6f}"
                                f" grad_elements_finite={grad_norms['all_elements_finite']}"
                                f" grad_norms_finite={grad_norms['all_norms_finite']}"
                                f" first_nonfinite_grad={grad_norms['first_nonfinite_grad_param']}"
                                f" first_norm_overflow={grad_norms['first_norm_overflow_param']}"
                            )
                            if "post_clip_total" in grad_norms:
                                message += (
                                    f" post_clip_grad_total={grad_norms['post_clip_total']:.6f}"
                                    f" post_clip_elements_finite={grad_norms['post_clip_all_elements_finite']}"
                                    f" post_clip_first_nonfinite_grad={grad_norms['post_clip_first_nonfinite_grad_param']}"
                                )
                    if self.config.debug_activation_stats and isinstance(prediction_output, dict):
                        message += " " + self._format_activation_stats(prediction_output)
                    if self.use_amp and new_scale < old_scale:
                        message += f" amp_scale_reduced={old_scale:.1f}->{new_scale:.1f}"
                    if gate_stats:
                        message += (
                            f" gate_mean={gate_stats['mean']:.6f}"
                            f" gate_min={gate_stats['min']:.6f}"
                            f" gate_max={gate_stats['max']:.6f}"
                        )
                    if dual_gate_stats and not fusion_stats:
                        message += (
                            f" dual_gate_mean={dual_gate_stats['mean']:.6f}"
                            f" dual_gate_min={dual_gate_stats['min']:.6f}"
                            f" dual_gate_max={dual_gate_stats['max']:.6f}"
                        )
                        if self.config.debug_dual_gate:
                            message += self._format_debug_gate_stats(dual_gate_stats, prefix="dual_gate")
                    if self.config.progress_bar:
                        self.log_file_only(message)
                    else:
                        self.log(message)
                if (
                    self.config.eval_every_steps is not None
                    and self.config.eval_every_steps > 0
                    and step % self.config.eval_every_steps == 0
                ):
                    interim_val = self.validate_epoch(
                        epoch,
                        max_batches_override=self.config.max_interim_val_batches,
                        validation_name="Interim Val",
                    )
                    parts = [
                        f"epoch {epoch:03d}",
                        f"global_step {self.global_optimizer_steps:06d}",
                        f"step {step:05d}",
                        f"interim_val_loss={interim_val['loss']:.6f}",
                        f"interim_val_mae={interim_val['mae']:.6f}",
                        f"interim_val_ssim={interim_val['ssim']:.4f}",
                        f"interim_val_batches={interim_val.get('batches', 0)}",
                        f"max_interim_val_batches={self.config.max_interim_val_batches}",
                    ]
                    if self.config.future_action_loss:
                        parts.extend(
                            [
                                f"interim_future_action_loss={interim_val['future_action_loss']:.6f}",
                                f"interim_future_action_reg_loss={interim_val['future_action_reg_loss']:.6f}",
                                f"interim_future_longitudinal_mae={interim_val['future_longitudinal_mae']:.6f}",
                                f"interim_future_steer_mae={interim_val['future_steer_mae']:.6f}",
                            ]
                        )
                        if "future_action_corr_loss" in interim_val:
                            parts.extend(
                                [
                                    f"interim_future_action_corr_loss={interim_val['future_action_corr_loss']:.6f}",
                                    f"interim_future_action_corr_longitudinal={interim_val['future_action_corr_longitudinal']:.6f}",
                                    f"interim_future_action_corr_steer={interim_val['future_action_corr_steer']:.6f}",
                                ]
                            )
                        if "future_action_delta_loss" in interim_val:
                            parts.extend(
                                [
                                    f"interim_future_action_delta_loss={interim_val['future_action_delta_loss']:.6f}",
                                    f"interim_future_action_delta_longitudinal_loss={interim_val['future_action_delta_longitudinal_loss']:.6f}",
                                    f"interim_future_action_delta_steer_loss={interim_val['future_action_delta_steer_loss']:.6f}",
                                ]
                            )
                    if self.config.future_action_cls_loss:
                        parts.extend(
                                [
                                    f"interim_future_action_cls_loss={interim_val['future_action_cls_loss']:.6f}",
                                    f"interim_future_longitudinal_cls_acc={interim_val['future_longitudinal_cls_acc']:.4f}",
                                    f"interim_future_steer_cls_acc={interim_val['future_steer_cls_acc']:.4f}",
                            ]
                        )
                    if self.config.future_action_head_variant in {"motion_context", "motion_context_v2"}:
                        parts.extend(
                            [
                                f"interim_future_action_motion_feature_norm={interim_val['future_action_motion_feature_norm']:.6f}",
                                f"interim_future_action_control_feature_norm={interim_val['future_action_control_feature_norm']:.6f}",
                                f"interim_future_action_latent_feature_norm={interim_val['future_action_latent_feature_norm']:.6f}",
                            ]
                        )
                    if "dual_conv1x1_wm_to_simvp_weight_norm_ratio" in interim_val:
                        parts.append(
                            "interim_dual_conv1x1_wm_to_simvp_weight_norm_ratio="
                            f"{interim_val['dual_conv1x1_wm_to_simvp_weight_norm_ratio']:.6f}"
                        )
                    elif "dual_gate_mean" in interim_val:
                        parts.append(f"interim_dual_gate_mean={interim_val['dual_gate_mean']:.6f}")
                    self.log(" ".join(parts))
                    self.model.train()
        finally:
            progress.close()

        metrics = {"loss": total_loss / max(total_samples, 1)}
        metrics["mae"] = total_train_mae / max(total_samples, 1)
        metrics["ssim"] = total_train_ssim / max(total_samples, 1) if self.config.ssim_loss_weight > 0 else None
        if self.config.ssim_loss_weight > 0:
            metrics.update(
                {
                    "rgb_loss": (total_recon_l1_loss + self.config.ssim_loss_weight * total_ssim_loss)
                    / max(total_samples, 1),
                    "recon_l1_loss": total_recon_l1_loss / max(total_samples, 1),
                    "ssim_loss": total_ssim_loss / max(total_samples, 1),
                    "ssim_loss_weight": self.config.ssim_loss_weight,
                    "total_loss": metrics["loss"],
                }
            )
        if self.config.aux_dynamics_loss:
            metrics.update(
                {
                    "rgb_loss": total_rgb_loss / max(total_samples, 1),
                    "aux_loss": total_aux_loss / max(total_samples, 1),
                    "total_loss": metrics["loss"],
                }
            )
        if self.config.wm_residual_loss:
            metrics["wm_residual_loss"] = total_wm_loss / max(total_samples, 1)
            metrics["total_loss"] = metrics["loss"]
        if self.has_dual_losses:
            metrics.update(
                {
                    "dual_wm_image_loss": total_dual_wm_image_loss / max(total_samples, 1),
                    "dual_simvp_image_loss": total_dual_simvp_image_loss / max(total_samples, 1),
                    "dual_align_loss": total_dual_align_loss / max(total_samples, 1),
                    "total_loss": metrics["loss"],
                }
            )
        if self.config.future_action_loss:
            metrics.update(
                {
                    "future_action_loss": total_future_action_loss / max(total_samples, 1),
                    "future_action_total_loss": total_future_action_loss / max(total_samples, 1),
                    "future_action_reg_loss": total_future_action_reg_loss / max(total_samples, 1),
                    "future_action_corr_loss": total_future_action_corr_loss / max(total_samples, 1),
                    "future_action_corr_longitudinal": total_future_action_corr_longitudinal / max(total_samples, 1),
                    "future_action_corr_steer": total_future_action_corr_steer / max(total_samples, 1),
                    "future_action_delta_loss": total_future_action_delta_loss / max(total_samples, 1),
                    "future_action_delta_longitudinal_loss": total_future_action_delta_longitudinal_loss / max(total_samples, 1),
                    "future_action_delta_steer_loss": total_future_action_delta_steer_loss / max(total_samples, 1),
                    "future_action_cls_loss": total_future_action_cls_loss / max(total_samples, 1),
                    "future_longitudinal_cls_loss": total_future_longitudinal_cls_loss / max(total_samples, 1),
                    "future_steer_cls_loss": total_future_steer_cls_loss / max(total_samples, 1),
                    "future_longitudinal_cls_acc": total_future_longitudinal_cls_acc / max(total_samples, 1),
                    "future_steer_cls_acc": total_future_steer_cls_acc / max(total_samples, 1),
                    "future_action_latent_feature_norm": total_future_latent_feature_norm / max(total_samples, 1),
                    "future_action_motion_feature_norm": total_motion_feature_norm / max(total_samples, 1),
                    "future_action_control_feature_norm": total_control_feature_norm / max(total_samples, 1),
                    "future_action_motion_energy_mean": total_motion_energy_mean / max(total_samples, 1),
                    "future_longitudinal_mae": total_future_longitudinal_mae / max(total_samples, 1),
                    "future_steer_mae": total_future_steer_mae / max(total_samples, 1),
                    "total_loss": metrics["loss"],
                }
            )
        metrics.update(self._dual_fusion_metric_fields())
        if gate_count:
            metrics.update(
                {
                    "wm_residual_gate_mean": gate_sum / gate_count,
                    "wm_residual_gate_min": gate_min,
                    "wm_residual_gate_max": gate_max,
                }
            )
        if dual_gate_count:
            metrics.update(
                {
                    "dual_gate_mean": dual_gate_sum / dual_gate_count,
                    "dual_gate_min": dual_gate_min,
                    "dual_gate_max": dual_gate_max,
                }
            )
            if self.config.debug_dual_gate:
                metrics.update(self._aggregate_debug_gate_stats(dual_gate_history, prefix="dual_gate"))
        return metrics

    @torch.no_grad()
    def validate_epoch(
        self,
        epoch: int,
        max_batches_override: int | None = None,
        validation_name: str = "Val",
    ) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_rgb_loss = 0.0
        total_recon_l1_loss = 0.0
        total_ssim_loss = 0.0
        total_aux_loss = 0.0
        total_wm_loss = 0.0
        total_dual_wm_image_loss = 0.0
        total_dual_simvp_image_loss = 0.0
        total_dual_align_loss = 0.0
        total_future_action_loss = 0.0
        total_future_action_reg_loss = 0.0
        total_future_action_corr_loss = 0.0
        total_future_action_corr_longitudinal = 0.0
        total_future_action_corr_steer = 0.0
        total_future_action_delta_loss = 0.0
        total_future_action_delta_longitudinal_loss = 0.0
        total_future_action_delta_steer_loss = 0.0
        total_future_action_cls_loss = 0.0
        total_future_longitudinal_cls_loss = 0.0
        total_future_steer_cls_loss = 0.0
        total_future_longitudinal_cls_acc = 0.0
        total_future_steer_cls_acc = 0.0
        total_future_latent_feature_norm = 0.0
        total_motion_feature_norm = 0.0
        total_control_feature_norm = 0.0
        total_motion_energy_mean = 0.0
        total_future_longitudinal_mae = 0.0
        total_future_steer_mae = 0.0
        dual_gate_sum = 0.0
        dual_gate_count = 0
        dual_gate_min = float("inf")
        dual_gate_max = float("-inf")
        dual_gate_history: list[tuple[int, dict[str, float]]] = []
        gate_sum = 0.0
        gate_count = 0
        gate_min = float("inf")
        gate_max = float("-inf")
        total_mae = 0.0
        total_ssim = 0.0
        total_samples = 0
        total_batches = 0
        val_steps = len(self.val_loader)
        max_batches = self.config.max_val_batches if max_batches_override is None else max_batches_override
        if max_batches is not None:
            val_steps = min(val_steps, max_batches)
        progress = self._create_progress(val_steps, f"{validation_name} {epoch:03d}")

        try:
            for step, batch in enumerate(self.val_loader, start=1):
                if max_batches is not None and step > max_batches:
                    break
                total_batches += 1
                past_frames = batch["past_frames"].to(self.device, non_blocking=True)
                future_frames = batch["future_frames"].to(self.device, non_blocking=True)
                batch_size = past_frames.shape[0]

                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    prediction_output = self.forward_model(batch, past_frames)
                    prediction, aux_loss, wm_loss, gate_stats, dual_losses, dual_gate_stats, future_action_metrics = self.compute_losses_from_output(
                        prediction_output, batch, future_frames
                    )
                    recon_l1_loss = frame_l1_loss(prediction, future_frames)
                    ssim_loss = (
                        frame_ssim_loss(prediction.float(), future_frames.float())
                        if self.config.ssim_loss_weight > 0
                        else future_frames.new_tensor(0.0)
                    )
                    rgb_loss = recon_l1_loss + self.config.ssim_loss_weight * ssim_loss
                    loss = (
                        rgb_loss
                        + self.config.aux_dynamics_weight * aux_loss
                        + self.config.wm_residual_loss_weight * wm_loss
                        + self.config.dual_wm_image_loss_weight * dual_losses["wm_image"]
                        + self.config.dual_simvp_image_loss_weight * dual_losses["simvp_image"]
                        + self.config.dual_align_loss_weight * dual_losses["align"]
                        + self.config.future_action_loss_weight * future_action_metrics["loss"]
                        + self.config.future_action_delta_loss_weight * future_action_metrics["delta_loss"]
                    )
                self._check_finite_tensors(
                    {
                        "total_loss": loss,
                        "rgb_loss": rgb_loss,
                        "recon_l1_loss": recon_l1_loss,
                        "ssim_loss": ssim_loss,
                        "aux_loss": aux_loss,
                        "wm_residual_loss": wm_loss,
                        "dual_wm_image_loss": dual_losses["wm_image"],
                        "dual_simvp_image_loss": dual_losses["simvp_image"],
                        "dual_align_loss": dual_losses["align"],
                        "future_action_loss": future_action_metrics["loss"],
                        "future_action_reg_loss": future_action_metrics["reg_loss"],
                        "future_action_corr_loss": future_action_metrics["corr_loss"],
                        "future_action_delta_loss": future_action_metrics["delta_loss"],
                        "future_action_cls_loss": future_action_metrics["cls_loss"],
                        "future_longitudinal_cls_loss": future_action_metrics["longitudinal_cls_loss"],
                        "future_steer_cls_loss": future_action_metrics["steer_cls_loss"],
                    },
                    context=f"validation epoch {epoch} step {step} losses",
                )

                metrics = video_metrics(prediction.float(), future_frames.float())
                loss_value = float(loss.detach().cpu())
                rgb_loss_value = float(rgb_loss.detach().cpu())
                recon_l1_loss_value = float(recon_l1_loss.detach().cpu())
                ssim_loss_value = float(ssim_loss.detach().cpu())
                aux_loss_value = float(aux_loss.detach().cpu())
                wm_loss_value = float(wm_loss.detach().cpu())
                dual_wm_image_loss_value = float(dual_losses["wm_image"].detach().cpu())
                dual_simvp_image_loss_value = float(dual_losses["simvp_image"].detach().cpu())
                dual_align_loss_value = float(dual_losses["align"].detach().cpu())
                future_action_loss_value = float(future_action_metrics["loss"].detach().cpu())
                future_action_reg_loss_value = float(future_action_metrics["reg_loss"].detach().cpu())
                future_action_corr_loss_value = float(future_action_metrics["corr_loss"].detach().cpu())
                future_action_corr_longitudinal_value = float(future_action_metrics["corr_longitudinal"].detach().cpu())
                future_action_corr_steer_value = float(future_action_metrics["corr_steer"].detach().cpu())
                future_action_delta_loss_value = float(future_action_metrics["delta_loss"].detach().cpu())
                future_action_delta_longitudinal_loss_value = float(
                    future_action_metrics["delta_longitudinal_loss"].detach().cpu()
                )
                future_action_delta_steer_loss_value = float(future_action_metrics["delta_steer_loss"].detach().cpu())
                future_action_cls_loss_value = float(future_action_metrics["cls_loss"].detach().cpu())
                future_longitudinal_cls_loss_value = float(future_action_metrics["longitudinal_cls_loss"].detach().cpu())
                future_steer_cls_loss_value = float(future_action_metrics["steer_cls_loss"].detach().cpu())
                future_longitudinal_cls_acc_value = float(future_action_metrics["longitudinal_cls_acc"].detach().cpu())
                future_steer_cls_acc_value = float(future_action_metrics["steer_cls_acc"].detach().cpu())
                future_latent_feature_norm_value = float(future_action_metrics["future_latent_feature_norm"].detach().cpu())
                motion_feature_norm_value = float(future_action_metrics["motion_feature_norm"].detach().cpu())
                control_feature_norm_value = float(future_action_metrics["control_feature_norm"].detach().cpu())
                motion_energy_mean_value = float(future_action_metrics["motion_energy_mean"].detach().cpu())
                future_longitudinal_mae_value = float(future_action_metrics["longitudinal_mae"].detach().cpu())
                future_steer_mae_value = float(future_action_metrics["steer_mae"].detach().cpu())
                total_loss += loss_value * batch_size
                total_mae += metrics["mae"] * batch_size
                total_ssim += metrics["ssim"] * batch_size
                total_samples += batch_size
                if self.config.ssim_loss_weight > 0:
                    total_recon_l1_loss += recon_l1_loss_value * batch_size
                    total_ssim_loss += ssim_loss_value * batch_size
                if self.config.aux_dynamics_loss:
                    total_rgb_loss += rgb_loss_value * batch_size
                    total_aux_loss += aux_loss_value * batch_size
                if self.config.wm_residual_loss:
                    total_wm_loss += wm_loss_value * batch_size
                if self.has_dual_losses:
                    total_dual_wm_image_loss += dual_wm_image_loss_value * batch_size
                    total_dual_simvp_image_loss += dual_simvp_image_loss_value * batch_size
                    total_dual_align_loss += dual_align_loss_value * batch_size
                if self.config.future_action_loss:
                    total_future_action_loss += future_action_loss_value * batch_size
                    total_future_action_reg_loss += future_action_reg_loss_value * batch_size
                    total_future_action_corr_loss += future_action_corr_loss_value * batch_size
                    total_future_action_corr_longitudinal += future_action_corr_longitudinal_value * batch_size
                    total_future_action_corr_steer += future_action_corr_steer_value * batch_size
                    total_future_action_delta_loss += future_action_delta_loss_value * batch_size
                    total_future_action_delta_longitudinal_loss += future_action_delta_longitudinal_loss_value * batch_size
                    total_future_action_delta_steer_loss += future_action_delta_steer_loss_value * batch_size
                    total_future_action_cls_loss += future_action_cls_loss_value * batch_size
                    total_future_longitudinal_cls_loss += future_longitudinal_cls_loss_value * batch_size
                    total_future_steer_cls_loss += future_steer_cls_loss_value * batch_size
                    total_future_longitudinal_cls_acc += future_longitudinal_cls_acc_value * batch_size
                    total_future_steer_cls_acc += future_steer_cls_acc_value * batch_size
                    total_future_latent_feature_norm += future_latent_feature_norm_value * batch_size
                    total_motion_feature_norm += motion_feature_norm_value * batch_size
                    total_control_feature_norm += control_feature_norm_value * batch_size
                    total_motion_energy_mean += motion_energy_mean_value * batch_size
                    total_future_longitudinal_mae += future_longitudinal_mae_value * batch_size
                    total_future_steer_mae += future_steer_mae_value * batch_size
                if gate_stats:
                    gate_sum += gate_stats["mean"] * batch_size
                    gate_count += batch_size
                    gate_min = min(gate_min, gate_stats["min"])
                    gate_max = max(gate_max, gate_stats["max"])
                if dual_gate_stats:
                    dual_gate_history.append((batch_size, dual_gate_stats))
                    dual_gate_sum += dual_gate_stats["mean"] * batch_size
                    dual_gate_count += batch_size
                    dual_gate_min = min(dual_gate_min, dual_gate_stats["min"])
                    dual_gate_max = max(dual_gate_max, dual_gate_stats["max"])

                progress_metrics = {
                    "loss": loss_value,
                    "avg": total_loss / max(total_samples, 1),
                    "img_mae": metrics["mae"],
                    "ssim": metrics["ssim"],
                }
                if self.config.ssim_loss_weight > 0:
                    progress_metrics.update({"l1": recon_l1_loss_value, "ssim_loss": ssim_loss_value})
                if self.config.aux_dynamics_loss:
                    progress_metrics.update({"rgb": rgb_loss_value, "aux": aux_loss_value})
                if self.config.wm_residual_loss:
                    progress_metrics["wm"] = wm_loss_value
                if self.has_dual_losses:
                    progress_metrics["dual"] = (
                        self.config.dual_wm_image_loss_weight * dual_wm_image_loss_value
                        + self.config.dual_simvp_image_loss_weight * dual_simvp_image_loss_value
                        + self.config.dual_align_loss_weight * dual_align_loss_value
                    )
                if self.config.future_action_loss:
                    progress_metrics["act"] = future_action_loss_value
                    progress_metrics["act_reg"] = future_action_reg_loss_value
                    if self.config.future_action_corr_loss_weight > 0:
                        progress_metrics["act_corr"] = future_action_corr_loss_value
                    if self.config.future_action_delta_loss and self.config.future_action_delta_loss_weight > 0:
                        progress_metrics["act_delta"] = future_action_delta_loss_value
                    if self.config.future_action_cls_loss:
                        progress_metrics["act_cls"] = future_action_cls_loss_value
                    if self.config.future_action_head_variant in {"motion_context", "motion_context_v2"}:
                        progress_metrics["mot_norm"] = motion_feature_norm_value
                        progress_metrics["ctl_norm"] = control_feature_norm_value
                        progress_metrics["lat_norm"] = future_latent_feature_norm_value
                fusion_stats = self._dual_fusion_metric_fields()
                if fusion_stats:
                    progress_metrics["wm/simvp"] = fusion_stats["dual_conv1x1_wm_to_simvp_weight_norm_ratio"]
                if gate_stats:
                    progress_metrics["gate"] = gate_stats["mean"]
                if dual_gate_stats and not fusion_stats:
                    progress_metrics["dual_gate"] = dual_gate_stats["mean"]
                progress.update(step, progress_metrics)
        finally:
            progress.close()

        output = {
            "loss": total_loss / max(total_samples, 1),
            "mae": total_mae / max(total_samples, 1),
            "ssim": total_ssim / max(total_samples, 1),
            "batches": total_batches,
            "max_batches": max_batches,
            "validation_name": validation_name,
        }
        if self.config.ssim_loss_weight > 0:
            output.update(
                {
                    "rgb_loss": (total_recon_l1_loss + self.config.ssim_loss_weight * total_ssim_loss)
                    / max(total_samples, 1),
                    "recon_l1_loss": total_recon_l1_loss / max(total_samples, 1),
                    "ssim_loss": total_ssim_loss / max(total_samples, 1),
                    "ssim_loss_weight": self.config.ssim_loss_weight,
                    "total_loss": output["loss"],
                }
            )
        if self.config.aux_dynamics_loss:
            output.update(
                {
                    "rgb_loss": total_rgb_loss / max(total_samples, 1),
                    "aux_loss": total_aux_loss / max(total_samples, 1),
                    "total_loss": output["loss"],
                }
            )
        if self.config.wm_residual_loss:
            output["wm_residual_loss"] = total_wm_loss / max(total_samples, 1)
            output["total_loss"] = output["loss"]
        if self.has_dual_losses:
            output.update(
                {
                    "dual_wm_image_loss": total_dual_wm_image_loss / max(total_samples, 1),
                    "dual_simvp_image_loss": total_dual_simvp_image_loss / max(total_samples, 1),
                    "dual_align_loss": total_dual_align_loss / max(total_samples, 1),
                    "total_loss": output["loss"],
                }
            )
        if self.config.future_action_loss:
            output.update(
                {
                    "future_action_loss": total_future_action_loss / max(total_samples, 1),
                    "future_action_total_loss": total_future_action_loss / max(total_samples, 1),
                    "future_action_reg_loss": total_future_action_reg_loss / max(total_samples, 1),
                    "future_action_corr_loss": total_future_action_corr_loss / max(total_samples, 1),
                    "future_action_corr_longitudinal": total_future_action_corr_longitudinal / max(total_samples, 1),
                    "future_action_corr_steer": total_future_action_corr_steer / max(total_samples, 1),
                    "future_action_delta_loss": total_future_action_delta_loss / max(total_samples, 1),
                    "future_action_delta_longitudinal_loss": total_future_action_delta_longitudinal_loss / max(total_samples, 1),
                    "future_action_delta_steer_loss": total_future_action_delta_steer_loss / max(total_samples, 1),
                    "future_action_cls_loss": total_future_action_cls_loss / max(total_samples, 1),
                    "future_longitudinal_cls_loss": total_future_longitudinal_cls_loss / max(total_samples, 1),
                    "future_steer_cls_loss": total_future_steer_cls_loss / max(total_samples, 1),
                    "future_longitudinal_cls_acc": total_future_longitudinal_cls_acc / max(total_samples, 1),
                    "future_steer_cls_acc": total_future_steer_cls_acc / max(total_samples, 1),
                    "future_action_latent_feature_norm": total_future_latent_feature_norm / max(total_samples, 1),
                    "future_action_motion_feature_norm": total_motion_feature_norm / max(total_samples, 1),
                    "future_action_control_feature_norm": total_control_feature_norm / max(total_samples, 1),
                    "future_action_motion_energy_mean": total_motion_energy_mean / max(total_samples, 1),
                    "future_longitudinal_mae": total_future_longitudinal_mae / max(total_samples, 1),
                    "future_steer_mae": total_future_steer_mae / max(total_samples, 1),
                    "total_loss": output["loss"],
                }
            )
        output.update(self._dual_fusion_metric_fields())
        if gate_count:
            output.update(
                {
                    "wm_residual_gate_mean": gate_sum / gate_count,
                    "wm_residual_gate_min": gate_min,
                    "wm_residual_gate_max": gate_max,
                }
            )
        if dual_gate_count:
            output.update(
                {
                    "dual_gate_mean": dual_gate_sum / dual_gate_count,
                    "dual_gate_min": dual_gate_min,
                    "dual_gate_max": dual_gate_max,
                }
            )
            if self.config.debug_dual_gate:
                output.update(self._aggregate_debug_gate_stats(dual_gate_history, prefix="dual_gate"))
        return output

    def forward_model(self, batch: dict, past_frames: torch.Tensor):
        if getattr(self.model, "requires_conditioning", False):
            past_actions = batch["past_actions"].to(self.device, non_blocking=True)
            past_speed = batch["past_speed"].to(self.device, non_blocking=True)
            if (
                self.config.aux_dynamics_loss
                or self.config.wm_residual_loss
                or self.has_dual_losses
                or self.config.future_action_loss
                or getattr(self.model, "wm_latent_residual", False)
                or getattr(self.model, "model_variant", None) in {"av_wm_dual", "av_wm_dual_bigwm"}
            ):
                return self.model(
                    past_frames,
                    past_actions,
                    past_speed,
                    return_latents=True,
                    decode_wm_frames=self.config.dual_wm_image_loss_weight > 0,
                    decode_simvp_frames=self.config.dual_simvp_image_loss_weight > 0,
                )
            return self.model(past_frames, past_actions, past_speed)
        return self.model(past_frames)

    def compute_losses_from_output(
        self,
        prediction_output,
        batch: dict,
        future_frames: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, float] | None,
        dict[str, torch.Tensor],
        dict[str, float] | None,
        dict[str, torch.Tensor],
    ]:
        zero = future_frames.new_tensor(0.0)
        empty_dual_losses = {"wm_image": zero, "simvp_image": zero, "align": zero}
        empty_future_action = {
            "loss": zero,
            "reg_loss": zero,
            "corr_loss": zero,
            "corr_longitudinal": zero,
            "corr_steer": zero,
            "delta_loss": zero,
            "delta_longitudinal_loss": zero,
            "delta_steer_loss": zero,
            "cls_loss": zero,
            "longitudinal_cls_loss": zero,
            "steer_cls_loss": zero,
            "longitudinal_cls_acc": zero,
            "steer_cls_acc": zero,
            "future_latent_feature_norm": zero,
            "motion_feature_norm": zero,
            "control_feature_norm": zero,
            "motion_energy_mean": zero,
            "longitudinal_mae": zero,
            "steer_mae": zero,
        }
        if (
            not self.config.aux_dynamics_loss
            and not self.config.wm_residual_loss
            and not self.has_dual_losses
            and not self.config.future_action_loss
        ):
            if isinstance(prediction_output, dict):
                return (
                    prediction_output["frames"],
                    zero,
                    zero,
                    self.extract_gate_stats(prediction_output),
                    empty_dual_losses,
                    self.extract_dual_gate_stats(prediction_output, detailed=self.config.debug_dual_gate),
                    empty_future_action,
                )
            return prediction_output, zero, zero, None, empty_dual_losses, None, empty_future_action
        if not isinstance(prediction_output, dict):
            raise RuntimeError("Aux/WM residual losses require model output dict with latent tensors.")
        prediction = prediction_output["frames"]
        aux_loss = zero
        if self.config.aux_dynamics_loss:
            future_latents = prediction_output["future_latents"]
            aux_head = getattr(self.model, "aux_dynamics_head", None)
            if aux_head is None:
                raise RuntimeError("Aux dynamics loss is enabled, but model has no aux_dynamics_head.")
            pred_dyn = aux_head(future_latents)
            target_dyn = self.build_future_dynamics_target(batch).to(pred_dyn.device)
            aux_loss = self._loss_by_type(pred_dyn, target_dyn, self.config.aux_dynamics_loss_type)

        wm_loss = zero
        if self.config.wm_residual_loss:
            wm_loss = self.compute_wm_residual_loss(prediction_output, future_frames)

        dual_losses = self.compute_dual_losses(prediction_output, future_frames)
        future_action_metrics = self.compute_future_action_loss(prediction_output, batch, future_frames)
        gate_stats = self.extract_gate_stats(prediction_output)
        dual_gate_stats = self.extract_dual_gate_stats(prediction_output, detailed=self.config.debug_dual_gate)
        return prediction, aux_loss, wm_loss, gate_stats, dual_losses, dual_gate_stats, future_action_metrics

    def _loss_by_type(self, prediction: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
        if loss_type == "smooth_l1":
            return F.smooth_l1_loss(prediction, target)
        if loss_type == "l1":
            return F.l1_loss(prediction, target)
        if loss_type in {"mse", "l2"}:
            return F.mse_loss(prediction, target)
        raise ValueError(f"Unknown loss type: {loss_type}")

    def compute_wm_residual_loss(self, prediction_output: dict, future_frames: torch.Tensor) -> torch.Tensor:
        residual = prediction_output.get("wm_latent_residual")
        simvp_future = prediction_output.get("simvp_future_latents")
        if residual is None or simvp_future is None:
            raise RuntimeError("WM residual loss requires wm_latent_residual and simvp_future_latents outputs.")
        if not hasattr(self.model, "encode_frames"):
            raise RuntimeError("WM residual loss requires model.encode_frames().")
        with torch.no_grad():
            future_target_latents = self.model.encode_frames(future_frames).detach()
            simvp_target = simvp_future.detach()
            target_residual = future_target_latents - simvp_target
        return self._loss_by_type(residual, target_residual, self.config.wm_residual_loss_type)

    def compute_dual_losses(self, prediction_output: dict, future_frames: torch.Tensor) -> dict[str, torch.Tensor]:
        zero = future_frames.new_tensor(0.0)
        losses = {"wm_image": zero, "simvp_image": zero, "align": zero}
        if not self.has_dual_losses:
            return losses
        if "wm_future_latents" not in prediction_output:
            raise RuntimeError("Dual losses require av_wm_dual outputs with wm_future_latents.")

        if self.config.dual_wm_image_loss_weight > 0:
            frames_wm = prediction_output.get("frames_wm")
            if frames_wm is None:
                raise RuntimeError("dual_wm_image_loss requires frames_wm output.")
            losses["wm_image"] = frame_l1_loss(frames_wm, future_frames)
        if self.config.dual_simvp_image_loss_weight > 0:
            frames_simvp = prediction_output.get("frames_simvp")
            if frames_simvp is None:
                raise RuntimeError("dual_simvp_image_loss requires frames_simvp output.")
            losses["simvp_image"] = frame_l1_loss(frames_simvp, future_frames)
        if self.config.dual_align_loss_weight > 0:
            z_simvp = prediction_output.get("simvp_future_latents")
            z_wm = prediction_output.get("wm_future_latents")
            if z_simvp is None or z_wm is None:
                raise RuntimeError("dual_align_loss requires simvp_future_latents and wm_future_latents.")
            losses["align"] = self.compute_dual_align_loss(z_simvp, z_wm)
        return losses

    def compute_dual_align_loss(self, z_simvp: torch.Tensor, z_wm: torch.Tensor) -> torch.Tensor:
        def align(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            if self.config.dual_align_loss_type == "smooth_l1":
                return F.smooth_l1_loss(prediction, target)
            if self.config.dual_align_loss_type == "l1":
                return F.l1_loss(prediction, target)
            if self.config.dual_align_loss_type == "cosine":
                pred_flat = prediction.flatten(start_dim=2)
                target_flat = target.flatten(start_dim=2)
                return 1.0 - F.cosine_similarity(pred_flat, target_flat, dim=-1).mean()
            raise ValueError(f"Unknown dual_align_loss_type {self.config.dual_align_loss_type!r}")

        direction = self.config.dual_align_direction
        if direction == "simvp_to_wm":
            return align(z_simvp, z_wm.detach())
        if direction == "wm_to_simvp":
            target = z_simvp.detach() if self.config.dual_detach_simvp_target else z_simvp
            return align(z_wm, target)
        if direction == "symmetric":
            return 0.5 * (align(z_simvp, z_wm.detach()) + align(z_wm, z_simvp.detach()))
        raise ValueError(f"Unknown dual_align_direction {direction!r}")

    def compute_future_action_loss(
        self,
        prediction_output: dict,
        batch: dict,
        future_frames: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        zero = future_frames.new_tensor(0.0)
        result = {
            "loss": zero,
            "reg_loss": zero,
            "corr_loss": zero,
            "corr_longitudinal": zero,
            "corr_steer": zero,
            "delta_loss": zero,
            "delta_longitudinal_loss": zero,
            "delta_steer_loss": zero,
            "cls_loss": zero,
            "longitudinal_cls_loss": zero,
            "steer_cls_loss": zero,
            "longitudinal_cls_acc": zero,
            "steer_cls_acc": zero,
            "future_latent_feature_norm": zero,
            "motion_feature_norm": zero,
            "control_feature_norm": zero,
            "motion_energy_mean": zero,
            "longitudinal_mae": zero,
            "steer_mae": zero,
        }
        if not self.config.future_action_loss:
            return result
        pred = prediction_output.get("pred_future_actions")
        if pred is None:
            raise RuntimeError("future_action_loss requires pred_future_actions output.")
        target = self.build_future_action_target(batch).to(pred.device)
        reg_loss = self._loss_by_type(pred, target, self.config.future_action_loss_type)
        result["reg_loss"] = reg_loss
        corr_loss = zero
        if self.config.future_action_corr_loss_weight > 0:
            corr_long = self._correlation_loss(pred[..., 0], target[..., 0])
            corr_steer = self._correlation_loss(pred[..., 1], target[..., 1])
            corr_loss = 0.5 * (corr_long + corr_steer)
            result["corr_loss"] = corr_loss
            result["corr_longitudinal"] = corr_long
            result["corr_steer"] = corr_steer
        result["loss"] = reg_loss + self.config.future_action_corr_loss_weight * corr_loss
        if (
            self.config.future_action_delta_loss
            and self.config.future_action_delta_loss_weight > 0
            and pred.shape[1] >= 2
        ):
            pred_delta = pred[:, 1:, :] - pred[:, :-1, :]
            target_delta = target[:, 1:, :] - target[:, :-1, :]
            delta_long = self._loss_by_type(
                pred_delta[..., 0],
                target_delta[..., 0],
                self.config.future_action_delta_loss_type,
            )
            delta_steer = self._loss_by_type(
                pred_delta[..., 1],
                target_delta[..., 1],
                self.config.future_action_delta_loss_type,
            )
            delta_loss = (
                self.config.future_action_delta_longitudinal_weight * delta_long
                + self.config.future_action_delta_steer_weight * delta_steer
            )
            result["delta_loss"] = delta_loss
            result["delta_longitudinal_loss"] = delta_long
            result["delta_steer_loss"] = delta_steer
        errors = torch.abs(pred - target)
        result["longitudinal_mae"] = errors[..., 0].mean()
        result["steer_mae"] = errors[..., 1].mean()
        feature_norms = prediction_output.get("future_action_feature_norms")
        if isinstance(feature_norms, dict):
            result["future_latent_feature_norm"] = feature_norms.get("future_latent_feature_norm", zero).to(pred.device)
            result["motion_feature_norm"] = feature_norms.get("motion_feature_norm", zero).to(pred.device)
            result["control_feature_norm"] = feature_norms.get("control_feature_norm", zero).to(pred.device)
            result["motion_energy_mean"] = feature_norms.get("motion_energy_mean", zero).to(pred.device)
        if self.config.future_action_cls_loss:
            long_logits = prediction_output.get("pred_future_longitudinal_logits")
            steer_logits = prediction_output.get("pred_future_steer_logits")
            if long_logits is None or steer_logits is None:
                raise RuntimeError("future_action_cls_loss requires future action classification logits.")
            original_target = self.build_future_action_original_target(batch).to(pred.device)
            long_labels = self._longitudinal_state_labels(
                original_target[..., 0],
                self.config.longitudinal_coast_threshold,
            )
            steer_labels = self._steering_direction_labels(
                original_target[..., 1],
                self.config.steer_straight_threshold,
            )
            long_loss = F.cross_entropy(long_logits.reshape(-1, 3), long_labels.reshape(-1))
            steer_loss = F.cross_entropy(steer_logits.reshape(-1, 3), steer_labels.reshape(-1))
            cls_loss = (
                self.config.future_action_longitudinal_cls_weight * long_loss
                + self.config.future_action_steer_cls_weight * steer_loss
            )
            result["longitudinal_cls_loss"] = long_loss
            result["steer_cls_loss"] = steer_loss
            result["cls_loss"] = cls_loss
            result["loss"] = result["loss"] + self.config.future_action_cls_weight * cls_loss
            result["longitudinal_cls_acc"] = (
                long_logits.argmax(dim=-1).eq(long_labels).float().mean()
            )
            result["steer_cls_acc"] = (
                steer_logits.argmax(dim=-1).eq(steer_labels).float().mean()
            )
        return result

    @staticmethod
    def _correlation_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        pred = prediction.float().reshape(-1)
        tgt = target.float().reshape(-1)
        pred_centered = pred - pred.mean()
        tgt_centered = tgt - tgt.mean()
        pred_energy = torch.sum(pred_centered * pred_centered)
        tgt_energy = torch.sum(tgt_centered * tgt_centered)
        if bool((pred_energy <= eps).detach().cpu()) or bool((tgt_energy <= eps).detach().cpu()):
            return prediction.new_tensor(0.0)
        denom = torch.sqrt(pred_energy * tgt_energy + eps)
        corr = torch.sum(pred_centered * tgt_centered) / denom
        return 1.0 - corr

    @staticmethod
    def extract_gate_stats(prediction_output) -> dict[str, float] | None:
        if not isinstance(prediction_output, dict):
            return None
        gate = prediction_output.get("wm_latent_residual_gate")
        if gate is None:
            return None
        gate_detached = gate.detach()
        return {
            "mean": float(gate_detached.mean().cpu()),
            "min": float(gate_detached.min().cpu()),
            "max": float(gate_detached.max().cpu()),
        }

    @staticmethod
    def extract_dual_gate_stats(prediction_output, detailed: bool = False) -> dict[str, float] | None:
        if not isinstance(prediction_output, dict):
            return None
        gate = prediction_output.get("dual_gate")
        if gate is None:
            return None
        gate_detached = gate.detach().float()
        flat = gate_detached.reshape(-1)
        stats = {
            "mean": float(flat.mean().cpu()),
            "min": float(flat.min().cpu()),
            "max": float(flat.max().cpu()),
        }
        if not detailed:
            return stats
        stats.update(
            {
                "std": float(flat.std(unbiased=False).cpu()),
                "p01": float(torch.quantile(flat, 0.01).cpu()),
                "p10": float(torch.quantile(flat, 0.10).cpu()),
                "p50": float(torch.quantile(flat, 0.50).cpu()),
                "p90": float(torch.quantile(flat, 0.90).cpu()),
                "p99": float(torch.quantile(flat, 0.99).cpu()),
                "frac_lt_0_05": float((flat < 0.05).float().mean().cpu()),
                "frac_near_0_25": float((torch.abs(flat - 0.25) < 1e-3).float().mean().cpu()),
                "frac_near_0_5": float((torch.abs(flat - 0.5) < 1e-3).float().mean().cpu()),
                "frac_gt_0_95": float((flat > 0.95).float().mean().cpu()),
            }
        )
        logits = prediction_output.get("dual_gate_logits")
        if logits is not None:
            logits_flat = logits.detach().float().reshape(-1)
            stats.update(
                {
                    "logits_mean": float(logits_flat.mean().cpu()),
                    "logits_std": float(logits_flat.std(unbiased=False).cpu()),
                    "logits_min": float(logits_flat.min().cpu()),
                    "logits_max": float(logits_flat.max().cpu()),
                }
            )
        return stats

    @staticmethod
    def _format_debug_gate_stats(stats: dict[str, float], prefix: str = "dual_gate") -> str:
        keys = (
            "std",
            "p01",
            "p10",
            "p50",
            "p90",
            "p99",
            "frac_lt_0_05",
            "frac_near_0_25",
            "frac_near_0_5",
            "frac_gt_0_95",
            "logits_mean",
            "logits_std",
            "logits_min",
            "logits_max",
        )
        return "".join(f" {prefix}_{key}={stats[key]:.6f}" for key in keys if key in stats)

    @staticmethod
    def _aggregate_debug_gate_stats(
        history: list[tuple[int, dict[str, float]]],
        prefix: str = "dual_gate",
    ) -> dict[str, float]:
        if not history:
            return {}
        total = sum(weight for weight, _ in history)
        if total <= 0:
            return {}
        keys = set().union(*(stats.keys() for _, stats in history))
        output = {}
        for key in keys:
            values = [(weight, stats[key]) for weight, stats in history if key in stats]
            if not values:
                continue
            if key.endswith("_min") or key == "min":
                output[f"{prefix}_{key}"] = min(value for _, value in values)
            elif key.endswith("_max") or key == "max":
                output[f"{prefix}_{key}"] = max(value for _, value in values)
            else:
                denom = sum(weight for weight, _ in values)
                output[f"{prefix}_{key}"] = sum(weight * value for weight, value in values) / max(denom, 1)
        return output

    def build_future_dynamics_target(self, batch: dict) -> torch.Tensor:
        future_actions = batch["future_actions"].to(self.device, non_blocking=True)
        future_speed = batch["future_speed"].to(self.device, non_blocking=True)
        longitudinal_steer_speed = controls_to_longitudinal_steer_speed(future_actions, future_speed)
        future_longitudinal = longitudinal_steer_speed[..., 0:1]
        future_steer = longitudinal_steer_speed[..., 1:2]
        future_speed = longitudinal_steer_speed[..., 2:3]
        return torch.cat([future_steer, future_longitudinal, future_speed], dim=-1)

    def build_future_action_target(self, batch: dict) -> torch.Tensor:
        original = self.build_future_action_original_target(batch)
        longitudinal = original[..., 0:1]
        steer = original[..., 1:2]
        if self.config.future_steer_target_scale <= 0:
            raise ValueError("future_steer_target_scale must be positive")
        if self.config.future_steer_target_scale != 1.0:
            steer = torch.clamp(steer / float(self.config.future_steer_target_scale), -1.0, 1.0)
        return torch.cat([longitudinal, steer], dim=-1)

    def build_future_action_original_target(self, batch: dict) -> torch.Tensor:
        future_actions = batch["future_actions"].to(self.device, non_blocking=True)
        future_speed = batch["future_speed"].to(self.device, non_blocking=True)
        converted = controls_to_longitudinal_steer_speed(future_actions, future_speed)
        return converted[..., [0, 1]]

    @staticmethod
    def _longitudinal_state_labels(values: torch.Tensor, threshold: float) -> torch.Tensor:
        labels = torch.ones_like(values, dtype=torch.long)
        labels = torch.where(values < -threshold, torch.zeros_like(labels), labels)
        labels = torch.where(values > threshold, torch.full_like(labels, 2), labels)
        return labels

    @staticmethod
    def _steering_direction_labels(values: torch.Tensor, threshold: float) -> torch.Tensor:
        labels = torch.ones_like(values, dtype=torch.long)
        labels = torch.where(values < -threshold, torch.zeros_like(labels), labels)
        labels = torch.where(values > threshold, torch.full_like(labels, 2), labels)
        return labels

    def _check_finite_tensors(self, tensors: dict[str, torch.Tensor], context: str) -> None:
        bad = []
        for name, tensor in tensors.items():
            if tensor is None:
                continue
            if not torch.isfinite(tensor.detach()).all():
                bad.append(name)
        if not bad:
            return
        message = f"Non-finite tensor(s) in {context}: {', '.join(bad)}"
        if self.config.stop_on_nan:
            raise FloatingPointError(message)
        self.log(message)

    def _remaining_train_steps(self) -> int | None:
        if self.config.max_train_steps is None:
            return None
        return max(int(self.config.max_train_steps) - int(self.global_optimizer_steps), 0)

    def _reached_max_train_steps(self) -> bool:
        remaining = self._remaining_train_steps()
        return remaining is not None and remaining <= 0

    def _gradient_diagnostics(self) -> dict[str, float | bool | str | None]:
        groups = {
            "simvp_backbone": [],
            "wm_branch": [],
            "dual_gate": [],
            "future_action_head": [],
            "other": [],
        }
        gate_weight_norms = []
        gate_bias_values = []
        all_elements_finite = True
        first_nonfinite_grad_param = None
        first_nonfinite_grad_kind = None
        first_norm_overflow_param = None
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            finite_mask = torch.isfinite(grad)
            if not bool(finite_mask.all().item()):
                all_elements_finite = False
                if first_nonfinite_grad_param is None:
                    first_nonfinite_grad_param = name
                    if bool(torch.isnan(grad).any().item()):
                        first_nonfinite_grad_kind = "nan"
                    elif bool(torch.isinf(grad).any().item()):
                        first_nonfinite_grad_kind = "inf"
                    else:
                        first_nonfinite_grad_kind = "nonfinite"
            # Use float64 for norm diagnostics. float32 norms can overflow to
            # inf even when every gradient element is finite; that is a norm
            # overflow, not proof of non-finite gradients.
            norm = float(grad.double().norm().cpu())
            if not torch.isfinite(torch.tensor(norm)).item() and first_norm_overflow_param is None:
                first_norm_overflow_param = name
            if name.startswith("dual_wm_dynamics.") and "gate_head" in name:
                groups["dual_gate"].append(norm)
            elif name.startswith(("dual_wm_dynamics.", "wm_residual_dynamics.")):
                groups["wm_branch"].append(norm)
            elif name.startswith("future_action_head."):
                groups["future_action_head"].append(norm)
            elif name.startswith("model."):
                groups["simvp_backbone"].append(norm)
            else:
                groups["other"].append(norm)
            if name.startswith("dual_wm_dynamics.gate_head."):
                if parameter.detach().ndim > 1:
                    gate_weight_norms.append(float(parameter.detach().double().norm().cpu()))
                elif name.endswith(".bias"):
                    gate_bias_values.append(parameter.detach().float().reshape(-1).cpu())
        result: dict[str, float | bool | str | None] = {
            key: self._combine_norms(values)
            for key, values in groups.items()
        }
        squared = sum(float(value) ** 2 for values in groups.values() for value in values)
        result["total"] = squared ** 0.5
        result["all_elements_finite"] = all_elements_finite
        result["all_norms_finite"] = all(torch.isfinite(torch.tensor(float(value))).item() for values in groups.values() for value in values)
        result["first_nonfinite_grad_param"] = first_nonfinite_grad_param
        result["first_nonfinite_grad_kind"] = first_nonfinite_grad_kind
        result["first_norm_overflow_param"] = first_norm_overflow_param
        result["dual_gate_param_weight_norm"] = self._combine_norms(gate_weight_norms)
        if gate_bias_values:
            bias = torch.cat(gate_bias_values)
            result["dual_gate_param_bias_mean"] = float(bias.mean())
            result["dual_gate_param_bias_std"] = float(bias.std(unbiased=False))
            result["dual_gate_param_bias_min"] = float(bias.min())
            result["dual_gate_param_bias_max"] = float(bias.max())
        else:
            result["dual_gate_param_bias_mean"] = 0.0
            result["dual_gate_param_bias_std"] = 0.0
            result["dual_gate_param_bias_min"] = 0.0
            result["dual_gate_param_bias_max"] = 0.0
        return result

    @staticmethod
    def _combine_norms(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(sum(float(value) ** 2 for value in values) ** 0.5)

    def _format_activation_stats(self, prediction_output: dict[str, Any]) -> str:
        pieces = []
        for key in ("simvp_future_latents", "wm_future_latents", "future_latents", "dual_gate", "frames"):
            value = prediction_output.get(key)
            if value is None or not torch.is_tensor(value):
                continue
            tensor = value.detach().float()
            pieces.append(
                f"{key}_mean={float(tensor.mean().cpu()):.4f}"
                f" {key}_std={float(tensor.std(unbiased=False).cpu()):.4f}"
                f" {key}_min={float(tensor.min().cpu()):.4f}"
                f" {key}_max={float(tensor.max().cpu()):.4f}"
            )
        return "activation_stats: " + " ".join(pieces) if pieces else ""

    def save(self, name: str, epoch: int, metrics: dict[str, float]) -> None:
        config = {
            "trainer": asdict(self.config),
            "run": self.run_config,
            "run_dir": str(self.run_dir),
        }
        save_checkpoint(
            self.checkpoint_dir / name,
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            metrics=metrics,
            config=config,
            scaler=self.scaler,
            best_val_loss=self.best_val_loss,
        )

    def log(self, message: str) -> None:
        print(message)
        self.log_file_only(message)

    def log_file_only(self, message: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    def _create_progress(self, total: int, desc: str):
        if not self.config.progress_bar:
            return _ManualProgress(total=total, desc=desc, enabled=False)
        try:
            from tqdm.auto import tqdm

            return _TqdmProgress(total=total, desc=desc, tqdm_cls=tqdm)
        except Exception:
            return _ManualProgress(total=total, desc=desc, enabled=True)

    def _write_config(self) -> None:
        write_json(
            self.config_path,
            {
                "trainer": asdict(self.config),
                "run": self.run_config,
                "run_dir": str(self.run_dir),
                "checkpoints": str(self.checkpoint_dir),
                "rollout_grids": str(self.rollout_grid_dir),
                "benchmark": str(self.benchmark_dir),
            },
        )

    def _resume_from_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])

        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

        scaler_state = checkpoint.get("scaler_state")
        if scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)

        checkpoint_epoch = int(checkpoint.get("epoch", 0))
        self.start_epoch = checkpoint_epoch + 1

        if checkpoint.get("best_val_loss") is not None:
            self.best_val_loss = float(checkpoint["best_val_loss"])
        else:
            metrics = checkpoint.get("metrics", {})
            if "loss" in metrics:
                self.best_val_loss = float(metrics["loss"])

        self.metrics_history = self._load_metrics_history()

    def _load_metrics_history(self) -> list[dict[str, Any]]:
        if not self.metrics_path.exists():
            return []
        with self.metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        history = payload.get("epochs", [])
        if payload.get("best_val_loss") is not None:
            self.best_val_loss = float(payload["best_val_loss"])
        return history

    def generate_plots(self) -> None:
        try:
            from src.utils.plotting import plot_training_curves

            outputs = plot_training_curves(run_dir=self.run_dir)
            self.log(f"Generated training plots in: {self.run_dir / 'plots'}")
            self.log(f"Plot shading enabled: {outputs['shading']}")
        except Exception as exc:
            self.log(f"Warning: failed to generate training plots: {exc}")


# Backward-compatibility alias for development-era imports.
SimVPTrainer = TeleopWMTrainer
