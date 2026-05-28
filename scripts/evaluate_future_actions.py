#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for lightweight installs.
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets import CarlaRolloutDataset
from src.models import TeleopWMPredictor, controls_to_longitudinal_steer_speed
from src.utils import infer_run_dir_from_checkpoint, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate future [longitudinal, steer] prediction head.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/path/to/mile_action_diverse/train"))
    parser.add_argument("--split", choices=["train", "val", "validation", "test"], default="val")
    parser.add_argument("--indices", type=int, nargs="+", default=None)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--sample-strategy", choices=["first", "uniform"], default="first")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--normalize-controls", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--speed-scale", type=float, default=None)
    parser.add_argument("--control-steer-input-scale", type=float, default=None)
    parser.add_argument("--future-action-future-motion-scale", type=float, default=None)
    parser.add_argument("--future-action-spatial-pooling", choices=["global", "grid"], default=None)
    parser.add_argument("--future-action-spatial-grid", default=None)
    parser.add_argument("--future-action-source", choices=["final", "wm", "simvp"], default=None)
    parser.add_argument(
        "--future-steer-target-scale",
        type=float,
        default=None,
        help=(
            "Scale used during future-action training for the steering target. "
            "If omitted, read from checkpoint config; predictions are unscaled for metrics."
        ),
    )
    parser.add_argument("--dual-wm-conditioning", choices=["add", "concat", "film"], default=None)
    parser.add_argument("--steer-near-zero-threshold", type=float, default=0.03)
    parser.add_argument("--steer-mild-threshold", type=float, default=0.10)
    parser.add_argument("--steer-sharp-threshold", type=float, default=0.30)
    parser.add_argument("--longitudinal-coast-threshold", type=float, default=0.05)
    return parser.parse_args()


def select_indices(dataset_length: int, requested: int, strategy: str) -> list[int]:
    k = min(int(requested), int(dataset_length))
    if k <= 0:
        return []
    if strategy == "first":
        return list(range(k))
    if strategy == "uniform":
        return np.linspace(0, dataset_length - 1, k, dtype=int).tolist()
    raise ValueError(f"Unknown sample strategy: {strategy}")


def load_model(path: Path, device: torch.device, args: argparse.Namespace) -> tuple[TeleopWMPredictor, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    run_cfg = checkpoint.get("config", {}).get("run", {})
    model_variant = run_cfg.get("model_variant", "rgb")
    model_variant = "av_simvp" if model_variant == "av" else model_variant
    if model_variant not in {"av_wm_dual", "av_wm_dual_bigwm"} or not run_cfg.get("future_action_loss", False):
        raise ValueError("evaluate_future_actions.py requires a TeleopWM checkpoint with future_action_loss enabled.")
    image_size = (
        int(args.height if args.height is not None else run_cfg.get("height", 160)),
        int(args.width if args.width is not None else run_cfg.get("width", 256)),
    )
    future_action_source = args.future_action_source or run_cfg.get("future_action_source", "final")
    control_steer_input_scale = (
        args.control_steer_input_scale
        if args.control_steer_input_scale is not None
        else run_cfg.get("control_steer_input_scale", 1.0)
    )
    future_action_future_motion_scale = (
        args.future_action_future_motion_scale
        if args.future_action_future_motion_scale is not None
        else run_cfg.get("future_action_future_motion_scale", 1.0)
    )
    future_action_spatial_pooling = args.future_action_spatial_pooling or run_cfg.get("future_action_spatial_pooling", "global")
    future_action_spatial_grid = args.future_action_spatial_grid or run_cfg.get("future_action_spatial_grid", "1x1")
    model = TeleopWMPredictor(
        past_len=9,
        future_len=8,
        channels=3,
        image_size=image_size,
        hid_s=run_cfg.get("hid_s", 32),
        hid_t=run_cfg.get("hid_t", 256),
        n_s=run_cfg.get("n_s", 4),
        n_t=run_cfg.get("n_t", 4),
        model_type=run_cfg.get("model_type", "gSTA"),
        drop_path=run_cfg.get("drop_path", 0.0),
        action_dim=run_cfg.get("action_dim", 3),
        speed_dim=run_cfg.get("speed_dim", 1),
        conditioning_dim=run_cfg.get("conditioning_dim", 32),
        model_variant=model_variant,
        simvp_conditioning=run_cfg.get("simvp_conditioning", run_cfg.get("conditioning_fusion", "none")),
        simvp_conditioning_stage=run_cfg.get(
            "simvp_conditioning_stage",
            "multipoint" if run_cfg.get("conditioning_injection", "single") == "multipoint" else "input",
        ),
        wm_latent_residual=run_cfg.get("wm_latent_residual", False),
        wm_residual_hidden_dim=run_cfg.get("wm_residual_hidden_dim", 128),
        wm_residual_scale=run_cfg.get("wm_residual_scale", 0.1),
        wm_residual_gated=run_cfg.get("wm_residual_gated", True),
        dual_fusion=run_cfg.get("dual_fusion", "gated_add"),
        dual_wm_scale=run_cfg.get("dual_wm_scale", 1.0),
        dual_wm_hidden_dim=run_cfg.get("dual_wm_hidden_dim", 128),
        dual_wm_num_layers=run_cfg.get("dual_wm_num_layers", 3),
        dual_wm_conditioning=args.dual_wm_conditioning or run_cfg.get("dual_wm_conditioning", "add"),
        dual_wm_gated=run_cfg.get("dual_wm_gated", True),
        aux_dynamics_hidden_dim=run_cfg.get("aux_dynamics_hidden_dim") if run_cfg.get("aux_dynamics_loss", False) else None,
        future_action_prediction=True,
        future_action_hidden_dim=run_cfg.get("future_action_hidden_dim", 128),
        future_action_num_layers=run_cfg.get("future_action_num_layers", 1),
        future_action_dropout=run_cfg.get("future_action_dropout", 0.0),
        future_action_source=future_action_source,
        future_action_classification=run_cfg.get("future_action_cls_loss", False),
        future_action_head_variant=run_cfg.get("future_action_head_variant", "default"),
        future_action_detach_latents=run_cfg.get("future_action_detach_latents", True),
        future_action_future_motion_scale=future_action_future_motion_scale,
        future_action_spatial_pooling=future_action_spatial_pooling,
        future_action_spatial_grid=future_action_spatial_grid,
        control_steer_input_scale=control_steer_input_scale,
    )
    load_checkpoint(path, model, map_location="cpu")
    model.to(device).eval()
    return model, run_cfg


def target_actions(sample: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    actions = sample["future_actions"].unsqueeze(0).to(device)
    speed = sample["future_speed"].unsqueeze(0).to(device)
    return controls_to_longitudinal_steer_speed(actions, speed)[..., [0, 1]]


def tensor_1d(values: torch.Tensor) -> torch.Tensor:
    return values.detach().float().reshape(-1)


def regression_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float | None]:
    pred = tensor_1d(prediction)
    tgt = tensor_1d(target)
    err = pred - tgt
    mse = torch.mean(err ** 2)
    mae = torch.mean(torch.abs(err))
    target_var = torch.sum((tgt - tgt.mean()) ** 2)
    r2 = None
    if float(target_var.cpu()) > 1e-12:
        r2 = float((1.0 - torch.sum(err ** 2) / target_var).cpu())
    corr = None
    pred_centered = pred - pred.mean()
    tgt_centered = tgt - tgt.mean()
    denom = torch.sqrt(torch.sum(pred_centered ** 2) * torch.sum(tgt_centered ** 2))
    if float(denom.cpu()) > 1e-12:
        corr = float((torch.sum(pred_centered * tgt_centered) / denom).cpu())
    return {
        "mse": float(mse.cpu()),
        "rmse": float(torch.sqrt(mse).cpu()),
        "mae": float(mae.cpu()),
        "r2": r2,
        "pearson_corr": corr,
    }


def sign_agreement(
    prediction: torch.Tensor,
    target: torch.Tensor,
    near_zero_threshold: float,
) -> dict[str, float | int | None]:
    pred = tensor_1d(prediction)
    tgt = tensor_1d(target)
    mask = torch.abs(tgt) >= near_zero_threshold
    count = int(mask.sum().cpu())
    if count == 0:
        return {"accuracy": None, "count": 0}
    agreement = torch.sign(pred[mask]) == torch.sign(tgt[mask])
    return {"accuracy": float(agreement.float().mean().cpu()), "count": count}


def steering_category(values: torch.Tensor, mild_threshold: float, sharp_threshold: float) -> torch.Tensor:
    abs_values = torch.abs(values)
    categories = torch.zeros_like(values, dtype=torch.long)
    categories[abs_values >= mild_threshold] = 1
    categories[abs_values >= sharp_threshold] = 2
    return categories


def longitudinal_state(values: torch.Tensor, coast_threshold: float) -> torch.Tensor:
    states = torch.ones_like(values, dtype=torch.long)
    states[values <= -coast_threshold] = 0
    states[values >= coast_threshold] = 2
    return states


def steering_direction(values: torch.Tensor, straight_threshold: float) -> torch.Tensor:
    states = torch.ones_like(values, dtype=torch.long)
    states[values <= -straight_threshold] = 0
    states[values >= straight_threshold] = 2
    return states


def confusion_matrix(prediction: torch.Tensor, target: torch.Tensor, num_classes: int = 3) -> list[list[int]]:
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    pred = tensor_1d(prediction).long()
    tgt = tensor_1d(target).long()
    for gt, pr in zip(tgt, pred, strict=False):
        if 0 <= int(gt) < num_classes and 0 <= int(pr) < num_classes:
            matrix[int(gt), int(pr)] += 1
    return matrix.tolist()


def classification_accuracy(prediction: torch.Tensor, target: torch.Tensor) -> float:
    pred = tensor_1d(prediction)
    tgt = tensor_1d(target)
    if pred.numel() == 0:
        return 0.0
    return float((pred == tgt).float().mean().cpu())


def plot_results(rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = sorted({int(row["future_step"]) for row in rows})
    long_mae = [np.mean([r["abs_err_longitudinal"] for r in rows if int(r["future_step"]) == step]) for step in steps]
    steer_mae = [np.mean([r["abs_err_steer"] for r in rows if int(r["future_step"]) == step]) for step in steps]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(steps))
    ax.bar(x - 0.18, long_mae, 0.36, label="longitudinal")
    ax.bar(x + 0.18, steer_mae, 0.36, label="steer")
    ax.set_xticks(x, [f"t+{step}" for step in steps])
    ax.set_ylabel("MAE")
    ax.set_title("Future Action Per-Step MAE")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "per_step_mae.png", dpi=160)
    plt.close(fig)

    for key, label, filename in [
        ("longitudinal", "Longitudinal", "longitudinal_prediction_vs_gt.png"),
        ("steer", "Steering", "steering_prediction_vs_gt.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        for sample_index in sorted({row["sample_index"] for row in rows})[:12]:
            sample_rows = [row for row in rows if row["sample_index"] == sample_index]
            ax.plot(steps, [r[f"gt_{key}"] for r in sample_rows], color="#333333", alpha=0.25)
            ax.plot(steps, [r[f"pred_{key}"] for r in sample_rows], color="#d95f02", alpha=0.25)
        ax.set_xlabel("Future step")
        ax.set_ylabel(label)
        ax.set_title(f"{label} Prediction vs Ground Truth")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    for key, label, filename in [
        ("longitudinal", "Longitudinal", "longitudinal_scatter_pred_vs_gt.png"),
        ("steer", "Steering", "steering_scatter_pred_vs_gt.png"),
    ]:
        gt = np.asarray([row[f"gt_{key}"] for row in rows], dtype=np.float32)
        pred = np.asarray([row[f"pred_{key}"] for row in rows], dtype=np.float32)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(gt, pred, s=10, alpha=0.45, edgecolors="none")
        lo = float(min(gt.min(), pred.min()))
        hi = float(max(gt.max(), pred.max()))
        pad = max((hi - lo) * 0.05, 1e-3)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#333333", linewidth=1.0, linestyle="--")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(f"Ground Truth {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.set_title(f"{label}: Prediction vs Ground Truth")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, run_cfg = load_model(args.checkpoint, device, args)
    run_dir = infer_run_dir_from_checkpoint(args.checkpoint)
    output_dir = args.output_dir or (run_dir / "future_action_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    future_action_source = getattr(model, "future_action_source", run_cfg.get("future_action_source", "final"))
    future_steer_target_scale = float(
        args.future_steer_target_scale
        if args.future_steer_target_scale is not None
        else run_cfg.get("future_steer_target_scale", 1.0)
    )
    if future_steer_target_scale <= 0:
        raise ValueError("future_steer_target_scale must be positive")
    print(f"future action source: {future_action_source}")
    print(f"future action head variant: {getattr(model, 'future_action_head_variant', run_cfg.get('future_action_head_variant', 'default'))}")
    print(f"future action hidden dim: {getattr(model, 'future_action_hidden_dim', run_cfg.get('future_action_hidden_dim', 128))}")
    print(f"future action detach latents: {getattr(model, 'future_action_detach_latents', run_cfg.get('future_action_detach_latents', True))}")
    print(f"future action future motion scale: {getattr(model, 'future_action_future_motion_scale', run_cfg.get('future_action_future_motion_scale', 1.0))}")
    print(f"future action spatial pooling: {getattr(model, 'future_action_spatial_pooling', run_cfg.get('future_action_spatial_pooling', 'global'))}")
    print(f"future action spatial grid: {getattr(model, 'future_action_spatial_grid', run_cfg.get('future_action_spatial_grid', '1x1'))}")
    print(f"future action token dim: {getattr(model, 'future_action_token_dim', None)}")
    print(f"future steer target scale: {future_steer_target_scale}")
    print(f"control steer input scale: {getattr(model, 'control_steer_input_scale', 1.0)}")
    image_size = (
        int(args.height if args.height is not None else run_cfg.get("height", 160)),
        int(args.width if args.width is not None else run_cfg.get("width", 256)),
    )
    normalize_controls = bool(args.normalize_controls if args.normalize_controls is not None else run_cfg.get("normalize_controls", False))
    speed_scale = float(args.speed_scale if args.speed_scale is not None else run_cfg.get("speed_scale", 20.0))
    dataset = CarlaRolloutDataset(
        args.data_root,
        split=args.split,
        image_size=image_size,
        normalize_controls=normalize_controls,
        speed_scale=speed_scale,
        include_metadata=True,
    )
    if args.indices is not None:
        indices = args.indices
        sample_strategy = "explicit_indices"
    else:
        indices = select_indices(len(dataset), args.num_samples, args.sample_strategy)
        sample_strategy = args.sample_strategy
    print(f"dataset length: {len(dataset)}")
    print(f"requested samples: {len(args.indices) if args.indices is not None else args.num_samples}")
    print(f"evaluated samples: {len(indices)}")
    print(f"sample strategy: {sample_strategy}")
    if indices:
        print(f"first index: {indices[0]}")
        print(f"last index: {indices[-1]}")
    csv_rows: list[dict[str, Any]] = []
    all_pred = []
    all_target = []
    all_long_logits = []
    all_steer_logits = []
    loss_values = []
    with torch.no_grad():
        sample_iter = enumerate(indices)
        if tqdm is not None:
            sample_iter = tqdm(sample_iter, total=len(indices), desc="Evaluating future actions", unit="sample")
        for sample_number, dataset_index in sample_iter:
            sample = dataset[dataset_index]
            past = sample["past_frames"].unsqueeze(0).to(device)
            past_actions = sample["past_actions"].unsqueeze(0).to(device)
            past_speed = sample["past_speed"].unsqueeze(0).to(device)
            output = model(past, past_actions, past_speed, return_latents=True)
            pred = output.get("pred_future_actions")
            if pred is None:
                raise RuntimeError("Checkpoint/model did not produce pred_future_actions.")
            target = target_actions(sample, device)
            pred_for_metrics = pred.clone()
            pred_for_metrics[..., 1] = pred_for_metrics[..., 1] * future_steer_target_scale
            long_logits = output.get("pred_future_longitudinal_logits")
            steer_logits = output.get("pred_future_steer_logits")
            if long_logits is not None:
                all_long_logits.append(long_logits.cpu())
            if steer_logits is not None:
                all_steer_logits.append(steer_logits.cpu())
            all_pred.append(pred_for_metrics.cpu())
            all_target.append(target.cpu())
            sample_abs_error = torch.mean(torch.abs(pred_for_metrics - target), dim=(0, 1))
            if hasattr(sample_iter, "set_postfix"):
                sample_iter.set_postfix(
                    {
                        "idx": dataset_index,
                        "long_mae": f"{float(sample_abs_error[0].cpu()):.4f}",
                        "steer_mae": f"{float(sample_abs_error[1].cpu()):.4f}",
                    },
                    refresh=False,
                )
            loss_values.append(float(torch.nn.functional.smooth_l1_loss(pred_for_metrics, target).cpu()))
            metadata = sample.get("metadata", {})
            for step in range(pred.shape[1]):
                gt_long = float(target[0, step, 0].cpu())
                pr_long = float(pred[0, step, 0].cpu())
                gt_steer = float(target[0, step, 1].cpu())
                pr_steer_scaled = float(pred[0, step, 1].cpu())
                pr_steer = pr_steer_scaled * future_steer_target_scale
                steer_gt_cat = int(
                    steering_category(
                        torch.tensor(gt_steer),
                        args.steer_mild_threshold,
                        args.steer_sharp_threshold,
                    ).item()
                )
                steer_pred_cat = int(
                    steering_category(
                        torch.tensor(pr_steer),
                        args.steer_mild_threshold,
                        args.steer_sharp_threshold,
                    ).item()
                )
                long_gt_state = int(longitudinal_state(torch.tensor(gt_long), args.longitudinal_coast_threshold).item())
                long_pred_state = int(longitudinal_state(torch.tensor(pr_long), args.longitudinal_coast_threshold).item())
                steer_gt_direction = int(steering_direction(torch.tensor(gt_steer), args.steer_near_zero_threshold).item())
                steer_pred_direction = int(steering_direction(torch.tensor(pr_steer), args.steer_near_zero_threshold).item())
                csv_rows.append(
                    {
                        "sample_index": sample_number,
                        "dataset_index": dataset_index,
                        "town": metadata.get("town", ""),
                        "run_id": metadata.get("run_id", metadata.get("run_key", "")),
                        "run_key": metadata.get("run_key", ""),
                        "future_step": step + 1,
                        "gt_longitudinal": gt_long,
                        "pred_longitudinal": pr_long,
                        "abs_err_longitudinal": abs(pr_long - gt_long),
                        "sq_err_longitudinal": (pr_long - gt_long) ** 2,
                        "gt_longitudinal_state": ["brake", "coast", "accelerate"][long_gt_state],
                        "pred_longitudinal_state": ["brake", "coast", "accelerate"][long_pred_state],
                        "gt_steer": gt_steer,
                        "pred_steer_scaled": pr_steer_scaled,
                        "pred_steer_unscaled": pr_steer,
                        "pred_steer": pr_steer,
                        "abs_err_steer": abs(pr_steer - gt_steer),
                        "sq_err_steer": (pr_steer - gt_steer) ** 2,
                        "gt_steer_category": ["straight", "mild_turn", "sharp_turn"][steer_gt_cat],
                        "pred_steer_category": ["straight", "mild_turn", "sharp_turn"][steer_pred_cat],
                        "gt_steer_direction": ["left", "straight", "right"][steer_gt_direction],
                        "pred_steer_direction": ["left", "straight", "right"][steer_pred_direction],
                    }
                )

    pred_all = torch.cat(all_pred, dim=0)
    target_all = torch.cat(all_target, dim=0)
    err = pred_all - target_all
    if pred_all.shape[1] >= 2:
        pred_delta = pred_all[:, 1:, :] - pred_all[:, :-1, :]
        target_delta = target_all[:, 1:, :] - target_all[:, :-1, :]
        delta_err = pred_delta - target_delta
        per_step_delta_mae_longitudinal = [
            float(torch.mean(torch.abs(delta_err[:, step, 0]))) for step in range(delta_err.shape[1])
        ]
        per_step_delta_mae_steer = [
            float(torch.mean(torch.abs(delta_err[:, step, 1]))) for step in range(delta_err.shape[1])
        ]
        mae_delta_longitudinal = float(torch.mean(torch.abs(delta_err[..., 0])).cpu())
        mae_delta_steer = float(torch.mean(torch.abs(delta_err[..., 1])).cpu())
    else:
        per_step_delta_mae_longitudinal = []
        per_step_delta_mae_steer = []
        mae_delta_longitudinal = 0.0
        mae_delta_steer = 0.0
    longitudinal_metrics = regression_metrics(pred_all[..., 0], target_all[..., 0])
    steering_metrics = regression_metrics(pred_all[..., 1], target_all[..., 1])
    steering_sign = sign_agreement(pred_all[..., 1], target_all[..., 1], args.steer_near_zero_threshold)
    longitudinal_sign = sign_agreement(pred_all[..., 0], target_all[..., 0], args.longitudinal_coast_threshold)
    steer_pred_categories = steering_category(pred_all[..., 1], args.steer_mild_threshold, args.steer_sharp_threshold)
    steer_target_categories = steering_category(target_all[..., 1], args.steer_mild_threshold, args.steer_sharp_threshold)
    long_pred_states = longitudinal_state(pred_all[..., 0], args.longitudinal_coast_threshold)
    long_target_states = longitudinal_state(target_all[..., 0], args.longitudinal_coast_threshold)
    steer_pred_directions = steering_direction(pred_all[..., 1], args.steer_near_zero_threshold)
    steer_target_directions = steering_direction(target_all[..., 1], args.steer_near_zero_threshold)
    metrics = {
        "checkpoint": str(args.checkpoint),
        "future_action_source": future_action_source,
        "future_action_head_variant": getattr(model, "future_action_head_variant", run_cfg.get("future_action_head_variant", "default")),
        "future_action_hidden_dim": getattr(model, "future_action_hidden_dim", run_cfg.get("future_action_hidden_dim", 128)),
        "future_action_detach_latents": getattr(model, "future_action_detach_latents", run_cfg.get("future_action_detach_latents", True)),
        "future_action_future_motion_scale": getattr(model, "future_action_future_motion_scale", run_cfg.get("future_action_future_motion_scale", 1.0)),
        "future_action_spatial_pooling": getattr(model, "future_action_spatial_pooling", run_cfg.get("future_action_spatial_pooling", "global")),
        "future_action_spatial_grid": getattr(model, "future_action_spatial_grid", run_cfg.get("future_action_spatial_grid", "1x1")),
        "future_action_token_dim": getattr(model, "future_action_token_dim", None),
        "future_steer_target_scale": future_steer_target_scale,
        "control_steer_input_scale": getattr(model, "control_steer_input_scale", run_cfg.get("control_steer_input_scale", 1.0)),
        "steering_metrics_units": "original_steering_units",
        "samples": len(indices),
        "dataset_length": len(dataset),
        "requested_samples": len(args.indices) if args.indices is not None else args.num_samples,
        "sample_strategy": sample_strategy,
        "first_index": indices[0] if indices else None,
        "last_index": indices[-1] if indices else None,
        "thresholds": {
            "steer_near_zero_threshold": args.steer_near_zero_threshold,
            "steer_mild_threshold": args.steer_mild_threshold,
            "steer_sharp_threshold": args.steer_sharp_threshold,
            "longitudinal_coast_threshold": args.longitudinal_coast_threshold,
        },
        "longitudinal": longitudinal_metrics,
        "steering": steering_metrics,
        "mae_longitudinal": longitudinal_metrics["mae"],
        "mae_steer": steering_metrics["mae"],
        "mse_longitudinal": longitudinal_metrics["mse"],
        "mse_steer": steering_metrics["mse"],
        "rmse_longitudinal": longitudinal_metrics["rmse"],
        "rmse_steer": steering_metrics["rmse"],
        "r2_longitudinal": longitudinal_metrics["r2"],
        "r2_steer": steering_metrics["r2"],
        "pearson_corr_longitudinal": longitudinal_metrics["pearson_corr"],
        "pearson_corr_steer": steering_metrics["pearson_corr"],
        "steering_sign_agreement_excluding_near_zero": steering_sign,
        "steering_turn_category_accuracy": classification_accuracy(steer_pred_categories, steer_target_categories),
        "longitudinal_sign_agreement_excluding_coast": longitudinal_sign,
        "longitudinal_state_accuracy": classification_accuracy(long_pred_states, long_target_states),
        "steering_direction_accuracy": classification_accuracy(steer_pred_directions, steer_target_directions),
        "steering_direction_confusion_matrix": confusion_matrix(steer_pred_directions, steer_target_directions),
        "longitudinal_state_confusion_matrix": confusion_matrix(long_pred_states, long_target_states),
        "smooth_l1": float(np.mean(loss_values)),
        "per_step_mae_longitudinal": [
            float(torch.mean(torch.abs(err[:, step, 0]))) for step in range(err.shape[1])
        ],
        "per_step_mae_steer": [
            float(torch.mean(torch.abs(err[:, step, 1]))) for step in range(err.shape[1])
        ],
        "mae_delta_longitudinal": mae_delta_longitudinal,
        "mae_delta_steer": mae_delta_steer,
        "per_step_delta_mae_longitudinal": per_step_delta_mae_longitudinal,
        "per_step_delta_mae_steer": per_step_delta_mae_steer,
    }
    for step, value in enumerate(per_step_delta_mae_longitudinal, start=2):
        metrics[f"longitudinal_delta_mae_step_{step}"] = value
    for step, value in enumerate(per_step_delta_mae_steer, start=2):
        metrics[f"steering_delta_mae_step_{step}"] = value
    for step in range(err.shape[1]):
        step_num = step + 1
        long_step = regression_metrics(pred_all[:, step, 0], target_all[:, step, 0])
        steer_step = regression_metrics(pred_all[:, step, 1], target_all[:, step, 1])
        metrics[f"longitudinal_r2_step_{step_num}"] = long_step["r2"]
        metrics[f"steering_r2_step_{step_num}"] = steer_step["r2"]
        metrics[f"longitudinal_corr_step_{step_num}"] = long_step["pearson_corr"]
        metrics[f"steering_corr_step_{step_num}"] = steer_step["pearson_corr"]
    if all_long_logits and all_steer_logits:
        long_logits_all = torch.cat(all_long_logits, dim=0)
        steer_logits_all = torch.cat(all_steer_logits, dim=0)
        long_cls_pred = long_logits_all.argmax(dim=-1)
        steer_cls_pred = steer_logits_all.argmax(dim=-1)
        metrics["future_action_cls_heads_present"] = True
        metrics["future_action_longitudinal_cls_accuracy"] = classification_accuracy(long_cls_pred, long_target_states)
        metrics["future_action_steer_cls_accuracy"] = classification_accuracy(steer_cls_pred, steer_target_directions)
        metrics["future_action_longitudinal_cls_confusion_matrix"] = confusion_matrix(long_cls_pred, long_target_states)
        metrics["future_action_steer_cls_confusion_matrix"] = confusion_matrix(steer_cls_pred, steer_target_directions)
    else:
        metrics["future_action_cls_heads_present"] = False
    (output_dir / "future_action_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "future_action_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    plot_results(csv_rows, output_dir)
    print(f"wrote future action evaluation: {output_dir}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
