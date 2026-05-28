#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import (
    align_maneuver_rows_to_dataset,
    category_counts,
    category_percentages,
    compute_rollout_weight_table,
    load_maneuver_metadata,
    load_rollout_metadata,
    load_rollout_weights,
    make_maneuver_balanced_sampler,
    make_maneuver_speed_balanced_sampler,
    make_weighted_sampler,
    maneuver_counts,
    maneuver_percentages,
    maneuver_speed_counts,
    maneuver_speed_percentages,
    parse_maneuver_weights,
    parse_maneuver_speed_weights,
    save_rollout_weight_outputs,
    save_maneuver_sampling_plot,
    save_maneuver_speed_sampling_plot,
    save_sampling_distribution_plot,
    simulate_maneuver_sampling,
    simulate_maneuver_speed_sampling,
    simulate_weighted_sampling,
)
from src.datasets import CarlaRolloutDataset
from src.models import TeleopWMPredictor, SimVPPredictor
from src.trainers import TeleopWMTrainer, TrainerConfig
from src.utils import seed_everything


RELEASE_DEFAULT_ARGS = {
    "--output-dir": "outputs/teleopwm",
    "--run-tag": "teleopwm",
    "--model-variant": "av_wm_dual_bigwm",
    "--simvp-conditioning": "none",
    "--dual-fusion": "conv1x1",
    "--dual-wm-hidden-dim": "512",
    "--dual-wm-num-layers": "3",
    "--dual-wm-conditioning": "film",
    "--future-action-source": "wm",
    "--future-action-head-variant": "motion_context_v2",
    "--future-action-hidden-dim": "256",
    "--future-action-spatial-pooling": "grid",
    "--future-action-spatial-grid": "2x4",
    "--future-action-future-motion-scale": "3.0",
    "--future-steer-target-scale": "0.30",
    "--control-steer-input-scale": "0.30",
    "--height": "320",
    "--width": "512",
}

RELEASE_DEFAULT_FLAGS = {
    "--future-action-loss",
    "--future-action-cls-loss",
}

RELEASE_DEFAULT_NEGATIVE_FLAGS = {
    "--no-future-action-detach-latents",
}


def _has_option(argv: list[str], option: str) -> bool:
    return any(item == option or item.startswith(f"{option}=") for item in argv)


def apply_release_defaults(argv: list[str]) -> list[str]:
    argv = list(argv)
    for option, value in RELEASE_DEFAULT_ARGS.items():
        if not _has_option(argv, option):
            argv.extend([option, value])
    for flag in RELEASE_DEFAULT_FLAGS:
        negative = f"--no-{flag[2:]}"
        if not _has_option(argv, flag) and not _has_option(argv, negative):
            argv.append(flag)
    for flag in RELEASE_DEFAULT_NEGATIVE_FLAGS:
        positive = f"--{flag.removeprefix('--no-')}"
        if not _has_option(argv, flag) and not _has_option(argv, positive):
            argv.append(flag)
    return argv


def print_public_help() -> None:
    print(
        """usage: train_teleopwm.py --data-root DATA_ROOT [options]

Train the final TeleopWM model.

Required:
  --data-root PATH                 Training split root.

Common options:
  --val-data-root PATH             Explicit validation split root.
  --output-dir PATH                Output root. Default: outputs/teleopwm
  --run-tag NAME                   Run tag. Default: teleopwm
  --batch-size N                   Batch size.
  --epochs N                       Number of epochs.
  --lr FLOAT                       Learning rate.
  --device cuda|cpu                Training device.
  --normalize-controls             Normalize controls/speed inputs.
  --speed-scale FLOAT              Speed normalization scale.
  --sampling-strategy STRATEGY     none or maneuver_speed_balanced.
  --maneuver-speed-metadata PATH   3x3 maneuver-speed metadata CSV.
  --max-train-steps N              Optional global optimizer-step cap.

TeleopWM defaults:
  final paper model, conv1x1 fusion, motion_context_v2 future-action head,
  grid 2x4 action tokens, 320x512 images, and TeleopWM output paths.

Advanced compatibility options are available in the source parser for older checkpoints."""
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TeleopWM training implementation.")
    parser.add_argument("--data-root", type=Path, default=Path("/path/to/mile_action_diverse/train/Town01"))
    parser.add_argument("--val-data-root", type=Path, default=None)
    parser.add_argument("--test-data-root", type=Path, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/teleopwm")
    parser.add_argument("--run-tag", default="teleopwm")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--normalize-controls", action="store_true")
    parser.add_argument("--speed-scale", type=float, default=20.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hid-s", type=int, default=32)
    parser.add_argument("--hid-t", type=int, default=256)
    parser.add_argument("--n-s", type=int, default=4)
    parser.add_argument("--n-t", type=int, default=4)
    parser.add_argument("--model-type", default="gSTA")
    parser.add_argument("--model-variant", choices=["rgb", "av", "av_simvp", "av_wm", "av_wm_dual", "av_wm_dual_bigwm"], default="rgb")
    parser.add_argument("--action-dim", type=int, default=3)
    parser.add_argument("--speed-dim", type=int, default=1)
    parser.add_argument("--conditioning-dim", type=int, default=32)
    parser.add_argument("--simvp-conditioning", choices=["none", "add", "concat", "film"], default=None)
    parser.add_argument("--simvp-conditioning-stage", choices=["input", "multipoint"], default=None)
    parser.add_argument("--conditioning-fusion", choices=["add", "concat", "film"], default=None)
    parser.add_argument("--conditioning-injection", choices=["single", "multipoint"], default=None)
    parser.add_argument("--wm-latent-residual", action="store_true")
    parser.add_argument("--wm-residual-hidden-dim", type=int, default=128)
    parser.add_argument("--wm-residual-scale", type=float, default=0.1)
    parser.add_argument("--wm-residual-gated", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wm-residual-loss", action="store_true")
    parser.add_argument("--wm-residual-loss-weight", type=float, default=0.05)
    parser.add_argument("--wm-residual-loss-type", choices=["smooth_l1", "l1"], default="smooth_l1")
    parser.add_argument("--ssim-loss-weight", type=float, default=0.0)
    parser.add_argument("--dual-fusion", choices=["add", "gated_add", "convex", "wm_only", "simvp_only", "conv1x1"], default="gated_add")
    parser.add_argument("--dual-wm-scale", type=float, default=1.0)
    parser.add_argument("--dual-wm-hidden-dim", type=int, default=128)
    parser.add_argument("--dual-wm-num-layers", type=int, default=3)
    parser.add_argument("--dual-wm-conditioning", choices=["add", "concat", "film"], default="add")
    parser.add_argument("--dual-wm-gated", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dual-wm-image-loss-weight", type=float, default=0.0)
    parser.add_argument("--dual-simvp-image-loss-weight", type=float, default=0.0)
    parser.add_argument("--dual-align-loss-weight", type=float, default=0.0)
    parser.add_argument("--dual-align-loss-type", choices=["smooth_l1", "l1", "cosine"], default="smooth_l1")
    parser.add_argument("--dual-align-direction", choices=["simvp_to_wm", "wm_to_simvp", "symmetric"], default="simvp_to_wm")
    parser.add_argument("--dual-detach-simvp-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-path", type=float, default=0.0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-interim-val-batches", type=int, default=None)
    parser.add_argument("--eval-every-steps", type=int, default=None)
    parser.add_argument("--stop-on-nan", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-grad-norms", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-activation-stats", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-dual-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--subset-strategy", choices=["first", "random", "maneuver_weighted"], default="first")
    parser.add_argument("--val-subset-strategy", choices=["first", "random", "maneuver_weighted"], default="first")
    parser.add_argument("--sampling-strategy", choices=["none", "action_balanced", "maneuver_balanced", "maneuver_speed_balanced"], default="none")
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument("--maneuver-metadata", type=Path, default=None)
    parser.add_argument("--val-maneuver-metadata", type=Path, default=None)
    parser.add_argument("--maneuver-weights", default="straight=1,mild_turn=1,sharp_turn=1")
    parser.add_argument("--maneuver-speed-metadata", type=Path, default=None)
    parser.add_argument("--val-maneuver-speed-metadata", type=Path, default=None)
    parser.add_argument(
        "--maneuver-speed-weights",
        default="straight_accel=1,straight_const=1,straight_decel=1,mild_turn_accel=1,mild_turn_const=1,mild_turn_decel=1,sharp_turn_accel=1,sharp_turn_const=1,sharp_turn_decel=1",
    )
    parser.add_argument("--speed-delta-threshold", type=float, default=0.3)
    parser.add_argument("--aux-dynamics-loss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--aux-dynamics-weight", type=float, default=0.05)
    parser.add_argument("--aux-dynamics-hidden-dim", type=int, default=128)
    parser.add_argument("--aux-dynamics-loss-type", choices=["smooth_l1", "l1"], default="smooth_l1")
    parser.add_argument("--future-action-loss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--future-action-loss-weight", type=float, default=0.1)
    parser.add_argument("--future-action-loss-type", choices=["smooth_l1", "l1", "mse", "l2"], default="smooth_l1")
    parser.add_argument("--future-action-hidden-dim", type=int, default=128)
    parser.add_argument("--future-action-num-layers", type=int, default=1)
    parser.add_argument("--future-action-dropout", type=float, default=0.0)
    parser.add_argument("--future-action-source", choices=["final", "wm", "simvp"], default="final")
    parser.add_argument("--future-steer-target-scale", type=float, default=1.0)
    parser.add_argument("--control-steer-input-scale", type=float, default=1.0)
    parser.add_argument("--future-action-head-variant", choices=["default", "motion_context", "motion_context_v2"], default="default")
    parser.add_argument("--future-action-detach-latents", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--future-action-future-motion-scale", type=float, default=1.0)
    parser.add_argument("--future-action-spatial-pooling", choices=["global", "grid"], default="global")
    parser.add_argument("--future-action-spatial-grid", default="1x1")
    parser.add_argument("--future-action-corr-loss-weight", type=float, default=0.0)
    parser.add_argument("--future-action-delta-loss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--future-action-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--future-action-delta-loss-type", choices=["smooth_l1", "l1", "mse"], default="smooth_l1")
    parser.add_argument("--future-action-delta-longitudinal-weight", type=float, default=1.0)
    parser.add_argument("--future-action-delta-steer-weight", type=float, default=1.0)
    parser.add_argument("--future-action-cls-loss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--future-action-cls-weight", type=float, default=0.1)
    parser.add_argument("--future-action-longitudinal-cls-weight", type=float, default=1.0)
    parser.add_argument("--future-action-steer-cls-weight", type=float, default=1.0)
    parser.add_argument("--longitudinal-coast-threshold", type=float, default=0.05)
    parser.add_argument("--steer-straight-threshold", type=float, default=0.03)
    return parser.parse_args()


def maybe_subset(dataset, max_samples: int | None):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, range(max_samples))


def split_dataset_indices(dataset_len: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    if dataset_len < 2:
        raise ValueError("Internal validation split requires at least two rollout windows")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"--val-fraction must be in (0, 1), got {val_fraction}")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(dataset_len)
    val_count = int(round(dataset_len * val_fraction))
    val_count = min(max(val_count, 1), dataset_len - 1)
    val_indices = sorted(int(index) for index in permutation[:val_count])
    train_indices = sorted(int(index) for index in permutation[val_count:])
    return train_indices, val_indices


def resolve_dataset_indices(dataset) -> tuple[object, list[int] | None]:
    """Return the base dataset and absolute indices represented by nested Subsets."""

    if not isinstance(dataset, Subset):
        return dataset, None
    base, parent_indices = resolve_dataset_indices(dataset.dataset)
    own_indices = [int(index) for index in dataset.indices]
    if parent_indices is None:
        return base, own_indices
    return base, [parent_indices[index] for index in own_indices]


def dataset_towns(dataset) -> list[str]:
    base, indices = resolve_dataset_indices(dataset)
    if not hasattr(base, "windows"):
        return []
    if indices is None:
        towns = {run.town for run in getattr(base, "run_infos", []) if run.town}
    else:
        towns = {base.windows[index].town for index in indices if base.windows[index].town}
    return sorted(str(town) for town in towns)


def dataset_run_counts(dataset) -> dict[str, int]:
    base, indices = resolve_dataset_indices(dataset)
    if not hasattr(base, "windows"):
        return {}
    if indices is None:
        iterator = range(len(base.windows))
    else:
        iterator = indices
    counts = Counter()
    for index in iterator:
        window = base.windows[int(index)]
        label = f"{window.town or ''}/{window.run_id}"
        counts[label] += 1
    return dict(sorted((label, int(count)) for label, count in counts.items()))


def dataset_absolute_indices(dataset) -> list[int]:
    base, indices = resolve_dataset_indices(dataset)
    if indices is None:
        return list(range(len(base)))
    return [int(index) for index in indices]


def select_subset_indices(
    dataset_len: int,
    max_samples: int | None,
    strategy: str,
    seed: int,
    maneuver_rows: list[dict] | None = None,
    maneuver_weight_config: dict[str, float] | None = None,
) -> list[int] | None:
    """Select deterministic candidate windows before DataLoader/sampler creation.

    Historical behavior was equivalent to ``strategy='first'``: the first
    ``max_samples`` rollout windows were selected before any weighted sampler
    ran. ``maneuver_weighted`` instead samples candidate windows from the full
    aligned maneuver metadata with probabilities proportional to maneuver
    weights, so small fast runs approximate the full weighted training
    distribution rather than the first town/run folders.

    This is stage 1 only: it creates the candidate subset. It is intentionally
    separate from stage 2 DataLoader weighted sampling, which is controlled by
    ``--sampling-strategy maneuver_balanced`` and can further rebalance batches
    during each training epoch.
    """

    if max_samples is None or max_samples >= dataset_len:
        return None
    if max_samples < 1:
        raise ValueError("max samples must be positive when provided")
    if strategy == "first":
        return list(range(max_samples))

    rng = np.random.default_rng(seed)
    if strategy == "random":
        return sorted(int(index) for index in rng.choice(dataset_len, size=max_samples, replace=False))

    if strategy != "maneuver_weighted":
        raise ValueError(f"Unknown subset strategy {strategy!r}")
    if maneuver_rows is None or maneuver_weight_config is None:
        raise ValueError("--subset-strategy maneuver_weighted requires --maneuver-metadata and --maneuver-weights")
    if len(maneuver_rows) != dataset_len:
        raise ValueError(
            f"Maneuver metadata has {len(maneuver_rows)} rows, but dataset has {dataset_len}; "
            "cannot build maneuver-weighted subset."
        )
    probabilities = []
    for row in maneuver_rows:
        label = row.get("maneuver_label") or row.get("category")
        if label not in maneuver_weight_config:
            raise ValueError(f"Unknown maneuver label in metadata: {label!r}")
        probabilities.append(float(maneuver_weight_config[label]))
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return sorted(int(index) for index in rng.choice(dataset_len, size=max_samples, replace=False, p=probabilities))


def apply_subset(dataset, indices: list[int] | None):
    return dataset if indices is None else Subset(dataset, indices)


def rows_for_indices(rows: list[dict] | None, indices: list[int] | None) -> list[dict] | None:
    if rows is None:
        return None
    if indices is None:
        return rows
    return [rows[index] for index in indices]


def row_group_counts(rows: list[dict], key: str) -> dict[str, int]:
    counts = Counter(str(row.get(key) or "") for row in rows)
    return dict(sorted((label, int(count)) for label, count in counts.items()))


def row_town_run_counts(rows: list[dict]) -> dict[str, int]:
    counts = Counter(f"{row.get('town') or ''}/{row.get('run_id') or row.get('run_key') or ''}" for row in rows)
    return dict(sorted((label, int(count)) for label, count in counts.items()))


def write_maneuver_sample_indices(path: Path, rows: list[dict], weights=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_index",
        "dataset_idx",
        "town",
        "run_key",
        "run_id",
        "start_idx",
        "end_idx",
        "maneuver_label",
        "longitudinal_label",
        "maneuver_speed_label",
        "delta_speed",
        "sampling_weight",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample_index, row in enumerate(rows):
            writer.writerow(
                {
                    "sample_index": sample_index,
                    "dataset_idx": row.get("dataset_idx"),
                    "town": row.get("town"),
                    "run_key": row.get("run_key"),
                    "run_id": row.get("run_id"),
                    "start_idx": row.get("start_idx"),
                    "end_idx": row.get("end_idx"),
                    "maneuver_label": row.get("maneuver_label") or row.get("category"),
                    "longitudinal_label": row.get("longitudinal_label"),
                    "maneuver_speed_label": row.get("maneuver_speed_label") or row.get("combined_label"),
                    "delta_speed": row.get("delta_speed"),
                    "sampling_weight": float(weights[sample_index]) if weights is not None else "",
                }
            )


def first_row_value(rows: list[dict], key: str, default=None):
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def save_maneuver_sampling_artifacts(
    trainer: TeleopWMTrainer,
    args: argparse.Namespace,
    full_train_rows: list[dict],
    train_rows: list[dict],
    train_weights,
    full_val_rows: list[dict],
    val_rows: list[dict],
    maneuver_weight_config: dict[str, float],
    sampling_diagnostics: dict | None,
    data_diagnostics: dict | None = None,
    full_train_maneuver_speed_rows: list[dict] | None = None,
    train_maneuver_speed_rows: list[dict] | None = None,
    train_maneuver_speed_weights=None,
    full_val_maneuver_speed_rows: list[dict] | None = None,
    val_maneuver_speed_rows: list[dict] | None = None,
    maneuver_speed_weight_config: dict[str, float] | None = None,
) -> None:
    sampling_dir = trainer.run_dir / "sampling"
    sampling_dir.mkdir(parents=True, exist_ok=True)
    effective_train_samples = (
        int(data_diagnostics.get("effective_train_samples"))
        if data_diagnostics and data_diagnostics.get("effective_train_samples") is not None
        else len(train_rows)
    )
    effective_val_samples = (
        int(data_diagnostics.get("effective_val_samples"))
        if data_diagnostics and data_diagnostics.get("effective_val_samples") is not None
        else len(val_rows)
    )

    plot_path = sampling_dir / "sampling_distribution.png"
    full_percentages = maneuver_percentages(full_train_rows)
    selected_percentages = maneuver_percentages(train_rows)
    combined_sampling_diagnostics = (
        sampling_diagnostics if args.sampling_strategy == "maneuver_speed_balanced" else None
    )
    if sampling_diagnostics is None or args.sampling_strategy == "maneuver_speed_balanced":
        sampling_diagnostics = {
            "sampled_counts": maneuver_counts(train_rows),
            "sampled_percentages": selected_percentages,
            "average_sampled_weight": None,
            "note": (
                "No three-class weighted DataLoader sampler was applied; epoch sampling follows "
                "normal shuffled batches or the separate 3x3 maneuver-speed sampler."
            ),
        }
    dataloader_percentages = sampling_diagnostics["sampled_percentages"]
    plot_title = (
        "Subset + Weighted Sampler Maneuver Distribution"
        if args.sampling_strategy == "maneuver_balanced"
        else "Subset Maneuver Distribution"
    )
    save_maneuver_sampling_plot(
        plot_path,
        full_percentages,
        dataloader_percentages,
        selected_percentages=selected_percentages,
        title=plot_title,
    )
    combined_plot_path = None
    if full_train_maneuver_speed_rows is not None and train_maneuver_speed_rows is not None:
        combined_plot_path = sampling_dir / "sampling_distribution_combined.png"
        combined_full_percentages = maneuver_speed_percentages(full_train_maneuver_speed_rows)
        combined_selected_percentages = maneuver_speed_percentages(train_maneuver_speed_rows)
        if args.sampling_strategy == "maneuver_speed_balanced" and combined_sampling_diagnostics is not None:
            combined_sampled_percentages = combined_sampling_diagnostics["sampled_percentages"]
        else:
            combined_sampled_percentages = combined_selected_percentages
        combined_plot_title = (
            "Subset + Weighted Sampler Maneuver-Speed Distribution"
            if args.sampling_strategy == "maneuver_speed_balanced"
            else "Subset Maneuver-Speed Distribution"
        )
        save_maneuver_speed_sampling_plot(
            combined_plot_path,
            combined_full_percentages,
            combined_sampled_percentages,
            selected_percentages=combined_selected_percentages,
            title=combined_plot_title,
        )

    train_csv = sampling_dir / "train_sample_indices.csv"
    val_csv = sampling_dir / "val_sample_indices.csv"
    csv_train_rows = train_maneuver_speed_rows if train_maneuver_speed_rows is not None else train_rows
    csv_val_rows = val_maneuver_speed_rows if val_maneuver_speed_rows is not None else val_rows
    csv_train_weights = train_maneuver_speed_weights if train_maneuver_speed_weights is not None else train_weights
    write_maneuver_sample_indices(train_csv, csv_train_rows, csv_train_weights)
    write_maneuver_sample_indices(val_csv, csv_val_rows, None)

    subset_selection = {
        "strategy": args.subset_strategy,
        "val_strategy": args.val_subset_strategy,
        "uses_maneuver_weights": args.subset_strategy == "maneuver_weighted",
        "maneuver_weights": maneuver_weight_config if args.subset_strategy == "maneuver_weighted" else None,
        "requested_max_train_samples": args.max_train_samples,
        "requested_max_val_samples": args.max_val_samples,
        "full_train_samples": len(full_train_rows),
        "full_val_samples": len(full_val_rows),
        "selected_train_samples": effective_train_samples,
        "selected_val_samples": effective_val_samples,
        "selected_train_metadata_rows": len(train_rows),
        "selected_val_metadata_rows": len(val_rows),
        "validation_metadata_available": bool(val_rows),
        "full_train_maneuver_counts": maneuver_counts(full_train_rows),
        "full_train_maneuver_percentages": full_percentages,
        "selected_train_maneuver_counts": maneuver_counts(train_rows),
        "selected_train_maneuver_percentages": selected_percentages,
        "selected_train_town_counts": row_group_counts(train_rows, "town"),
        "selected_train_run_counts": row_town_run_counts(train_rows),
        "selected_val_maneuver_counts": maneuver_counts(val_rows),
        "selected_val_maneuver_percentages": maneuver_percentages(val_rows),
        "selected_val_town_counts": row_group_counts(val_rows, "town"),
        "selected_val_run_counts": row_town_run_counts(val_rows),
        "note": (
            "Subset selection happens before DataLoader creation. maneuver_weighted uses maneuver "
            "weights to create the candidate subset; it is not the same as DataLoader weighted sampling."
        ),
    }
    combined_subset_selection = None
    combined_dataloader_sampling = None
    if full_train_maneuver_speed_rows is not None and train_maneuver_speed_rows is not None:
        combined_sampled_counts = maneuver_speed_counts(train_maneuver_speed_rows)
        combined_sampled_percentages = maneuver_speed_percentages(train_maneuver_speed_rows)
        if args.sampling_strategy == "maneuver_speed_balanced" and combined_sampling_diagnostics is not None:
            combined_sampled_counts = combined_sampling_diagnostics["sampled_counts"]
            combined_sampled_percentages = combined_sampling_diagnostics["sampled_percentages"]
        combined_subset_selection = {
            "full_train_combined_counts": maneuver_speed_counts(full_train_maneuver_speed_rows),
            "full_train_combined_percentages": maneuver_speed_percentages(full_train_maneuver_speed_rows),
            "selected_train_combined_counts": maneuver_speed_counts(train_maneuver_speed_rows),
            "selected_train_combined_percentages": maneuver_speed_percentages(train_maneuver_speed_rows),
            "selected_val_combined_counts": maneuver_speed_counts(val_maneuver_speed_rows or []),
            "selected_val_combined_percentages": maneuver_speed_percentages(val_maneuver_speed_rows or []),
            "speed_delta_threshold": args.speed_delta_threshold,
            "speed_delta_definition": first_row_value(
                full_train_maneuver_speed_rows,
                "speed_delta_definition",
                "mean_future_speed_minus_mean_past_speed",
            ),
            "note": "Combined labels cross lateral maneuver class with rollout-level speed evolution.",
        }
        combined_dataloader_sampling = {
            "strategy": args.sampling_strategy,
            "weighted_sampler_enabled": args.sampling_strategy == "maneuver_speed_balanced",
            "maneuver_speed_weights": (
                maneuver_speed_weight_config
                if args.sampling_strategy == "maneuver_speed_balanced"
                else None
            ),
            "sampled_combined_counts": combined_sampled_counts,
            "sampled_combined_percentages": combined_sampled_percentages,
            "average_sampled_weight": (
                combined_sampling_diagnostics.get("average_sampled_weight")
                if args.sampling_strategy == "maneuver_speed_balanced" and combined_sampling_diagnostics
                else None
            ),
            "note": (
                "WeightedRandomSampler draws from the selected subset by 3x3 maneuver-speed labels."
                if args.sampling_strategy == "maneuver_speed_balanced"
                else "No 3x3 weighted DataLoader sampler applied."
            ),
        }
    dataloader_sampling = {
        "strategy": args.sampling_strategy,
        "weighted_sampler_enabled": args.sampling_strategy in {"maneuver_balanced", "maneuver_speed_balanced"},
        "maneuver_weights": maneuver_weight_config if args.sampling_strategy == "maneuver_balanced" else None,
        "sampled_maneuver_counts": sampling_diagnostics["sampled_counts"],
        "sampled_maneuver_percentages": dataloader_percentages,
        "average_sampled_weight": sampling_diagnostics["average_sampled_weight"],
        "note": (
            "WeightedRandomSampler simulates/draws from the selected subset during training epochs."
            if args.sampling_strategy == "maneuver_balanced"
            else (
                "WeightedRandomSampler is active for 3x3 maneuver-speed labels; see maneuver_speed_dataloader_sampling."
                if args.sampling_strategy == "maneuver_speed_balanced"
                else "No weighted DataLoader sampler applied; training uses normal shuffled batches from the selected subset."
            )
        ),
    }
    summary = {
        "sampling_strategy": args.sampling_strategy,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "run_tag": args.run_tag,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(args.data_root),
        "val_data_root": str(args.val_data_root) if args.val_data_root is not None else None,
        "test_data_root": str(args.test_data_root) if args.test_data_root is not None else None,
        "maneuver_metadata": str(args.maneuver_metadata),
        "val_maneuver_metadata": str(args.val_maneuver_metadata) if args.val_maneuver_metadata is not None else None,
        "maneuver_weights": maneuver_weight_config,
        "maneuver_speed_metadata": str(
            args.maneuver_speed_metadata
            or (args.maneuver_metadata if args.sampling_strategy == "maneuver_speed_balanced" else "")
        )
        or None,
        "val_maneuver_speed_metadata": str(args.val_maneuver_speed_metadata) if args.val_maneuver_speed_metadata is not None else None,
        "maneuver_speed_weights": maneuver_speed_weight_config,
        "speed_delta_threshold": args.speed_delta_threshold,
        "subset_strategy": args.subset_strategy,
        "val_subset_strategy": args.val_subset_strategy,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "validation": data_diagnostics or {},
        "subset_selection": subset_selection,
        "dataloader_sampling": dataloader_sampling,
        "maneuver_speed_subset_selection": combined_subset_selection,
        "maneuver_speed_dataloader_sampling": combined_dataloader_sampling,
        "full_train_samples": len(full_train_rows),
        "full_val_samples": len(full_val_rows),
        "effective_train_samples": effective_train_samples,
        "effective_val_samples": effective_val_samples,
        "selected_train_metadata_rows": len(train_rows),
        "selected_val_metadata_rows": len(val_rows),
        "full_original_maneuver_counts": maneuver_counts(full_train_rows),
        "full_original_maneuver_percentages": full_percentages,
        "selected_subset_maneuver_counts": maneuver_counts(train_rows),
        "selected_subset_maneuver_percentages": selected_percentages,
        "selected_subset_town_counts": row_group_counts(train_rows, "town"),
        "selected_subset_run_counts": row_town_run_counts(train_rows),
        "validation_subset_maneuver_counts": maneuver_counts(val_rows),
        "validation_subset_maneuver_percentages": maneuver_percentages(val_rows),
        "validation_subset_town_counts": row_group_counts(val_rows, "town"),
        "validation_subset_run_counts": row_town_run_counts(val_rows),
        "sampled_maneuver_counts": sampling_diagnostics["sampled_counts"],
        "sampled_maneuver_percentages": dataloader_percentages,
        "average_sampled_weight": sampling_diagnostics["average_sampled_weight"],
        "train_sample_indices_csv": str(train_csv),
        "val_sample_indices_csv": str(val_csv),
        "sampling_distribution_png": str(plot_path),
        "sampling_distribution_combined_png": str(combined_plot_path) if combined_plot_path is not None else None,
        "note": (
            "full_original_* is the complete aligned metadata distribution before subsetting. "
            "selected_subset_* is the candidate dataset after max_*_samples and subset_strategy. "
            "sampled_* simulates one WeightedRandomSampler epoch over the selected train subset. "
            "CSV files record selected train/validation rollout windows; sampler draws are generated "
            "by PyTorch during training."
        ),
    }
    summary_path = sampling_dir / "sampling_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    trainer.log("=" * 50)
    trainer.log("Subset Selection Stage")
    trainer.log("=" * 50)
    if data_diagnostics:
        trainer.log(f"train data root: {data_diagnostics.get('train_data_root')}")
        trainer.log(f"val data root: {data_diagnostics.get('val_data_root')}")
        trainer.log(f"validation mode: {data_diagnostics.get('validation_mode')}")
        trainer.log(f"train towns: {', '.join(data_diagnostics.get('train_towns') or [])}")
        trainer.log(f"val towns: {', '.join(data_diagnostics.get('val_towns') or [])}")
        trainer.log(f"selected train towns: {', '.join(data_diagnostics.get('selected_train_towns') or [])}")
        trainer.log(f"selected val towns: {', '.join(data_diagnostics.get('selected_val_towns') or [])}")
    trainer.log(f"subset strategy: {args.subset_strategy}")
    trainer.log(f"validation subset strategy: {args.val_subset_strategy}")
    trainer.log(f"maneuver metadata: {args.maneuver_metadata}")
    if subset_selection["uses_maneuver_weights"]:
        trainer.log(f"subset maneuver weights: {maneuver_weight_config}")
        trainer.log("subset selection uses maneuver weighting BEFORE DataLoader creation.")
    else:
        trainer.log("subset selection does not use maneuver weights.")
    trainer.log(f"full dataset percentages: {summary['full_original_maneuver_percentages']}")
    trainer.log(f"selected subset percentages: {summary['selected_subset_maneuver_percentages']}")
    trainer.log(f"selected subset counts: {summary['selected_subset_maneuver_counts']}")
    trainer.log(f"effective train/val samples: {effective_train_samples} / {effective_val_samples}")
    if not val_rows:
        trainer.log("validation maneuver metadata unavailable; val_sample_indices.csv contains only rows when --val-maneuver-metadata is provided.")
    trainer.log("=" * 50)
    trainer.log("Dataloader Sampling Stage")
    trainer.log("=" * 50)
    trainer.log(f"sampling strategy: {args.sampling_strategy}")
    if args.sampling_strategy == "maneuver_balanced":
        trainer.log("weighted dataloader sampler ENABLED.")
        trainer.log(f"dataloader maneuver weights: {maneuver_weight_config}")
        trainer.log(f"sampled maneuver counts: {summary['sampled_maneuver_counts']}")
        trainer.log(f"sampled maneuver percentages: {summary['sampled_maneuver_percentages']}")
    elif args.sampling_strategy == "maneuver_speed_balanced":
        trainer.log("weighted dataloader sampler ENABLED for 3x3 maneuver-speed labels.")
        trainer.log("Three-class maneuver percentages above are shown for compatibility; combined diagnostics follow below.")
    else:
        trainer.log("No weighted dataloader sampler applied.")
        trainer.log("Training uses normal shuffled batches from the selected subset.")
    trainer.log(f"effective train/val samples: {effective_train_samples} / {effective_val_samples}")
    trainer.log(f"saved sampling summary: {summary_path}")
    trainer.log(f"saved train sample indices: {train_csv}")
    trainer.log(f"saved val sample indices: {val_csv}")
    trainer.log(f"saved maneuver sampling distribution plot: {plot_path}")
    if combined_subset_selection is not None:
        trainer.log("=" * 50)
        trainer.log("Combined Maneuver-Speed Diagnostics")
        trainer.log("=" * 50)
        trainer.log(f"speed delta definition: {combined_subset_selection['speed_delta_definition']}")
        trainer.log(f"speed delta threshold: {combined_subset_selection['speed_delta_threshold']}")
        trainer.log(
            f"full combined percentages: {combined_subset_selection['full_train_combined_percentages']}"
        )
        trainer.log(
            f"selected combined percentages: {combined_subset_selection['selected_train_combined_percentages']}"
        )
        if combined_dataloader_sampling and args.sampling_strategy == "maneuver_speed_balanced":
            trainer.log(f"dataloader combined weights: {maneuver_speed_weight_config}")
            trainer.log(f"sampled combined counts: {combined_dataloader_sampling['sampled_combined_counts']}")
            trainer.log(f"sampled combined percentages: {combined_dataloader_sampling['sampled_combined_percentages']}")
        if combined_plot_path is not None:
            trainer.log(f"saved maneuver-speed sampling distribution plot: {combined_plot_path}")


def apply_resume_config(args: argparse.Namespace) -> None:
    if args.resume is None:
        return

    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    run_cfg = checkpoint.get("config", {}).get("run", {})
    explicit_flags = set()
    for item in sys.argv[1:]:
        if item.startswith("--"):
            explicit_flags.add(item.split("=")[0])

    inherited = {
        "data_root": ("--data-root", Path),
        "val_data_root": ("--val-data-root", Path),
        "test_data_root": ("--test-data-root", Path),
        "val_fraction": ("--val-fraction", float),
        "split_seed": ("--split-seed", int),
        "height": ("--height", int),
        "width": ("--width", int),
        "hid_s": ("--hid-s", int),
        "hid_t": ("--hid-t", int),
        "n_s": ("--n-s", int),
        "n_t": ("--n-t", int),
        "model_type": ("--model-type", str),
        "model_variant": ("--model-variant", str),
        "action_dim": ("--action-dim", int),
        "speed_dim": ("--speed-dim", int),
        "conditioning_dim": ("--conditioning-dim", int),
        "simvp_conditioning": ("--simvp-conditioning", str),
        "simvp_conditioning_stage": ("--simvp-conditioning-stage", str),
        "conditioning_fusion": ("--conditioning-fusion", str),
        "conditioning_injection": ("--conditioning-injection", str),
        "wm_latent_residual": ("--wm-latent-residual", bool),
        "wm_residual_hidden_dim": ("--wm-residual-hidden-dim", int),
        "wm_residual_scale": ("--wm-residual-scale", float),
        "wm_residual_gated": ("--wm-residual-gated", bool),
        "wm_residual_loss": ("--wm-residual-loss", bool),
        "wm_residual_loss_weight": ("--wm-residual-loss-weight", float),
        "wm_residual_loss_type": ("--wm-residual-loss-type", str),
        "ssim_loss_weight": ("--ssim-loss-weight", float),
        "dual_fusion": ("--dual-fusion", str),
        "dual_wm_scale": ("--dual-wm-scale", float),
        "dual_wm_hidden_dim": ("--dual-wm-hidden-dim", int),
        "dual_wm_num_layers": ("--dual-wm-num-layers", int),
        "dual_wm_conditioning": ("--dual-wm-conditioning", str),
        "dual_wm_gated": ("--dual-wm-gated", bool),
        "dual_wm_image_loss_weight": ("--dual-wm-image-loss-weight", float),
        "dual_simvp_image_loss_weight": ("--dual-simvp-image-loss-weight", float),
        "dual_align_loss_weight": ("--dual-align-loss-weight", float),
        "dual_align_loss_type": ("--dual-align-loss-type", str),
        "dual_align_direction": ("--dual-align-direction", str),
        "dual_detach_simvp_target": ("--dual-detach-simvp-target", bool),
        "normalize_controls": ("--normalize-controls", bool),
        "speed_scale": ("--speed-scale", float),
        "drop_path": ("--drop-path", float),
        "sampling_strategy": ("--sampling-strategy", str),
        "balanced_sampling": ("--balanced-sampling", bool),
        "subset_strategy": ("--subset-strategy", str),
        "val_subset_strategy": ("--val-subset-strategy", str),
        "weights_path": ("--weights-path", Path),
        "maneuver_metadata": ("--maneuver-metadata", Path),
        "val_maneuver_metadata": ("--val-maneuver-metadata", Path),
        "maneuver_weights": ("--maneuver-weights", str),
        "maneuver_speed_metadata": ("--maneuver-speed-metadata", Path),
        "val_maneuver_speed_metadata": ("--val-maneuver-speed-metadata", Path),
        "maneuver_speed_weights": ("--maneuver-speed-weights", str),
        "speed_delta_threshold": ("--speed-delta-threshold", float),
        "aux_dynamics_loss": ("--aux-dynamics-loss", bool),
        "aux_dynamics_weight": ("--aux-dynamics-weight", float),
        "aux_dynamics_hidden_dim": ("--aux-dynamics-hidden-dim", int),
        "aux_dynamics_loss_type": ("--aux-dynamics-loss-type", str),
        "future_action_loss": ("--future-action-loss", bool),
        "future_action_loss_weight": ("--future-action-loss-weight", float),
        "future_action_loss_type": ("--future-action-loss-type", str),
        "future_action_hidden_dim": ("--future-action-hidden-dim", int),
        "future_action_num_layers": ("--future-action-num-layers", int),
        "future_action_dropout": ("--future-action-dropout", float),
        "future_action_source": ("--future-action-source", str),
        "future_steer_target_scale": ("--future-steer-target-scale", float),
        "control_steer_input_scale": ("--control-steer-input-scale", float),
        "future_action_head_variant": ("--future-action-head-variant", str),
        "future_action_detach_latents": ("--future-action-detach-latents", bool),
        "future_action_future_motion_scale": ("--future-action-future-motion-scale", float),
        "future_action_spatial_pooling": ("--future-action-spatial-pooling", str),
        "future_action_spatial_grid": ("--future-action-spatial-grid", str),
        "future_action_corr_loss_weight": ("--future-action-corr-loss-weight", float),
        "future_action_delta_loss": ("--future-action-delta-loss", bool),
        "future_action_delta_loss_weight": ("--future-action-delta-loss-weight", float),
        "future_action_delta_loss_type": ("--future-action-delta-loss-type", str),
        "future_action_delta_longitudinal_weight": ("--future-action-delta-longitudinal-weight", float),
        "future_action_delta_steer_weight": ("--future-action-delta-steer-weight", float),
        "future_action_cls_loss": ("--future-action-cls-loss", bool),
        "future_action_cls_weight": ("--future-action-cls-weight", float),
        "future_action_longitudinal_cls_weight": ("--future-action-longitudinal-cls-weight", float),
        "future_action_steer_cls_weight": ("--future-action-steer-cls-weight", float),
        "longitudinal_coast_threshold": ("--longitudinal-coast-threshold", float),
        "steer_straight_threshold": ("--steer-straight-threshold", float),
        "max_train_steps": ("--max-train-steps", int),
        "max_val_batches": ("--max-val-batches", int),
        "max_interim_val_batches": ("--max-interim-val-batches", int),
        "eval_every_steps": ("--eval-every-steps", int),
        "stop_on_nan": ("--stop-on-nan", bool),
        "grad_clip_norm": ("--grad-clip-norm", float),
        "log_grad_norms": ("--log-grad-norms", bool),
        "debug_activation_stats": ("--debug-activation-stats", bool),
        "debug_dual_gate": ("--debug-dual-gate", bool),
    }
    for attr, (flag, caster) in inherited.items():
        if flag in explicit_flags or f"--no-{flag[2:]}" in explicit_flags or attr not in run_cfg:
            continue
        if run_cfg[attr] is None:
            setattr(args, attr, None)
        else:
            setattr(args, attr, caster(run_cfg[attr]))
    if args.resume is not None and "conditioning_fusion" not in run_cfg and getattr(args, "model_variant", "rgb") == "av":
        args.conditioning_fusion = "add"
    if args.resume is not None and "conditioning_injection" not in run_cfg and getattr(args, "model_variant", "rgb") == "av":
        args.conditioning_injection = "single"


def normalize_model_args(args: argparse.Namespace) -> None:
    explicit_flags = {
        item.split("=")[0]
        for item in sys.argv[1:]
        if item.startswith("--")
    }
    explicit_sampling_strategy = any(
        item == "--sampling-strategy" or item.startswith("--sampling-strategy=")
        for item in sys.argv[1:]
    )
    if args.balanced_sampling:
        if explicit_sampling_strategy and args.sampling_strategy != "action_balanced":
            raise ValueError("--balanced-sampling is an alias for --sampling-strategy action_balanced; do not combine it with another strategy.")
        args.sampling_strategy = "action_balanced"
    args.balanced_sampling = args.sampling_strategy == "action_balanced"

    if args.model_variant == "av":
        args.model_variant = "av_simvp"
    if args.model_variant == "av_wm_dual_bigwm":
        if "--dual-wm-hidden-dim" not in explicit_flags:
            args.dual_wm_hidden_dim = 512
        if "--dual-wm-num-layers" not in explicit_flags:
            args.dual_wm_num_layers = 3
    if args.simvp_conditioning is None:
        args.simvp_conditioning = args.conditioning_fusion if args.conditioning_fusion is not None else "concat"
    if args.simvp_conditioning_stage is None:
        if args.conditioning_injection == "multipoint":
            args.simvp_conditioning_stage = "multipoint"
        else:
            args.simvp_conditioning_stage = "input"
    if args.conditioning_fusion is None:
        args.conditioning_fusion = None if args.simvp_conditioning == "none" else args.simvp_conditioning
    if args.conditioning_injection is None:
        args.conditioning_injection = "multipoint" if args.simvp_conditioning_stage == "multipoint" else "single"
    if args.model_variant == "av_wm" and not args.wm_latent_residual:
        raise ValueError("--model-variant av_wm currently requires --wm-latent-residual")


def main() -> int:
    if any(item in {"-h", "--help"} for item in sys.argv[1:]):
        print_public_help()
        return 0
    sys.argv = [sys.argv[0], *apply_release_defaults(sys.argv[1:])]
    args = parse_args()
    apply_resume_config(args)
    normalize_model_args(args)
    if args.aux_dynamics_loss and args.model_variant not in {"av_simvp", "av_wm", "av_wm_dual", "av_wm_dual_bigwm"}:
        raise ValueError("--aux-dynamics-loss is only valid with AV model variants")
    if args.future_action_loss and args.model_variant not in {"av_wm_dual", "av_wm_dual_bigwm"}:
        raise ValueError("--future-action-loss is only valid with --model-variant av_wm_dual or av_wm_dual_bigwm")
    if args.future_action_loss_weight < 0:
        raise ValueError("--future-action-loss-weight must be nonnegative")
    if args.future_action_future_motion_scale <= 0:
        raise ValueError("--future-action-future-motion-scale must be positive")
    if args.future_action_corr_loss_weight < 0:
        raise ValueError("--future-action-corr-loss-weight must be nonnegative")
    if args.future_action_delta_loss and not args.future_action_loss:
        raise ValueError("--future-action-delta-loss requires --future-action-loss")
    if args.future_action_delta_loss_weight < 0:
        raise ValueError("--future-action-delta-loss-weight must be nonnegative")
    if args.future_action_delta_longitudinal_weight < 0 or args.future_action_delta_steer_weight < 0:
        raise ValueError("future action delta channel weights must be nonnegative")
    if args.future_steer_target_scale <= 0:
        raise ValueError("--future-steer-target-scale must be positive")
    if args.control_steer_input_scale <= 0:
        raise ValueError("--control-steer-input-scale must be positive")
    if args.future_action_cls_loss and not args.future_action_loss:
        raise ValueError("--future-action-cls-loss requires --future-action-loss")
    if args.future_action_cls_weight < 0:
        raise ValueError("--future-action-cls-weight must be nonnegative")
    if args.future_action_longitudinal_cls_weight < 0 or args.future_action_steer_cls_weight < 0:
        raise ValueError("future action classification channel weights must be nonnegative")
    if args.longitudinal_coast_threshold < 0 or args.steer_straight_threshold < 0:
        raise ValueError("future action classification thresholds must be nonnegative")
    if args.wm_residual_loss and args.model_variant != "av_wm":
        raise ValueError("--wm-residual-loss is only valid with --model-variant av_wm, not av_wm_dual")
    if args.ssim_loss_weight < 0:
        raise ValueError("--ssim-loss-weight must be nonnegative")
    if args.grad_clip_norm is not None and args.grad_clip_norm <= 0:
        raise ValueError("--grad-clip-norm must be positive when provided")
    if args.max_train_steps is not None and args.max_train_steps < 1:
        raise ValueError("--max-train-steps must be positive when provided")
    if args.max_val_batches is not None and args.max_val_batches < 1:
        raise ValueError("--max-val-batches must be positive when provided")
    if args.max_interim_val_batches is not None and args.max_interim_val_batches < 1:
        raise ValueError("--max-interim-val-batches must be positive when provided")
    if args.eval_every_steps is not None and args.eval_every_steps < 1:
        raise ValueError("--eval-every-steps must be positive when provided")
    for name in ("dual_wm_image_loss_weight", "dual_simvp_image_loss_weight", "dual_align_loss_weight"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")
    if (
        args.model_variant not in {"av_wm_dual", "av_wm_dual_bigwm"}
        and (args.dual_wm_image_loss_weight > 0 or args.dual_simvp_image_loss_weight > 0 or args.dual_align_loss_weight > 0)
    ):
        raise ValueError("Dual losses require --model-variant av_wm_dual or av_wm_dual_bigwm")
    if args.split_seed is None:
        args.split_seed = args.seed
    seed_everything(args.seed)

    image_size = (args.height, args.width)
    if args.val_data_root is not None:
        validation_mode = "explicit_val_root"
        full_train_dataset = CarlaRolloutDataset(
            args.data_root,
            split="train",
            image_size=image_size,
            normalize_controls=args.normalize_controls,
            speed_scale=args.speed_scale,
        )
        full_val_dataset = CarlaRolloutDataset(
            args.val_data_root,
            split="val",
            image_size=image_size,
            normalize_controls=args.normalize_controls,
            speed_scale=args.speed_scale,
        )
        internal_train_indices = None
        internal_val_indices = None
    else:
        validation_mode = "internal_split"
        base_dataset = CarlaRolloutDataset(
            args.data_root,
            split="train",
            image_size=image_size,
            normalize_controls=args.normalize_controls,
            speed_scale=args.speed_scale,
        )
        internal_train_indices, internal_val_indices = split_dataset_indices(
            len(base_dataset),
            args.val_fraction,
            args.split_seed,
        )
        full_train_dataset = Subset(base_dataset, internal_train_indices)
        full_val_dataset = Subset(base_dataset, internal_val_indices)

    maneuver_weight_config = parse_maneuver_weights(args.maneuver_weights)
    maneuver_speed_weight_config = parse_maneuver_speed_weights(args.maneuver_speed_weights)
    train_maneuver_rows_all = None
    val_maneuver_rows_all = None
    train_maneuver_speed_rows_all = None
    val_maneuver_speed_rows_all = None
    if args.maneuver_metadata is not None:
        train_maneuver_rows_all = load_maneuver_metadata(args.maneuver_metadata)
    if args.val_maneuver_metadata is not None:
        val_maneuver_rows_all = load_maneuver_metadata(args.val_maneuver_metadata)
    train_maneuver_speed_metadata_path = args.maneuver_speed_metadata
    if train_maneuver_speed_metadata_path is None and args.sampling_strategy == "maneuver_speed_balanced":
        train_maneuver_speed_metadata_path = args.maneuver_metadata
    val_maneuver_speed_metadata_path = args.val_maneuver_speed_metadata
    if train_maneuver_speed_metadata_path is not None:
        train_maneuver_speed_rows_all = load_maneuver_metadata(train_maneuver_speed_metadata_path)
    if val_maneuver_speed_metadata_path is not None:
        val_maneuver_speed_rows_all = load_maneuver_metadata(val_maneuver_speed_metadata_path)

    needs_train_maneuver_rows = (
        args.sampling_strategy == "maneuver_balanced"
        or args.subset_strategy == "maneuver_weighted"
    )
    needs_train_maneuver_speed_rows = args.sampling_strategy == "maneuver_speed_balanced"
    needs_val_maneuver_rows = args.val_subset_strategy == "maneuver_weighted"
    if needs_train_maneuver_rows and train_maneuver_rows_all is None:
        raise ValueError("maneuver-balanced sampling or train maneuver_weighted subsetting requires --maneuver-metadata")
    if needs_train_maneuver_speed_rows and train_maneuver_speed_rows_all is None:
        raise ValueError(
            "--sampling-strategy maneuver_speed_balanced requires --maneuver-speed-metadata "
            "or a --maneuver-metadata file generated by scripts/build_maneuver_metadata.py with combined labels."
        )
    if needs_val_maneuver_rows:
        if args.val_data_root is not None and val_maneuver_rows_all is None:
            raise ValueError(
                "--val-subset-strategy maneuver_weighted with --val-data-root requires "
                "--val-maneuver-metadata generated for the validation root."
            )
        if args.val_data_root is None and train_maneuver_rows_all is None:
            raise ValueError("--val-subset-strategy maneuver_weighted requires --maneuver-metadata")

    full_train_maneuver_rows = None
    full_val_maneuver_rows = None
    full_train_maneuver_speed_rows = None
    full_val_maneuver_speed_rows = None
    if args.val_data_root is None and train_maneuver_rows_all is not None:
        base_maneuver_rows = align_maneuver_rows_to_dataset(train_maneuver_rows_all, base_dataset)
        full_train_maneuver_rows = rows_for_indices(base_maneuver_rows, internal_train_indices)
        full_val_maneuver_rows = rows_for_indices(base_maneuver_rows, internal_val_indices)
    if args.val_data_root is None and train_maneuver_speed_rows_all is not None:
        base_maneuver_speed_rows = align_maneuver_rows_to_dataset(train_maneuver_speed_rows_all, base_dataset)
        full_train_maneuver_speed_rows = rows_for_indices(base_maneuver_speed_rows, internal_train_indices)
        full_val_maneuver_speed_rows = rows_for_indices(base_maneuver_speed_rows, internal_val_indices)
    else:
        if train_maneuver_rows_all is not None:
            full_train_maneuver_rows = align_maneuver_rows_to_dataset(train_maneuver_rows_all, full_train_dataset)
        if val_maneuver_rows_all is not None:
            full_val_maneuver_rows = align_maneuver_rows_to_dataset(val_maneuver_rows_all, full_val_dataset)
        if train_maneuver_speed_rows_all is not None:
            full_train_maneuver_speed_rows = align_maneuver_rows_to_dataset(train_maneuver_speed_rows_all, full_train_dataset)
        if val_maneuver_speed_rows_all is not None:
            full_val_maneuver_speed_rows = align_maneuver_rows_to_dataset(val_maneuver_speed_rows_all, full_val_dataset)

    train_indices = select_subset_indices(
        len(full_train_dataset),
        args.max_train_samples,
        args.subset_strategy,
        args.seed,
        maneuver_rows=full_train_maneuver_rows,
        maneuver_weight_config=maneuver_weight_config,
    )
    val_indices = select_subset_indices(
        len(full_val_dataset),
        args.max_val_samples,
        args.val_subset_strategy,
        args.seed + 1,
        maneuver_rows=full_val_maneuver_rows,
        maneuver_weight_config=maneuver_weight_config,
    )
    train_dataset = apply_subset(full_train_dataset, train_indices)
    val_dataset = apply_subset(full_val_dataset, val_indices)
    selected_train_maneuver_rows = rows_for_indices(full_train_maneuver_rows, train_indices)
    selected_val_maneuver_rows = rows_for_indices(full_val_maneuver_rows, val_indices)
    selected_train_maneuver_speed_rows = rows_for_indices(full_train_maneuver_speed_rows, train_indices)
    selected_val_maneuver_speed_rows = rows_for_indices(full_val_maneuver_speed_rows, val_indices)
    full_train_towns = dataset_towns(full_train_dataset)
    full_val_towns = dataset_towns(full_val_dataset)
    selected_train_towns = dataset_towns(train_dataset)
    selected_val_towns = dataset_towns(val_dataset)
    data_diagnostics = {
        "train_data_root": str(args.data_root),
        "val_data_root": str(args.val_data_root if args.val_data_root is not None else args.data_root),
        "test_data_root": str(args.test_data_root) if args.test_data_root is not None else None,
        "validation_mode": validation_mode,
        "val_fraction": args.val_fraction if validation_mode == "internal_split" else None,
        "split_seed": args.split_seed,
        "train_towns": full_train_towns,
        "val_towns": full_val_towns,
        "selected_train_towns": selected_train_towns,
        "selected_val_towns": selected_val_towns,
        "full_train_samples": len(full_train_dataset),
        "full_val_samples": len(full_val_dataset),
        "effective_train_samples": len(train_dataset),
        "effective_val_samples": len(val_dataset),
        "train_run_counts": dataset_run_counts(train_dataset),
        "val_run_counts": dataset_run_counts(val_dataset),
        "selected_train_dataset_indices": dataset_absolute_indices(train_dataset),
        "selected_val_dataset_indices": dataset_absolute_indices(val_dataset),
        "internal_split_train_indices": len(internal_train_indices) if internal_train_indices is not None else None,
        "internal_split_val_indices": len(internal_val_indices) if internal_val_indices is not None else None,
        "internal_split_overlap": (
            len(set(internal_train_indices).intersection(internal_val_indices))
            if internal_train_indices is not None and internal_val_indices is not None
            else None
        ),
    }

    print("=" * 50)
    print("Dataset Roots")
    print("=" * 50)
    print(f"train data root: {data_diagnostics['train_data_root']}")
    print(f"val data root: {data_diagnostics['val_data_root']}")
    print(f"validation mode: {validation_mode}")
    print(f"train towns: {', '.join(full_train_towns) if full_train_towns else '(unknown)'}")
    print(f"val towns: {', '.join(full_val_towns) if full_val_towns else '(unknown)'}")
    print(f"selected train towns: {', '.join(selected_train_towns) if selected_train_towns else '(unknown)'}")
    print(f"selected val towns: {', '.join(selected_val_towns) if selected_val_towns else '(unknown)'}")
    print(f"train samples: {len(train_dataset)}")
    print(f"val samples: {len(val_dataset)}")

    if selected_train_maneuver_rows is not None:
        print("=" * 50)
        print("Subset Selection Stage")
        print("=" * 50)
        print(f"subset strategy: {args.subset_strategy}")
        print(f"validation subset strategy: {args.val_subset_strategy}")
        if args.subset_strategy == "maneuver_weighted":
            print(f"subset maneuver weights: {maneuver_weight_config}")
            print("subset selection uses maneuver weighting BEFORE DataLoader creation.")
        else:
            print("subset selection does not use maneuver weights.")
        print(f"train samples: {len(train_dataset)}/{len(full_train_dataset)}")
        print(f"val samples: {len(val_dataset)}/{len(full_val_dataset)}")
        print(f"full dataset percentages: {maneuver_percentages(full_train_maneuver_rows or [])}")
        print(f"selected subset percentages: {maneuver_percentages(selected_train_maneuver_rows)}")
    else:
        print(
            "subset selection: "
            f"train={args.subset_strategy} {len(train_dataset)}/{len(full_train_dataset)} "
            f"val={args.val_subset_strategy} {len(val_dataset)}/{len(full_val_dataset)}"
        )

    train_sampler = None
    sampling_diagnostics = None
    rollout_weight_rows = None
    rollout_weight_thresholds = None
    rollout_weights = None
    maneuver_rows = None
    maneuver_weights = None
    maneuver_val_rows = None
    maneuver_speed_rows = None
    maneuver_speed_weights = None
    maneuver_speed_val_rows = None
    if args.sampling_strategy == "action_balanced":
        if args.weights_path is not None:
            rollout_weights = load_rollout_weights(args.weights_path)
            if len(rollout_weights) != len(train_dataset):
                absolute_indices = dataset_absolute_indices(train_dataset)
                if absolute_indices and len(rollout_weights) > max(absolute_indices):
                    rollout_weights = rollout_weights[absolute_indices]
                elif isinstance(train_dataset, Subset) and len(rollout_weights) == len(train_dataset.dataset):
                    rollout_weights = rollout_weights[list(train_dataset.indices)]
                else:
                    raise ValueError(
                        f"weights length {len(rollout_weights)} does not match train dataset length {len(train_dataset)}"
                    )
            metadata_path = args.weights_path.with_name("rollout_metadata.csv")
            rollout_weight_rows = load_rollout_metadata(metadata_path) if metadata_path.exists() else None
            rollout_weight_thresholds = {}
        else:
            rollout_weight_rows, rollout_weights, rollout_weight_thresholds = compute_rollout_weight_table(train_dataset)
        train_sampler = make_weighted_sampler(rollout_weights, seed=args.seed)
        if rollout_weight_rows is not None and len(rollout_weight_rows) == len(train_dataset):
            sampling_diagnostics = simulate_weighted_sampling(
                rollout_weight_rows,
                rollout_weights,
                num_samples=len(train_dataset),
                seed=args.seed,
            )
            print("balanced sampling enabled")
            print(f"category counts: {category_counts(rollout_weight_rows)}")
            print(f"category percentages: {category_percentages(rollout_weight_rows)}")
            print(f"sampled percentages: {sampling_diagnostics['sampled_percentages']}")
            print(f"average sampled weight: {sampling_diagnostics['average_sampled_weight']:.4f}")
        else:
            print("balanced sampling enabled")
            print("category diagnostics unavailable because rollout_metadata.csv was not found or did not match dataset length")
    elif args.sampling_strategy == "maneuver_balanced":
        if selected_train_maneuver_rows is None:
            raise ValueError("--sampling-strategy maneuver_balanced requires --maneuver-metadata")
        train_sampler, maneuver_weights, maneuver_rows = make_maneuver_balanced_sampler(
            selected_train_maneuver_rows,
            train_dataset,
            maneuver_weight_config,
            seed=args.seed,
        )
        sampling_diagnostics = simulate_maneuver_sampling(
            maneuver_rows,
            maneuver_weights,
            num_samples=len(train_dataset),
            seed=args.seed,
        )
        maneuver_val_rows = selected_val_maneuver_rows
        print("=" * 50)
        print("Dataloader Sampling Stage")
        print("=" * 50)
        print("weighted dataloader sampler ENABLED.")
        print(f"dataloader maneuver weights: {maneuver_weight_config}")
        print(f"selected subset counts: {maneuver_counts(maneuver_rows)}")
        print(f"selected subset percentages: {maneuver_percentages(maneuver_rows)}")
        print(f"sampled percentages: {sampling_diagnostics['sampled_percentages']}")
        print(f"average sampled weight: {sampling_diagnostics['average_sampled_weight']:.4f}")
    elif args.sampling_strategy == "maneuver_speed_balanced":
        if selected_train_maneuver_speed_rows is None:
            raise ValueError("--sampling-strategy maneuver_speed_balanced requires --maneuver-speed-metadata")
        train_sampler, maneuver_speed_weights, maneuver_speed_rows = make_maneuver_speed_balanced_sampler(
            selected_train_maneuver_speed_rows,
            train_dataset,
            maneuver_speed_weight_config,
            seed=args.seed,
        )
        sampling_diagnostics = simulate_maneuver_speed_sampling(
            maneuver_speed_rows,
            maneuver_speed_weights,
            num_samples=len(train_dataset),
            seed=args.seed,
        )
        maneuver_speed_val_rows = selected_val_maneuver_speed_rows
        print("=" * 50)
        print("Dataloader Sampling Stage")
        print("=" * 50)
        print("weighted dataloader sampler ENABLED.")
        print(f"dataloader maneuver-speed weights: {maneuver_speed_weight_config}")
        print(f"speed delta threshold: {args.speed_delta_threshold}")
        print(f"selected subset combined counts: {maneuver_speed_counts(maneuver_speed_rows)}")
        print(f"selected subset combined percentages: {maneuver_speed_percentages(maneuver_speed_rows)}")
        print(f"sampled combined percentages: {sampling_diagnostics['sampled_percentages']}")
        print(f"average sampled weight: {sampling_diagnostics['average_sampled_weight']:.4f}")
    elif selected_train_maneuver_rows is not None or selected_train_maneuver_speed_rows is not None:
        print("=" * 50)
        print("Dataloader Sampling Stage")
        print("=" * 50)
        print(f"sampling strategy: {args.sampling_strategy}")
        print("No weighted dataloader sampler applied.")
        print("Training uses normal shuffled batches from the selected subset.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model_kwargs = dict(
        past_len=9,
        future_len=8,
        channels=3,
        image_size=image_size,
        hid_s=args.hid_s,
        hid_t=args.hid_t,
        n_s=args.n_s,
        n_t=args.n_t,
        model_type=args.model_type,
        drop_path=args.drop_path,
    )
    if args.model_variant in {"av_simvp", "av_wm", "av_wm_dual", "av_wm_dual_bigwm"}:
        model = TeleopWMPredictor(
            **model_kwargs,
            model_variant=args.model_variant,
            action_dim=args.action_dim,
            speed_dim=args.speed_dim,
            conditioning_dim=args.conditioning_dim,
            simvp_conditioning=args.simvp_conditioning,
            simvp_conditioning_stage=args.simvp_conditioning_stage,
            wm_latent_residual=args.wm_latent_residual,
            wm_residual_hidden_dim=args.wm_residual_hidden_dim,
            wm_residual_scale=args.wm_residual_scale,
            wm_residual_gated=args.wm_residual_gated,
            dual_fusion=args.dual_fusion,
            dual_wm_scale=args.dual_wm_scale,
            dual_wm_hidden_dim=args.dual_wm_hidden_dim,
            dual_wm_num_layers=args.dual_wm_num_layers,
            dual_wm_conditioning=args.dual_wm_conditioning,
            dual_wm_gated=args.dual_wm_gated,
            aux_dynamics_hidden_dim=args.aux_dynamics_hidden_dim if args.aux_dynamics_loss else None,
            future_action_prediction=args.future_action_loss,
            future_action_hidden_dim=args.future_action_hidden_dim,
            future_action_num_layers=args.future_action_num_layers,
            future_action_dropout=args.future_action_dropout,
            future_action_source=args.future_action_source,
            future_action_classification=args.future_action_cls_loss,
            future_action_head_variant=args.future_action_head_variant,
            future_action_detach_latents=args.future_action_detach_latents,
            future_action_future_motion_scale=args.future_action_future_motion_scale,
            future_action_spatial_pooling=args.future_action_spatial_pooling,
            future_action_spatial_grid=args.future_action_spatial_grid,
            control_steer_input_scale=args.control_steer_input_scale,
        )
    else:
        model = SimVPPredictor(**model_kwargs)

    trainer_config = TrainerConfig(
        output_dir=args.output_dir,
        run_tag=args.run_tag,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        amp=not args.no_amp,
        log_interval=args.log_interval,
        grad_clip_norm=args.grad_clip_norm,
        max_train_steps=args.max_train_steps,
        max_val_batches=args.max_val_batches,
        max_interim_val_batches=args.max_interim_val_batches,
        eval_every_steps=args.eval_every_steps,
        stop_on_nan=args.stop_on_nan,
        log_grad_norms=args.log_grad_norms,
        debug_activation_stats=args.debug_activation_stats,
        debug_dual_gate=args.debug_dual_gate,
        resume_checkpoint=str(args.resume) if args.resume is not None else None,
        aux_dynamics_loss=args.aux_dynamics_loss,
        aux_dynamics_weight=args.aux_dynamics_weight,
        aux_dynamics_loss_type=args.aux_dynamics_loss_type,
        progress_bar=not args.no_progress_bar,
        wm_residual_loss=args.wm_residual_loss,
        wm_residual_loss_weight=args.wm_residual_loss_weight,
        wm_residual_loss_type=args.wm_residual_loss_type,
        ssim_loss_weight=args.ssim_loss_weight,
        dual_wm_image_loss_weight=args.dual_wm_image_loss_weight,
        dual_simvp_image_loss_weight=args.dual_simvp_image_loss_weight,
        dual_align_loss_weight=args.dual_align_loss_weight,
        dual_align_loss_type=args.dual_align_loss_type,
        dual_align_direction=args.dual_align_direction,
        dual_detach_simvp_target=args.dual_detach_simvp_target,
        future_action_loss=args.future_action_loss,
        future_action_loss_weight=args.future_action_loss_weight,
        future_action_loss_type=args.future_action_loss_type,
        future_action_source=args.future_action_source,
        future_steer_target_scale=args.future_steer_target_scale,
        control_steer_input_scale=args.control_steer_input_scale,
        future_action_head_variant=args.future_action_head_variant,
        future_action_detach_latents=args.future_action_detach_latents,
        future_action_future_motion_scale=args.future_action_future_motion_scale,
        future_action_corr_loss_weight=args.future_action_corr_loss_weight,
        future_action_delta_loss=args.future_action_delta_loss,
        future_action_delta_loss_weight=args.future_action_delta_loss_weight,
        future_action_delta_loss_type=args.future_action_delta_loss_type,
        future_action_delta_longitudinal_weight=args.future_action_delta_longitudinal_weight,
        future_action_delta_steer_weight=args.future_action_delta_steer_weight,
        future_action_cls_loss=args.future_action_cls_loss,
        future_action_cls_weight=args.future_action_cls_weight,
        future_action_longitudinal_cls_weight=args.future_action_longitudinal_cls_weight,
        future_action_steer_cls_weight=args.future_action_steer_cls_weight,
        longitudinal_coast_threshold=args.longitudinal_coast_threshold,
        steer_straight_threshold=args.steer_straight_threshold,
    )
    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    run_config["train_samples"] = len(train_dataset)
    run_config["val_samples"] = len(val_dataset)
    run_config["validation_mode"] = validation_mode
    run_config["train_data_root"] = str(args.data_root)
    run_config["val_data_root"] = str(args.val_data_root if args.val_data_root is not None else args.data_root)
    run_config["test_data_root"] = str(args.test_data_root) if args.test_data_root is not None else None
    run_config["train_towns"] = full_train_towns
    run_config["val_towns"] = full_val_towns
    run_config["selected_train_towns"] = selected_train_towns
    run_config["selected_val_towns"] = selected_val_towns
    run_config["full_train_samples"] = len(full_train_dataset)
    run_config["full_val_samples"] = len(full_val_dataset)
    run_config["split_seed"] = args.split_seed
    run_config["val_fraction"] = args.val_fraction
    run_config["internal_split_overlap"] = data_diagnostics["internal_split_overlap"]
    run_config["aux_dynamics_target"] = "steer_longitudinal_speed"
    run_config["future_action_target"] = "longitudinal_steer"
    run_config["conditioning_representation"] = "longitudinal_steer_speed"
    run_config["conditioning_fusion"] = args.conditioning_fusion
    run_config["conditioning_injection"] = args.conditioning_injection

    trainer = TeleopWMTrainer(model, train_loader, val_loader, trainer_config, run_config)
    trainer.log("=" * 50)
    trainer.log("Dataset Roots")
    trainer.log("=" * 50)
    trainer.log(f"train data root: {data_diagnostics['train_data_root']}")
    trainer.log(f"val data root: {data_diagnostics['val_data_root']}")
    trainer.log(f"validation mode: {validation_mode}")
    trainer.log(f"train towns: {', '.join(full_train_towns) if full_train_towns else '(unknown)'}")
    trainer.log(f"val towns: {', '.join(full_val_towns) if full_val_towns else '(unknown)'}")
    trainer.log(f"selected train towns: {', '.join(selected_train_towns) if selected_train_towns else '(unknown)'}")
    trainer.log(f"selected val towns: {', '.join(selected_val_towns) if selected_val_towns else '(unknown)'}")
    trainer.log(f"train samples: {len(train_dataset)}")
    trainer.log(f"val samples: {len(val_dataset)}")
    if validation_mode == "internal_split":
        trainer.log(f"val fraction: {args.val_fraction}")
        trainer.log(f"split seed: {args.split_seed}")
        trainer.log(f"internal split overlap: {data_diagnostics['internal_split_overlap']}")
    if args.sampling_strategy == "action_balanced" and rollout_weights is not None:
        sampling_dir = trainer.run_dir / "sampling"
        if args.weights_path is None and rollout_weight_rows is not None:
            weights_path, metadata_path, summary_path = save_rollout_weight_outputs(
                sampling_dir,
                rollout_weight_rows,
                rollout_weights,
                rollout_weight_thresholds or {},
            )
            trainer.log(f"saved rollout weights: {weights_path}")
            trainer.log(f"saved rollout metadata: {metadata_path}")
            trainer.log(f"saved rollout weight summary: {summary_path}")
        elif args.weights_path is not None:
            trainer.log(f"loaded rollout weights: {args.weights_path}")
        if sampling_diagnostics is not None and rollout_weight_rows is not None:
            plot_path = trainer.run_dir / "sampling_distribution.png"
            save_sampling_distribution_plot(
                plot_path,
                category_percentages(rollout_weight_rows),
                sampling_diagnostics["sampled_percentages"],
            )
            trainer.log(f"saved sampling distribution plot: {plot_path}")
    should_save_maneuver_diagnostics = (
        (selected_train_maneuver_rows is not None or selected_train_maneuver_speed_rows is not None)
        and (
            args.subset_strategy == "maneuver_weighted"
            or args.val_subset_strategy == "maneuver_weighted"
            or args.sampling_strategy == "maneuver_balanced"
            or args.sampling_strategy == "maneuver_speed_balanced"
        )
    )
    if should_save_maneuver_diagnostics:
        save_maneuver_sampling_artifacts(
            trainer,
            args,
            full_train_maneuver_rows or selected_train_maneuver_rows or full_train_maneuver_speed_rows or selected_train_maneuver_speed_rows,
            maneuver_rows or selected_train_maneuver_rows or selected_train_maneuver_speed_rows,
            maneuver_weights if args.sampling_strategy == "maneuver_balanced" else None,
            full_val_maneuver_rows or selected_val_maneuver_rows or full_val_maneuver_speed_rows or selected_val_maneuver_speed_rows or [],
            maneuver_val_rows or selected_val_maneuver_rows or selected_val_maneuver_speed_rows or [],
            maneuver_weight_config or parse_maneuver_weights(args.maneuver_weights),
            sampling_diagnostics,
            data_diagnostics=data_diagnostics,
            full_train_maneuver_speed_rows=full_train_maneuver_speed_rows,
            train_maneuver_speed_rows=maneuver_speed_rows or selected_train_maneuver_speed_rows,
            train_maneuver_speed_weights=maneuver_speed_weights,
            full_val_maneuver_speed_rows=full_val_maneuver_speed_rows,
            val_maneuver_speed_rows=maneuver_speed_val_rows or selected_val_maneuver_speed_rows,
            maneuver_speed_weight_config=maneuver_speed_weight_config,
        )
    trainer.fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
