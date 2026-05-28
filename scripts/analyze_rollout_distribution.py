#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets import CarlaRolloutDataset


INFORMATIVENESS_LEVELS = [
    "low_control_info",
    "medium_control_info",
    "high_control_info",
]

DOMINANT_FACTORS = [
    "steering_dominant",
    "longitudinal_dominant",
    "speed_dominant",
    "mixed_dynamic",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze control diversity over CARLA rollout windows.")
    parser.add_argument("--data-root", type=Path, default=Path("/path/to/mile_action_diverse/train/Town01"))
    parser.add_argument("--split", default="train", choices=["train", "val", "validation", "test"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rollout_distribution"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--normalize-controls", action="store_true")
    parser.add_argument("--speed-scale", type=float, default=20.0)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steering-weight", type=float, default=2.0)
    parser.add_argument("--longitudinal-weight", type=float, default=1.5)
    parser.add_argument("--speed-weight", type=float, default=1.0)
    parser.add_argument(
        "--sample-mode",
        choices=["random", "first"],
        default="random",
        help="How to choose max-samples rollout windows when limiting analysis.",
    )
    parser.add_argument(
        "--examples-per-category",
        type=int,
        default=3,
        help="Save up to N representative rollout grids per informativeness level and dominant factor. Use 0 to disable.",
    )
    return parser.parse_args()


def resolve_data_root(root: Path) -> Path:
    root = root.expanduser()
    if (root / "0000").exists() or (root / "0002").exists() or any(root.glob("Town*/0000")):
        return root
    if any((root / "train").glob("Town*/0000")):
        return root / "train"
    for candidate in (root / "train" / "Town01", root / "Town01"):
        if (candidate / "0000").exists() or (candidate / "0002").exists():
            return candidate
    return root


def make_output_dir(base_dir: Path, split: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = base_dir / f"{timestamp}_{split}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def choose_indices(total: int, max_samples: int | None, seed: int, sample_mode: str) -> list[int]:
    if max_samples is None or max_samples >= total:
        return list(range(total))
    if sample_mode == "first":
        return list(range(max_samples))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(total, size=max_samples, replace=False).astype(int).tolist())


def thresholds_for() -> dict[str, float]:
    return {
        "brake_active_threshold": 0.05,
    }


def scoring_weights(args: argparse.Namespace) -> dict[str, float]:
    return {
        "steering": args.steering_weight,
        "longitudinal": args.longitudinal_weight,
        "speed": args.speed_weight,
    }


def scoring_formula() -> dict[str, str]:
    return {
        "steering_informativeness": "full_abs_steer_mean + full_steer_std + 2.0 * full_steer_delta_mean + full_steer_delta_max",
        "longitudinal_control": "throttle - brake",
        "longitudinal_informativeness": "full_abs_longitudinal_mean + full_longitudinal_std + 2.0 * full_longitudinal_delta_mean + full_longitudinal_delta_max",
        "speed_informativeness": "full_speed_std + 2.0 * full_speed_delta_mean + full_speed_delta_max",
        "overall_control_informativeness": (
            "steering_weight * steering_informativeness + "
            "longitudinal_weight * longitudinal_informativeness + "
            "speed_weight * speed_informativeness"
        ),
        "informativeness_levels": "low <= p50, medium > p50 and <= p80, high > p80",
        "dominant_factor": "largest weighted component, unless top two weighted components are within 15%, then mixed_dynamic",
    }


def finite_float(value: float) -> float:
    if math.isfinite(float(value)):
        return float(value)
    return 0.0


def diff_abs_mean(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return finite_float(np.mean(np.abs(np.diff(values, axis=0))))


def diff_abs_max(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return finite_float(np.max(np.abs(np.diff(values, axis=0))))


def segment_stats(prefix: str, actions: np.ndarray, speed: np.ndarray, thresholds: dict[str, float]) -> dict[str, float]:
    throttle = actions[:, 0]
    steer = actions[:, 1]
    brake = actions[:, 2]
    longitudinal = throttle - brake
    speed_1d = speed[:, 0]
    action_delta = np.diff(actions, axis=0) if len(actions) > 1 else np.zeros((0, actions.shape[1]), dtype=np.float32)
    speed_delta = np.diff(speed_1d, axis=0) if len(speed_1d) > 1 else np.zeros((0,), dtype=np.float32)
    action_l2 = np.linalg.norm(actions, axis=1)
    control_variation_score = (
        diff_abs_mean(steer)
        + diff_abs_mean(longitudinal)
        + diff_abs_mean(speed_1d)
    )

    return {
        f"{prefix}_steer_mean": finite_float(np.mean(steer)),
        f"{prefix}_steer_std": finite_float(np.std(steer)),
        f"{prefix}_steer_min": finite_float(np.min(steer)),
        f"{prefix}_steer_max": finite_float(np.max(steer)),
        f"{prefix}_abs_steer_mean": finite_float(np.mean(np.abs(steer))),
        f"{prefix}_abs_steer_max": finite_float(np.max(np.abs(steer))),
        f"{prefix}_steer_delta_mean": diff_abs_mean(steer),
        f"{prefix}_steer_delta_max": diff_abs_max(steer),
        f"{prefix}_throttle_mean": finite_float(np.mean(throttle)),
        f"{prefix}_throttle_std": finite_float(np.std(throttle)),
        f"{prefix}_throttle_delta_mean": diff_abs_mean(throttle),
        f"{prefix}_throttle_delta_max": diff_abs_max(throttle),
        f"{prefix}_brake_mean": finite_float(np.mean(brake)),
        f"{prefix}_brake_active_ratio": finite_float(np.mean(brake > thresholds["brake_active_threshold"])),
        f"{prefix}_brake_delta_mean": diff_abs_mean(brake),
        f"{prefix}_longitudinal_mean": finite_float(np.mean(longitudinal)),
        f"{prefix}_longitudinal_std": finite_float(np.std(longitudinal)),
        f"{prefix}_longitudinal_min": finite_float(np.min(longitudinal)),
        f"{prefix}_longitudinal_max": finite_float(np.max(longitudinal)),
        f"{prefix}_abs_longitudinal_mean": finite_float(np.mean(np.abs(longitudinal))),
        f"{prefix}_abs_longitudinal_max": finite_float(np.max(np.abs(longitudinal))),
        f"{prefix}_longitudinal_delta_mean": diff_abs_mean(longitudinal),
        f"{prefix}_longitudinal_delta_max": diff_abs_max(longitudinal),
        f"{prefix}_speed_mean": finite_float(np.mean(speed_1d)),
        f"{prefix}_speed_std": finite_float(np.std(speed_1d)),
        f"{prefix}_speed_delta_mean": diff_abs_mean(speed_1d),
        f"{prefix}_speed_delta_max": diff_abs_max(speed_1d),
        f"{prefix}_action_l2_mean": finite_float(np.mean(action_l2)),
        f"{prefix}_action_l2_std": finite_float(np.std(action_l2)),
        f"{prefix}_control_variation_score": finite_float(control_variation_score),
        f"{prefix}_action_delta_l2_mean": finite_float(np.mean(np.linalg.norm(action_delta, axis=1))) if len(action_delta) else 0.0,
        f"{prefix}_speed_signed_delta_mean": finite_float(np.mean(speed_delta)) if len(speed_delta) else 0.0,
    }


def add_informativeness_scores(row: dict[str, Any], weights: dict[str, float]) -> None:
    steering = (
        row["full_abs_steer_mean"]
        + row["full_steer_std"]
        + 2.0 * row["full_steer_delta_mean"]
        + row["full_steer_delta_max"]
    )
    longitudinal = (
        row["full_abs_longitudinal_mean"]
        + row["full_longitudinal_std"]
        + 2.0 * row["full_longitudinal_delta_mean"]
        + row["full_longitudinal_delta_max"]
    )
    speed = row["full_speed_std"] + 2.0 * row["full_speed_delta_mean"] + row["full_speed_delta_max"]

    weighted_components = {
        "steering_dominant": weights["steering"] * steering,
        "longitudinal_dominant": weights["longitudinal"] * longitudinal,
        "speed_dominant": weights["speed"] * speed,
    }
    sorted_components = sorted(weighted_components.items(), key=lambda item: item[1], reverse=True)
    top_name, top_value = sorted_components[0]
    second_value = sorted_components[1][1]
    if top_value > 0.0 and second_value / top_value >= 0.85:
        dominant = "mixed_dynamic"
    else:
        dominant = top_name

    row["steering_informativeness"] = finite_float(steering)
    row["longitudinal_informativeness"] = finite_float(longitudinal)
    row["speed_informativeness"] = finite_float(speed)
    row["weighted_steering_informativeness"] = finite_float(weighted_components["steering_dominant"])
    row["weighted_longitudinal_informativeness"] = finite_float(weighted_components["longitudinal_dominant"])
    row["weighted_speed_informativeness"] = finite_float(weighted_components["speed_dominant"])
    row["overall_control_informativeness"] = finite_float(sum(weighted_components.values()))
    row["dominant_factor"] = dominant


def assign_informativeness_levels(rows: list[dict[str, Any]]) -> dict[str, float]:
    scores = np.asarray([row["overall_control_informativeness"] for row in rows], dtype=np.float64)
    p50 = finite_float(np.percentile(scores, 50))
    p80 = finite_float(np.percentile(scores, 80))
    for row in rows:
        score = row["overall_control_informativeness"]
        if score <= p50:
            level = "low_control_info"
        elif score <= p80:
            level = "medium_control_info"
        else:
            level = "high_control_info"
        row["informativeness_level"] = level
        row["category"] = level
    return {
        "p50": p50,
        "p80": p80,
        "p95": finite_float(np.percentile(scores, 95)),
        "p99": finite_float(np.percentile(scores, 99)),
    }


def compute_rollout_stats(
    dataset: CarlaRolloutDataset,
    dataset_idx: int,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    window = dataset.windows[dataset_idx]
    actions = dataset.actions_by_run[window.run_key][window.start : window.end]
    speed = dataset.speed_by_run[window.run_key][window.start : window.end]
    past_actions = actions[: dataset.past_len]
    future_actions = actions[dataset.past_len :]
    past_speed = speed[: dataset.past_len]
    future_speed = speed[dataset.past_len :]

    row: dict[str, Any] = {
        "dataset_idx": dataset_idx,
        "town": window.town,
        "run_key": window.run_key,
        "run_id": window.run_id,
        "run_path": str(window.run_path),
        "start_idx": window.start,
        "end_idx": window.end - 1,
        "past_start_idx": window.start,
        "past_end_idx": window.start + dataset.past_len - 1,
        "future_start_idx": window.start + dataset.past_len,
        "future_end_idx": window.end - 1,
    }
    row.update(segment_stats("full", actions, speed, thresholds))
    row.update(segment_stats("past", past_actions, past_speed, thresholds))
    row.update(segment_stats("future", future_actions, future_speed, thresholds))
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def numeric_columns(rows: list[dict[str, Any]]) -> list[str]:
    excluded = {"dataset_idx", "start_idx", "end_idx", "past_start_idx", "past_end_idx", "future_start_idx", "future_end_idx"}
    columns = []
    for key, value in rows[0].items():
        if key in excluded or key in {"run_id", "category", "informativeness_level", "dominant_factor"}:
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            columns.append(key)
    return columns


def summarize(
    rows: list[dict[str, Any]],
    thresholds: dict[str, float],
    percentile_thresholds: dict[str, float],
    args: argparse.Namespace,
    data_root: Path,
) -> dict[str, Any]:
    level_counts_raw = Counter(row["informativeness_level"] for row in rows)
    dominant_counts_raw = Counter(row["dominant_factor"] for row in rows)
    total = len(rows)
    level_counts = {level: level_counts_raw.get(level, 0) for level in INFORMATIVENESS_LEVELS}
    level_percentages = {
        level: 100.0 * count / max(total, 1)
        for level, count in level_counts.items()
    }
    dominant_counts = {factor: dominant_counts_raw.get(factor, 0) for factor in DOMINANT_FACTORS}
    dominant_percentages = {
        factor: 100.0 * count / max(total, 1)
        for factor, count in dominant_counts.items()
    }

    global_stats: dict[str, dict[str, float]] = {}
    for column in numeric_columns(rows):
        values = np.asarray([row[column] for row in rows], dtype=np.float64)
        global_stats[column] = {
            "mean": finite_float(np.mean(values)),
            "std": finite_float(np.std(values)),
            "min": finite_float(np.min(values)),
            "max": finite_float(np.max(values)),
            "p1": finite_float(np.percentile(values, 1)),
            "p50": finite_float(np.percentile(values, 50)),
            "p80": finite_float(np.percentile(values, 80)),
            "p95": finite_float(np.percentile(values, 95)),
            "p99": finite_float(np.percentile(values, 99)),
        }

    return {
        "split": args.split,
        "data_root": str(data_root),
        "total_rollouts_available": None,
        "total_rollouts": total,
        "total_rollouts_analyzed": total,
        "max_samples": args.max_samples,
        "sample_mode": args.sample_mode,
        "seed": args.seed,
        "normalize_controls": args.normalize_controls,
        "speed_scale": args.speed_scale,
        "informativeness_level_counts": level_counts,
        "informativeness_level_percentages": level_percentages,
        "dominant_factor_counts": dominant_counts,
        "dominant_factor_percentages": dominant_percentages,
        "category_counts": level_counts,
        "category_percentages": level_percentages,
        "global_stats": global_stats,
        "thresholds": thresholds,
        "percentile_thresholds": percentile_thresholds,
        "scoring_formula": scoring_formula(),
        "scoring_weights": scoring_weights(args),
    }


def import_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_hist(plt, values: np.ndarray, path: Path, title: str, xlabel: str, bins: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins, color="#386cb0", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Rollout count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_bar(plt, labels: list[str], values: list[float], path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color="#7fc97f")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_correlation_heatmap(plt, rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "full_steer_delta_mean",
        "full_longitudinal_delta_mean",
        "full_speed_delta_mean",
        "full_control_variation_score",
        "overall_control_informativeness",
        "full_abs_steer_mean",
        "full_abs_longitudinal_mean",
    ]
    labels = [
        "steer variation",
        "longitudinal variation",
        "speed variation",
        "control variation",
        "informativeness",
        "abs steer mean",
        "abs longitudinal mean",
    ]
    matrix = np.asarray([[row[column] for column in columns] for row in rows], dtype=np.float64)
    if len(rows) < 2:
        corr = np.eye(len(columns))
    else:
        corr = np.corrcoef(matrix, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_title("Control Dynamics Correlation")
    for row_idx in range(corr.shape[0]):
        for col_idx in range(corr.shape[1]):
            ax.text(col_idx, row_idx, f"{corr[row_idx, col_idx]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_summary_plot(plt, rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path, bins: int) -> None:
    levels = INFORMATIVENESS_LEVELS
    level_percentages = [summary["informativeness_level_percentages"][level] for level in levels]
    factors = DOMINANT_FACTORS
    factor_percentages = [summary["dominant_factor_percentages"][factor] for factor in factors]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()

    axes[0].bar(levels, level_percentages, color="#7fc97f")
    axes[0].set_title("Control Informativeness Level Percentages")
    axes[0].set_ylabel("Percent")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(factors, factor_percentages, color="#fdc086")
    axes[1].set_title("Dominant Control Factor Percentages")
    axes[1].set_ylabel("Percent")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].hist([row["steering_informativeness"] for row in rows], bins=bins, color="#386cb0")
    axes[2].set_title("Steering Informativeness Distribution")
    axes[2].set_xlabel("Steering informativeness")
    axes[2].set_ylabel("Rollout count")
    axes[2].grid(True, alpha=0.3)

    axes[3].hist([row["longitudinal_informativeness"] for row in rows], bins=bins, color="#beaed4")
    axes[3].set_title("Longitudinal Informativeness Distribution")
    axes[3].set_xlabel("Longitudinal informativeness")
    axes[3].set_ylabel("Rollout count")
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "rollout_dynamics_summary.png", dpi=170)
    plt.close(fig)


def save_plots(rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path, bins: int) -> None:
    plt = import_matplotlib()
    arrays = {key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in numeric_columns(rows)}

    save_hist(plt, arrays["full_steer_mean"], output_dir / "steering_hist.png", "Steering Mean per Rollout", "Mean steer", bins)
    save_hist(plt, arrays["full_abs_steer_mean"], output_dir / "abs_steering_hist.png", "Absolute Steering Mean per Rollout", "Mean |steer|", bins)
    save_hist(plt, arrays["full_steer_delta_mean"], output_dir / "steering_delta_hist.png", "Steering Temporal Variation", "Mean |delta steer|", bins)
    save_hist(plt, arrays["full_speed_mean"], output_dir / "speed_hist.png", "Speed Mean per Rollout", "Mean speed", bins)
    save_hist(plt, arrays["full_speed_delta_mean"], output_dir / "speed_delta_hist.png", "Speed Temporal Variation", "Mean |delta speed|", bins)
    save_hist(plt, arrays["full_throttle_mean"], output_dir / "throttle_hist.png", "Throttle Mean per Rollout", "Mean throttle", bins)
    save_hist(plt, arrays["full_brake_active_ratio"], output_dir / "brake_activity_hist.png", "Brake Activity per Rollout", "Brake active ratio", bins)
    save_hist(plt, arrays["full_longitudinal_mean"], output_dir / "longitudinal_hist.png", "Longitudinal Control Mean per Rollout", "Mean throttle - brake", bins)
    save_hist(plt, arrays["full_abs_longitudinal_mean"], output_dir / "abs_longitudinal_hist.png", "Absolute Longitudinal Control Mean per Rollout", "Mean |throttle - brake|", bins)
    save_hist(plt, arrays["full_longitudinal_delta_mean"], output_dir / "longitudinal_delta_hist.png", "Longitudinal Control Temporal Variation", "Mean |delta longitudinal|", bins)
    save_hist(plt, arrays["full_control_variation_score"], output_dir / "control_variation_hist.png", "Control Variation Score", "Variation score", bins)
    save_hist(plt, arrays["overall_control_informativeness"], output_dir / "control_informativeness_hist.png", "Overall Control Informativeness", "Overall informativeness", bins)
    save_hist(plt, arrays["steering_informativeness"], output_dir / "steering_informativeness_hist.png", "Steering Informativeness", "Steering informativeness", bins)
    save_hist(plt, arrays["longitudinal_informativeness"], output_dir / "longitudinal_informativeness_hist.png", "Longitudinal Informativeness", "Longitudinal informativeness", bins)
    save_hist(plt, arrays["speed_informativeness"], output_dir / "speed_informativeness_hist.png", "Speed Informativeness", "Speed informativeness", bins)

    level_counts = [summary["informativeness_level_counts"][level] for level in INFORMATIVENESS_LEVELS]
    level_percentages = [summary["informativeness_level_percentages"][level] for level in INFORMATIVENESS_LEVELS]
    save_bar(plt, INFORMATIVENESS_LEVELS, level_counts, output_dir / "informativeness_level_counts.png", "Control Informativeness Level Counts", "Count")
    save_bar(plt, INFORMATIVENESS_LEVELS, level_percentages, output_dir / "informativeness_level_percentages.png", "Control Informativeness Level Percentages", "Percent")

    factor_counts = [summary["dominant_factor_counts"][factor] for factor in DOMINANT_FACTORS]
    factor_percentages = [summary["dominant_factor_percentages"][factor] for factor in DOMINANT_FACTORS]
    save_bar(plt, DOMINANT_FACTORS, factor_counts, output_dir / "dominant_factor_counts.png", "Dominant Control Factor Counts", "Count")
    save_bar(plt, DOMINANT_FACTORS, factor_percentages, output_dir / "dominant_factor_percentages.png", "Dominant Control Factor Percentages", "Percent")
    save_correlation_heatmap(plt, rows, output_dir / "correlation_heatmap.png")
    save_summary_plot(plt, rows, summary, output_dir, bins)


def tensor_to_image(frame) -> Image.Image:
    array = frame.permute(1, 2, 0).numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def save_rollout_example(dataset: CarlaRolloutDataset, row: dict[str, Any], group_name: str, output_path: Path) -> None:
    dataset_idx = int(row["dataset_idx"])
    sample = dataset[dataset_idx]
    metadata = sample["metadata"]
    rows = [
        ("past", sample["past_frames"], sample["past_actions"], sample["past_speed"], metadata["past_indices"]),
        ("future", sample["future_frames"], sample["future_actions"], sample["future_speed"], metadata["future_indices"]),
    ]
    cell_w, cell_h = 160, 100
    label_h = 70
    title_h = 82
    width = max(frames.shape[0] for _, frames, _, _, _ in rows) * cell_w
    height = title_h + len(rows) * (cell_h + label_h)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (4, 4),
        f"group={group_name} level={row['informativeness_level']} dominant={row['dominant_factor']} "
        f"dataset_idx={dataset_idx} run={metadata['run_id']} start={metadata['start']}",
        fill=(0, 0, 0),
    )
    draw.text(
        (4, 24),
        f"overall={row['overall_control_informativeness']:.4f} "
        f"steer={row['steering_informativeness']:.4f} "
        f"long={row['longitudinal_informativeness']:.4f} "
        f"speed={row['speed_informativeness']:.4f}",
        fill=(0, 0, 0),
    )
    draw.text(
        (4, 44),
        f"weighted steer={row['weighted_steering_informativeness']:.4f} "
        f"long={row['weighted_longitudinal_informativeness']:.4f} "
        f"speed={row['weighted_speed_informativeness']:.4f}",
        fill=(0, 0, 0),
    )
    draw.text(
        (4, 64),
        f"past={metadata['past_indices'][0]}..{metadata['past_indices'][-1]} future={metadata['future_indices'][0]}..{metadata['future_indices'][-1]}",
        fill=(0, 0, 0),
    )

    for row_idx, (label, frames, actions, speed, indices) in enumerate(rows):
        y = title_h + row_idx * (cell_h + label_h)
        for col, frame in enumerate(frames):
            x = col * cell_w
            action = actions[col].numpy()
            longitudinal = action[0] - action[2]
            draw.text((x + 4, y + 2), f"{label} t={col} abs={indices[col]}", fill=(0, 0, 0))
            draw.text((x + 4, y + 18), f"thr={action[0]:.2f} steer={action[1]:.2f}", fill=(0, 0, 0))
            draw.text((x + 4, y + 34), f"brake={action[2]:.2f} long={longitudinal:.2f}", fill=(0, 0, 0))
            draw.text((x + 4, y + 50), f"speed={speed[col, 0].item():.2f}", fill=(0, 0, 0))
            image = tensor_to_image(frame)
            image.thumbnail((cell_w, cell_h))
            canvas.paste(image, (x, y + label_h))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def save_examples(dataset: CarlaRolloutDataset, rows: list[dict[str, Any]], output_dir: Path, examples_per_category: int) -> None:
    if examples_per_category <= 0:
        return
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for group_name in (row["informativeness_level"], row["dominant_factor"]):
            if len(by_group[group_name]) < examples_per_category:
                by_group[group_name].append(row)

    for group_name in INFORMATIVENESS_LEVELS + DOMINANT_FACTORS:
        (output_dir / "examples" / group_name).mkdir(parents=True, exist_ok=True)
        group_rows = by_group.get(group_name, [])
        for example_idx, row in enumerate(group_rows):
            save_rollout_example(
                dataset,
                row,
                group_name,
                output_dir / "examples" / group_name / f"rollout_{example_idx:02d}_idx_{int(row['dataset_idx']):06d}.png",
            )


def main() -> int:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    output_dir = make_output_dir(args.output_dir, args.split)
    thresholds = thresholds_for()
    weights = scoring_weights(args)

    dataset = CarlaRolloutDataset(
        data_root,
        split=args.split,
        image_size=(160, 256),
        normalize_controls=args.normalize_controls,
        speed_scale=args.speed_scale,
        include_metadata=True,
    )
    indices = choose_indices(len(dataset), args.max_samples, args.seed, args.sample_mode)
    print(f"data_root: {data_root}")
    print(f"split: {args.split}")
    print(f"available rollouts: {len(dataset)}")
    print(f"analyzing rollouts: {len(indices)}")
    print(f"output_dir: {output_dir}")

    rows = [compute_rollout_stats(dataset, dataset_idx, thresholds) for dataset_idx in indices]
    if not rows:
        raise RuntimeError("No rollout windows selected for analysis.")
    for row in rows:
        add_informativeness_scores(row, weights)
    percentile_thresholds = assign_informativeness_levels(rows)

    summary = summarize(rows, thresholds, percentile_thresholds, args, data_root)
    summary["total_rollouts_available"] = len(dataset)

    csv_path = output_dir / "rollout_stats.csv"
    json_path = output_dir / "rollout_distribution_summary.json"
    write_csv(csv_path, rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    save_plots(rows, summary, output_dir, args.bins)
    save_examples(dataset, rows, output_dir, args.examples_per_category)

    print(f"wrote CSV: {csv_path}")
    print(f"wrote summary: {json_path}")
    print("informativeness level percentages:")
    for level, percentage in summary["informativeness_level_percentages"].items():
        print(f"  {level}: {percentage:.2f}% ({summary['informativeness_level_counts'][level]})")
    print("dominant factor percentages:")
    for factor, percentage in summary["dominant_factor_percentages"].items():
        print(f"  {factor}: {percentage:.2f}% ({summary['dominant_factor_counts'][factor]})")
    print("informativeness percentiles:")
    overall_stats = summary["global_stats"]["overall_control_informativeness"]
    steering_stats = summary["global_stats"]["steering_informativeness"]
    longitudinal_stats = summary["global_stats"]["longitudinal_informativeness"]
    speed_stats = summary["global_stats"]["speed_informativeness"]
    print(
        "  overall_control_informativeness: "
        f"p50={overall_stats['p50']:.6f} p80={overall_stats['p80']:.6f} "
        f"p95={overall_stats['p95']:.6f} p99={overall_stats['p99']:.6f}"
    )
    print(
        "  steering_informativeness: "
        f"p50={steering_stats['p50']:.6f} p80={steering_stats['p80']:.6f} "
        f"p95={steering_stats['p95']:.6f} "
        f"p99={steering_stats['p99']:.6f}"
    )
    print(
        "  longitudinal_informativeness: "
        f"p50={longitudinal_stats['p50']:.6f} p80={longitudinal_stats['p80']:.6f} "
        f"p95={longitudinal_stats['p95']:.6f} p99={longitudinal_stats['p99']:.6f}"
    )
    print(
        "  speed_informativeness: "
        f"p50={speed_stats['p50']:.6f} p80={speed_stats['p80']:.6f} "
        f"p95={speed_stats['p95']:.6f} p99={speed_stats['p99']:.6f}"
    )
    print("selected global stats:")
    for key in [
        "full_abs_steer_mean",
        "full_steer_delta_mean",
        "full_speed_mean",
        "full_speed_delta_mean",
        "full_abs_longitudinal_mean",
        "full_longitudinal_delta_mean",
        "full_brake_active_ratio",
        "full_control_variation_score",
        "overall_control_informativeness",
        "steering_informativeness",
        "longitudinal_informativeness",
        "speed_informativeness",
    ]:
        stats = summary["global_stats"][key]
        print(f"  {key}: mean={stats['mean']:.6f} std={stats['std']:.6f} p50={stats['p50']:.6f} p99={stats['p99']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
