from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Subset, WeightedRandomSampler

from src.datasets import CarlaRolloutDataset


CATEGORY_WEIGHTS = {
    "low_dynamic": 1.0,
    "turn_dynamic": 3.0,
    "speed_transition": 2.5,
    "mixed_dynamic": 4.0,
    "stop_and_go": 4.0,
}

CATEGORY_ORDER = [
    "low_dynamic",
    "turn_dynamic",
    "speed_transition",
    "mixed_dynamic",
    "stop_and_go",
]


def _base_dataset(dataset) -> CarlaRolloutDataset:
    if isinstance(dataset, Subset):
        dataset = dataset.dataset
    if not isinstance(dataset, CarlaRolloutDataset):
        raise TypeError(f"Expected CarlaRolloutDataset or Subset, got {type(dataset).__name__}")
    return dataset


def _base_index(dataset, sample_index: int) -> int:
    if isinstance(dataset, Subset):
        return int(dataset.indices[sample_index])
    return int(sample_index)


def _diff_abs_mean(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(values, axis=0))))


def _diff_abs_max(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(values, axis=0))))


def _score_window(actions: np.ndarray, speed: np.ndarray) -> dict[str, float]:
    throttle = actions[:, 0]
    steer = actions[:, 1]
    brake = actions[:, 2]
    longitudinal = throttle - brake
    speed_1d = speed[:, 0]

    steer_delta_mean = _diff_abs_mean(steer)
    steer_delta_max = _diff_abs_max(steer)
    steer_std = float(np.std(steer))

    longitudinal_delta_mean = _diff_abs_mean(longitudinal)
    longitudinal_delta_max = _diff_abs_max(longitudinal)
    longitudinal_std = float(np.std(longitudinal))

    speed_delta_mean = _diff_abs_mean(speed_1d)
    speed_delta_max = _diff_abs_max(speed_1d)
    speed_std = float(np.std(speed_1d))

    steering_score = steer_delta_mean + 0.5 * steer_delta_max + 0.25 * steer_std
    longitudinal_score = (
        longitudinal_delta_mean
        + 0.5 * longitudinal_delta_max
        + 0.25 * longitudinal_std
    )
    speed_score = speed_delta_mean + 0.5 * speed_delta_max + 0.25 * speed_std
    overall_score = 2.0 * steering_score + 1.5 * longitudinal_score + speed_score

    return {
        "steer_abs_mean": float(np.mean(np.abs(steer))),
        "steer_std": steer_std,
        "steer_delta_mean": steer_delta_mean,
        "steer_delta_max": steer_delta_max,
        "longitudinal_abs_mean": float(np.mean(np.abs(longitudinal))),
        "longitudinal_std": longitudinal_std,
        "longitudinal_delta_mean": longitudinal_delta_mean,
        "longitudinal_delta_max": longitudinal_delta_max,
        "longitudinal_min": float(np.min(longitudinal)),
        "longitudinal_max": float(np.max(longitudinal)),
        "speed_std": speed_std,
        "speed_delta_mean": speed_delta_mean,
        "speed_delta_max": speed_delta_max,
        "steering_score": float(steering_score),
        "longitudinal_score": float(longitudinal_score),
        "speed_score": float(speed_score),
        "overall_variation": float(steer_delta_mean + longitudinal_delta_mean + speed_delta_mean),
        "overall_informativeness": float(overall_score),
    }


def _percentile_thresholds(rows: list[dict[str, Any]]) -> dict[str, float]:
    arrays = {
        "steering_score": np.asarray([row["steering_score"] for row in rows], dtype=np.float64),
        "longitudinal_score": np.asarray([row["longitudinal_score"] for row in rows], dtype=np.float64),
        "speed_score": np.asarray([row["speed_score"] for row in rows], dtype=np.float64),
        "overall_informativeness": np.asarray(
            [row["overall_informativeness"] for row in rows], dtype=np.float64
        ),
        "longitudinal_abs_mean": np.asarray(
            [row["longitudinal_abs_mean"] for row in rows], dtype=np.float64
        ),
    }
    return {
        "steering_top20": float(np.percentile(arrays["steering_score"], 80)),
        "longitudinal_top20": float(np.percentile(arrays["longitudinal_score"], 80)),
        "speed_top20": float(np.percentile(arrays["speed_score"], 80)),
        "steering_bottom50": float(np.percentile(arrays["steering_score"], 50)),
        "longitudinal_bottom50": float(np.percentile(arrays["longitudinal_score"], 50)),
        "speed_bottom50": float(np.percentile(arrays["speed_score"], 50)),
        "overall_bottom50": float(np.percentile(arrays["overall_informativeness"], 50)),
        "overall_top20": float(np.percentile(arrays["overall_informativeness"], 80)),
        "stop_go_longitudinal_abs_threshold": float(
            np.percentile(arrays["longitudinal_abs_mean"], 50)
        ),
    }


def _assign_category(row: dict[str, Any], thresholds: dict[str, float]) -> str:
    high_steer = row["steering_score"] >= thresholds["steering_top20"]
    high_longitudinal = row["longitudinal_score"] >= thresholds["longitudinal_top20"]
    high_speed = row["speed_score"] >= thresholds["speed_top20"]

    low_steer = row["steering_score"] <= thresholds["steering_bottom50"]
    low_longitudinal = row["longitudinal_score"] <= thresholds["longitudinal_bottom50"]
    low_speed = row["speed_score"] <= thresholds["speed_bottom50"]
    low_overall = row["overall_informativeness"] <= thresholds["overall_bottom50"]

    sign_threshold = thresholds["stop_go_longitudinal_abs_threshold"]
    stop_and_go = (
        row["longitudinal_min"] < -sign_threshold
        and row["longitudinal_max"] > sign_threshold
        and high_longitudinal
    )
    if stop_and_go:
        return "stop_and_go"
    if high_steer and (high_longitudinal or high_speed):
        return "mixed_dynamic"
    if high_steer and low_longitudinal:
        return "turn_dynamic"
    if (high_longitudinal or high_speed) and low_steer:
        return "speed_transition"
    if low_overall or (low_steer and low_longitudinal and low_speed):
        return "low_dynamic"
    if high_steer:
        return "turn_dynamic"
    if high_longitudinal or high_speed:
        return "speed_transition"
    return "low_dynamic"


def compute_rollout_weight_table(dataset) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, float]]:
    """Compute rollout metadata and per-sample weights for a dataset or subset.

    The returned weights are aligned with the input dataset's sample indices,
    which is exactly what WeightedRandomSampler expects.
    """

    base = _base_dataset(dataset)
    rows: list[dict[str, Any]] = []
    for sample_index in range(len(dataset)):
        dataset_index = _base_index(dataset, sample_index)
        window = base.windows[dataset_index]
        actions = base.actions_by_run[window.run_key][window.start : window.end]
        speed = base.speed_by_run[window.run_key][window.start : window.end]
        row: dict[str, Any] = {
            "rollout_id": sample_index,
            "dataset_idx": dataset_index,
            "town": window.town,
            "run_key": window.run_key,
            "run_id": window.run_id,
            "run_path": str(window.run_path),
            "start_idx": window.start,
            "end_idx": window.end - 1,
        }
        row.update(_score_window(actions, speed))
        rows.append(row)

    if not rows:
        raise ValueError("Cannot compute rollout weights for an empty dataset.")

    thresholds = _percentile_thresholds(rows)
    raw_weights = []
    for row in rows:
        category = _assign_category(row, thresholds)
        raw_weight = CATEGORY_WEIGHTS[category]
        row["category"] = category
        row["raw_weight"] = float(raw_weight)
        raw_weights.append(raw_weight)

    weights = np.asarray(raw_weights, dtype=np.float64)
    weights = weights / max(float(np.mean(weights)), 1e-12)
    for row, weight in zip(rows, weights):
        row["weight"] = float(weight)
    return rows, weights.astype(np.float32), thresholds


def save_rollout_weight_outputs(
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    weights: np.ndarray,
    thresholds: dict[str, float],
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "rollout_weights.npy"
    metadata_path = output_dir / "rollout_metadata.csv"
    summary_path = output_dir / "rollout_weight_summary.json"

    np.save(weights_path, weights)
    fieldnames = list(rows[0].keys()) if rows else []
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "num_rollouts": len(rows),
        "category_weights": CATEGORY_WEIGHTS,
        "thresholds": thresholds,
        "category_counts": category_counts(rows),
        "category_percentages": category_percentages(rows),
        "weight_mean": float(np.mean(weights)),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
    }
    _write_json(summary_path, summary)
    return weights_path, metadata_path, summary_path


def load_rollout_weights(weights_path: str | Path, expected_length: int | None = None) -> np.ndarray:
    weights = np.load(Path(weights_path)).astype(np.float32)
    if weights.ndim != 1:
        raise ValueError(f"Expected 1D rollout weights, got shape {weights.shape}")
    if expected_length is not None and len(weights) != expected_length:
        raise ValueError(
            f"Rollout weights length {len(weights)} does not match dataset length {expected_length}"
        )
    return weights


def load_rollout_metadata(metadata_path: str | Path) -> list[dict[str, Any]]:
    path = Path(metadata_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def make_weighted_sampler(weights: np.ndarray, seed: int | None = None) -> WeightedRandomSampler:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def category_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row["category"] for row in rows)
    return {category: int(counts.get(category, 0)) for category in CATEGORY_ORDER}


def category_percentages(rows: list[dict[str, Any]]) -> dict[str, float]:
    counts = category_counts(rows)
    total = max(len(rows), 1)
    return {category: 100.0 * count / total for category, count in counts.items()}


def simulate_weighted_sampling(
    rows: list[dict[str, Any]],
    weights: np.ndarray,
    num_samples: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    num_samples = len(weights) if num_samples is None else num_samples
    rng = np.random.default_rng(seed)
    probabilities = weights.astype(np.float64)
    probabilities = probabilities / probabilities.sum()
    sampled_indices = rng.choice(len(weights), size=num_samples, replace=True, p=probabilities)
    sampled_categories = [rows[int(index)]["category"] for index in sampled_indices]
    sampled_counts = Counter(sampled_categories)
    sampled_percentages = {
        category: 100.0 * sampled_counts.get(category, 0) / max(num_samples, 1)
        for category in CATEGORY_ORDER
    }
    return {
        "num_samples": int(num_samples),
        "sampled_indices": sampled_indices.astype(int).tolist(),
        "sampled_counts": {
            category: int(sampled_counts.get(category, 0)) for category in CATEGORY_ORDER
        },
        "sampled_percentages": sampled_percentages,
        "average_sampled_weight": float(np.mean(weights[sampled_indices])),
    }


def save_sampling_distribution_plot(
    path: str | Path,
    original_percentages: dict[str, float],
    sampled_percentages: dict[str, float],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = CATEGORY_ORDER
    x = np.arange(len(labels))
    width = 0.38
    original = [original_percentages.get(label, 0.0) for label in labels]
    sampled = [sampled_percentages.get(label, 0.0) for label in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, original, width, label="original", color="#80b1d3")
    ax.bar(x + width / 2, sampled, width, label="weighted sampler", color="#fb8072")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Percent")
    ax.set_title("Rollout Sampling Distribution")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
