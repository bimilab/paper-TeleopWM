from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Subset, WeightedRandomSampler


MANEUVER_LABELS = ("straight", "mild_turn", "sharp_turn")
LONGITUDINAL_LABELS = ("accel", "const", "decel")
MANEUVER_SPEED_LABELS = tuple(
    f"{maneuver}_{longitudinal}"
    for maneuver in MANEUVER_LABELS
    for longitudinal in LONGITUDINAL_LABELS
)
MANEUVER_WEIGHTS = {
    "straight": 1.0,
    "mild_turn": 1.0,
    "sharp_turn": 1.0,
}
MANEUVER_SPEED_WEIGHTS = {label: 1.0 for label in MANEUVER_SPEED_LABELS}


def parse_maneuver_weights(text: str | None) -> dict[str, float]:
    weights = dict(MANEUVER_WEIGHTS)
    if not text:
        return weights
    for item in text.split(","):
        if not item.strip():
            continue
        label, value = item.split("=", 1)
        label = label.strip()
        if label not in weights:
            raise ValueError(f"Unknown maneuver label {label!r}. Expected one of {MANEUVER_LABELS}")
        weights[label] = float(value)
    return weights


def parse_maneuver_speed_weights(text: str | None) -> dict[str, float]:
    """Parse 3x3 lateral-longitudinal weights.

    Defaults are intentionally neutral. Users can provide experiment-specific
    weights without changing the metadata distribution or the existing
    three-class maneuver-balanced sampler.
    """

    weights = dict(MANEUVER_SPEED_WEIGHTS)
    if not text:
        return weights
    for item in text.split(","):
        if not item.strip():
            continue
        label, value = item.split("=", 1)
        label = label.strip()
        if label not in weights:
            raise ValueError(f"Unknown maneuver-speed label {label!r}. Expected one of {MANEUVER_SPEED_LABELS}")
        weights[label] = float(value)
    return weights


def load_maneuver_metadata(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _subset_indices(dataset) -> list[int] | None:
    if isinstance(dataset, Subset):
        return [int(index) for index in dataset.indices]
    return None


def align_maneuver_rows_to_dataset(rows: list[dict[str, Any]], dataset) -> list[dict[str, Any]]:
    if len(rows) == len(dataset):
        return rows
    indices = _subset_indices(dataset)
    if indices is not None and len(rows) >= max(indices, default=-1) + 1:
        return [rows[index] for index in indices]
    raise ValueError(
        f"Maneuver metadata has {len(rows)} rows, but dataset has {len(dataset)} samples. "
        "Regenerate metadata for the same data root/window settings or pass a matching file."
    )


def maneuver_weights_for_rows(rows: list[dict[str, Any]], maneuver_weights: dict[str, float]) -> np.ndarray:
    raw = []
    for row in rows:
        label = row.get("maneuver_label") or row.get("category")
        if label not in maneuver_weights:
            raise ValueError(f"Unknown maneuver label in metadata: {label!r}")
        raw.append(float(maneuver_weights[label]))
    weights = np.asarray(raw, dtype=np.float64)
    weights = weights / max(float(np.mean(weights)), 1e-12)
    return weights.astype(np.float32)


def maneuver_speed_label_for_row(row: dict[str, Any]) -> str:
    label = row.get("maneuver_speed_label") or row.get("combined_label")
    if label:
        return str(label)
    maneuver = row.get("maneuver_label") or row.get("category")
    longitudinal = row.get("longitudinal_label")
    if maneuver and longitudinal:
        return f"{maneuver}_{longitudinal}"
    raise ValueError(
        "Maneuver-speed metadata is missing combined labels. Regenerate metadata with "
        "scripts/build_maneuver_metadata.py so rows include maneuver_speed_label and longitudinal_label."
    )


def maneuver_speed_weights_for_rows(
    rows: list[dict[str, Any]],
    maneuver_speed_weights: dict[str, float],
) -> np.ndarray:
    raw = []
    for row in rows:
        label = maneuver_speed_label_for_row(row)
        if label not in maneuver_speed_weights:
            raise ValueError(f"Unknown maneuver-speed label in metadata: {label!r}")
        raw.append(float(maneuver_speed_weights[label]))
    weights = np.asarray(raw, dtype=np.float64)
    weights = weights / max(float(np.mean(weights)), 1e-12)
    return weights.astype(np.float32)


def make_maneuver_balanced_sampler(
    rows: list[dict[str, Any]],
    dataset,
    maneuver_weights: dict[str, float],
    seed: int | None = None,
) -> tuple[WeightedRandomSampler, np.ndarray, list[dict[str, Any]]]:
    """Create stage-2 maneuver-balanced DataLoader sampling weights.

    This operates on the dataset passed to the DataLoader, which may already be
    a subset selected by stage 1. It does not choose the candidate subset; it
    controls how examples are drawn during each training epoch.
    """

    aligned_rows = align_maneuver_rows_to_dataset(rows, dataset)
    weights = maneuver_weights_for_rows(aligned_rows, maneuver_weights)
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
    return sampler, weights, aligned_rows


def make_maneuver_speed_balanced_sampler(
    rows: list[dict[str, Any]],
    dataset,
    maneuver_speed_weights: dict[str, float],
    seed: int | None = None,
) -> tuple[WeightedRandomSampler, np.ndarray, list[dict[str, Any]]]:
    """Create stage-2 weighted sampling across lateral-longitudinal bins.

    This is distinct from subset selection: it draws training batches from the
    already selected candidate dataset using the 3x3 combined maneuver-speed
    labels stored in metadata.
    """

    aligned_rows = align_maneuver_rows_to_dataset(rows, dataset)
    weights = maneuver_speed_weights_for_rows(aligned_rows, maneuver_speed_weights)
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
    return sampler, weights, aligned_rows


def maneuver_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter((row.get("maneuver_label") or row.get("category")) for row in rows)
    return {label: int(counts.get(label, 0)) for label in MANEUVER_LABELS}


def maneuver_percentages(rows: list[dict[str, Any]]) -> dict[str, float]:
    counts = maneuver_counts(rows)
    total = max(len(rows), 1)
    return {label: 100.0 * count / total for label, count in counts.items()}


def maneuver_speed_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(maneuver_speed_label_for_row(row) for row in rows)
    return {label: int(counts.get(label, 0)) for label in MANEUVER_SPEED_LABELS}


def maneuver_speed_percentages(rows: list[dict[str, Any]]) -> dict[str, float]:
    counts = maneuver_speed_counts(rows)
    total = max(len(rows), 1)
    return {label: 100.0 * count / total for label, count in counts.items()}


def simulate_maneuver_sampling(
    rows: list[dict[str, Any]],
    weights: np.ndarray,
    num_samples: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    num_samples = len(weights) if num_samples is None else num_samples
    probabilities = weights.astype(np.float64)
    probabilities = probabilities / probabilities.sum()
    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(len(weights), size=num_samples, replace=True, p=probabilities)
    sampled_labels = [rows[int(index)].get("maneuver_label") or rows[int(index)].get("category") for index in sampled_indices]
    sampled_counts = Counter(sampled_labels)
    return {
        "num_samples": int(num_samples),
        "sampled_indices": sampled_indices.astype(int).tolist(),
        "sampled_counts": {label: int(sampled_counts.get(label, 0)) for label in MANEUVER_LABELS},
        "sampled_percentages": {
            label: 100.0 * sampled_counts.get(label, 0) / max(num_samples, 1)
            for label in MANEUVER_LABELS
        },
        "average_sampled_weight": float(np.mean(weights[sampled_indices])),
    }


def simulate_maneuver_speed_sampling(
    rows: list[dict[str, Any]],
    weights: np.ndarray,
    num_samples: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    num_samples = len(weights) if num_samples is None else num_samples
    probabilities = weights.astype(np.float64)
    probabilities = probabilities / probabilities.sum()
    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(len(weights), size=num_samples, replace=True, p=probabilities)
    sampled_labels = [maneuver_speed_label_for_row(rows[int(index)]) for index in sampled_indices]
    sampled_counts = Counter(sampled_labels)
    return {
        "num_samples": int(num_samples),
        "sampled_indices": sampled_indices.astype(int).tolist(),
        "sampled_counts": {label: int(sampled_counts.get(label, 0)) for label in MANEUVER_SPEED_LABELS},
        "sampled_percentages": {
            label: 100.0 * sampled_counts.get(label, 0) / max(num_samples, 1)
            for label in MANEUVER_SPEED_LABELS
        },
        "average_sampled_weight": float(np.mean(weights[sampled_indices])),
    }


def save_maneuver_sampling_plot(
    path: str | Path,
    original_percentages: dict[str, float],
    sampled_percentages: dict[str, float],
    selected_percentages: dict[str, float] | None = None,
    title: str = "Maneuver Sampling Distribution",
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(MANEUVER_LABELS)
    x = np.arange(len(labels))
    width = 0.26 if selected_percentages is not None else 0.38
    original = [original_percentages.get(label, 0.0) for label in labels]
    selected = [selected_percentages.get(label, 0.0) for label in labels] if selected_percentages is not None else None
    sampled = [sampled_percentages.get(label, 0.0) for label in labels]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if selected is None:
        ax.bar(x - width / 2, original, width, label="original", color="#80b1d3")
        ax.bar(x + width / 2, sampled, width, label="weighted sampler", color="#fb8072")
    else:
        ax.bar(x - width, original, width, label="full dataset", color="#80b1d3")
        ax.bar(x, selected, width, label="selected subset", color="#b3de69")
        ax.bar(x + width, sampled, width, label="weighted sampler", color="#fb8072")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Percent")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_maneuver_speed_sampling_plot(
    path: str | Path,
    original_percentages: dict[str, float],
    sampled_percentages: dict[str, float],
    selected_percentages: dict[str, float] | None = None,
    title: str = "Maneuver-Speed Sampling Distribution",
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(MANEUVER_SPEED_LABELS)
    x = np.arange(len(labels))
    width = 0.26 if selected_percentages is not None else 0.38
    original = [original_percentages.get(label, 0.0) for label in labels]
    selected = [selected_percentages.get(label, 0.0) for label in labels] if selected_percentages is not None else None
    sampled = [sampled_percentages.get(label, 0.0) for label in labels]

    fig, ax = plt.subplots(figsize=(12, 5.2))
    if selected is None:
        ax.bar(x - width / 2, original, width, label="original", color="#80b1d3")
        ax.bar(x + width / 2, sampled, width, label="weighted sampler", color="#fb8072")
    else:
        ax.bar(x - width, original, width, label="full dataset", color="#80b1d3")
        ax.bar(x, selected, width, label="selected subset", color="#b3de69")
        ax.bar(x + width, sampled, width, label="weighted sampler", color="#fb8072")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Percent")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
