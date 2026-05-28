from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.datasets import CarlaRolloutDataset


MANEUVER_LABELS = ("straight", "mild_turn", "sharp_turn")
LONGITUDINAL_LABELS = ("accel", "const", "decel")
MANEUVER_SPEED_LABELS = tuple(
    f"{maneuver}_{longitudinal}"
    for maneuver in MANEUVER_LABELS
    for longitudinal in LONGITUDINAL_LABELS
)
SPEED_DELTA_DEFINITION = "mean_future_speed_minus_mean_past_speed"
SCALAR_HEADING_COLUMNS = (
    "heading",
    "yaw",
    "compass",
    "theta",
    "rotation_yaw",
    "vehicle_yaw",
)
VECTOR_HEADING_COLUMNS = (("imu", 6),)


def wrap_degrees(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle differences to [-180, 180)."""

    return (np.asarray(angle) + 180.0) % 360.0 - 180.0


def heading_to_degrees(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Heading array has no finite values")
    # MILE stores compass in radians in imu[6]. Scalar custom exports may be degrees.
    if float(np.nanmax(np.abs(finite))) <= (2.0 * np.pi + 0.25):
        return np.degrees(values)
    return values


def signed_cumulative_heading_change_deg(heading_deg: np.ndarray) -> float:
    """Return robust signed cumulative heading change over a window.

    We sum wrapped frame-to-frame heading deltas so transitions such as
    179 -> -179 degrees correctly contribute +2 degrees instead of -358.
    """

    if len(heading_deg) < 2:
        return 0.0
    deltas = wrap_degrees(np.diff(heading_deg))
    return float(np.sum(deltas))


def mean_abs_yaw_rate_deg_s(heading_deg: np.ndarray, fps: float) -> float:
    if len(heading_deg) < 2:
        return 0.0
    deltas = np.abs(wrap_degrees(np.diff(heading_deg)))
    return float(np.mean(deltas) * fps)


def choose_heading_series(df: pd.DataFrame, requested: str | None = None) -> tuple[np.ndarray, str]:
    """Find a heading/yaw-like series and return it in degrees.

    Preference is explicit scalar columns, then MILE's imu[6] compass field.
    """

    if requested:
        values = extract_heading_series(df, requested)
        return heading_to_degrees(values), requested

    lower_to_column = {column.lower(): column for column in df.columns}
    for candidate in SCALAR_HEADING_COLUMNS:
        if candidate in lower_to_column:
            column = lower_to_column[candidate]
            values = np.asarray(df[column].to_numpy(), dtype=np.float64)
            return heading_to_degrees(values), column

    for column, index in VECTOR_HEADING_COLUMNS:
        if column in df.columns:
            values = np.asarray(
                df[column].map(lambda value: np.asarray(value, dtype=np.float64)[index]).to_numpy(),
                dtype=np.float64,
            )
            return heading_to_degrees(values), f"{column}[{index}]"

    raise KeyError(
        "Could not find a heading field. Tried scalar columns "
        f"{SCALAR_HEADING_COLUMNS} and vector fields {VECTOR_HEADING_COLUMNS}. "
        f"Available columns: {list(df.columns)}"
    )


def extract_heading_series(df: pd.DataFrame, spec: str) -> np.ndarray:
    if "[" in spec and spec.endswith("]"):
        column, index_text = spec[:-1].split("[", 1)
        index = int(index_text)
        if column not in df.columns:
            raise KeyError(f"Missing heading column {column!r}")
        return np.asarray(
            df[column].map(lambda value: np.asarray(value, dtype=np.float64)[index]).to_numpy(),
            dtype=np.float64,
        )
    if spec not in df.columns:
        raise KeyError(f"Missing heading column {spec!r}")
    return np.asarray(df[spec].to_numpy(), dtype=np.float64)


def label_maneuver(abs_future_delta_deg: float, straight_threshold: float, sharp_threshold: float) -> str:
    if abs_future_delta_deg < straight_threshold:
        return "straight"
    if abs_future_delta_deg < sharp_threshold:
        return "mild_turn"
    return "sharp_turn"


def label_longitudinal_speed(delta_speed: float, threshold: float) -> str:
    """Classify rollout-level longitudinal dynamics from raw speed evolution."""

    if delta_speed > threshold:
        return "accel"
    if delta_speed < -threshold:
        return "decel"
    return "const"


def build_maneuver_rows(
    dataset: CarlaRolloutDataset,
    fps: float = 15.0,
    heading_field: str | None = None,
    straight_threshold_deg: float = 3.0,
    sharp_threshold_deg: float = 10.0,
    speed_delta_threshold: float = 0.3,
    progress: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one maneuver metadata row per rollout window.

    Maneuvers are labeled from future-horizon heading change, not absolute
    steering magnitude. This targets scene curvature over the exact prediction
    window, which is the distribution pressure relevant for action-conditioned
    world-model training.
    """

    heading_by_run: dict[str, np.ndarray] = {}
    heading_sources: dict[str, str] = {}
    for run in dataset.run_infos:
        df = pd.read_pickle(run.path / "pd_dataframe.pkl")
        heading_deg, source = choose_heading_series(df, heading_field)
        heading_by_run[run.run_key] = heading_deg
        heading_sources[run.run_key] = source

    iterator = range(len(dataset.windows))
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, desc="maneuver windows")
        except Exception:
            pass

    rows: list[dict[str, Any]] = []
    for dataset_idx in iterator:
        window = dataset.windows[dataset_idx]
        heading = heading_by_run[window.run_key][window.start : window.end]
        speed = dataset.speed_by_run[window.run_key][window.start : window.end, 0]

        future_start = dataset.past_len
        future_heading = heading[future_start:]
        past_speed = speed[:future_start]
        future_speed = speed[future_start:]
        full_delta = signed_cumulative_heading_change_deg(heading)
        future_delta = signed_cumulative_heading_change_deg(future_heading)
        mean_abs_yaw_rate = mean_abs_yaw_rate_deg_s(future_heading, fps=fps)
        label = label_maneuver(
            abs(future_delta),
            straight_threshold=straight_threshold_deg,
            sharp_threshold=sharp_threshold_deg,
        )
        past_mean_speed = float(np.mean(past_speed)) if len(past_speed) else 0.0
        future_mean_speed = float(np.mean(future_speed)) if len(future_speed) else 0.0
        delta_speed = future_mean_speed - past_mean_speed
        longitudinal_label = label_longitudinal_speed(delta_speed, speed_delta_threshold)
        combined_label = f"{label}_{longitudinal_label}"
        rows.append(
            {
                "dataset_idx": dataset_idx,
                "town": window.town or "",
                "run_key": window.run_key,
                "run_id": window.run_id,
                "run_path": str(window.run_path),
                "start_idx": window.start,
                "end_idx": window.end - 1,
                "future_start_idx": window.start + dataset.past_len,
                "future_end_idx": window.end - 1,
                "heading_field": heading_sources[window.run_key],
                "mean_speed": float(np.mean(speed)),
                "past_mean_speed": past_mean_speed,
                "future_mean_speed": future_mean_speed,
                "delta_speed": float(delta_speed),
                "speed_delta_definition": SPEED_DELTA_DEFINITION,
                "speed_delta_threshold": speed_delta_threshold,
                "mean_abs_yaw_rate": mean_abs_yaw_rate,
                "full_heading_change_deg": full_delta,
                "cumulative_heading_change_deg": future_delta,
                "abs_cumulative_heading_change_deg": abs(future_delta),
                "maneuver_label": label,
                "longitudinal_label": longitudinal_label,
                "maneuver_speed_label": combined_label,
                "combined_label": combined_label,
                "category": label,
            }
        )

    info = {
        "heading_sources": dict(Counter(heading_sources.values())),
        "fps": fps,
        "straight_threshold_deg": straight_threshold_deg,
        "sharp_threshold_deg": sharp_threshold_deg,
        "speed_delta_threshold": speed_delta_threshold,
        "speed_delta_definition": SPEED_DELTA_DEFINITION,
    }
    return rows, info


def summarize_maneuver_rows(rows: list[dict[str, Any]], info: dict[str, Any]) -> dict[str, Any]:
    counts = Counter(row["maneuver_label"] for row in rows)
    longitudinal_counts = Counter(row.get("longitudinal_label", "") for row in rows)
    combined_counts = Counter(row.get("maneuver_speed_label") or row.get("combined_label", "") for row in rows)
    total = max(len(rows), 1)
    class_stats: dict[str, dict[str, float]] = {}
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[row["maneuver_label"]].append(row)

    for label in MANEUVER_LABELS:
        label_rows = by_label.get(label, [])
        class_stats[label] = {
            "count": int(counts.get(label, 0)),
            "percentage": 100.0 * counts.get(label, 0) / total,
            "mean_speed": float(np.mean([r["mean_speed"] for r in label_rows])) if label_rows else 0.0,
            "mean_abs_yaw_rate": float(np.mean([r["mean_abs_yaw_rate"] for r in label_rows])) if label_rows else 0.0,
            "mean_abs_heading_change_deg": (
                float(np.mean([r["abs_cumulative_heading_change_deg"] for r in label_rows]))
                if label_rows
                else 0.0
            ),
        }

    return {
        "total_windows": len(rows),
        "maneuver_counts": {label: int(counts.get(label, 0)) for label in MANEUVER_LABELS},
        "maneuver_percentages": {
            label: 100.0 * counts.get(label, 0) / total for label in MANEUVER_LABELS
        },
        "longitudinal_counts": {label: int(longitudinal_counts.get(label, 0)) for label in LONGITUDINAL_LABELS},
        "longitudinal_percentages": {
            label: 100.0 * longitudinal_counts.get(label, 0) / total for label in LONGITUDINAL_LABELS
        },
        "maneuver_speed_counts": {
            label: int(combined_counts.get(label, 0)) for label in MANEUVER_SPEED_LABELS
        },
        "maneuver_speed_percentages": {
            label: 100.0 * combined_counts.get(label, 0) / total for label in MANEUVER_SPEED_LABELS
        },
        "class_stats": class_stats,
        **info,
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
