from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.models import controls_to_longitudinal_steer_speed


PERCENTILES = (0.1, 1, 5, 25, 50, 75, 95, 99, 99.5, 99.9)
NEAR_ZERO_THRESHOLDS = (0.01, 0.03, 0.05, 0.10)


@dataclass
class ActionDistributionStats:
    """Accumulate frame-level action arrays directly from MILE dataframes."""

    values: dict[str, list[np.ndarray]] = field(default_factory=lambda: {
        "raw_throttle": [],
        "raw_brake": [],
        "raw_steer": [],
        "longitudinal": [],
        "converted_steer": [],
        "speed": [],
    })
    runs: int = 0
    frames: int = 0

    def update_dataframe(
        self,
        dataframe: pd.DataFrame,
        action_column: str = "action",
        speed_column: str = "speed",
        normalize_controls: bool = False,
        speed_scale: float = 20.0,
    ) -> None:
        actions = extract_actions(dataframe, action_column=action_column)
        speed = extract_speed(dataframe, speed_column=speed_column)
        if normalize_controls:
            if speed_scale <= 0:
                raise ValueError(f"speed_scale must be positive, got {speed_scale}")
            speed = speed / float(speed_scale)

        action_tensor = torch.from_numpy(actions.astype(np.float32))
        speed_tensor = torch.from_numpy(speed.astype(np.float32))
        converted = controls_to_longitudinal_steer_speed(action_tensor.unsqueeze(0), speed_tensor.unsqueeze(0))
        converted_np = converted.squeeze(0).detach().cpu().numpy()

        self.values["raw_throttle"].append(actions[:, 0].astype(np.float64))
        self.values["raw_steer"].append(actions[:, 1].astype(np.float64))
        self.values["raw_brake"].append(actions[:, 2].astype(np.float64))
        self.values["longitudinal"].append(converted_np[:, 0].astype(np.float64))
        self.values["converted_steer"].append(converted_np[:, 1].astype(np.float64))
        self.values["speed"].append(converted_np[:, 2].astype(np.float64))
        self.runs += 1
        self.frames += int(actions.shape[0])

    def update_arrays(self, actions: np.ndarray, speed: np.ndarray) -> None:
        dataframe = pd.DataFrame({"action": list(actions), "speed": list(speed)})
        self.update_dataframe(dataframe)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            name: np.concatenate(chunks, axis=None).astype(np.float64)
            if chunks
            else np.asarray([], dtype=np.float64)
            for name, chunks in self.values.items()
        }

    def compute(self) -> dict[str, Any]:
        arrays = self.arrays()
        summary: dict[str, Any] = {
            "runs": self.runs,
            "frames": self.frames,
            "variables": {
                name: describe_array(values)
                for name, values in arrays.items()
                if values.size > 0
            },
        }
        steer_abs = np.abs(arrays["converted_steer"])
        long_abs = np.abs(arrays["longitudinal"])
        summary["recommended_scales"] = {
            "steer_scale_absmax": safe_float(np.max(steer_abs)) if steer_abs.size else None,
            "steer_scale_p99": percentile(steer_abs, 99),
            "steer_scale_p99_5": percentile(steer_abs, 99.5),
            "longitudinal_scale_absmax": safe_float(np.max(long_abs)) if long_abs.size else None,
            "longitudinal_scale_p99": percentile(long_abs, 99),
            "longitudinal_scale_p99_5": percentile(long_abs, 99.5),
        }
        return summary

    def save_json(self, path: str | Path, payload: dict[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def save_csv(self, path: str | Path, summaries: dict[str, dict[str, Any]]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for split, payload in summaries.items():
            for variable, stats in payload["variables"].items():
                row = {"split": split, "variable": variable}
                row.update(flatten_dict(stats))
                rows.append(row)
        if not rows:
            return
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def plot_histograms(self, output_dir: str | Path, split_name: str) -> list[str]:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for name, values in self.arrays().items():
            if values.size == 0:
                continue
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            ax.hist(values, bins=100, color="#4C78A8", alpha=0.85)
            ax.axvline(0.0, color="#333333", linewidth=1.0)
            ax.set_title(f"{split_name}: {name}")
            ax.set_xlabel(name)
            ax.set_ylabel("Frame count")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            filename = f"{name}_hist.png" if split_name in {"", "train"} else f"{split_name}_{name}_hist.png"
            path = output_dir / filename
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(str(path))
        return paths


class ActionDistributionAnalyzer:
    """Analyze frame-level action distribution by reading pd_dataframe.pkl files."""

    def __init__(
        self,
        split_name: str,
        action_column: str = "action",
        speed_column: str = "speed",
        normalize_controls: bool = False,
        speed_scale: float = 20.0,
    ) -> None:
        self.split_name = split_name
        self.action_column = action_column
        self.speed_column = speed_column
        self.normalize_controls = normalize_controls
        self.speed_scale = speed_scale
        self.stats = ActionDistributionStats()
        self.run_paths: list[str] = []

    def update_dataframe(self, dataframe: pd.DataFrame, run_path: str | Path | None = None) -> None:
        self.stats.update_dataframe(
            dataframe,
            action_column=self.action_column,
            speed_column=self.speed_column,
            normalize_controls=self.normalize_controls,
            speed_scale=self.speed_scale,
        )
        if run_path is not None:
            self.run_paths.append(str(run_path))

    def update_run(self, run_dir: str | Path) -> None:
        run_dir = Path(run_dir)
        dataframe = pd.read_pickle(run_dir / "pd_dataframe.pkl")
        self.update_dataframe(dataframe, run_path=run_dir)

    def compute(self) -> dict[str, Any]:
        payload = self.stats.compute()
        payload.update(
            {
                "split": self.split_name,
                "run_paths": self.run_paths,
                "action_column": self.action_column,
                "speed_column": self.speed_column,
                "normalize_controls": self.normalize_controls,
                "speed_scale": self.speed_scale,
            }
        )
        return payload

    def save_json(self, path: str | Path, payload: dict[str, Any]) -> None:
        self.stats.save_json(path, payload)

    def save_csv(self, path: str | Path, summaries: dict[str, dict[str, Any]]) -> None:
        self.stats.save_csv(path, summaries)

    def plot_histograms(self, output_dir: str | Path) -> list[str]:
        return self.stats.plot_histograms(output_dir, self.split_name)


def discover_run_dirs(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(path.parent for path in root.rglob("pd_dataframe.pkl") if (path.parent / "pd_dataframe.pkl").is_file())


def extract_actions(dataframe: pd.DataFrame, action_column: str = "action") -> np.ndarray:
    if action_column in dataframe.columns:
        actions = np.stack(
            dataframe[action_column].map(lambda value: np.asarray(value, dtype=np.float32)).to_numpy()
        )
    elif {"throttle", "steer", "brake"}.issubset(dataframe.columns):
        actions = dataframe[["throttle", "steer", "brake"]].to_numpy(dtype=np.float32)
    else:
        raise KeyError(
            f"Could not find action column {action_column!r} or throttle/steer/brake columns. "
            f"Available columns: {list(dataframe.columns)}"
        )
    if actions.ndim != 2 or actions.shape[1] < 3:
        raise ValueError(f"Expected actions [N,3+] with [throttle, steer, brake], got {actions.shape}")
    return actions[:, :3]


def extract_speed(dataframe: pd.DataFrame, speed_column: str = "speed") -> np.ndarray:
    if speed_column not in dataframe.columns:
        raise KeyError(f"Missing speed column {speed_column!r}. Available columns: {list(dataframe.columns)}")
    speed = np.stack(
        dataframe[speed_column].map(lambda value: np.asarray(value, dtype=np.float32)).to_numpy()
    )
    if speed.ndim == 1:
        speed = speed[:, None]
    if speed.ndim != 2 or speed.shape[1] != 1:
        raise ValueError(f"Expected speed [N,1], got {speed.shape}")
    return speed


def describe_array(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"count": 0}
    abs_values = np.abs(values)
    return {
        "count": int(values.size),
        "min": safe_float(np.min(values)),
        "max": safe_float(np.max(values)),
        "mean": safe_float(np.mean(values)),
        "std": safe_float(np.std(values)),
        "median": safe_float(np.median(values)),
        "abs_mean": safe_float(np.mean(abs_values)),
        "abs_max": safe_float(np.max(abs_values)),
        "percentiles": {f"p{str(p).replace('.', '_')}": percentile(values, p) for p in PERCENTILES},
        "abs_percentiles": {f"p{str(p).replace('.', '_')}": percentile(abs_values, p) for p in PERCENTILES},
        "fraction_near_zero": {
            f"abs_lt_{threshold:g}": safe_float(np.mean(abs_values < threshold))
            for threshold in NEAR_ZERO_THRESHOLDS
        },
    }


def percentile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return safe_float(np.percentile(values, q))


def safe_float(value: Any) -> float:
    return float(np.asarray(value).item())


def flatten_dict(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, name))
        elif isinstance(value, list):
            flattened[name] = json.dumps(value)
        else:
            flattened[name] = value
    return flattened
