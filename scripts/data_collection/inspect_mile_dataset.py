#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data import category_percentages, compute_rollout_weight_table
from src.datasets import CarlaRolloutDataset


REQUIRED_COLUMNS = ["image_path", "action", "speed"]
EXPECTED_DIRS = ["image", "birdview", "routemap"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a MILE dataset root for teleop_wm compatibility.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/data_diagnostics/mile_dataset_inspection.json"))
    parser.add_argument("--normalize-controls", action="store_true")
    parser.add_argument("--speed-scale", type=float, default=20.0)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--rollout-split", default=None, choices=["train", "val", "validation", "test"])
    return parser.parse_args()


def find_run_dirs(root: Path) -> list[Path]:
    root = root.expanduser()
    if (root / "pd_dataframe.pkl").exists():
        return [root]
    run_dirs = sorted(path.parent for path in root.rglob("pd_dataframe.pkl"))
    return run_dirs


def to_array_series(series: pd.Series) -> np.ndarray:
    return np.stack(series.map(lambda value: np.asarray(value, dtype=np.float32)).to_numpy())


def inspect_run(run_dir: Path) -> dict[str, Any]:
    df_path = run_dir / "pd_dataframe.pkl"
    df = pd.read_pickle(df_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    missing_dirs = [name for name in EXPECTED_DIRS if not (run_dir / name).exists()]
    image_count = len(list((run_dir / "image").glob("*.png"))) if (run_dir / "image").exists() else 0
    row_count = len(df)
    image_paths_exist = True
    if "image_path" in df.columns:
        image_paths_exist = all((run_dir / rel_path).exists() for rel_path in df["image_path"].head(20))

    action_stats = {}
    speed_stats = {}
    nan_counts = {}
    if "action" in df.columns:
        actions = to_array_series(df["action"])
        nan_counts["action"] = int(np.isnan(actions).sum())
        action_stats = {
            "shape": list(actions.shape),
            "min": actions.min(axis=0).astype(float).tolist(),
            "max": actions.max(axis=0).astype(float).tolist(),
            "mean": actions.mean(axis=0).astype(float).tolist(),
            "std": actions.std(axis=0).astype(float).tolist(),
        }
    if "speed" in df.columns:
        speed = to_array_series(df["speed"])
        if speed.ndim == 1:
            speed = speed[:, None]
        nan_counts["speed"] = int(np.isnan(speed).sum())
        speed_stats = {
            "shape": list(speed.shape),
            "min": speed.min(axis=0).astype(float).tolist(),
            "max": speed.max(axis=0).astype(float).tolist(),
            "mean": speed.mean(axis=0).astype(float).tolist(),
            "std": speed.std(axis=0).astype(float).tolist(),
        }

    return {
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "town": run_dir.parent.name,
        "row_count": row_count,
        "image_count": image_count,
        "row_image_count_match": row_count == image_count,
        "image_paths_exist_sample": image_paths_exist,
        "missing_columns": missing_columns,
        "missing_dirs": missing_dirs,
        "columns": list(df.columns),
        "action_stats": action_stats,
        "speed_stats": speed_stats,
        "nan_counts": nan_counts,
    }


def try_rollout_stats(root: Path, split: str | None, normalize_controls: bool, speed_scale: float) -> dict[str, Any] | None:
    if split is None:
        return None
    try:
        dataset = CarlaRolloutDataset(
            root,
            split=split,
            normalize_controls=normalize_controls,
            speed_scale=speed_scale,
        )
        rows, _, thresholds = compute_rollout_weight_table(dataset)
        return {
            "split": split,
            "num_rollouts": len(dataset),
            "category_percentages": category_percentages(rows),
            "thresholds": thresholds,
        }
    except Exception as exc:
        return {"error": str(exc), "split": split}


def main() -> int:
    args = parse_args()
    run_dirs = find_run_dirs(args.data_root)
    if args.max_runs is not None:
        run_dirs = run_dirs[: args.max_runs]
    run_reports = [inspect_run(run_dir) for run_dir in run_dirs]
    compatibility_failures = [
        report
        for report in run_reports
        if report["missing_columns"]
        or report["missing_dirs"]
        or not report["row_image_count_match"]
        or not report["image_paths_exist_sample"]
        or any(count > 0 for count in report["nan_counts"].values())
    ]
    town_counts = Counter(report["town"] for report in run_reports)
    payload = {
        "data_root": str(args.data_root.expanduser()),
        "num_runs": len(run_reports),
        "town_counts": dict(town_counts),
        "num_compatibility_failures": len(compatibility_failures),
        "compatibility_failures": compatibility_failures[:20],
        "runs": run_reports,
        "rollout_stats": try_rollout_stats(
            args.data_root.expanduser(),
            args.rollout_split,
            args.normalize_controls,
            args.speed_scale,
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"data_root: {args.data_root.expanduser()}")
    print(f"runs inspected: {len(run_reports)}")
    print(f"town counts: {dict(town_counts)}")
    print(f"compatibility failures: {len(compatibility_failures)}")
    print(f"wrote report: {args.output_json}")
    return 0 if not compatibility_failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
