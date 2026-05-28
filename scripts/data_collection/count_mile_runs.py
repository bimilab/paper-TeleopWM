#!/usr/bin/env python3
"""Count completed MILE run folders by town."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def count_images(image_dir: Path) -> int:
    if not image_dir.is_dir():
        return 0
    return sum(1 for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def dataframe_rows(pkl_path: Path) -> int | None:
    if not pkl_path.is_file():
        return None
    try:
        import pandas as pd

        return int(len(pd.read_pickle(pkl_path)))
    except Exception:
        return None


def inspect_run(run_dir: Path, min_frames: int, expected_frames: int, expected_tolerance: float) -> dict[str, Any]:
    image_dir = run_dir / "image"
    pkl_path = run_dir / "pd_dataframe.pkl"
    image_count = count_images(image_dir)
    row_count = dataframe_rows(pkl_path)
    has_dataframe = pkl_path.is_file()
    has_images = image_dir.is_dir()
    min_complete = has_dataframe and has_images and image_count >= min_frames
    lower_expected = int(expected_frames * (1.0 - expected_tolerance))
    upper_expected = int(expected_frames * (1.0 + expected_tolerance))
    near_expected = has_dataframe and has_images and lower_expected <= image_count <= upper_expected
    rows_match_images = row_count is None or row_count == image_count
    return {
        "run_id": run_dir.name,
        "path": str(run_dir),
        "has_dataframe": has_dataframe,
        "has_image_dir": has_images,
        "image_count": image_count,
        "dataframe_rows": row_count,
        "rows_match_images": rows_match_images,
        "min_complete": min_complete,
        "near_expected_frames": near_expected,
    }


def inspect_dataset(
    dataset_root: Path,
    towns: list[str] | None,
    min_frames: int,
    expected_frames: int,
    expected_tolerance: float,
) -> dict[str, Any]:
    if towns is None:
        towns = sorted(path.name for path in dataset_root.glob("Town*") if path.is_dir())

    summary: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "min_frames": min_frames,
        "expected_frames": expected_frames,
        "expected_tolerance": expected_tolerance,
        "towns": {},
    }
    for town in towns:
        town_dir = dataset_root / town
        run_dirs = sorted(path for path in town_dir.iterdir() if path.is_dir()) if town_dir.is_dir() else []
        runs = [inspect_run(path, min_frames, expected_frames, expected_tolerance) for path in run_dirs]
        summary["towns"][town] = {
            "town_dir": str(town_dir),
            "total_run_dirs": len(runs),
            "complete_runs": sum(1 for item in runs if item["min_complete"]),
            "near_expected_runs": sum(1 for item in runs if item["near_expected_frames"]),
            "runs": runs,
        }
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print(f"Dataset root: {summary['dataset_root']}")
    print(f"Completion rule: pd_dataframe.pkl + image/ + >= {summary['min_frames']} frames")
    print("")
    print(f"{'Town':<8} {'complete':>9} {'4500-ish':>9} {'run_dirs':>8}")
    print("-" * 40)
    for town, info in summary["towns"].items():
        print(
            f"{town:<8} "
            f"{info['complete_runs']:>9} "
            f"{info['near_expected_runs']:>9} "
            f"{info['total_run_dirs']:>8}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Root containing TownXX folders.")
    parser.add_argument("--towns", nargs="*", default=None, help="Optional towns to inspect.")
    parser.add_argument("--min-frames", type=int, default=18, help="Minimum frames for a usable rollout run.")
    parser.add_argument("--expected-frames", type=int, default=4500, help="Nominal full MILE episode length.")
    parser.add_argument(
        "--expected-tolerance",
        type=float,
        default=0.20,
        help="Fractional tolerance for the 4500-ish full-run check.",
    )
    parser.add_argument("--json-out", default=None, help="Optional path for a JSON report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = inspect_dataset(
        dataset_root=Path(args.dataset_root).expanduser(),
        towns=args.towns,
        min_frames=args.min_frames,
        expected_frames=args.expected_frames,
        expected_tolerance=args.expected_tolerance,
    )
    print_summary(summary)
    if args.json_out:
        out_path = Path(args.json_out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved JSON report: {out_path}")


if __name__ == "__main__":
    main()
