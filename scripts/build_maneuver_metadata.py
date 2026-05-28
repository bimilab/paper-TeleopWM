#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.maneuver_metadata import build_maneuver_rows, summarize_maneuver_rows, write_json
from src.datasets import CarlaRolloutDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build heading-change maneuver metadata for rollout windows.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maneuver_metadata"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--past-len", type=int, default=9)
    parser.add_argument("--future-len", type=int, default=8)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--heading-field", default=None, help="Optional heading column spec, e.g. heading or imu[6].")
    parser.add_argument("--straight-threshold-deg", type=float, default=3.0)
    parser.add_argument("--sharp-threshold-deg", type=float, default=10.0)
    parser.add_argument(
        "--speed-delta-threshold",
        type=float,
        default=0.3,
        help=(
            "Speed delta threshold for longitudinal labels. delta_speed is "
            "mean(future_speed) - mean(past_speed), using the raw dataset speed."
        ),
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    dataset = CarlaRolloutDataset(
        args.data_root,
        split=args.split,
        past_len=args.past_len,
        future_len=args.future_len,
        include_metadata=False,
    )
    rows, info = build_maneuver_rows(
        dataset,
        fps=args.fps,
        heading_field=args.heading_field,
        straight_threshold_deg=args.straight_threshold_deg,
        sharp_threshold_deg=args.sharp_threshold_deg,
        speed_delta_threshold=args.speed_delta_threshold,
        progress=not args.no_progress,
    )
    summary = summarize_maneuver_rows(rows, info)

    csv_path = args.output_dir / "maneuver_metadata.csv"
    stats_path = args.output_dir / "stats.json"
    write_csv(csv_path, rows)
    write_json(stats_path, summary)

    print(f"dataset layout: {dataset.layout}")
    print(f"runs: {len(dataset.run_infos)}")
    print(f"windows: {len(dataset)}")
    print(f"heading source counts: {summary['heading_sources']}")
    print("maneuver counts:", summary["maneuver_counts"])
    print("maneuver percentages:", summary["maneuver_percentages"])
    print("longitudinal counts:", summary["longitudinal_counts"])
    print("longitudinal percentages:", summary["longitudinal_percentages"])
    print("maneuver-speed counts:", summary["maneuver_speed_counts"])
    print("maneuver-speed percentages:", summary["maneuver_speed_percentages"])
    print(f"speed delta definition: {summary['speed_delta_definition']}")
    print(f"speed delta threshold: {summary['speed_delta_threshold']}")
    for label, stats in summary["class_stats"].items():
        print(
            f"{label}: mean_speed={stats['mean_speed']:.4f} "
            f"mean_abs_yaw_rate={stats['mean_abs_yaw_rate']:.4f} deg/s "
            f"mean_abs_heading_change={stats['mean_abs_heading_change_deg']:.4f} deg"
        )
    print(f"wrote: {csv_path}")
    print(f"wrote: {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
