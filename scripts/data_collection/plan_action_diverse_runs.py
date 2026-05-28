#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_TOWNS = ["Town01", "Town03", "Town04"]
DEFAULT_WEATHERS = ["ClearNoon", "WetNoon", "ClearSunset"]
OBJECTIVES = [
    ("turn_intersections", "left/right turns through intersections"),
    ("braking_into_turn", "deceleration before turning"),
    ("accelerate_after_turn", "longitudinal transition after steering event"),
    ("stop_and_go", "traffic-light or actor-induced stopping and restart"),
    ("speed_transition", "visible acceleration/deceleration on straight segment"),
    ("mixed_dynamic", "combined lateral and longitudinal change"),
    ("lane_follow_reference", "steady lane following reference run"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a proposed action-diverse MILE collection manifest.")
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/data_collection/action_diverse_manifest.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/data_collection/action_diverse_manifest.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("/path/to/action_diverse_mile"))
    parser.add_argument("--towns", nargs="+", default=DEFAULT_TOWNS)
    parser.add_argument("--weathers", nargs="+", default=DEFAULT_WEATHERS)
    parser.add_argument("--routes-per-town", type=int, default=8)
    parser.add_argument("--runs-per-route", type=int, default=3)
    parser.add_argument("--traffic-density", default="mile_default")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    run_counter = 0
    for town in args.towns:
        for route_id in range(args.routes_per_town):
            for repeat in range(args.runs_per_route):
                objective, expected_behavior = OBJECTIVES[(route_id + repeat) % len(OBJECTIVES)]
                weather = args.weathers[(route_id + repeat) % len(args.weathers)]
                run_id = f"{run_counter:05d}"
                rows.append(
                    {
                        "run_id": run_id,
                        "town": town,
                        "route": route_id,
                        "weather": weather,
                        "traffic_density": args.traffic_density,
                        "objective_tag": objective,
                        "expected_behavior": expected_behavior,
                        "output_path": str(args.dataset_root / "train" / town / run_id),
                    }
                )
                run_counter += 1

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_root": str(args.dataset_root),
                "towns": args.towns,
                "weathers": args.weathers,
                "routes_per_town": args.routes_per_town,
                "runs_per_route": args.runs_per_route,
                "total_runs": len(rows),
                "rows": rows,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    print(f"wrote CSV: {args.output_csv}")
    print(f"wrote JSON: {args.output_json}")
    print(f"total planned runs: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
