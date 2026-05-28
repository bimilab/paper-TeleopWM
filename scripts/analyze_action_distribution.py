#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.analysis import ActionDistributionAnalyzer, ActionDistributionStats
from src.analysis.action_distribution import discover_run_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze frame-level MILE action/speed distributions directly from pd_dataframe.pkl files."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--val-data-root", type=Path, default=None)
    parser.add_argument("--test-data-root", type=Path, default=None)
    parser.add_argument("--normalize-controls", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--speed-scale", type=float, default=20.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/action_distribution"))
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--action-column", default="action")
    parser.add_argument("--speed-column", default="speed")
    return parser.parse_args()


def maybe_tqdm(iterable, total: int | None = None, desc: str = ""):
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)
    except Exception:
        return iterable


def analyze_root(
    split_name: str,
    root: Path,
    args: argparse.Namespace,
) -> tuple[dict, ActionDistributionAnalyzer]:
    run_dirs = discover_run_dirs(root)
    if args.max_runs is not None:
        run_dirs = run_dirs[: args.max_runs]
    analyzer = ActionDistributionAnalyzer(
        split_name=split_name,
        action_column=args.action_column,
        speed_column=args.speed_column,
        normalize_controls=args.normalize_controls,
        speed_scale=args.speed_scale,
    )
    for run_dir in maybe_tqdm(run_dirs, total=len(run_dirs), desc=split_name):
        analyzer.update_run(run_dir)
    payload = analyzer.compute()
    payload.update(
        {
            "data_root": str(root),
            "run_count": len(run_dirs),
            "max_runs": args.max_runs,
        }
    )
    return payload, analyzer


def merge_analyzers(split_name: str, analyzers: list[ActionDistributionAnalyzer]) -> ActionDistributionAnalyzer:
    combined = ActionDistributionAnalyzer(split_name=split_name)
    for analyzer in analyzers:
        for name, chunks in analyzer.stats.values.items():
            combined.stats.values[name].extend(chunks)
        combined.stats.runs += analyzer.stats.runs
        combined.stats.frames += analyzer.stats.frames
        combined.run_paths.extend(analyzer.run_paths)
    return combined


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_specs = [("train", args.data_root)]
    if args.val_data_root is not None:
        split_specs.append(("val", args.val_data_root))
    if args.test_data_root is not None:
        split_specs.append(("test", args.test_data_root))

    summaries = {}
    analyzers = []
    plots = []
    for split_name, root in split_specs:
        payload, analyzer = analyze_root(split_name, root, args)
        summaries[split_name] = payload
        analyzers.append(analyzer)
        plots.extend(analyzer.plot_histograms(args.output_dir))

    combined = merge_analyzers("all", analyzers)
    summaries["all"] = combined.compute()
    plots.extend(combined.plot_histograms(args.output_dir))

    output = {
        "config": {
            "normalize_controls": args.normalize_controls,
            "speed_scale": args.speed_scale,
            "max_runs": args.max_runs,
            "action_column": args.action_column,
            "speed_column": args.speed_column,
            "data_root": str(args.data_root),
            "val_data_root": str(args.val_data_root) if args.val_data_root is not None else None,
            "test_data_root": str(args.test_data_root) if args.test_data_root is not None else None,
            "analysis_mode": "frame_level_dataframe_scan_no_images",
        },
        "splits": summaries,
        "plots": plots,
    }

    summary_json = args.output_dir / "action_distribution_summary.json"
    summary_csv = args.output_dir / "action_distribution_summary.csv"
    ActionDistributionStats().save_json(summary_json, output)
    ActionDistributionStats().save_csv(summary_csv, summaries)

    print(f"wrote json: {summary_json}")
    print(f"wrote csv: {summary_csv}")
    for split_name, payload in summaries.items():
        scales = payload["recommended_scales"]
        long_stats = payload["variables"]["longitudinal"]
        steer_stats = payload["variables"]["converted_steer"]
        speed_stats = payload["variables"]["speed"]
        print(f"{split_name}: runs={payload['runs']} frames={payload['frames']}")
        print(
            f"  longitudinal mean={long_stats['mean']:.6f} std={long_stats['std']:.6f} "
            f"abs_mean={long_stats['abs_mean']:.6f} p99_abs={long_stats['abs_percentiles']['p99']:.6f}"
        )
        print(
            f"  steer mean={steer_stats['mean']:.6f} std={steer_stats['std']:.6f} "
            f"abs_mean={steer_stats['abs_mean']:.6f} p99_abs={steer_stats['abs_percentiles']['p99']:.6f}"
        )
        print(
            f"  speed mean={speed_stats['mean']:.6f} std={speed_stats['std']:.6f} "
            f"p99={speed_stats['percentiles']['p99']:.6f}"
        )
        print(f"  recommended_scales={json.dumps(scales, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
