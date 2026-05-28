#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from paper_utils import load_json, save_figure, setup_matplotlib


METRICS = {
    "latency": ("latency_ms_per_batch", "Latency (ms / batch)"),
    "fps": ("samples_per_second", "Samples / second"),
    "future_fps": ("future_frames_per_second", "Predicted frames / second"),
    "vram": ("peak_vram_mb", "Peak VRAM (MB)"),
    "params": ("parameters", "Parameters"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready benchmark plots.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--run-names", nargs="+", default=None)
    parser.add_argument("--metric", choices=METRICS.keys(), default="latency")
    parser.add_argument("--output", type=Path, default=Path("paper/figures/generated/benchmark.pdf"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--font-size", type=int, default=9)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    names = args.run_names or [path.parent.parent.name for path in args.inputs]
    if len(names) != len(args.inputs):
        raise ValueError("--run-names length must match --inputs length")
    key, ylabel = METRICS[args.metric]
    values = [float(load_json(path).get(key, 0.0)) for path in args.inputs]
    if args.metric == "params":
        values = [value / 1e6 for value in values]
        ylabel = "Parameters (M)"

    plt = setup_matplotlib(args.font_size)
    fig, ax = plt.subplots(figsize=(max(3.2, 0.7 * len(names)), 2.4))
    ax.bar(range(len(names)), values, color="#4c78a8")
    ax.set_xticks(range(len(names)), names, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, args.output, dpi=args.dpi)
    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
