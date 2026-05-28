#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from paper_utils import load_json, save_figure, setup_matplotlib


LABELS = ["straight", "mild_turn", "sharp_turn"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready maneuver sampling figures.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("paper/figures/generated/sampling.pdf"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--font-size", type=int, default=9)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_json(args.input_json)
    original = [float(data.get("original_maneuver_percentages", {}).get(label, 0.0)) for label in LABELS]
    sampled = [float(data.get("sampled_maneuver_percentages", {}).get(label, 0.0)) for label in LABELS]

    plt = setup_matplotlib(args.font_size)
    x = np.arange(len(LABELS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    ax.bar(x - width / 2, original, width, label="Original", color="#4c78a8")
    ax.bar(x + width / 2, sampled, width, label="Sampled", color="#f58518")
    ax.set_xticks(x, [label.replace("_", " ") for label in LABELS])
    ax.set_ylabel("Rollouts (%)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, args.output, dpi=args.dpi)
    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
