#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.plotting import plot_training_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves from metrics.jsonl.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--metrics-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = plot_training_curves(
        run_dir=args.run_dir,
        metrics_jsonl=args.metrics_jsonl,
        output_dir=args.output_dir,
    )
    print("Generated training plots:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
