#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot paper-quality future-action evaluation curves.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures/generated"))
    parser.add_argument("--output-name", default="future_action_eval")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--font-scale", type=float, default=1.0)
    parser.add_argument("--format", choices=["png", "pdf", "both"], default="both")
    parser.add_argument("--mae-ymax", type=float, default=1.0)
    parser.add_argument("--corr-ymin", type=float, default=0.0)
    parser.add_argument("--corr-ymax", type=float, default=1.0)
    return parser.parse_args()


def load_metrics(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "future_action_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing metrics JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def setup_style(font_scale: float) -> None:
    base = 9.5 * font_scale
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Nimbus Roman",
                "Liberation Serif",
                "TeX Gyre Termes",
                "DejaVu Serif",
            ],
            "font.size": base,
            "axes.titlesize": base * 1.05,
            "axes.labelsize": base,
            "xtick.labelsize": base * 0.92,
            "ytick.labelsize": base * 0.92,
            "legend.fontsize": base * 0.88,
            "figure.titlesize": base * 1.05,
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.45,
            "lines.linewidth": 1.7,
            "lines.markersize": 4.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def require_sequence(metrics: dict[str, Any], key: str, length: int = 8) -> list[float]:
    values = metrics.get(key)
    if not isinstance(values, list) or len(values) < length:
        raise ValueError(f"Expected {key} to contain at least {length} values")
    return [float(v) for v in values[:length]]


def corr_steps(metrics: dict[str, Any], prefix: str, length: int = 8) -> list[float]:
    values = []
    for step in range(1, length + 1):
        key = f"{prefix}_corr_step_{step}"
        if key not in metrics:
            raise ValueError(f"Missing metric: {key}")
        values.append(float(metrics[key]))
    return values


def fmt_metric(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def add_metric_annotation(ax: plt.Axes, text: str, *, x: float = 0.03, y: float = 0.96, alpha: float = 0.92) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.24",
            "facecolor": "white",
            "edgecolor": "0.78",
            "linewidth": 0.6,
            "alpha": alpha,
        },
    )


def plot(metrics: dict[str, Any], args: argparse.Namespace) -> plt.Figure:
    setup_style(args.font_scale)
    steps = list(range(1, 9))
    xlabels = [f"t+{step}" for step in steps]
    long_color = "#1f4e79"
    steer_color = "#b45f06"

    mae_long = require_sequence(metrics, "per_step_mae_longitudinal")
    mae_steer = require_sequence(metrics, "per_step_mae_steer")
    corr_long = corr_steps(metrics, "longitudinal")
    corr_steer = corr_steps(metrics, "steering")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.35), constrained_layout=True)

    ax = axes[0]
    ax.plot(steps, mae_long, marker="o", color=long_color, label="Longitudinal")
    ax.plot(steps, mae_steer, marker="s", color=steer_color, label="Steering")
    ax.set_title("(a) Per-step action error", loc="left")
    ax.set_ylabel("MAE")
    ax.set_xlabel("Future step")
    ax.set_ylim(0.0, args.mae_ymax)
    ax.set_xticks(steps, xlabels)
    ax.grid(True, axis="y", color="0.86")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.78))
    add_metric_annotation(
        ax,
        "Overall MAE\n"
        f"long: {fmt_metric(metrics.get('mae_longitudinal'))}\n"
        f"steer: {fmt_metric(metrics.get('mae_steer'))}",
    )

    ax = axes[1]
    ax.plot(steps, corr_long, marker="o", color=long_color, label="Longitudinal")
    ax.plot(steps, corr_steer, marker="s", color=steer_color, label="Steering")
    ax.set_title("(b) Per-step action correlation", loc="left")
    ax.set_ylabel("Pearson correlation")
    ax.set_xlabel("Future step")
    ax.set_ylim(args.corr_ymin, args.corr_ymax)
    ax.set_xticks(steps, xlabels)
    ax.grid(True, axis="y", color="0.86")
    ax.legend(frameon=False, loc="lower left")
    add_metric_annotation(
        ax,
        "Overall corr.\n"
        f"long: {fmt_metric(metrics.get('pearson_corr_longitudinal'))}\n"
        f"steer: {fmt_metric(metrics.get('pearson_corr_steer'))}",
        x=0.03,
        y=0.44,
        alpha=0.86,
    )
    return fig


def output_paths(args: argparse.Namespace) -> list[Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.format == "both":
        suffixes = ("pdf", "png")
    else:
        suffixes = (args.format,)
    return [args.output_dir / f"{args.output_name}.{suffix}" for suffix in suffixes]


def main() -> int:
    args = parse_args()
    metrics = load_metrics(args.input_dir)
    fig = plot(metrics, args)
    for path in output_paths(args):
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
