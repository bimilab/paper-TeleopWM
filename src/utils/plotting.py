from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any


CURVE_COLUMNS = [
    "epoch",
    "train_loss",
    "val_loss",
    "train_mae",
    "val_mae",
    "train_ssim",
    "val_ssim",
    "best_val_loss",
    "train_rgb_loss",
    "train_aux_loss",
    "train_total_loss",
    "val_rgb_loss",
    "val_aux_loss",
    "val_total_loss",
]

ACTION_HEAD_CONFIG_FIELDS = [
    "future_action_head_variant",
    "future_action_hidden_dim",
    "future_action_spatial_pooling",
    "future_action_spatial_grid",
    "future_action_token_dim",
    "future_action_future_motion_scale",
    "future_action_source",
    "future_action_detach_latents",
    "future_steer_target_scale",
    "control_steer_input_scale",
    "future_action_delta_loss",
    "future_action_delta_loss_weight",
    "future_action_delta_loss_type",
    "future_action_delta_longitudinal_weight",
    "future_action_delta_steer_weight",
    "future_action_corr_loss_weight",
    "future_action_cls_loss",
    "future_action_cls_weight",
    "future_action_longitudinal_cls_weight",
    "future_action_steer_cls_weight",
    "longitudinal_coast_threshold",
    "steer_straight_threshold",
]


def load_metrics_jsonl(metrics_jsonl: str | Path, run_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Load metrics.jsonl, sort by epoch, and keep the last record for duplicate epochs."""

    metrics_jsonl = Path(metrics_jsonl)
    run_dir_path = Path(run_dir) if run_dir is not None else metrics_jsonl.parent
    config = _load_run_config(run_dir_path)
    config_metadata = _action_head_config_metadata(config)
    by_epoch: dict[int, dict[str, Any]] = {}
    with metrics_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            epoch = int(record["epoch"])
            by_epoch[epoch] = record

    rows = []
    for epoch in sorted(by_epoch):
        record = by_epoch[epoch]
        train = record.get("train", {})
        val = record.get("val", {})
        row: dict[str, Any] = {
            "epoch": epoch,
            "best_val_loss": _float_or_none(record.get("best_val_loss")),
        }
        row.update(_flatten_metrics("train", train))
        row.update(_flatten_metrics("val", val))
        row.update(
            {
                "train_loss": _float_or_none(train.get("loss")),
                "val_loss": _float_or_none(val.get("loss")),
                "train_mae": _float_or_none(train.get("mae")),
                "val_mae": _float_or_none(val.get("mae")),
                "train_ssim": _float_or_none(train.get("ssim")),
                "val_ssim": _float_or_none(val.get("ssim")),
                "train_rgb_loss": _float_or_none(train.get("rgb_loss")),
                "train_aux_loss": _float_or_none(train.get("aux_loss")),
                "train_total_loss": _float_or_none(train.get("total_loss")),
                "val_rgb_loss": _float_or_none(val.get("rgb_loss")),
                "val_aux_loss": _float_or_none(val.get("aux_loss")),
                "val_total_loss": _float_or_none(val.get("total_loss")),
            }
        )
        row.update(config_metadata)
        rows.append(row)
    _merge_train_log_metrics(rows, run_dir_path / "train.log")
    _add_derived_ratios(rows)
    return rows


def plot_training_curves(
    run_dir: str | Path | None = None,
    metrics_jsonl: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Path | bool]:
    """Generate training diagnostic plots and CSV from a metrics.jsonl file."""

    if run_dir is None and metrics_jsonl is None:
        raise ValueError("Either run_dir or metrics_jsonl must be provided")

    run_dir_path = Path(run_dir) if run_dir is not None else None
    metrics_path = Path(metrics_jsonl) if metrics_jsonl is not None else run_dir_path / "metrics.jsonl"
    default_plot_root = run_dir_path if run_dir_path is not None else metrics_path.parent
    plot_dir = Path(output_dir) if output_dir is not None else default_plot_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = load_metrics_jsonl(metrics_path, run_dir=run_dir_path)
    csv_path = plot_dir / "training_curves.csv"
    diagnostics_dir = plot_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    _write_action_head_summary(rows, run_dir_path, diagnostics_dir / "action_head_config_summary.txt")
    write_curves_csv(rows, csv_path)

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "savefig.dpi": 160,
        }
    )

    shading = len(rows) >= 5
    loss_path = plot_dir / "loss_curves.png"
    mae_path = plot_dir / "val_mae.png"
    ssim_path = plot_dir / "val_ssim.png"
    summary_path = plot_dir / "training_summary.png"
    aux_path = plot_dir / "aux_loss_curves.png"

    epochs = [row["epoch"] for row in rows]
    train_loss = _column(rows, "train_loss")
    val_loss = _column(rows, "val_loss")
    val_mae = _column(rows, "val_mae")
    val_ssim = _column(rows, "val_ssim")
    train_rgb_loss = _column(rows, "train_rgb_loss")
    train_aux_loss = _column(rows, "train_aux_loss")
    train_total_loss = _column(rows, "train_total_loss")
    val_rgb_loss = _column(rows, "val_rgb_loss")
    val_aux_loss = _column(rows, "val_aux_loss")
    val_total_loss = _column(rows, "val_total_loss")

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    _plot_curve(ax, epochs, train_loss, "train loss", "#1f77b4", shading)
    _plot_curve(ax, epochs, val_loss, "val loss", "#d62728", shading)
    ax.set_title("Training vs Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1 loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(loss_path)
    plt.close(fig)

    _save_single_metric(plt, epochs, val_mae, "Validation MAE", "MAE", "#2ca02c", mae_path, shading)
    _save_single_metric(plt, epochs, val_ssim, "Validation SSIM", "SSIM", "#9467bd", ssim_path, shading)

    has_aux = _has_any(train_aux_loss) or _has_any(val_aux_loss)
    if has_aux:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        _plot_curve(axes[0], epochs, train_rgb_loss, "train RGB", "#1f77b4", shading)
        _plot_curve(axes[0], epochs, val_rgb_loss, "val RGB", "#d62728", shading)
        _plot_curve(axes[0], epochs, train_total_loss, "train total", "#17becf", shading)
        _plot_curve(axes[0], epochs, val_total_loss, "val total", "#8c564b", shading)
        axes[0].set_title("RGB and Total Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        _plot_curve(axes[1], epochs, train_aux_loss, "train aux", "#ff7f0e", shading)
        _plot_curve(axes[1], epochs, val_aux_loss, "val aux", "#bcbd22", shading)
        axes[1].set_title("Auxiliary Dynamics Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Aux loss")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(aux_path)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    _plot_curve(axes[0], epochs, train_loss, "train loss", "#1f77b4", shading)
    _plot_curve(axes[0], epochs, val_loss, "val loss", "#d62728", shading)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("L1 loss")
    axes[0].legend()

    _plot_curve(axes[1], epochs, val_mae, "val MAE", "#2ca02c", shading)
    axes[1].set_title("Validation MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()

    _plot_curve(axes[2], epochs, val_ssim, "val SSIM", "#9467bd", shading)
    axes[2].set_title("Validation SSIM")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("SSIM")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(summary_path)
    plt.close(fig)

    diagnostic_outputs = _plot_diagnostics(plt, rows, diagnostics_dir, shading)

    return {
        "summary": summary_path,
        "loss": loss_path,
        "val_mae": mae_path,
        "val_ssim": ssim_path,
        "aux_loss": aux_path if has_aux else False,
        "csv": csv_path,
        "diagnostics_dir": diagnostics_dir,
        **diagnostic_outputs,
        "shading": shading,
    }


def write_curves_csv(rows: list[dict[str, Any]], csv_path: str | Path) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _csv_fieldnames(rows)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in fieldnames})


def plot_columns_if_present(
    plt,
    rows: list[dict[str, Any]],
    columns: list[tuple[str, str, str]],
    title: str,
    ylabel: str,
    output_path: Path,
    shading: bool,
) -> bool:
    """Plot the requested columns if at least one is present and numeric."""

    epochs = _column(rows, "epoch")
    present = [(key, label, color) for key, label, color in columns if _has_any(_column(rows, key))]
    if not present:
        print(f"warning: skipping {output_path.name}; no matching columns found")
        return False
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for key, label, color in present:
        _plot_curve(ax, epochs, _column(rows, key), label, color, shading)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True


def safe_ratio(numerator: float | None, denominator: float | None, eps: float = 1e-12) -> float | None:
    if numerator is None or denominator is None:
        return None
    if abs(float(denominator)) <= eps:
        return None
    return float(numerator) / float(denominator)


def _plot_diagnostics(plt, rows: list[dict[str, Any]], diagnostics_dir: Path, shading: bool) -> dict[str, Path | bool]:
    outputs: dict[str, Path | bool] = {}

    image_path = diagnostics_dir / "image_learning_curves.png"
    if _has_any(_column(rows, "train_mae")) or _has_any(_column(rows, "val_mae")) or _has_any(_column(rows, "val_ssim")):
        _plot_image_learning_curves(plt, rows, image_path, shading)
        outputs["diagnostic_image_learning"] = image_path
    else:
        print(f"warning: skipping {image_path.name}; no image metric columns found")
        outputs["diagnostic_image_learning"] = False

    grouped_specs = {
        "diagnostic_future_action_losses": (
            "Future Action Losses",
            "Loss",
            diagnostics_dir / "future_action_losses.png",
            [
                ("train_future_action_loss", "train action", "#1f77b4"),
                ("val_future_action_loss", "val action", "#d62728"),
                ("train_future_action_reg_loss", "train reg", "#17becf"),
                ("val_future_action_reg_loss", "val reg", "#ff9896"),
                ("train_future_action_corr_loss", "train corr", "#2ca02c"),
                ("val_future_action_corr_loss", "val corr", "#98df8a"),
                ("train_future_action_cls_loss", "train cls", "#9467bd"),
                ("val_future_action_cls_loss", "val cls", "#c5b0d5"),
                ("train_future_action_delta_loss", "train delta", "#ff7f0e"),
                ("val_future_action_delta_loss", "val delta", "#ffbb78"),
            ],
        ),
        "diagnostic_future_action_mae": (
            "Future Action MAE",
            "MAE",
            diagnostics_dir / "future_action_mae.png",
            [
                ("train_future_longitudinal_mae", "train longitudinal", "#1f77b4"),
                ("val_future_longitudinal_mae", "val longitudinal", "#d62728"),
                ("train_future_steer_mae", "train steer", "#2ca02c"),
                ("val_future_steer_mae", "val steer", "#ff7f0e"),
            ],
        ),
        "diagnostic_future_action_correlations": (
            "Future Action Correlations",
            "Correlation",
            diagnostics_dir / "future_action_correlations.png",
            [
                ("train_future_action_corr_longitudinal", "train longitudinal", "#1f77b4"),
                ("val_future_action_corr_longitudinal", "val longitudinal", "#d62728"),
                ("train_future_action_corr_steer", "train steer", "#2ca02c"),
                ("val_future_action_corr_steer", "val steer", "#ff7f0e"),
            ],
        ),
        "diagnostic_future_action_classification": (
            "Future Action Classification",
            "Accuracy / Loss",
            diagnostics_dir / "future_action_classification.png",
            [
                ("train_future_longitudinal_cls_acc", "train long acc", "#1f77b4"),
                ("val_future_longitudinal_cls_acc", "val long acc", "#d62728"),
                ("train_future_steer_cls_acc", "train steer acc", "#2ca02c"),
                ("val_future_steer_cls_acc", "val steer acc", "#ff7f0e"),
                ("train_future_action_cls_loss", "train cls loss", "#9467bd"),
                ("val_future_action_cls_loss", "val cls loss", "#8c564b"),
            ],
        ),
        "diagnostic_future_action_delta_losses": (
            "Future Action Delta Losses",
            "Loss",
            diagnostics_dir / "future_action_delta_losses.png",
            [
                ("train_future_action_delta_loss", "train total", "#1f77b4"),
                ("val_future_action_delta_loss", "val total", "#d62728"),
                ("train_future_action_delta_longitudinal_loss", "train longitudinal", "#2ca02c"),
                ("val_future_action_delta_longitudinal_loss", "val longitudinal", "#98df8a"),
                ("train_future_action_delta_steer_loss", "train steer", "#ff7f0e"),
                ("val_future_action_delta_steer_loss", "val steer", "#ffbb78"),
            ],
        ),
        "diagnostic_branch_norms": (
            "Future Action Branch Feature Norms",
            "Feature norm",
            diagnostics_dir / "branch_norms.png",
            [
                ("train_future_action_motion_feature_norm", "train motion", "#1f77b4"),
                ("val_future_action_motion_feature_norm", "val motion", "#d62728"),
                ("train_future_action_control_feature_norm", "train control", "#2ca02c"),
                ("val_future_action_control_feature_norm", "val control", "#98df8a"),
                ("train_future_action_latent_feature_norm", "train latent", "#ff7f0e"),
                ("val_future_action_latent_feature_norm", "val latent", "#ffbb78"),
            ],
        ),
        "diagnostic_branch_norm_ratios": (
            "Future Action Branch Norm Ratios",
            "Ratio",
            diagnostics_dir / "branch_norm_ratios.png",
            [
                ("train_action_control_to_motion_norm_ratio", "train control/motion", "#1f77b4"),
                ("val_action_control_to_motion_norm_ratio", "val control/motion", "#d62728"),
                ("train_action_latent_to_motion_norm_ratio", "train latent/motion", "#2ca02c"),
                ("val_action_latent_to_motion_norm_ratio", "val latent/motion", "#98df8a"),
                ("train_action_control_to_latent_norm_ratio", "train control/latent", "#ff7f0e"),
                ("val_action_control_to_latent_norm_ratio", "val control/latent", "#ffbb78"),
            ],
        ),
        "diagnostic_wm_fusion_ratio": (
            "Conv1x1 WM/SimVP Fusion Weight Ratio",
            "WM half / SimVP half weight norm",
            diagnostics_dir / "wm_fusion_ratio.png",
            [
                ("train_dual_conv1x1_wm_to_simvp_weight_norm_ratio", "train wm/simvp", "#1f77b4"),
                ("val_dual_conv1x1_wm_to_simvp_weight_norm_ratio", "val wm/simvp", "#d62728"),
                ("dual_conv1x1_wm_to_simvp_weight_norm_ratio", "wm/simvp", "#2ca02c"),
            ],
        ),
        "diagnostic_gradient_norms": (
            "Gradient Norms",
            "Norm",
            diagnostics_dir / "gradient_norms.png",
            _gradient_columns(rows),
        ),
    }
    for output_key, (title, ylabel, path, columns) in grouped_specs.items():
        outputs[output_key] = path if plot_columns_if_present(plt, rows, columns, title, ylabel, path, shading) else False
    return outputs


def _plot_image_learning_curves(plt, rows: list[dict[str, Any]], path: Path, shading: bool) -> None:
    epochs = _column(rows, "epoch")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    _plot_curve(axes[0], epochs, _column(rows, "train_mae"), "train MAE", "#1f77b4", shading)
    _plot_curve(axes[0], epochs, _column(rows, "val_mae"), "val MAE", "#d62728", shading)
    axes[0].set_title("Image MAE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MAE")
    axes[0].legend()

    _plot_curve(axes[1], epochs, _column(rows, "train_ssim"), "train SSIM", "#2ca02c", shading)
    _plot_curve(axes[1], epochs, _column(rows, "val_ssim"), "val SSIM", "#9467bd", shading)
    axes[1].set_title("Image SSIM")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("SSIM")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_curve(ax, epochs, values, label: str, color: str, shading: bool) -> None:
    pairs = [(epoch, value) for epoch, value in zip(epochs, values) if value is not None]
    if not pairs:
        return
    plot_epochs = [epoch for epoch, _ in pairs]
    plot_values = [value for _, value in pairs]
    ax.plot(plot_epochs, plot_values, label=label, color=color, linewidth=2.2)
    if shading:
        std = _rolling_std(plot_values)
        lower = [value - err for value, err in zip(plot_values, std)]
        upper = [value + err for value, err in zip(plot_values, std)]
        ax.fill_between(plot_epochs, lower, upper, color=color, alpha=0.15, linewidth=0)


def _save_single_metric(plt, epochs, values, title: str, ylabel: str, color: str, path: Path, shading: bool) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    _plot_curve(ax, epochs, values, title, color, shading)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        numeric = _float_or_none(value)
        if numeric is not None:
            out[f"{prefix}_{key}"] = numeric
    return out


def _load_run_config(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"warning: could not read config metadata from {config_path}: {exc}")
        return {}


def _merged_config(config: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for section in ("run", "trainer"):
        values = config.get(section, {})
        if isinstance(values, dict):
            merged.update(values)
    return merged


def _action_head_config_metadata(config: dict[str, Any]) -> dict[str, Any]:
    merged = _merged_config(config)
    metadata: dict[str, Any] = {}
    for key in ACTION_HEAD_CONFIG_FIELDS:
        metadata[key] = merged.get(key, "n/a")
    if metadata.get("future_action_spatial_pooling") == "n/a":
        metadata["future_action_spatial_pooling"] = "global"
    if metadata.get("future_action_spatial_grid") == "n/a":
        metadata["future_action_spatial_grid"] = "1x1"
    if metadata.get("future_action_token_dim") == "n/a":
        metadata["future_action_token_dim"] = _infer_action_token_dim(merged)
    return metadata


def _infer_action_token_dim(config: dict[str, Any]) -> int | str:
    latent_dim = config.get("hid_s", config.get("latent_dim"))
    try:
        latent_dim = int(latent_dim)
    except (TypeError, ValueError):
        return "n/a"
    pooling = str(config.get("future_action_spatial_pooling", "global"))
    if pooling == "global":
        return latent_dim
    grid = _parse_grid(config.get("future_action_spatial_grid", "1x1"))
    if grid is None:
        return "n/a"
    return latent_dim * grid[0] * grid[1]


def _parse_grid(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    match = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*$", str(value))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _merge_train_log_metrics(rows: list[dict[str, Any]], train_log_path: Path) -> None:
    """Merge parseable epoch summary key=value lines from train.log into metric rows.

    metrics.jsonl is the primary source. This fallback catches newer scalar logs that
    may appear only in the human-readable log and merges multiple summary lines into
    one row per epoch.
    """

    if not train_log_path.exists() or not rows:
        return
    by_epoch = {int(row["epoch"]): row for row in rows if row.get("epoch") is not None}
    pattern = re.compile(r"(?:^|\s)([A-Za-z0-9_./-]+)=([^\s]+)")
    try:
        lines = train_log_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(f"warning: could not read train.log metrics from {train_log_path}: {exc}")
        return
    for line in lines:
        match = re.search(r"\bepoch\s+(\d+)\b", line)
        if not match:
            continue
        epoch = int(match.group(1))
        row = by_epoch.get(epoch)
        if row is None:
            continue
        for key, value in pattern.findall(line):
            if key == "epoch":
                continue
            numeric = _float_or_none(value)
            if numeric is not None:
                row.setdefault(key, numeric)


def _add_derived_ratios(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        for prefix in ("train", "val"):
            motion = row.get(f"{prefix}_future_action_motion_feature_norm")
            control = row.get(f"{prefix}_future_action_control_feature_norm")
            latent = row.get(f"{prefix}_future_action_latent_feature_norm")
            row[f"{prefix}_action_control_to_motion_norm_ratio"] = safe_ratio(control, motion)
            row[f"{prefix}_action_latent_to_motion_norm_ratio"] = safe_ratio(latent, motion)
            row[f"{prefix}_action_control_to_latent_norm_ratio"] = safe_ratio(control, latent)


def _write_action_head_summary(rows: list[dict[str, Any]], run_dir: Path | None, output_path: Path) -> None:
    config = _action_head_config_metadata(_load_run_config(run_dir))
    if rows:
        for key in ACTION_HEAD_CONFIG_FIELDS:
            value = rows[0].get(key)
            if value is not None:
                config[key] = value
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Action Head Configuration Summary\n")
        handle.write("=================================\n")
        if run_dir is not None:
            handle.write(f"run_dir: {run_dir}\n")
        for key in ACTION_HEAD_CONFIG_FIELDS:
            handle.write(f"{key}: {config.get(key, 'n/a')}\n")


def _csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    available = set()
    for row in rows:
        available.update(row.keys())
    ordered = [column for column in CURVE_COLUMNS if column in available]
    ordered.extend(column for column in ACTION_HEAD_CONFIG_FIELDS if column in available and column not in ordered)
    ordered.extend(sorted(column for column in available if column not in ordered))
    return ordered


def _column(rows: list[dict[str, Any]], key: str) -> list[Any]:
    return [row.get(key) for row in rows]


def _gradient_columns(rows: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    preferred = [
        ("train_grad_norm_total", "train total grad", "#1f77b4"),
        ("train_total_grad_norm", "train total grad", "#1f77b4"),
        ("train_clipped_grad_norm", "train clipped grad", "#d62728"),
        ("train_grad_clip_norm", "train clipped grad", "#d62728"),
        ("grad_norm_total", "total grad", "#2ca02c"),
        ("total_grad_norm", "total grad", "#2ca02c"),
        ("clipped_grad_norm", "clipped grad", "#ff7f0e"),
    ]
    present_keys = {
        key
        for row in rows
        for key, value in row.items()
        if "grad" in key.lower() and _float_or_none(value) is not None
    }
    columns = [entry for entry in preferred if entry[0] in present_keys]
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    for key in sorted(present_keys):
        if key in {entry[0] for entry in columns}:
            continue
        columns.append((key, key.replace("_", " "), palette[len(columns) % len(palette)]))
    return columns


def _rolling_std(values: list[float], window: int = 5) -> list[float]:
    out = []
    half = window // 2
    for idx in range(len(values)):
        start = max(0, idx - half)
        end = min(len(values), idx + half + 1)
        local = values[start:end]
        if len(local) < 2:
            out.append(0.0)
            continue
        mean = sum(local) / len(local)
        variance = sum((value - mean) ** 2 for value in local) / (len(local) - 1)
        out.append(variance**0.5)
    return out


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_any(values: list[float | None]) -> bool:
    return any(value is not None for value in values)
