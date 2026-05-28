#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_future_actions import load_model, target_actions
from src.datasets import CarlaRolloutDataset


GT_COLOR = (30, 90, 180)
PRED_COLOR = (220, 90, 40)
AXIS_COLOR = (190, 190, 190)
TICK_COLOR = (105, 105, 105)
TEXT_COLOR = (30, 30, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render paper-ready TeleopWM rollout figures with compact future-action indicators."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "validation", "test"], default="test")
    parser.add_argument("--indices", type=int, nargs="+", default=None)
    parser.add_argument("--ranked-csv", type=Path, default=None, help="CSV containing a dataset_index column.")
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures/generated/rollouts"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--frame-width", type=int, default=150)
    parser.add_argument("--font-size", type=int, default=13)
    parser.add_argument("--no-action-indicators", action="store_true")
    parser.add_argument(
        "--display-steer-multiplier",
        type=float,
        default=-1.0,
        help=(
            "Multiplier used only for visual steering indicators. The default "
            "matches the left/right display convention used in the paper figures; "
            "metrics and saved action tensors are unchanged."
        ),
    )
    parser.add_argument(
        "--show-action-values",
        action="store_true",
        help="Print compact numeric GT/Pred longitudinal and steering values below each timestep.",
    )
    parser.add_argument("--future-action-source", choices=["final", "wm", "simvp"], default=None)
    parser.add_argument("--future-steer-target-scale", type=float, default=None)
    parser.add_argument("--control-steer-input-scale", type=float, default=None)
    parser.add_argument("--future-action-future-motion-scale", type=float, default=None)
    parser.add_argument("--future-action-spatial-pooling", choices=["global", "grid"], default=None)
    parser.add_argument("--future-action-spatial-grid", default=None)
    parser.add_argument("--dual-wm-conditioning", choices=["add", "concat", "film"], default=None)
    return parser.parse_args()


def tensor_to_image(frame: torch.Tensor) -> Image.Image:
    array = frame.detach().cpu().clamp(0.0, 1.0)
    if array.ndim == 3 and array.shape[0] in {1, 3}:
        array = array.permute(1, 2, 0)
    array = (array.numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array)


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def read_indices_from_csv(path: Path, max_cases: int | None = None) -> list[int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if "dataset_index" not in rows[0]:
        raise ValueError(f"{path} must contain a dataset_index column")
    indices = [int(row["dataset_index"]) for row in rows]
    return indices[:max_cases] if max_cases is not None else indices


def make_dataset_and_model(args: argparse.Namespace) -> tuple[CarlaRolloutDataset, torch.nn.Module, dict[str, Any], torch.device]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    run_cfg = checkpoint.get("config", {}).get("run", {})
    image_size = (
        int(args.height if args.height is not None else run_cfg.get("height", 320)),
        int(args.width if args.width is not None else run_cfg.get("width", 512)),
    )
    normalize_controls = bool(run_cfg.get("normalize_controls", False))
    speed_scale = float(run_cfg.get("speed_scale", 20.0))
    dataset = CarlaRolloutDataset(
        args.data_root,
        split=args.split,
        image_size=image_size,
        normalize_controls=normalize_controls,
        speed_scale=speed_scale,
        include_metadata=True,
    )
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, run_cfg = load_model(args.checkpoint, device, args)
    return dataset, model, run_cfg, device


def forward_case(
    dataset: CarlaRolloutDataset,
    model: torch.nn.Module,
    device: torch.device,
    dataset_index: int,
    future_steer_target_scale: float,
) -> dict[str, Any]:
    sample = dataset[dataset_index]
    past = sample["past_frames"].unsqueeze(0).to(device)
    actions = sample["past_actions"].unsqueeze(0).to(device)
    speed = sample["past_speed"].unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(past, actions, speed, return_latents=True)
    frames = output["frames"].detach().cpu().clamp(0.0, 1.0).squeeze(0)
    gt_frames = sample["future_frames"].detach().cpu().clamp(0.0, 1.0)
    pred_actions = output.get("pred_future_actions")
    if pred_actions is None:
        pred_actions_unscaled = torch.zeros(gt_frames.shape[0], 2)
    else:
        pred_actions_unscaled = pred_actions.detach().cpu().squeeze(0)
        pred_actions_unscaled[..., 1] = pred_actions_unscaled[..., 1] * future_steer_target_scale
    gt_actions = target_actions(sample, device).detach().cpu().squeeze(0)
    future_len = int(gt_frames.shape[0])
    if frames.shape[0] != future_len:
        raise ValueError(
            f"Prediction/future frame horizon mismatch for dataset index {dataset_index}: "
            f"pred={frames.shape[0]} future={future_len}"
        )
    for key in ("future_actions", "future_speed"):
        if sample[key].shape[0] != future_len:
            raise ValueError(
                f"{key}/future frame horizon mismatch for dataset index {dataset_index}: "
                f"{key}={sample[key].shape[0]} future={future_len}"
            )
    if gt_actions.shape[0] != future_len or pred_actions_unscaled.shape[0] != future_len:
        raise ValueError(
            f"Action/frame horizon mismatch for dataset index {dataset_index}: "
            f"gt_actions={gt_actions.shape[0]} pred_actions={pred_actions_unscaled.shape[0]} "
            f"future={future_len}"
        )
    return {
        "dataset_index": dataset_index,
        "sample": sample,
        "gt_frames": gt_frames,
        "pred_frames": frames,
        "gt_actions": gt_actions,
        "pred_actions": pred_actions_unscaled,
        "metadata": sample.get("metadata", {}),
    }


def draw_action_cell(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    gt_action: torch.Tensor,
    pred_action: torch.Tensor,
    font: ImageFont.ImageFont,
    *,
    display_steer_multiplier: float = -1.0,
    show_values: bool = False,
) -> None:
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    bar_x = x0 + 16
    bar_w = w - 32
    steer_y = y0 + 34
    long_y = y0 + 60
    tick_y = long_y + 13
    value_y = tick_y + 13

    draw_overlay_bar(
        draw,
        bar_x,
        steer_y,
        bar_w,
        float(gt_action[1]) * display_steer_multiplier,
        float(pred_action[1]) * display_steer_multiplier,
        font,
    )
    draw_overlay_bar(
        draw,
        bar_x,
        long_y,
        bar_w,
        float(gt_action[0]),
        float(pred_action[0]),
        font,
    )
    draw_scale_ticks(draw, bar_x, tick_y, bar_w, font)

    if show_values:
        gt_long = float(gt_action[0])
        gt_steer = float(gt_action[1]) * display_steer_multiplier
        pred_long = float(pred_action[0])
        pred_steer = float(pred_action[1]) * display_steer_multiplier
        draw.text((x0 + 8, value_y), f"G s{gt_steer:+.2f} l{gt_long:+.2f}", fill=GT_COLOR, font=font)
        draw.text((x0 + 8, value_y + 12), f"P s{pred_steer:+.2f} l{pred_long:+.2f}", fill=PRED_COLOR, font=font)


def unit_x(x: int, width: int, value: float) -> int:
    clipped = max(-1.0, min(1.0, value))
    return int(round(x + (clipped + 1.0) * 0.5 * width))


def draw_scale_ticks(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    font: ImageFont.ImageFont,
) -> None:
    for value, label in ((-1.0, "-1"), (0.0, "0"), (1.0, "+1")):
        tx = unit_x(x, width, value)
        draw.line((tx, y - 3, tx, y + 1), fill=TICK_COLOR, width=1)
        label_w = int(draw.textlength(label, font=font))
        draw.text((tx - label_w // 2, y + 1), label, fill=TICK_COLOR, font=font)


def draw_overlay_bar(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    gt_value: float,
    pred_value: float,
    font: ImageFont.ImageFont,
) -> None:
    center = unit_x(x, width, 0.0)
    gt_x = unit_x(x, width, gt_value)
    pred_x = unit_x(x, width, pred_value)
    track_h = 9
    fill_h = 5
    draw.rounded_rectangle((x, y - track_h // 2, x + width, y + track_h // 2), radius=3, fill=(245, 245, 245), outline=AXIS_COLOR, width=1)
    draw.line((center, y - track_h // 2 - 3, center, y + track_h // 2 + 3), fill=(110, 110, 110), width=1)

    if pred_x >= center:
        pred_box = (center, y - fill_h // 2, pred_x, y + fill_h // 2)
    else:
        pred_box = (pred_x, y - fill_h // 2, center, y + fill_h // 2)
    draw.rectangle(pred_box, fill=PRED_COLOR)

    # GT is drawn as a blue outlined bracket on the same scale as the prediction,
    # keeping the comparison compact while making the target visibly distinct.
    draw.line((gt_x, y - track_h // 2 - 4, gt_x, y + track_h // 2 + 4), fill=GT_COLOR, width=2)
    draw.line((gt_x - 4, y - track_h // 2 - 4, gt_x + 4, y - track_h // 2 - 4), fill=GT_COLOR, width=2)
    draw.line((gt_x - 4, y + track_h // 2 + 4, gt_x + 4, y + track_h // 2 + 4), fill=GT_COLOR, width=2)


def render_case_figure(
    case: dict[str, Any],
    output_path: Path,
    *,
    frame_width: int = 150,
    font_size: int = 13,
    dpi: int = 300,
    include_action_indicators: bool = True,
    display_steer_multiplier: float = -1.0,
    show_action_values: bool = False,
    title: str | None = None,
) -> None:
    gt_frames = case["gt_frames"]
    pred_frames = case["pred_frames"]
    future_len = int(gt_frames.shape[0])
    font = load_font(font_size)
    small_font = load_font(max(9, font_size - 3))
    label_w = 132
    pad = 6
    frame_images = [tensor_to_image(frame) for frame in gt_frames]
    frame_h = int(frame_width * frame_images[0].height / frame_images[0].width)
    action_h = 112 if include_action_indicators and show_action_values else 88 if include_action_indicators else 0
    title_h = 24 if title else 0
    width = label_w + future_len * frame_width + (future_len + 1) * pad
    height = title_h + 2 * frame_h + action_h + (4 if include_action_indicators else 3) * pad
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    y = pad
    if title:
        draw.text((pad, y), title, fill=(0, 0, 0), font=font)
        y += title_h
    for row_name, frames in (("GT", gt_frames), ("Pred", pred_frames)):
        draw.text((pad, y + frame_h // 2 - font_size // 2), row_name, fill=(0, 0, 0), font=font)
        for step, frame in enumerate(frames):
            img = tensor_to_image(frame)
            img = img.resize((frame_width, frame_h), Image.Resampling.BILINEAR)
            x = label_w + pad + step * (frame_width + pad)
            canvas.paste(img, (x, y))
            draw.text((x + 4, y + 4), f"t+{step + 1}", fill=(255, 255, 255), font=small_font)
        y += frame_h + pad
    if include_action_indicators:
        draw.text((pad, y + 2), "Action", fill=TEXT_COLOR, font=font)
        draw.text((pad, y + 19), "Blue = GT", fill=GT_COLOR, font=small_font)
        draw.text((pad, y + 32), "Orange = Pred", fill=PRED_COLOR, font=small_font)
        draw.text((pad, y + 44), "Steer: left / right", fill=TEXT_COLOR, font=small_font)
        draw.text((pad, y + 70), "Long: brake / accel", fill=TEXT_COLOR, font=small_font)
        for step in range(future_len):
            x = label_w + pad + step * (frame_width + pad)
            draw_action_cell(
                draw,
                (x, y, x + frame_width, y + action_h),
                case["gt_actions"][step],
                case["pred_actions"][step],
                small_font,
                display_steer_multiplier=display_steer_multiplier,
                show_values=show_action_values,
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, dpi=(dpi, dpi))


def main() -> int:
    args = parse_args()
    if args.indices is None and args.ranked_csv is None:
        raise ValueError("Provide --indices or --ranked-csv.")
    indices = args.indices or read_indices_from_csv(args.ranked_csv, args.max_cases)
    if args.max_cases is not None:
        indices = indices[: args.max_cases]
    dataset, model, run_cfg, device = make_dataset_and_model(args)
    future_steer_target_scale = (
        args.future_steer_target_scale
        if args.future_steer_target_scale is not None
        else float(run_cfg.get("future_steer_target_scale", 1.0))
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for rank, dataset_index in enumerate(indices, start=1):
        case = forward_case(dataset, model, device, dataset_index, future_steer_target_scale)
        title = f"case {rank} | idx {dataset_index}"
        render_case_figure(
            case,
            args.output_dir / f"paper_rollout_case_{rank:02d}_{dataset_index:06d}.png",
            frame_width=args.frame_width,
            font_size=args.font_size,
            dpi=args.dpi,
            include_action_indicators=not args.no_action_indicators,
            display_steer_multiplier=args.display_steer_multiplier,
            show_action_values=args.show_action_values,
            title=title,
        )
        render_case_figure(
            case,
            args.output_dir / f"paper_rollout_case_{rank:02d}_{dataset_index:06d}_visual_only.png",
            frame_width=args.frame_width,
            font_size=args.font_size,
            dpi=args.dpi,
            include_action_indicators=False,
            title=title,
        )
    print(f"wrote rollout figures: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
