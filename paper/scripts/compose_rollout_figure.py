#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from render_rollout_with_action_indicators import (  # noqa: E402
    TEXT_COLOR,
    draw_action_cell,
    forward_case,
    make_dataset_and_model,
    tensor_to_image,
)


DEFAULT_INDICES = [1987, 1796, 1760, 1058]
DEFAULT_LABELS = [
    "Straight acceleration",
    "Mild turn",
    "Sharp turn",
    "Challenging intersection",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose publication-quality multi-case TeleopWM rollout figures."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "validation", "test"], default="test")
    parser.add_argument("--indices", type=int, nargs="+", default=DEFAULT_INDICES)
    parser.add_argument("--case-labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures/generated"))
    parser.add_argument("--output-name", default="main_rollout_action_figure")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--frame-width", type=int, default=150)
    parser.add_argument("--font-size", type=int, default=18)
    parser.add_argument("--font-scale", type=float, default=1.0)
    parser.add_argument(
        "--paper-font-preset",
        choices=["small", "normal", "large", "xlarge"],
        default=None,
        help="Convenience preset for --font-scale: small=1.0, normal=1.2, large=1.5, xlarge=1.8.",
    )
    parser.add_argument("--case-title-font-size", type=int, default=None)
    parser.add_argument("--row-label-font-size", type=int, default=None)
    parser.add_argument("--step-font-size", type=int, default=None)
    parser.add_argument("--action-font-size", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--action-height", type=int, default=None)
    parser.add_argument("--action-scale", type=float, default=1.25)
    parser.add_argument("--display-steer-multiplier", type=float, default=-1.0)
    parser.add_argument("--show-action-values", action="store_true")
    parser.add_argument("--no-action-indicators", action="store_true")
    parser.add_argument("--hide-indices", action="store_true", default=True)
    parser.add_argument("--show-indices", action="store_false", dest="hide_indices")
    parser.add_argument("--title", default=None)
    parser.add_argument("--format", choices=["png", "pdf", "svg", "both", "all"], default="both")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--visual-only", action="store_true")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--future-action-source", choices=["final", "wm", "simvp"], default=None)
    parser.add_argument("--future-steer-target-scale", type=float, default=None)
    parser.add_argument("--control-steer-input-scale", type=float, default=None)
    parser.add_argument("--future-action-future-motion-scale", type=float, default=None)
    parser.add_argument("--future-action-spatial-pooling", choices=["global", "grid"], default=None)
    parser.add_argument("--future-action-spatial-grid", default=None)
    parser.add_argument("--dual-wm-conditioning", choices=["add", "concat", "film"], default=None)
    return parser.parse_args()


def effective_font_scale(args: argparse.Namespace) -> float:
    preset_scales = {
        "small": 1.0,
        "normal": 1.2,
        "large": 1.5,
        "xlarge": 1.8,
    }
    if args.paper_font_preset is not None:
        return preset_scales[args.paper_font_preset]
    return float(args.font_scale)


def scaled_font_size(base_size: float, scale: float) -> int:
    return max(1, int(round(base_size * scale)))


def load_serif_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "TeX Gyre Termes",
        "TeXGyreTermes-Regular.ttf",
        "NimbusRoman-Regular.otf",
        "Nimbus Roman",
        "LiberationSerif-Regular.ttf",
        "Times New Roman.ttf",
        "Times.ttf",
        "DejaVuSerif.ttf",
    )
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def alpha_label(index: int) -> str:
    return chr(ord("A") + index)


def normalize_case_labels(indices: list[int], labels: list[str]) -> list[str]:
    if len(labels) < len(indices):
        labels = labels + [f"Case {i + 1}" for i in range(len(labels), len(indices))]
    return labels[: len(indices)]


def case_title(case_number: int, label: str, dataset_index: int, *, hide_indices: bool) -> str:
    prefix = f"({alpha_label(case_number)})"
    if hide_indices:
        return f"{prefix} {label}"
    return f"{prefix} {label}  [idx {dataset_index}]"


def draw_frame_row(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    frames: torch.Tensor,
    *,
    row_label: str,
    y: int,
    label_x: int,
    frame_x0: int,
    frame_width: int,
    frame_height: int,
    gap: int,
    font: ImageFont.ImageFont,
    step_font: ImageFont.ImageFont,
    show_steps: bool,
) -> None:
    draw.text((label_x, y + frame_height // 2 - font.size // 2), row_label, fill=TEXT_COLOR, font=font)
    for step, frame in enumerate(frames):
        x = frame_x0 + step * (frame_width + gap)
        img = tensor_to_image(frame).resize((frame_width, frame_height), Image.Resampling.BILINEAR)
        canvas.paste(img, (x, y))
        if show_steps:
            label = f"t+{step + 1}"
            tx = x + 5
            ty = y + 4
            # A very light shadow keeps the labels readable over bright road/sky
            # pixels without making them look like dense annotations.
            draw.text((tx + 1, ty + 1), label, fill=(0, 0, 0), font=step_font)
            draw.text((tx, ty), label, fill=(255, 255, 255), font=step_font)
            draw.text((tx + 1, ty), label, fill=(255, 255, 255), font=step_font)


def draw_action_row(
    draw: ImageDraw.ImageDraw,
    case: dict[str, Any],
    *,
    y: int,
    label_x: int,
    frame_x0: int,
    frame_width: int,
    action_height: int,
    gap: int,
    font: ImageFont.ImageFont,
    display_steer_multiplier: float,
    show_action_values: bool,
    action_scale: float,
) -> None:
    draw.text((label_x, y + action_height // 2 - font.size // 2), "Act", fill=TEXT_COLOR, font=font)
    for step in range(int(case["gt_actions"].shape[0])):
        x = frame_x0 + step * (frame_width + gap)
        inset = max(0, int(round((1.0 - min(action_scale, 1.0)) * frame_width * 0.5)))
        if action_scale > 1.0:
            expanded = min(gap // 2, int(round((action_scale - 1.0) * frame_width * 0.5)))
            x0 = x - expanded
            x1 = x + frame_width + expanded
        else:
            x0 = x + inset
            x1 = x + frame_width - inset
        draw_action_cell(
            draw,
            (x0, y, x1, y + action_height),
            case["gt_actions"][step],
            case["pred_actions"][step],
            font,
            display_steer_multiplier=display_steer_multiplier,
            show_values=show_action_values,
        )


def compose_figure(
    cases: list[dict[str, Any]],
    labels: list[str],
    indices: list[int],
    args: argparse.Namespace,
) -> Image.Image:
    include_actions = not args.no_action_indicators and not args.visual_only
    steps = max(1, min(args.steps, 8))
    font_scale = effective_font_scale(args)
    base_font_size = scaled_font_size(args.font_size, font_scale)
    case_title_size = args.case_title_font_size or scaled_font_size(args.font_size + 3, font_scale)
    row_label_size = args.row_label_font_size or scaled_font_size(args.font_size + 1, font_scale)
    step_size = args.step_font_size or scaled_font_size(args.font_size - 1, font_scale)
    action_size = args.action_font_size or scaled_font_size(args.font_size - 2, font_scale)
    title_font = load_serif_font(scaled_font_size(args.font_size + 4, font_scale))
    label_font = load_serif_font(row_label_size)
    step_font = load_serif_font(max(10, step_size))
    action_font = load_serif_font(max(10, action_size))
    case_font = load_serif_font(case_title_size)

    pad_x = 18
    pad_y = 14
    label_w = max(66, row_label_size * 4)
    gap = 8
    case_gap = 12
    frame_w = args.frame_width
    frame_h = int(frame_w * args.height / args.width)
    default_action_h = 130 if args.show_action_values else 104
    action_h = args.action_height if args.action_height is not None else default_action_h
    case_title_h = case_title_size + 11
    title_h = base_font_size + 18 if args.title else 0
    content_w = steps * frame_w + (steps - 1) * gap
    width = pad_x * 2 + label_w + content_w
    case_h = case_title_h + 2 * frame_h + 2 * gap + (action_h + gap if include_actions else 0)
    height = pad_y * 2 + title_h + len(cases) * case_h + (len(cases) - 1) * case_gap

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    y = pad_y
    if args.title:
        draw.text((pad_x, y), args.title, fill=TEXT_COLOR, font=title_font)
        y += title_h

    label_x = pad_x
    frame_x0 = pad_x + label_w
    for case_i, case in enumerate(cases):
        title = case_title(case_i, labels[case_i], indices[case_i], hide_indices=args.hide_indices)
        draw.text((label_x, y), title, fill=TEXT_COLOR, font=case_font)
        y += case_title_h

        gt = case["gt_frames"][:steps]
        pred = case["pred_frames"][:steps]
        render_case = {
            **case,
            "gt_actions": case["gt_actions"][:steps],
            "pred_actions": case["pred_actions"][:steps],
        }
        draw_frame_row(
            canvas,
            draw,
            gt,
            row_label="GT",
            y=y,
            label_x=label_x,
            frame_x0=frame_x0,
            frame_width=frame_w,
            frame_height=frame_h,
            gap=gap,
            font=label_font,
            step_font=step_font,
            show_steps=True,
        )
        y += frame_h + gap + 2
        draw_frame_row(
            canvas,
            draw,
            pred,
            row_label="Pred",
            y=y,
            label_x=label_x,
            frame_x0=frame_x0,
            frame_width=frame_w,
            frame_height=frame_h,
            gap=gap,
            font=label_font,
            step_font=step_font,
            show_steps=False,
        )
        y += frame_h + gap + 2
        if include_actions:
            draw_action_row(
                draw,
                render_case,
                y=y,
                label_x=label_x,
                frame_x0=frame_x0,
                frame_width=frame_w,
                action_height=action_h,
                gap=gap,
                font=action_font,
                display_steer_multiplier=args.display_steer_multiplier,
                show_action_values=args.show_action_values,
                action_scale=args.action_scale,
            )
            y += action_h + gap
        y += case_gap
    return canvas


def output_paths(args: argparse.Namespace) -> list[Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.format == "both":
        suffixes = ("png", "pdf")
    elif args.format == "all":
        suffixes = ("png", "pdf", "svg")
    else:
        suffixes = (args.format,)
    return [args.output_dir / f"{args.output_name}.{suffix}" for suffix in suffixes]


def save_outputs(image: Image.Image, args: argparse.Namespace) -> list[Path]:
    paths = output_paths(args)
    for path in paths:
        if path.suffix.lower() == ".pdf":
            image.save(path, "PDF", resolution=args.dpi)
        elif path.suffix.lower() == ".svg":
            # Pillow cannot write real vector content from raster frames; this SVG
            # embeds the composed bitmap so LaTeX workflows can still reference it.
            png_path = path.with_suffix(".png")
            if png_path not in paths:
                image.save(png_path, dpi=(args.dpi, args.dpi))
            import base64

            encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
            svg = (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{image.width}" height="{image.height}" '
                f'viewBox="0 0 {image.width} {image.height}">'
                f'<image href="data:image/png;base64,{encoded}" width="{image.width}" height="{image.height}"/>'
                "</svg>\n"
            )
            path.write_text(svg, encoding="utf-8")
        else:
            image.save(path, dpi=(args.dpi, args.dpi))
    return paths


def main() -> int:
    args = parse_args()
    indices = list(args.indices)
    if args.max_cases is not None:
        indices = indices[: args.max_cases]
    labels = normalize_case_labels(indices, list(args.case_labels))
    dataset, model, run_cfg, device = make_dataset_and_model(args)
    future_steer_target_scale = (
        args.future_steer_target_scale
        if args.future_steer_target_scale is not None
        else float(run_cfg.get("future_steer_target_scale", 1.0))
    )
    cases: list[dict[str, Any]] = []
    for idx in indices:
        case = forward_case(dataset, model, device, idx, future_steer_target_scale)
        cases.append(case)
    figure = compose_figure(cases, labels, indices, args)
    written = save_outputs(figure, args)
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
