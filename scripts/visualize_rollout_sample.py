#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.carla_rollout_dataset import ACTION_NAMES, CarlaRolloutDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one rollout sample and print actions.")
    parser.add_argument("--root", type=Path, default=Path("/path/to/mile_action_diverse/train/Town01"))
    parser.add_argument("--split", default="train", choices=["train", "val", "validation", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/rollout_samples/sample.png"))
    return parser.parse_args()


def tensor_to_image(frame) -> Image.Image:
    array = frame.permute(1, 2, 0).numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def draw_row(frames, actions, speed, title: str, indices: list[int]) -> Image.Image:
    cell_w, cell_h = 160, 160
    label_h = 70
    row = Image.new("RGB", (cell_w * len(frames), cell_h + label_h), "white")
    for col, frame in enumerate(frames):
        image = tensor_to_image(frame)
        image.thumbnail((cell_w, cell_h))
        x = col * cell_w
        row.paste(image, (x, 0))
        draw = ImageDraw.Draw(row)
        action = actions[col].numpy()
        draw.text((x + 4, cell_h + 2), f"{title} t={indices[col]}", fill=(0, 0, 0))
        draw.text((x + 4, cell_h + 18), f"thr {action[0]:.2f} str {action[1]:.2f}", fill=(0, 0, 0))
        draw.text((x + 4, cell_h + 34), f"brk {action[2]:.2f}", fill=(0, 0, 0))
        draw.text((x + 4, cell_h + 50), f"speed {speed[col, 0].item():.2f}", fill=(0, 0, 0))
    return row


def main() -> int:
    args = parse_args()
    dataset = CarlaRolloutDataset(args.root, split=args.split, include_metadata=True)
    sample = dataset[args.index]
    metadata = sample["metadata"]

    past_indices = metadata["past_indices"]
    future_indices = metadata["future_indices"]
    past_row = draw_row(
        sample["past_frames"], sample["past_actions"], sample["past_speed"], "past", past_indices
    )
    future_row = draw_row(
        sample["future_frames"],
        sample["future_actions"],
        sample["future_speed"],
        "future",
        future_indices,
    )

    width = max(past_row.width, future_row.width)
    sheet = Image.new("RGB", (width, past_row.height + future_row.height), "white")
    sheet.paste(past_row, (0, 0))
    sheet.paste(future_row, (0, past_row.height))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)

    print(f"split: {args.split}")
    print(f"dataset length: {len(dataset)}")
    print(f"sample index: {args.index}")
    print(f"run_id: {metadata['run_id']}")
    print(f"start frame: {metadata['start']}")
    print(f"past indices: {past_indices}")
    print(f"future indices: {future_indices}")
    print(f"action names: {ACTION_NAMES}")
    print("past actions:")
    print(sample["past_actions"])
    print("past speed:")
    print(sample["past_speed"])
    print("future actions:")
    print(sample["future_actions"])
    print("future speed:")
    print(sample["future_speed"])
    print(f"wrote rollout visualization: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
