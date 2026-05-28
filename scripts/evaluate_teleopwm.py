#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets import CarlaRolloutDataset
from src.metrics import video_metrics
from src.models import TeleopWMPredictor, SimVPPredictor
from src.utils import infer_run_dir_from_checkpoint, load_checkpoint, write_json


def print_public_help() -> None:
    print(
        """usage: evaluate_teleopwm.py --checkpoint CHECKPOINT --data-root DATA_ROOT [options]

Evaluate TeleopWM future RGB rollouts.

Required:
  --checkpoint PATH                TeleopWM checkpoint.
  --data-root PATH                 Dataset split or town root.

Common options:
  --split val|test                 Split label for reporting.
  --output-dir PATH                Evaluation output directory.
  --max-samples N                  Number of rollout windows to evaluate.
  --sample-strategy first|uniform  Sample selection when indices are omitted.
  --indices I [I ...]              Explicit dataset indices.
  --device cuda|cpu                Evaluation device.

Advanced checkpoint-compatibility options are available in the source parser."""
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TeleopWM rollout evaluator implementation.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/path/to/mile_action_diverse/train/Town01"))
    parser.add_argument("--split", default="val", choices=["val", "validation", "test"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--normalize-controls", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--speed-scale", type=float, default=None)
    parser.add_argument("--control-steer-input-scale", type=float, default=None)
    parser.add_argument("--dual-wm-conditioning", choices=["add", "concat", "film"], default=None)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--sample-strategy", choices=["first", "uniform"], default="first")
    parser.add_argument("--num-grids", type=int, default=4)
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit dataset indices to evaluate and visualize exactly.",
    )
    parser.add_argument(
        "--grid-stride",
        type=int,
        default=1,
        help="Save every Nth evaluated sample as a grid. Use >1 to avoid near-duplicate sliding windows.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def select_indices(dataset_length: int, requested: int, strategy: str) -> list[int]:
    k = min(int(requested), int(dataset_length))
    if k <= 0:
        return []
    if strategy == "first":
        return list(range(k))
    if strategy == "uniform":
        return np.linspace(0, dataset_length - 1, k, dtype=int).tolist()
    raise ValueError(f"Unknown sample strategy: {strategy}")


def checkpoint_image_size(run_cfg: dict[str, Any], args: argparse.Namespace) -> tuple[int, int]:
    height = args.height if args.height is not None else run_cfg.get("height", 160)
    width = args.width if args.width is not None else run_cfg.get("width", 256)
    return int(height), int(width)


def model_from_checkpoint(path: Path, device: torch.device, args: argparse.Namespace) -> SimVPPredictor:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    run_cfg = checkpoint.get("config", {}).get("run", {})
    image_size = checkpoint_image_size(run_cfg, args)
    model_kwargs = dict(
        past_len=9,
        future_len=8,
        channels=3,
        image_size=image_size,
        hid_s=run_cfg.get("hid_s", 32),
        hid_t=run_cfg.get("hid_t", 256),
        n_s=run_cfg.get("n_s", 4),
        n_t=run_cfg.get("n_t", 4),
        model_type=run_cfg.get("model_type", "gSTA"),
        drop_path=run_cfg.get("drop_path", 0.0),
    )
    model_variant = run_cfg.get("model_variant", "rgb")
    model_variant = "av_simvp" if model_variant == "av" else model_variant
    if model_variant in {"av_simvp", "av_wm", "av_wm_dual", "av_wm_dual_bigwm"}:
        simvp_conditioning = run_cfg.get("simvp_conditioning", run_cfg.get("conditioning_fusion", "add"))
        simvp_conditioning_stage = run_cfg.get(
            "simvp_conditioning_stage",
            "multipoint" if run_cfg.get("conditioning_injection", "single") == "multipoint" else "input",
        )
        model = TeleopWMPredictor(
            **model_kwargs,
            model_variant=model_variant,
            action_dim=run_cfg.get("action_dim", 3),
            speed_dim=run_cfg.get("speed_dim", 1),
            conditioning_dim=run_cfg.get("conditioning_dim", 32),
            simvp_conditioning=simvp_conditioning,
            simvp_conditioning_stage=simvp_conditioning_stage,
            wm_latent_residual=run_cfg.get("wm_latent_residual", False),
            wm_residual_hidden_dim=run_cfg.get("wm_residual_hidden_dim", 128),
            wm_residual_scale=run_cfg.get("wm_residual_scale", 0.1),
            wm_residual_gated=run_cfg.get("wm_residual_gated", True),
            dual_fusion=run_cfg.get("dual_fusion", "gated_add"),
            dual_wm_scale=run_cfg.get("dual_wm_scale", 1.0),
            dual_wm_hidden_dim=run_cfg.get("dual_wm_hidden_dim", 128),
            dual_wm_num_layers=run_cfg.get("dual_wm_num_layers", 3),
            dual_wm_conditioning=args.dual_wm_conditioning or run_cfg.get("dual_wm_conditioning", "add"),
            dual_wm_gated=run_cfg.get("dual_wm_gated", True),
            aux_dynamics_hidden_dim=run_cfg.get("aux_dynamics_hidden_dim") if run_cfg.get("aux_dynamics_loss", False) else None,
            future_action_prediction=run_cfg.get("future_action_loss", False),
            future_action_hidden_dim=run_cfg.get("future_action_hidden_dim", 128),
            future_action_num_layers=run_cfg.get("future_action_num_layers", 1),
            future_action_dropout=run_cfg.get("future_action_dropout", 0.0),
            future_action_source=run_cfg.get("future_action_source", "final"),
            future_action_classification=run_cfg.get("future_action_cls_loss", False),
            future_action_head_variant=run_cfg.get("future_action_head_variant", "default"),
            future_action_detach_latents=run_cfg.get("future_action_detach_latents", True),
            future_action_future_motion_scale=run_cfg.get("future_action_future_motion_scale", 1.0),
            future_action_spatial_pooling=run_cfg.get("future_action_spatial_pooling", "global"),
            future_action_spatial_grid=run_cfg.get("future_action_spatial_grid", "1x1"),
            control_steer_input_scale=(
                args.control_steer_input_scale
                if args.control_steer_input_scale is not None
                else run_cfg.get("control_steer_input_scale", 1.0)
            ),
        )
    else:
        model = SimVPPredictor(**model_kwargs)
    load_checkpoint(path, model, map_location="cpu")
    model.to(device)
    model.eval()
    return model


def forward_model(model, past, batch_or_sample, device):
    if getattr(model, "requires_conditioning", False):
        past_actions = batch_or_sample["past_actions"].to(device, non_blocking=True)
        past_speed = batch_or_sample["past_speed"].to(device, non_blocking=True)
        if past_actions.ndim == 2:
            past_actions = past_actions.unsqueeze(0)
        if past_speed.ndim == 2:
            past_speed = past_speed.unsqueeze(0)
        return model(past, past_actions, past_speed)
    return model(past)


def tensor_to_image(frame: torch.Tensor) -> Image.Image:
    array = frame.detach().cpu().permute(1, 2, 0).numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def make_grid(
    past: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    path: Path,
    metadata: dict[str, object] | None = None,
    dataset_index: int | None = None,
) -> None:
    rows = [("past", past), ("gt", target), ("pred", prediction)]
    cell_w, cell_h = 160, 100
    label_h = 36
    title_h = 48 if metadata is not None or dataset_index is not None else 0
    width = max(tensor.shape[0] for _, tensor in rows) * cell_w
    height = title_h + len(rows) * (cell_h + label_h)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    if title_h:
        title_parts = []
        if dataset_index is not None:
            title_parts.append(f"dataset_idx={dataset_index}")
        if metadata is not None:
            title_parts.extend(
                [
                    f"town={metadata.get('town')}",
                    f"run={metadata['run_id']}",
                    f"start={metadata['start']}",
                    f"end={metadata['future_indices'][-1]}",
                ]
            )
        draw.text((4, 4), "  ".join(title_parts), fill=(0, 0, 0))
        if metadata is not None:
            draw.text(
                (4, 24),
                f"past={metadata['past_indices'][0]}..{metadata['past_indices'][-1]}  "
                f"future={metadata['future_indices'][0]}..{metadata['future_indices'][-1]}",
                fill=(0, 0, 0),
            )

    for row_idx, (label, frames) in enumerate(rows):
        y = title_h + row_idx * (cell_h + label_h)
        for col, frame in enumerate(frames):
            x = col * cell_w
            draw.text((x + 4, y + 2), label, fill=(0, 0, 0))
            temporal_text = f"t={col}"
            if metadata is not None:
                if label == "past":
                    temporal_text += f" abs={metadata['past_indices'][col]}"
                else:
                    temporal_text += f" abs={metadata['future_indices'][col]}"
            draw.text((x + 4, y + 18), temporal_text, fill=(0, 0, 0))
            image = tensor_to_image(frame)
            image.thumbnail((cell_w, cell_h))
            canvas.paste(image, (x, y + label_h))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def append_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")


def evaluate_explicit_indices(
    args: argparse.Namespace,
    dataset: CarlaRolloutDataset,
    model: SimVPPredictor,
    device: torch.device,
    output_dir: Path,
    run_dir: Path,
) -> int:
    results = []
    for dataset_index in args.indices:
        if not 0 <= dataset_index < len(dataset):
            raise IndexError(f"Dataset index {dataset_index} out of range for split {args.split} length {len(dataset)}")

        print(f"Evaluating dataset index: {dataset_index}")
        sample = dataset[dataset_index]
        metadata = sample["metadata"]
        print(
            f"  town={metadata.get('town')} "
            f"run={metadata['run_id']} "
            f"start={metadata['start']} "
            f"end={metadata['future_indices'][-1]}"
        )
        print(f"  past={metadata['past_indices']}")
        print(f"  future={metadata['future_indices']}")

        past = sample["past_frames"].unsqueeze(0).to(device)
        target = sample["future_frames"].unsqueeze(0).to(device)
        prediction = forward_model(model, past, sample, device).clamp(0.0, 1.0)
        metrics = video_metrics(prediction.float(), target.float())

        grid_path = output_dir / args.split / f"rollout_idx_{dataset_index:06d}.png"
        make_grid(
            past.squeeze(0).cpu(),
            prediction.squeeze(0).cpu(),
            target.squeeze(0).cpu(),
            grid_path,
            metadata=metadata,
            dataset_index=dataset_index,
        )
        print(f"  wrote: {grid_path}")

        results.append(
            {
                "dataset_index": dataset_index,
                "run_id": metadata["run_id"],
                "start": metadata["start"],
                "end": metadata["future_indices"][-1],
                "past_indices": metadata["past_indices"],
                "future_indices": metadata["future_indices"],
                "mae": metrics["mae"],
                "ssim": metrics["ssim"],
                "grid": str(grid_path),
            }
        )

    mean_mae = sum(item["mae"] for item in results) / max(len(results), 1)
    mean_ssim = sum(item["ssim"] for item in results) / max(len(results), 1)
    output_payload: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "mode": "explicit_indices",
        "dataset_length": len(dataset),
        "requested_samples": len(args.indices),
        "sample_strategy": "explicit_indices",
        "first_index": args.indices[0] if args.indices else None,
        "last_index": args.indices[-1] if args.indices else None,
        "indices": args.indices,
        "samples": len(results),
        "mae": mean_mae,
        "ssim": mean_ssim,
        "rollout_grids": str(output_dir / args.split),
        "rollouts": results,
    }
    eval_path = run_dir / f"eval_{args.split}_indices.json"
    write_json(eval_path, output_payload)
    append_text(
        run_dir / "train.log",
        [
            "",
            "[evaluation explicit indices]",
            f"checkpoint: {args.checkpoint}",
            f"split: {args.split}",
            f"indices: {args.indices}",
            "sample_strategy: explicit_indices",
            f"samples: {len(results)}",
            f"mae: {mean_mae:.6f}",
            f"ssim: {mean_ssim:.4f}",
            f"rollout_grids: {output_dir / args.split}",
        ],
    )

    print(f"samples: {len(results)}")
    print(f"mae: {mean_mae:.6f}")
    print(f"ssim: {mean_ssim:.4f}")
    print(f"wrote metrics: {eval_path}")
    return 0


@torch.no_grad()
def main() -> int:
    if any(item in {"-h", "--help"} for item in sys.argv[1:]):
        print_public_help()
        return 0
    args = parse_args()
    run_dir = infer_run_dir_from_checkpoint(args.checkpoint)
    output_dir = args.output_dir if args.output_dir is not None else run_dir / "rollout_grids"
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    run_cfg = checkpoint.get("config", {}).get("run", {})
    image_size = checkpoint_image_size(run_cfg, args)
    normalize_controls = (
        args.normalize_controls
        if args.normalize_controls is not None
        else bool(run_cfg.get("normalize_controls", False))
    )
    speed_scale = args.speed_scale if args.speed_scale is not None else float(run_cfg.get("speed_scale", 20.0))
    dataset = CarlaRolloutDataset(
        args.data_root,
        split=args.split,
        image_size=image_size,
        normalize_controls=normalize_controls,
        speed_scale=speed_scale,
        include_metadata=True,
    )
    model = model_from_checkpoint(args.checkpoint, device, args)

    if args.indices is not None:
        return evaluate_explicit_indices(args, dataset, model, device, output_dir, run_dir)

    dataset_length = len(dataset)
    requested_samples = args.max_samples if args.max_samples is not None else dataset_length
    selected_indices = select_indices(dataset_length, requested_samples, args.sample_strategy)
    if len(selected_indices) < dataset_length or args.sample_strategy != "first":
        dataset = Subset(dataset, selected_indices)
    print(f"dataset length: {dataset_length}")
    print(f"requested samples: {requested_samples}")
    print(f"evaluated samples: {len(selected_indices)}")
    print(f"sample strategy: {args.sample_strategy}")
    if selected_indices:
        print(f"first index: {selected_indices[0]}")
        print(f"last index: {selected_indices[-1]}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    totals = {"mae": 0.0, "ssim": 0.0}
    total_samples = 0
    seen_samples = 0
    grids_written = 0

    for batch in loader:
        past = batch["past_frames"].to(device, non_blocking=True)
        target = batch["future_frames"].to(device, non_blocking=True)
        prediction = forward_model(model, past, batch, device).clamp(0.0, 1.0)
        metrics = video_metrics(prediction.float(), target.float())
        batch_size = past.shape[0]
        totals["mae"] += metrics["mae"] * batch_size
        totals["ssim"] += metrics["ssim"] * batch_size
        total_samples += batch_size

        for local_idx in range(batch_size):
            if grids_written >= args.num_grids:
                break
            if seen_samples % args.grid_stride == 0:
                make_grid(
                    past[local_idx].cpu(),
                    prediction[local_idx].cpu(),
                    target[local_idx].cpu(),
                    output_dir / args.split / f"rollout_{grids_written:03d}.png",
                )
                grids_written += 1
            seen_samples += 1

    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "dataset_length": dataset_length,
        "requested_samples": requested_samples,
        "sample_strategy": args.sample_strategy,
        "first_index": selected_indices[0] if selected_indices else None,
        "last_index": selected_indices[-1] if selected_indices else None,
        "indices": selected_indices,
        "samples": total_samples,
        "mae": totals["mae"] / max(total_samples, 1),
        "ssim": totals["ssim"] / max(total_samples, 1),
        "rollout_grids": str(output_dir / args.split),
    }
    write_json(run_dir / f"eval_{args.split}.json", result)
    append_text(
        run_dir / "train.log",
        [
            "",
            "[evaluation]",
            f"checkpoint: {args.checkpoint}",
            f"split: {args.split}",
            f"dataset_length: {dataset_length}",
            f"requested_samples: {requested_samples}",
            f"sample_strategy: {args.sample_strategy}",
            f"first_index: {selected_indices[0] if selected_indices else None}",
            f"last_index: {selected_indices[-1] if selected_indices else None}",
            f"samples: {result['samples']}",
            f"mae: {result['mae']:.6f}",
            f"ssim: {result['ssim']:.4f}",
            f"rollout_grids: {result['rollout_grids']}",
        ],
    )

    print(f"checkpoint: {args.checkpoint}")
    print(f"split: {args.split}")
    print(f"dataset length: {dataset_length}")
    print(f"requested samples: {requested_samples}")
    print(f"sample strategy: {args.sample_strategy}")
    print(f"samples: {total_samples}")
    print(f"mae: {result['mae']:.6f}")
    print(f"ssim: {result['ssim']:.4f}")
    print(f"wrote grids: {output_dir / args.split}")
    print(f"wrote metrics: {run_dir / f'eval_{args.split}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
