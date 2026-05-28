#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models import TeleopWMPredictor, SimVPPredictor
from src.utils import infer_run_dir_from_checkpoint, load_checkpoint, write_json


def print_public_help() -> None:
    print(
        """usage: benchmark_teleopwm.py [--checkpoint CHECKPOINT] [options]

Benchmark TeleopWM inference latency, throughput, and memory.

Common options:
  --checkpoint PATH                Optional checkpoint to load.
  --output-dir PATH                Benchmark output directory.
  --batch-size N                   Batch size for benchmark.
  --warmup N                       Warmup iterations.
  --iters N                        Timed iterations.
  --device cuda|cpu                Benchmark device.
  --height N --width N             Input resolution if no checkpoint is used.

Advanced checkpoint-compatibility options are available in the source parser."""
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TeleopWM runtime benchmark implementation.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--normalize-controls", action="store_true")
    parser.add_argument("--speed-scale", type=float, default=20.0)
    parser.add_argument("--hid-s", type=int, default=32)
    parser.add_argument("--hid-t", type=int, default=256)
    parser.add_argument("--n-s", type=int, default=4)
    parser.add_argument("--n-t", type=int, default=4)
    parser.add_argument("--model-type", default="gSTA")
    parser.add_argument("--model-variant", choices=["rgb", "av", "av_simvp", "av_wm", "av_wm_dual", "av_wm_dual_bigwm"], default="rgb")
    parser.add_argument("--action-dim", type=int, default=3)
    parser.add_argument("--speed-dim", type=int, default=1)
    parser.add_argument("--conditioning-dim", type=int, default=32)
    parser.add_argument("--simvp-conditioning", choices=["none", "add", "concat", "film"], default=None)
    parser.add_argument("--simvp-conditioning-stage", choices=["input", "multipoint"], default=None)
    parser.add_argument("--conditioning-fusion", choices=["add", "concat", "film"], default=None)
    parser.add_argument("--conditioning-injection", choices=["single", "multipoint"], default=None)
    parser.add_argument("--wm-latent-residual", action="store_true")
    parser.add_argument("--wm-residual-hidden-dim", type=int, default=128)
    parser.add_argument("--wm-residual-scale", type=float, default=0.1)
    parser.add_argument("--wm-residual-gated", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dual-fusion", choices=["add", "gated_add", "convex", "wm_only", "simvp_only", "conv1x1"], default="gated_add")
    parser.add_argument("--dual-wm-scale", type=float, default=1.0)
    parser.add_argument("--dual-wm-hidden-dim", type=int, default=128)
    parser.add_argument("--dual-wm-num-layers", type=int, default=3)
    parser.add_argument("--dual-wm-conditioning", choices=["add", "concat", "film"], default="add")
    parser.add_argument("--dual-wm-gated", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aux-dynamics-hidden-dim", type=int, default=None)
    parser.add_argument("--future-action-loss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--future-action-hidden-dim", type=int, default=128)
    parser.add_argument("--future-action-num-layers", type=int, default=1)
    parser.add_argument("--future-action-dropout", type=float, default=0.0)
    parser.add_argument("--future-action-source", choices=["final", "wm", "simvp"], default="final")
    parser.add_argument("--future-action-cls-loss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--future-action-head-variant", choices=["default", "motion_context", "motion_context_v2"], default="default")
    parser.add_argument("--future-action-detach-latents", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--future-action-future-motion-scale", type=float, default=1.0)
    parser.add_argument("--future-action-spatial-pooling", choices=["global", "grid"], default="global")
    parser.add_argument("--future-action-spatial-grid", default="1x1")
    parser.add_argument("--control-steer-input-scale", type=float, default=None)
    return parser.parse_args()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def checkpoint_image_size(run_cfg: dict[str, Any], args: argparse.Namespace) -> tuple[int, int]:
    height = args.height if args.height is not None else run_cfg.get("height", 160)
    width = args.width if args.width is not None else run_cfg.get("width", 256)
    return int(height), int(width)


def main() -> int:
    if any(item in {"-h", "--help"} for item in sys.argv[1:]):
        print_public_help()
        return 0
    args = parse_args()
    explicit_flags = {item.split("=")[0] for item in sys.argv[1:] if item.startswith("--")}
    if args.model_variant == "av_wm_dual_bigwm":
        if "--dual-wm-hidden-dim" not in explicit_flags:
            args.dual_wm_hidden_dim = 512
        if "--dual-wm-num-layers" not in explicit_flags:
            args.dual_wm_num_layers = 3
    run_dir = infer_run_dir_from_checkpoint(args.checkpoint) if args.checkpoint else Path("outputs/teleopwm_benchmark")
    benchmark_dir = args.output_dir if args.output_dir is not None else run_dir / "benchmark"
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    image_size = (args.height or 160, args.width or 256)
    model_variant = "av_simvp" if args.model_variant == "av" else args.model_variant
    action_dim = args.action_dim
    speed_dim = args.speed_dim
    conditioning_dim = args.conditioning_dim
    simvp_conditioning = args.simvp_conditioning or args.conditioning_fusion or "concat"
    simvp_conditioning_stage = args.simvp_conditioning_stage or ("multipoint" if args.conditioning_injection == "multipoint" else "input")
    conditioning_fusion = None if simvp_conditioning == "none" else simvp_conditioning
    conditioning_injection = "multipoint" if simvp_conditioning_stage == "multipoint" else "single"
    aux_dynamics_hidden_dim = args.aux_dynamics_hidden_dim
    model_kwargs = dict(
        past_len=9,
        future_len=8,
        channels=3,
        image_size=image_size,
        hid_s=args.hid_s,
        hid_t=args.hid_t,
        n_s=args.n_s,
        n_t=args.n_t,
        model_type=args.model_type,
    )
    model = (
        TeleopWMPredictor(
            **model_kwargs,
            action_dim=action_dim,
            speed_dim=speed_dim,
            conditioning_dim=conditioning_dim,
            model_variant=model_variant,
            simvp_conditioning=simvp_conditioning,
            simvp_conditioning_stage=simvp_conditioning_stage,
            wm_latent_residual=args.wm_latent_residual,
            wm_residual_hidden_dim=args.wm_residual_hidden_dim,
            wm_residual_scale=args.wm_residual_scale,
            wm_residual_gated=args.wm_residual_gated,
            dual_fusion=args.dual_fusion,
            dual_wm_scale=args.dual_wm_scale,
            dual_wm_hidden_dim=args.dual_wm_hidden_dim,
            dual_wm_num_layers=args.dual_wm_num_layers,
            dual_wm_conditioning=args.dual_wm_conditioning,
            dual_wm_gated=args.dual_wm_gated,
            aux_dynamics_hidden_dim=aux_dynamics_hidden_dim,
            future_action_prediction=args.future_action_loss,
            future_action_hidden_dim=args.future_action_hidden_dim,
            future_action_num_layers=args.future_action_num_layers,
            future_action_dropout=args.future_action_dropout,
            future_action_source=args.future_action_source,
            future_action_classification=args.future_action_cls_loss,
            future_action_head_variant=getattr(args, "future_action_head_variant", "default"),
            future_action_detach_latents=getattr(args, "future_action_detach_latents", True),
            future_action_future_motion_scale=getattr(args, "future_action_future_motion_scale", 1.0),
            future_action_spatial_pooling=getattr(args, "future_action_spatial_pooling", "global"),
            future_action_spatial_grid=getattr(args, "future_action_spatial_grid", "1x1"),
            control_steer_input_scale=args.control_steer_input_scale or 1.0,
        )
        if model_variant in {"av_simvp", "av_wm", "av_wm_dual", "av_wm_dual_bigwm"}
        else SimVPPredictor(**model_kwargs)
    )
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        run_cfg = checkpoint.get("config", {}).get("run", {})
        image_size = checkpoint_image_size(run_cfg, args)
        model_variant = run_cfg.get("model_variant", "rgb")
        model_variant = "av_simvp" if model_variant == "av" else model_variant
        action_dim = run_cfg.get("action_dim", action_dim)
        speed_dim = run_cfg.get("speed_dim", speed_dim)
        conditioning_dim = run_cfg.get("conditioning_dim", conditioning_dim)
        simvp_conditioning = run_cfg.get("simvp_conditioning", run_cfg.get("conditioning_fusion", "add"))
        simvp_conditioning_stage = run_cfg.get(
            "simvp_conditioning_stage",
            "multipoint" if run_cfg.get("conditioning_injection", "single") == "multipoint" else "input",
        )
        conditioning_fusion = None if simvp_conditioning == "none" else simvp_conditioning
        conditioning_injection = "multipoint" if simvp_conditioning_stage == "multipoint" else "single"
        wm_latent_residual = bool(run_cfg.get("wm_latent_residual", False))
        wm_residual_hidden_dim = int(run_cfg.get("wm_residual_hidden_dim", args.wm_residual_hidden_dim))
        wm_residual_scale = float(run_cfg.get("wm_residual_scale", args.wm_residual_scale))
        wm_residual_gated = bool(run_cfg.get("wm_residual_gated", args.wm_residual_gated))
        dual_fusion = run_cfg.get("dual_fusion", args.dual_fusion)
        dual_wm_scale = float(run_cfg.get("dual_wm_scale", args.dual_wm_scale))
        dual_wm_hidden_dim = int(run_cfg.get("dual_wm_hidden_dim", args.dual_wm_hidden_dim))
        dual_wm_num_layers = int(run_cfg.get("dual_wm_num_layers", args.dual_wm_num_layers))
        dual_wm_conditioning = run_cfg.get("dual_wm_conditioning", args.dual_wm_conditioning)
        dual_wm_gated = bool(run_cfg.get("dual_wm_gated", args.dual_wm_gated))
        aux_dynamics_hidden_dim = run_cfg.get("aux_dynamics_hidden_dim") if run_cfg.get("aux_dynamics_loss", False) else None
        future_action_prediction = bool(run_cfg.get("future_action_loss", False))
        future_action_hidden_dim = int(run_cfg.get("future_action_hidden_dim", args.future_action_hidden_dim))
        future_action_num_layers = int(run_cfg.get("future_action_num_layers", args.future_action_num_layers))
        future_action_dropout = float(run_cfg.get("future_action_dropout", args.future_action_dropout))
        future_action_source = run_cfg.get("future_action_source", args.future_action_source)
        future_action_head_variant = run_cfg.get("future_action_head_variant", args.future_action_head_variant)
        future_action_detach_latents = bool(run_cfg.get("future_action_detach_latents", args.future_action_detach_latents))
        future_action_future_motion_scale = float(
            run_cfg.get("future_action_future_motion_scale", args.future_action_future_motion_scale)
        )
        future_action_spatial_pooling = run_cfg.get("future_action_spatial_pooling", args.future_action_spatial_pooling)
        future_action_spatial_grid = run_cfg.get("future_action_spatial_grid", args.future_action_spatial_grid)
        control_steer_input_scale = (
            args.control_steer_input_scale
            if args.control_steer_input_scale is not None
            else run_cfg.get("control_steer_input_scale", 1.0)
        )
        normalize_controls = bool(run_cfg.get("normalize_controls", args.normalize_controls))
        speed_scale = float(run_cfg.get("speed_scale", args.speed_scale))
        if run_cfg:
            model_kwargs = dict(
                past_len=9,
                future_len=8,
                channels=3,
                image_size=image_size,
                hid_s=run_cfg.get("hid_s", args.hid_s),
                hid_t=run_cfg.get("hid_t", args.hid_t),
                n_s=run_cfg.get("n_s", args.n_s),
                n_t=run_cfg.get("n_t", args.n_t),
                model_type=run_cfg.get("model_type", args.model_type),
                drop_path=run_cfg.get("drop_path", 0.0),
            )
            model = (
                TeleopWMPredictor(
                    **model_kwargs,
                    action_dim=action_dim,
                    speed_dim=speed_dim,
                    conditioning_dim=conditioning_dim,
                    model_variant=model_variant,
                    simvp_conditioning=simvp_conditioning,
                    simvp_conditioning_stage=simvp_conditioning_stage,
                    wm_latent_residual=wm_latent_residual,
                    wm_residual_hidden_dim=wm_residual_hidden_dim,
                    wm_residual_scale=wm_residual_scale,
                    wm_residual_gated=wm_residual_gated,
                    dual_fusion=dual_fusion,
                    dual_wm_scale=dual_wm_scale,
                    dual_wm_hidden_dim=dual_wm_hidden_dim,
                    dual_wm_num_layers=dual_wm_num_layers,
                    dual_wm_conditioning=dual_wm_conditioning,
                    dual_wm_gated=dual_wm_gated,
                    aux_dynamics_hidden_dim=aux_dynamics_hidden_dim,
                    future_action_prediction=future_action_prediction,
                    future_action_hidden_dim=future_action_hidden_dim,
                    future_action_num_layers=future_action_num_layers,
                    future_action_dropout=future_action_dropout,
                    future_action_source=future_action_source,
                    future_action_classification=run_cfg.get("future_action_cls_loss", False),
                    future_action_head_variant=future_action_head_variant,
                    future_action_detach_latents=future_action_detach_latents,
                    future_action_future_motion_scale=future_action_future_motion_scale,
                    future_action_spatial_pooling=future_action_spatial_pooling,
                    future_action_spatial_grid=future_action_spatial_grid,
                    control_steer_input_scale=control_steer_input_scale,
                )
                if model_variant in {"av_simvp", "av_wm", "av_wm_dual", "av_wm_dual_bigwm"}
                else SimVPPredictor(**model_kwargs)
            )
        load_checkpoint(args.checkpoint, model, map_location="cpu")

    model.to(device)
    model.eval()
    height, width = image_size
    x = torch.randn(args.batch_size, 9, 3, height, width, device=device)
    actions = torch.randn(args.batch_size, 9, action_dim, device=device)
    speed = torch.randn(args.batch_size, 9, speed_dim, device=device)
    normalize_controls = locals().get("normalize_controls", args.normalize_controls)
    speed_scale = locals().get("speed_scale", args.speed_scale)
    if normalize_controls:
        speed = speed / speed_scale

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def run_once():
        if not getattr(model, "requires_conditioning", False):
            return model(x)
        if getattr(model, "future_action_head", None) is not None:
            return model(x, actions, speed, return_latents=True)
        return model(x, actions, speed)

    last_output = None
    with torch.no_grad():
        for _ in range(args.warmup):
            last_output = run_once()
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(args.iters):
            last_output = run_once()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    latency_ms = elapsed * 1000.0 / args.iters
    samples_per_second = args.batch_size * args.iters / elapsed
    future_frames_per_second = args.batch_size * 8 * args.iters / elapsed
    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    result: dict[str, Any] = {
        "device": str(device),
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "model_variant": model_variant,
        "action_dim": action_dim,
        "speed_dim": speed_dim,
        "conditioning_dim": conditioning_dim,
        "simvp_conditioning": simvp_conditioning,
        "simvp_conditioning_stage": simvp_conditioning_stage,
        "conditioning_fusion": conditioning_fusion,
        "conditioning_injection": conditioning_injection,
        "conditioning_representation": "longitudinal_steer_speed" if model_variant in {"av_simvp", "av_wm", "av_wm_dual", "av_wm_dual_bigwm"} else None,
        "wm_latent_residual": locals().get("wm_latent_residual", args.wm_latent_residual),
        "wm_residual_hidden_dim": locals().get("wm_residual_hidden_dim", args.wm_residual_hidden_dim),
        "wm_residual_scale": locals().get("wm_residual_scale", args.wm_residual_scale),
        "wm_residual_gated": locals().get("wm_residual_gated", args.wm_residual_gated),
        "dual_fusion": locals().get("dual_fusion", args.dual_fusion),
        "dual_wm_scale": locals().get("dual_wm_scale", args.dual_wm_scale),
        "dual_wm_hidden_dim": locals().get("dual_wm_hidden_dim", args.dual_wm_hidden_dim),
        "dual_wm_num_layers": locals().get("dual_wm_num_layers", args.dual_wm_num_layers),
        "dual_wm_conditioning": locals().get("dual_wm_conditioning", args.dual_wm_conditioning),
        "dual_wm_gated": locals().get("dual_wm_gated", args.dual_wm_gated),
        "dual_conv1x1_fusion_stats": (
            model.dual_conv1x1_fusion_stats()
            if hasattr(model, "dual_conv1x1_fusion_stats")
            else None
        ),
        "aux_dynamics_hidden_dim": aux_dynamics_hidden_dim,
        "has_future_action_head": getattr(model, "future_action_head", None) is not None,
        "has_future_action_cls_heads": bool(
            getattr(getattr(model, "future_action_head", None), "classification", False)
        ),
        "future_action_source": getattr(model, "future_action_source", locals().get("future_action_source", args.future_action_source)),
        "future_action_head_variant": getattr(model, "future_action_head_variant", locals().get("future_action_head_variant", args.future_action_head_variant)),
        "future_action_hidden_dim": getattr(model, "future_action_hidden_dim", locals().get("future_action_hidden_dim", args.future_action_hidden_dim)),
        "future_action_detach_latents": getattr(model, "future_action_detach_latents", locals().get("future_action_detach_latents", args.future_action_detach_latents)),
        "future_action_future_motion_scale": getattr(model, "future_action_future_motion_scale", locals().get("future_action_future_motion_scale", args.future_action_future_motion_scale)),
        "future_action_spatial_pooling": getattr(model, "future_action_spatial_pooling", locals().get("future_action_spatial_pooling", args.future_action_spatial_pooling)),
        "future_action_spatial_grid": getattr(model, "future_action_spatial_grid", locals().get("future_action_spatial_grid", args.future_action_spatial_grid)),
        "future_action_token_dim": getattr(model, "future_action_token_dim", None),
        "control_steer_input_scale": getattr(model, "control_steer_input_scale", locals().get("control_steer_input_scale", args.control_steer_input_scale or 1.0)),
        "pred_future_actions_shape": (
            list(last_output["pred_future_actions"].shape)
            if isinstance(last_output, dict) and "pred_future_actions" in last_output
            else None
        ),
        "normalize_controls": normalize_controls,
        "speed_scale": speed_scale,
        "parameters": count_parameters(model),
        "batch_size": args.batch_size,
        "height": height,
        "width": width,
        "warmup": args.warmup,
        "iters": args.iters,
        "latency_ms_per_batch": latency_ms,
        "samples_per_second": samples_per_second,
        "future_frames_per_second": future_frames_per_second,
        "peak_vram_mb": peak_vram_mb,
    }
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    write_json(benchmark_dir / "benchmark.json", result)
    with (benchmark_dir / "benchmark.txt").open("w", encoding="utf-8") as handle:
        for key, value in result.items():
            handle.write(f"{key}: {value}\n")

    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"parameters: {count_parameters(model):,}")
    print(f"batch_size: {args.batch_size}")
    print(f"latency_ms_per_batch: {latency_ms:.3f}")
    print(f"samples_per_second: {samples_per_second:.3f}")
    print(f"future_frames_per_second: {future_frames_per_second:.3f}")
    print(f"peak_vram_mb: {peak_vram_mb:.1f}")
    print(f"wrote benchmark: {benchmark_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
