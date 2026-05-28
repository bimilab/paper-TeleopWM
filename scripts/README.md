# Public Scripts

This directory contains the release-facing TeleopWM command-line tools.

## Main Commands

- `train_teleopwm.py` — train the final TeleopWM model.
- `evaluate_teleopwm.py` — evaluate future RGB rollout quality.
- `evaluate_future_actions.py` — evaluate future longitudinal and steering prediction.
- `benchmark_teleopwm.py` — benchmark inference latency, throughput, and memory.
- `build_maneuver_metadata.py` — build maneuver-speed metadata for balanced sampling.
- `plot_training_curves.py` — summarize training logs.
- `sanity_check.py` — minimal import and model-construction check.

## Typical Workflow

```bash
python scripts/sanity_check.py
python scripts/train_teleopwm.py --help
python scripts/evaluate_teleopwm.py --help
python scripts/benchmark_teleopwm.py --help
```

The public scripts use TeleopWM names. Compatibility shims for older checkpoints are retained internally and are not part of the recommended public workflow.
