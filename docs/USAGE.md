# TeleopWM Usage Summary

This page summarizes the public command-line workflow. For detailed instructions, use the focused docs:

- [SETUP.md](SETUP.md)
- [DATASET.md](DATASET.md)
- [TRAINING.md](TRAINING.md)
- [EVALUATION.md](EVALUATION.md)
- [CHECKPOINTS.md](CHECKPOINTS.md)
- [REPRODUCTION.md](REPRODUCTION.md)
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

## Public Commands

```bash
python scripts/sanity_check.py
python scripts/train_teleopwm.py --help
python scripts/evaluate_teleopwm.py --help
python scripts/benchmark_teleopwm.py --help
```

## Train

```bash
python scripts/train_teleopwm.py \
  --data-root /path/to/mile_action_diverse/train \
  --val-data-root /path/to/mile_action_diverse/val \
  --maneuver-speed-metadata outputs/maneuver_metadata/maneuver_metadata.csv \
  --sampling-strategy maneuver_speed_balanced \
  --normalize-controls \
  --speed-scale 20.0 \
  --batch-size 4 \
  --epochs 5 \
  --device cuda
```

## Evaluate Rollouts

```bash
python scripts/evaluate_teleopwm.py \
  --checkpoint /path/to/checkpoints/best.pt \
  --data-root /path/to/mile_action_diverse/test/Town05 \
  --split test \
  --sample-strategy uniform \
  --max-samples 64 \
  --device cuda
```

## Evaluate Future Actions

```bash
python scripts/evaluate_future_actions.py \
  --checkpoint /path/to/checkpoints/best.pt \
  --data-root /path/to/mile_action_diverse/test/Town05 \
  --split test \
  --sample-strategy uniform \
  --num-samples 256 \
  --device cuda
```

## Benchmark

```bash
python scripts/benchmark_teleopwm.py \
  --checkpoint /path/to/checkpoints/best.pt \
  --device cuda \
  --batch-size 1
```

## Generate a Paper Rollout Figure

```bash
python paper/scripts/compose_rollout_figure.py \
  --checkpoint /path/to/checkpoints/best.pt \
  --data-root /path/to/mile_action_diverse/test/Town05 \
  --indices 1987 1796 1760 1058 \
  --output-dir paper/figures/generated
```
