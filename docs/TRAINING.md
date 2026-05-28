# Training

This document describes the final TeleopWM training workflow used by the release pipeline.

## Final Model Entry Point

Use:

```bash
python scripts/train_teleopwm.py
```

The public wrapper applies the final paper-model defaults, including:

- 320x512 image resolution,
- TeleopWM latent dynamics branch,
- conv1x1 dual fusion,
- FiLM-based control integration in the latent dynamics branch,
- motion-context-v2 future-action head,
- 2x4 spatial action tokens,
- future-action classification enabled,
- steering input/target scaling at 0.30.

## Prepare Sampling Metadata

The recommended training recipe uses maneuver-speed balanced sampling:

```bash
python scripts/build_maneuver_metadata.py \
  --data-root /path/to/mile_action_diverse/train \
  --output-dir outputs/maneuver_metadata \
  --speed-delta-threshold 0.3
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
  --lr 1e-3 \
  --device cuda
```

Outputs are written to:

```text
outputs/teleopwm/<timestamp>_<run_tag>/
```

The run directory contains checkpoints, logs, metrics, plots, and sampling summaries.

## Useful Training Options

For short verification runs:

```bash
python scripts/train_teleopwm.py \
  --data-root /path/to/train \
  --val-data-root /path/to/val \
  --max-train-steps 10 \
  --max-val-batches 1 \
  --device cuda
```

For CPU-only checks, reduce image size and batch size if needed. CPU training is not representative of final runtime.

## Notes

- Public documentation uses TeleopWM names only.
- The internal checkpoint config may contain older metadata fields so paper checkpoints remain loadable.
- Do not place training data inside `outputs/`.
