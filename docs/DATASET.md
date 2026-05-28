# TeleopWM Dataset

TeleopWM uses CARLA/MILE-style driving rollouts. Each rollout supplies recent camera frames, future frames, controls, speed, and metadata used to construct sliding prediction windows.

## Expected Layout

A typical release dataset is organized by split and town:

```text
data/mile_action_diverse/
  train/
    Town01/
    Town03/
    Town04/
  val/
    Town02/
  test/
    Town05/
```

Each run directory should contain RGB frame files and a `pd_dataframe.pkl` metadata file compatible with `src.datasets.CarlaRolloutDataset`.

## Window Definition

The final TeleopWM model uses:

- 9 past RGB frames,
- 8 future RGB frames,
- past controls `[throttle, steer, brake]`,
- scalar speed,
- future controls for the future-action objective.

The held-out evaluation used in the paper is Town05.

## Control Input Representation

Raw controls are stored as:

```text
[throttle, steer, brake]
```

TeleopWM converts these internally to:

```text
[longitudinal, scaled_steer, speed]
```

where:

```text
longitudinal = throttle - brake
scaled_steer = clamp(steer / control_steer_input_scale, -1, 1)
```

The final public defaults use:

```text
control_steer_input_scale = 0.30
future_steer_target_scale = 0.30
```

The first scale affects model control inputs. The second affects the future-action regression target during training and is kept separate.

## Maneuver-Speed Metadata

The final training recipe uses 3x3 maneuver-speed balanced sampling. This balances:

- lateral maneuver class: `straight`, `mild_turn`, `sharp_turn`
- longitudinal trend: `accel`, `const`, `decel`

Generate metadata with:

```bash
python scripts/build_maneuver_metadata.py \
  --data-root /path/to/mile_action_diverse/train \
  --output-dir outputs/maneuver_metadata \
  --speed-delta-threshold 0.3
```

Then pass it to training:

```bash
python scripts/train_teleopwm.py \
  --data-root /path/to/mile_action_diverse/train \
  --val-data-root /path/to/mile_action_diverse/val \
  --maneuver-speed-metadata outputs/maneuver_metadata/maneuver_metadata.csv \
  --sampling-strategy maneuver_speed_balanced
```

## Dataset Notes

- Keep train, validation, and test towns separate.
- Do not mix generated outputs or checkpoints into the dataset root.
- Use full paths when running on shared machines to avoid accidentally evaluating the wrong split.
- If you only want to verify installation, use `scripts/sanity_check.py`; it does not require a dataset.
