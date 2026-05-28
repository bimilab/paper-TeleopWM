# Evaluation

TeleopWM includes public scripts for rollout quality, future-action prediction, runtime benchmarking, and paper figure generation.

## RGB Rollout Evaluation

```bash
python scripts/evaluate_teleopwm.py \
  --checkpoint /path/to/checkpoints/best.pt \
  --data-root /path/to/mile_action_diverse/test/Town05 \
  --split test \
  --sample-strategy uniform \
  --max-samples 64 \
  --device cuda
```

Outputs include metrics and rollout grids in the checkpoint run directory unless an explicit output directory is provided.

## Future-Action Evaluation

```bash
python scripts/evaluate_future_actions.py \
  --checkpoint /path/to/checkpoints/best.pt \
  --data-root /path/to/mile_action_diverse/test/Town05 \
  --split test \
  --sample-strategy uniform \
  --num-samples 256 \
  --device cuda
```

This reports longitudinal and steering metrics, including MAE, RMSE, R2, Pearson correlation, per-step metrics, and action plots.

## Runtime Benchmark

```bash
python scripts/benchmark_teleopwm.py \
  --checkpoint /path/to/checkpoints/best.pt \
  --device cuda \
  --batch-size 1 \
  --warmup 20 \
  --iters 200
```

Benchmark results depend on GPU, PyTorch version, resolution, and driver state. Re-measure on your target machine before reporting numbers.

## Paper Figure Generation

```bash
python paper/scripts/compose_rollout_figure.py \
  --checkpoint /path/to/checkpoints/best.pt \
  --data-root /path/to/mile_action_diverse/test/Town05 \
  --indices 1987 1796 1760 1058 \
  --output-dir paper/figures/generated \
  --output-name main_rollout_action_figure \
  --device cuda
```

Additional paper plotting scripts are documented in [paper/README.md](../paper/README.md).

## Sampling Strategy for Evaluation

When explicit indices are not provided, evaluation scripts support:

- `--sample-strategy first`
- `--sample-strategy uniform`

Use `uniform` for representative quick evaluations across a split. Use explicit indices for reproducible paper figures.
