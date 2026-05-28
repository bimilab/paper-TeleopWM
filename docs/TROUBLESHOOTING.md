# Troubleshooting

This page lists common issues when running TeleopWM.

## `ModuleNotFoundError` for `src`

Run commands from the repository root:

```bash
cd /path/to/teleop_wm
python scripts/sanity_check.py
```

## CUDA Is Not Available

Use CPU for sanity checks:

```bash
python scripts/sanity_check.py
python scripts/benchmark_teleopwm.py --device cpu --iters 5 --warmup 1
```

Full-resolution training and final benchmark numbers require CUDA.

## Multiprocessing or DataLoader Permission Errors

Some sandboxed or restricted machines block multiprocessing sockets. Use:

```bash
--num-workers 0
```

for evaluation or small checks.

## Dataset Not Found

Verify that the split root points to the actual town/run directory:

```bash
python scripts/evaluate_teleopwm.py \
  --checkpoint /path/to/checkpoints/best.pt \
  --data-root /path/to/mile_action_diverse/test/Town05 \
  --split test
```

See [DATASET.md](DATASET.md) for expected layout.

## Checkpoint Loads but Metrics Look Different

Evaluation results can vary with:

- sample strategy and selected indices,
- image resolution,
- GPU and PyTorch version,
- whether a full split or a subset is evaluated.

Use explicit `--indices` for exact qualitative reproduction, and `--sample-strategy uniform` for quick representative checks.

## Out of Memory

Try:

- reducing evaluation batch size,
- using `--num-workers 0`,
- closing other GPU workloads,
- evaluating fewer samples for quick checks.
