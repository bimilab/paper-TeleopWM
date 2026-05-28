# Checkpoints

This document explains TeleopWM checkpoint conventions and compatibility.

## Expected Checkpoint

Training writes checkpoints under:

```text
outputs/teleopwm/<timestamp>_<run_tag>/checkpoints/
```

The main checkpoint used for evaluation is:

```text
best.pt
```

## Loading

Use public scripts:

```bash
python scripts/evaluate_teleopwm.py \
  --checkpoint /path/to/checkpoints/best.pt \
  --data-root /path/to/mile_action_diverse/test/Town05 \
  --split test
```

The evaluator reads the checkpoint config and reconstructs the appropriate TeleopWM model.

## Compatibility Notes

The release API is:

```python
from src.models import TeleopWM, TeleopWMConfig, build_teleopwm_model
```

The final implementation lives in:

```text
src/models/teleopwm_predictor.py
```

Some saved checkpoint configs may contain older metadata fields. These are handled internally and should not affect public usage.

## Public Downloads

- Final paper checkpoint/model: https://huggingface.co/bimilab/TeleopWM
- Dataset: https://huggingface.co/datasets/bimilab/TeleopWM-Dataset

## Common Issues

- If checkpoint loading fails with missing keys, verify that the checkpoint was trained with the public TeleopWM release code or a compatible TeleopWM paper checkpoint..
- If evaluation is slow or multiprocessing fails on a restricted machine, try `--num-workers 0`.
- If CUDA memory is insufficient, reduce batch size for evaluation and benchmarking.
