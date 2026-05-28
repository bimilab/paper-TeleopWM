# Setup

This document describes the minimal environment needed to run the public TeleopWM pipeline.

## Requirements

TeleopWM is a PyTorch project. Full-resolution training and benchmarking should be run on a CUDA GPU. CPU is useful for import checks and small sanity runs.

Recommended:

- Python 3.10+
- PyTorch with CUDA for training/evaluation
- CARLA/MILE-style rollout dataset
- enough disk space for checkpoints and rollout grids

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv teleopwm
source teleopwm/bin/activate
pip install -r requirements.txt
```

If you need a specific CUDA build of PyTorch, install PyTorch following the official PyTorch instructions first, then install the remaining requirements.

## Sanity Check

Run:

```bash
python scripts/sanity_check.py
```

Expected behavior:

- imports the public TeleopWM API,
- constructs the release model,
- prints the model parameter count.

This check does not require a dataset or checkpoint.

## External Backbone

TeleopWM uses a SimVP visual backbone. The public code keeps the backbone wrapper in:

```text
src/models/simvp_predictor.py
```

Install the official backbone source before running model construction:

```bash
mkdir -p external
git clone https://github.com/Lupin1998/SimVPv2 external/SimVPv2
```

This is the only public-facing place where SimVP naming is expected. The final model implementation is exposed through:

```text
src/models/teleopwm_predictor.py
src/models/teleopwm.py
```

## Output Directories

Generated artifacts are ignored by git. Common output locations:

```text
outputs/teleopwm/
outputs/maneuver_metadata/
paper/figures/generated/
```
