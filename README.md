# TeleopWM: A Real-Time Predictive World Model for Latency-Resilient Vision-Based Teleoperation

**Paper**  
## *TeleopWM: A Real-Time Predictive World Model for Latency-Resilient Vision-Based Teleoperation*

**Authors:** Aws Khalil and Jaerock Kwon  
**Affiliation:** Bio-Inspired Machine Intelligence (BIMI) Lab, University of Michigan-Dearborn  
**Status:** Under review  
**Project Page:** https://bimilab.github.io/paper-TeleopWM/  

---

## Research Overview

### Problem

Vision-based teleoperation is highly sensitive to communication latency. Even modest delay can make the operator's visual feedback stale, forcing control decisions to be made from an outdated scene. This is especially challenging in dynamic driving scenarios, where the vehicle, road geometry, and nearby traffic can change within a few hundred milliseconds.

The core problem addressed in this work is:

> **Can a lightweight world model predict near-future visual feedback and action trends quickly enough to support latency-resilient vision-based teleoperation?**

TeleopWM focuses on short-horizon predictive continuity rather than open-ended video generation. The objective is to provide a real-time predictive display that helps bridge the perception delay experienced by a remote operator.

---

<p align="center">
  <img src="docs/images/TeleopWM-Method-v2.svg" width="780">
  <br><em>Figure 1: Overview of the TeleopWM predictive world-model pipeline.</em>
</p>

---

### Method

**TeleopWM** is a compact latent world model for CARLA/MILE-style vision-based teleoperation rollouts. Given recent RGB frames and teleoperation controls, the model predicts:

- the next 8 RGB frames,
- future longitudinal and steering trends,
- action-aware latent dynamics for predictive display.

The final release model uses:

- a SimVP visual backbone,
- a TeleopWM latent dynamics branch,
- lightweight action/speed input encoding,
- 3x3 maneuver-speed balanced sampling,
- a spatial-token future-action head for steering and longitudinal prediction,
- runtime-oriented inference suitable for real-time predictive display studies.

The SimVP terminology in this repository is limited to the official visual backbone wrapper. The public model API is exposed as `TeleopWM`, `TeleopWMConfig`, and `TeleopWMPredictor`.

---

### Key Contribution

The main contributions of this work are summarized as follows:

- We propose TeleopWM, a lightweight predictive latent framework for latency-resilient vision-based teleoperation that jointly supports predictive display and future action forecasting.
- We introduce a motion-aware future action prediction strategy that estimates future driving behavior from latent motion dynamics rather than static latent appearance representations.
- We demonstrate that TeleopWM maintains lightweight real-time inference characteristics while producing stable predictive visual rollouts and multi-step future action forecasts under teleoperation-oriented constraints.

This repository is intended to support reproduction of the final TeleopWM paper pipeline.

---

## Documentation

Please follow the documentation in this order:

1. **[SETUP.md](docs/SETUP.md)** — installation and environment preparation  
2. **[DATASET.md](docs/DATASET.md)** — expected dataset layout and metadata generation  
3. **[TRAINING.md](docs/TRAINING.md)** — final TeleopWM training recipe  
4. **[EVALUATION.md](docs/EVALUATION.md)** — rollout, action, and runtime evaluation  
5. **[CHECKPOINTS.md](docs/CHECKPOINTS.md)** — checkpoint loading and compatibility notes  
6. **[REPRODUCTION.md](docs/REPRODUCTION.md)** — end-to-end paper reproduction checklist  
7. **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** — common issues and fixes  

Additional notes:

- **[scripts/README.md](scripts/README.md)** — public command-line entry points  
- **[paper/README.md](paper/README.md)** — paper figure generation utilities  

Each document is self-contained and focuses on one stage of use.

---

## Repository Structure

```text
src/                 TeleopWM datasets, model, trainer, metrics, and utilities
scripts/             Public training, evaluation, benchmark, and sanity commands
configs/             Optional release configuration files and examples
docs/                Setup, dataset, training, evaluation, and checkpoint docs
paper/               Paper figure utilities and plotting scripts
imgs/                Paper method figure and README images
external/            External dependencies such as the SimVP backbone source
tests/               Minimal release sanity tests
```

---

## Results & Key Metrics

The final TeleopWM pipeline is designed for short-horizon predictive display at 320x512 resolution with an 8-frame horizon.

| Category | Metric | Value |
|---|---:|---:|
| Rollout prediction | Horizon | 8 frames / approximately 533 ms at 15 FPS |
| Future action prediction | Outputs | longitudinal and steering trends |
| Runtime | Inference latency | 38.9 ms / rollout |
| Runtime | Prediction rate | 205.5 FPS |
| Runtime | Peak VRAM | 1.24 GB |

These numbers are release/reference values from the final paper configuration. They should be re-measured on your hardware using `scripts/benchmark_teleopwm.py`.

---

## Quick Start

### Installation

```bash
python3 -m venv teleopwm
source teleopwm/bin/activate
pip install -r requirements.txt
```

### Sanity Check

```bash
python scripts/sanity_check.py
```

### Evaluate a Checkpoint

```bash
python scripts/evaluate_teleopwm.py \
  --checkpoint /path/to/checkpoints/best.pt \
  --data-root /path/to/mile_action_diverse/test/Town05 \
  --split test \
  --sample-strategy uniform \
  --max-samples 64 \
  --device cuda
```

### Benchmark Runtime

```bash
python scripts/benchmark_teleopwm.py \
  --checkpoint /path/to/checkpoints/best.pt \
  --device cuda \
  --batch-size 1 \
  --warmup 20 \
  --iters 200
```

### Train the Final Model

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

See [docs/TRAINING.md](docs/TRAINING.md) for the full training workflow.

---

## Checkpoints and Data

- Checkpoint/model: https://huggingface.co/bimilab/TeleopWM
- Dataset: https://huggingface.co/datasets/bimilab/TeleopWM-Dataset
- Project page: https://bimilab.github.io/paper-TeleopWM/
- Repository: https://github.com/bimilab/paper-TeleopWM
- Video: https://youtu.be/WeKqqZuwBl0

See [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md) and [docs/DATASET.md](docs/DATASET.md) for expected formats.

---

## Citation

If you use TeleopWM or build on this release, please cite:

```bibtex
@misc{teleopwm2026,
  title={TeleopWM: A Real-Time Predictive World Model for Latency-Resilient Vision-Based Teleoperation},
  author={Khalil, Aws and Kwon, Jaerock},
  year={2026},
  note={Under review}
}
```

See [CITATION.cff](CITATION.cff) for citation metadata.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

---

## Authors

Aws Khalil and Jaerock Kwon  
Bio-Inspired Machine Intelligence (BIMI) Lab  
University of Michigan-Dearborn
