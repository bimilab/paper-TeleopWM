# TeleopWM Reproduction Checklist

This page provides a compact end-to-end checklist for reproducing the public TeleopWM evaluation and paper-figure workflow.

## 1. Download the Dataset

Download the TeleopWM dataset from Hugging Face:

https://huggingface.co/datasets/bimilab/TeleopWM-Dataset

Place the extracted data in a local directory such as:

```text
/path/to/mile_action_diverse
```

The evaluation examples below assume the Town05 test split is available at:

```text
/path/to/mile_action_diverse/test/Town05
```

## 2. Download the Checkpoint

Download the released TeleopWM checkpoint from Hugging Face:

https://huggingface.co/bimilab/TeleopWM

Example:

```bash
huggingface-cli download bimilab/TeleopWM \
  best.pt config.json \
  --local-dir /path/to/checkpoints/TeleopWM
```

The examples below refer to:

```text
/path/to/checkpoints/TeleopWM/best.pt
```

## 3. Install the Environment

Follow [SETUP.md](SETUP.md). A minimal setup is:

```bash
python -m venv teleopwm
source teleopwm/bin/activate
pip install -r requirements.txt
```

## 4. Run the Sanity Check

```bash
python scripts/sanity_check.py
```

This verifies that the public TeleopWM API can be constructed and run on small random tensors.

## 5. Run Rollout Evaluation

```bash
python scripts/evaluate_teleopwm.py \
  --checkpoint /path/to/checkpoints/TeleopWM/best.pt \
  --data-root /path/to/mile_action_diverse/test/Town05 \
  --split test \
  --sample-strategy uniform \
  --max-samples 64 \
  --device cuda
```

Use `--device cpu` if CUDA is unavailable.

## 6. Run Future-Action Evaluation

```bash
python scripts/evaluate_future_actions.py \
  --checkpoint /path/to/checkpoints/TeleopWM/best.pt \
  --data-root /path/to/mile_action_diverse/test/Town05 \
  --split test \
  --sample-strategy uniform \
  --num-samples 256 \
  --device cuda
```

## 7. Reproduce Paper Figures

Qualitative rollout figure:

```bash
python paper/scripts/compose_rollout_figure.py \
  --checkpoint /path/to/checkpoints/TeleopWM/best.pt \
  --data-root /path/to/mile_action_diverse/test/Town05 \
  --indices 1987 1796 1760 1058 \
  --output-dir paper/figures/generated \
  --output-name main_rollout_action_figure
```

Future-action evaluation figure:

```bash
python paper/scripts/plot_future_action_eval.py \
  --input-dir /path/to/future_action_eval \
  --output-dir paper/figures/generated \
  --output-name future_action_eval
```

Benchmark figure:

```bash
python paper/scripts/generate_benchmark_figure.py \
  --input /path/to/benchmark.json \
  --output-dir paper/figures/generated
```

Sampling figure:

```bash
python paper/scripts/generate_sampling_figure.py \
  --input /path/to/sampling_summary.json \
  --output-dir paper/figures/generated
```

## 8. Run Runtime Benchmark

```bash
python scripts/benchmark_teleopwm.py \
  --checkpoint /path/to/checkpoints/TeleopWM/best.pt \
  --device cuda \
  --batch-size 1
```

Runtime values should be re-measured on the target hardware used for deployment or comparison.
