# TER - Self-Supervised Test-Time Training for Image Classification

M1 TER project on test-time adaptation for image classification.

The project keeps the original structure: SimCLR, ResNet18, CIFAR-10, CIFAR-10-C, and ActMAD adaptation. The main changes are in the training loop, reproducibility, checkpoints, and logs.

## Installation

Create a Python environment and install the main dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `torch` or `torchvision` do not install correctly on your machine, install them with the command for your system from the official PyTorch website, then run:

```bash
pip install numpy pyyaml
```

## Self-Supervised Training

```bash
python run.py --task self_supervised --config configs/self_supervised.yaml
```

Equivalent alias:

```bash
python run.py --task ssl --config configs/self_supervised.yaml
```

This step creates:

```text
results/self_supervised/best_backbone.pt
results/self_supervised/last_backbone.pt
results/self_supervised/simclr_backbone.pt
results/self_supervised/classifier.pt
results/self_supervised/source_stats.pt
results/self_supervised/training_log.csv
```

The configuration no longer uses a fixed 100 epochs. It uses:

```yaml
max_epochs: 50
patience: 7
min_delta: 0.001
```

The model stops earlier if the SimCLR loss does not improve enough.

## Test-Time Training Evaluation

Before this step, self-supervised training must already have been run.

CIFAR-10-C should be placed here:

```text
data/raw/cifar10c/CIFAR-10-C/
```

with the `.npy` files, for example:

```text
gaussian_noise.npy
shot_noise.npy
labels.npy
```

CIFAR-10-C is optional for a quick TER test. If these files are missing and `allow_synthetic_fallback: true` is set in the config, the code uses lightweight synthetic corruptions generated from the CIFAR-10 test batch. This fallback is for demonstration and TER testing, not a replacement for a full CIFAR-10-C evaluation.

Run the evaluation:

```bash
python run.py --task test_time_training --config configs/test_time_training.yaml
```

Equivalent alias:

```bash
python run.py --task ttt --config configs/test_time_training.yaml
```

The results are saved in:

```text
results/test_time_training/results.csv
```

## Evaluate All CIFAR-10-C Corruptions

```bash
python scripts/eval_corruption.py
```

This script runs through the 15 CIFAR-10-C corruptions at severity 5 and also writes:

```text
results/test_time_training/results.csv
```

## Report

The LaTeX report is in:

```text
report/report.tex
report/references.bib
report/report.pdf
```

The numerical results are not invented in the report. The experimental table should be filled after the commands are actually run.
