# Self-Supervised Test-Time Training for Image Classification

Master 1 TER project on test-time adaptation for image classification under distribution shift.

The project uses SimCLR pretraining on CIFAR-10, a frozen ResNet18 backbone with linear evaluation, and ActMAD-style Test-Time Training on CIFAR-10-C corruptions or local synthetic fallback corruptions.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If PyTorch does not install correctly for the machine, install `torch` and `torchvision` from the official PyTorch instructions, then install the remaining dependencies:

```bash
pip install numpy pyyaml
```

## Data

CIFAR-10 is downloaded automatically when `dataset.download: true` is set in `configs/self_supervised.yaml`.

The official CIFAR-10-C files are optional for a light TER run. For a full CIFAR-10-C evaluation, place the `.npy` files here:

```text
data/raw/CIFAR-10-C/
```

Example files:

```text
gaussian_noise.npy
shot_noise.npy
labels.npy
```

If CIFAR-10-C is missing and `allow_synthetic_fallback: true` is enabled, the project generates lightweight corruptions from the CIFAR-10 test batch. This fallback is useful for checking the pipeline, but it is not a replacement for the official CIFAR-10-C benchmark.

## Train SSL and Linear Classifier

```bash
python run.py --task self_supervised --config configs/self_supervised.yaml
```

Alias:

```bash
python run.py --task ssl --config configs/self_supervised.yaml
```

This step trains the SimCLR backbone, saves the best and last checkpoints, trains the linear classifier on frozen features, and computes source activation statistics for TTT.

## Run TTT Evaluation

```bash
python run.py --task test_time_training --config configs/test_time_training.yaml
```

Alias:

```bash
python run.py --task ttt --config configs/test_time_training.yaml
```

The default TTT config uses severity 5, which is a hard corruption setting.

To run the same evaluation at moderate severity:

```bash
python run.py --task test_time_training --config configs/test_time_training_severity3.yaml
```

Severity 3 is moderate corruption. Severity 5 is hard corruption.

## Run TTT Steps Sweep

```bash
python scripts/sweep_ttt_steps.py
```

The sweep compares `steps_per_batch = 3, 5, 10` using the same backbone, classifier, corruptions, severity, batch size, and adaptation learning rate.

## Extra Corruption Evaluation

```bash
python scripts/eval_corruption.py
```

This script evaluates the full list of CIFAR-10-C corruption names with the same local fallback behavior.

## Outputs

```text
results/self_supervised/training_log.csv
results/self_supervised/best_backbone.pt
results/self_supervised/last_backbone.pt
results/self_supervised/simclr_backbone.pt
results/self_supervised/classifier.pt
results/self_supervised/source_stats.pt
results/test_time_training/results.csv
results/test_time_training/ttt_steps_sweep.csv
```

The `results/` folder is ignored by Git because it contains generated checkpoints, logs, and evaluation outputs.

## Report

The report source is in:

```text
report/report.tex
report/references.bib
```

The compiled report is:

```text
report/report.pdf
```

The report only states measured results from the project runs. New experiments should be added after rerunning the corresponding commands.
