# Test-Time Training with Self-Supervised Learning

M1 TER project on test-time training for image classification under distribution shift.

The experiments use CIFAR-10 for self-supervised pretraining and CIFAR-10-C for
corruption robustness evaluation. The current implementation trains a SimCLR
backbone, fits a linear classifier on frozen features, saves source activation
statistics, and evaluates ActMAD adaptation at test time.

## Authors

Rayane KHATIM, Mehdi AGHAEI  
Universite Paris Cite

## Repository

```text
Project_final_master1/
├── configs/        Experiment configurations
├── data/           Local datasets
├── docs/           Project notes and supporting documents
├── notebooks/      Exploration notebooks
├── papers/         Reference papers
├── presentation/   Presentation slides
├── report/         Final report, figures, logs, and result tables
├── results/        Saved checkpoints and source statistics
├── scripts/        Utility scripts
├── src/            Training, adaptation, datasets, and metrics
├── run.py          Main experiment entrypoint
├── LICENSE
└── README.md
```

Main source files:

- `src/self_supervised.py`: SimCLR model, pretraining, linear evaluation, and source-statistics export.
- `src/test_time_training.py`: ActMAD source statistics, adaptation loss, and TTT evaluation adapter.
- `src/datasets.py`: CIFAR-10 and CIFAR-10-C dataloaders.
- `src/metrics.py`: evaluation metrics.

## Data

CIFAR-10 is loaded from:

```text
data/raw/cifar-10-batches-py/
```

CIFAR-10-C is expected at:

```text
data/raw/CIFAR-10-C/
```

The selected CIFAR-10-C corruption and severity are set in:

```text
configs/test_time_training.yaml
```

## Setup

From the project directory:

```bash
source ../.venv/bin/activate
```

Install the required Python packages if the environment has not already been prepared:

```bash
pip install torch torchvision numpy pyyaml
```

## Running Experiments

Run self-supervised training first:

```bash
python run.py --task self_supervised --config configs/self_supervised.yaml
```

This writes:

```text
results/self_supervised/simclr_backbone.pt
results/self_supervised/classifier.pt
results/self_supervised/source_stats.pt
```

Then run test-time training:

```bash
python run.py --task test_time_training --config configs/test_time_training.yaml
```

Short task names are also supported:

```bash
python run.py --task ssl --config configs/self_supervised.yaml
python run.py --task ttt --config configs/test_time_training.yaml
```

## Smoke Runs

Small configurations are available for quick checks:

```bash
python run.py --task ssl --config configs/smoke_self_supervised.yml
python run.py --task ttt --config configs/smoke_test_time_training.yml
```

The smoke TTT run expects the smoke SSL checkpoints in:

```text
results/smoke/self_supervised/
```

## Report

The final report is in:

```text
report/report.pdf
```

Supporting outputs are stored in:

```text
report/figures/
report/logs/
report/tables/
```
