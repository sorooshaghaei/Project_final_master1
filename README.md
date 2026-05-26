# Project_final_master1
TER M1 project Test-Time Training and Self-Supervised Learning

## Authors
Rayane KHATIM, Mehdi AGHAEI  
Universite Paris Cite

## Clean Structure
```
Project_final_master1/
├── README.md
├── LICENSE
├── run.py
├── papers/
│   ├── 1_sun20.pdf
│   ├── 3_When_test_time_Adaptation.pdf
│   ├── 2_NeurIPS-2021-ttt-when-does-self-supervised-test-time-training-fail-or-thrive-Paper.pdf
│   ├── 4_ActMAD_Activation_Matching_to_Align_Distributions_for_Test-Time-Training.pdf
│   └── TER2.pdf
├── notebooks/
│   ├── basicUnderstanding.ipynb
│   ├── choose_2_paper_for_TER.ipynb
│   ├── Test_Time_Training.pdf
│   └── choose_paper_for_TER.ipynb
├── src/
│   ├── __init__.py
│   ├── datasets.py
│   ├── test_time_training.py
│   ├── self_supervised.py
│   └── metrics.py
├── configs/
│   ├── self_supervised.yaml
│   ├── test_time_training.yaml
│   ├── smoke_self_supervised.yml
│   └── smoke_test_time_training.yml
└── data/
    └── raw/
```

## pipeline
- `papers/`:TER PDFs and reference papers
- `notebooks/`: exploration notebooks and paper-choice notebook
- `src/test_time_training.py`: test-time adaptation 
- `src/self_supervised.py`: self-supervised training 
- `src/datasets.py`: dataset metadata/registry
- `src/metrics.py`: evaluation helpers (accuracy, etc.)
- `configs/*.yaml`: experiment parameters (model, data path, hyperparameters)
- `configs/smoke_*.yml`: short smoke configurations kept for quick checks
- `scripts/`: helper scripts for dataset download and corruption evaluation.
- `data/raw`: dataset
- `run.py`: single entrypoint to run either track.

## Run

The main self-supervised configuration is `configs/self_supervised.yaml`. It trains SimCLR for **200 epochs** with batch size `256` and learning rate `0.03`.

```bash
cd Project_final_master1
python run.py --task self_supervised --config configs/self_supervised.yaml
python run.py --task ttt --config configs/test_time_training.yaml
```

Run `self_supervised` first. It creates the files expected by TTT:
`results/self_supervised/simclr_backbone.pt`,
`results/self_supervised/classifier.pt`, and
`results/self_supervised/source_stats.pt`.

Short aliases are also accepted:
```bash
python run.py --task ssl --config configs/self_supervised.yaml
python run.py --task ttt --config configs/test_time_training.yaml
```

## Evaluation

Run the all-corruption CIFAR-10-C evaluation from the `scripts` directory:

```bash
cd scripts
python eval_corruption.py
```

This prints the baseline and ActMAD TTT accuracy for all 15 CIFAR-10-C corruptions at severity 5 and saves:

```text
results/test_time_training/corruption_eval.csv
```

Run the adaptation-step sweep with:

```bash
cd Project_final_master1
python scripts/eval_corruption.py
python scripts/sweep_ttt_steps.py
```

This evaluates `steps_per_batch` values 3, 5, and 10 on the configured `shot_noise` setting and saves:

```text
results/test_time_training/ttt_steps_sweep.csv
```

The final all-corruption run improved accuracy on 14/15 corruptions, with an average gain of +15.40 percentage points. The only negative case was `brightness` (-1.55 percentage points).

## Data Paths

CIFAR-10 is downloaded by torchvision under:

```text
data/raw/cifar-10-batches-py
```

For the TTT config, CIFAR-10-C uses this root:

```text
data/raw/cifar10c
```

The loader expects the official `.npy` files inside:

```text
data/raw/cifar10c/CIFAR-10-C/
```

For example:

```text
data/raw/cifar10c/CIFAR-10-C/shot_noise.npy
data/raw/cifar10c/CIFAR-10-C/labels.npy
```
