# Project_final_master1
TER M1 project focused on Test-Time Training and Self-Supervised Learning.

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
│   └── TER2.pdf
├── docs/
│   └── project_tracking.md
├── notebooks/
│   ├── basicUnderstanding.ipynb
│   └── choose_paper_for_TER.ipynb
├── src/
│   ├── __init__.py
│   ├── datasets.py
│   ├── test_time_training.py
│   ├── self_supervised.py
│   └── metrics.py
├── configs/
│   ├── test_time_training.yaml
│   └── self_supervised.yaml
└── data/
    ├── raw/
    ├── interim/
    └── processed/
```

## What Each Part Means
- `papers/`: your TER PDFs and reference papers.
- `docs/project_tracking.md`: links for presentations/progress PPTs + weekly decisions.
- `notebooks/`: exploration notebooks and paper-choice notebook.
- `src/test_time_training.py`: code skeleton for test-time adaptation/training.
- `src/self_supervised.py`: code skeleton for self-supervised training.
- `src/datasets.py`: dataset metadata/registry.
- `src/metrics.py`: evaluation helpers (accuracy, etc.).
- `configs/*.yaml`: experiment parameters (model, data path, hyperparameters).
- `data/raw`, `data/interim`, `data/processed`: dataset lifecycle.
- `run.py`: single entrypoint to run either track.

## Run
```bash
python run.py --task test_time_training --config configs/test_time_training.yaml
python run.py --task self_supervised --config configs/self_supervised.yaml
```

Short aliases are also accepted:
```bash
python run.py --task ttt --config configs/test_time_training.yaml
python run.py --task ssl --config configs/self_supervised.yaml
```
