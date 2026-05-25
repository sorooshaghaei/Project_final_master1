# Project Notes

This file records the main project artifacts and commands used for the final TER repository.

## Main Artifacts

| Artifact | Path | Role |
|---|---|---|
| Main entry point | `run.py` | Runs SSL training and TTT evaluation |
| SSL config | `configs/self_supervised.yaml` | SimCLR, linear evaluation, checkpoints, and source statistics |
| TTT config | `configs/test_time_training.yaml` | Severity 5 TTT evaluation |
| Severity 3 config | `configs/test_time_training_severity3.yaml` | Moderate corruption TTT evaluation |
| Report source | `report/report.tex` | TER report |
| Report PDF | `report/report.pdf` | Compiled report |

## Reproducible Commands

```bash
python run.py --task self_supervised --config configs/self_supervised.yaml
python run.py --task test_time_training --config configs/test_time_training.yaml
python run.py --task test_time_training --config configs/test_time_training_severity3.yaml
python scripts/sweep_ttt_steps.py
```

## Data Note

CIFAR-10-C is optional for a light TER run. If the official files are missing and synthetic fallback is enabled, the code uses locally generated CIFAR-10 test corruptions. This checks the pipeline but does not replace a full CIFAR-10-C benchmark.
