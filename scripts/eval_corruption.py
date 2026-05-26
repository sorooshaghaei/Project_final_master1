import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.self_supervised import SimCLRModel, SSLConfig
from src.test_time_training import TTTAdapter, TTTConfig
from src.datasets import build_cifar10c_loader, missing_cifar10c_files
from src.metrics import evaluate_ttt

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu" # use mps when available

    # load the pretrained model and classifier
    ssl_cfg = SSLConfig()
    simclr = SimCLRModel(ssl_cfg)
    backbone_path = PROJECT_ROOT / "results/self_supervised/simclr_backbone.pt"
    classifier_path = PROJECT_ROOT / "results/self_supervised/classifier.pt"
    source_stats_path = PROJECT_ROOT / "results/self_supervised/source_stats.pt"
    simclr.backbone.load_state_dict(torch.load(backbone_path, map_location=device))
    clf = nn.Linear(simclr.feat_dim, 10)
    clf.load_state_dict(torch.load(classifier_path, map_location=device))
    model = nn.Sequential(simclr.backbone, clf).to(device)

    # load source stats for actmad
    source_stats = torch.load(source_stats_path, map_location=device)
    # set up ttt adaptation
    ttt_cfg = TTTConfig(lr=1e-4, steps_per_batch=10)
    adapter = TTTAdapter(model, ttt_cfg, source_stats)

    # evaluate selected corruptions
    CORRUPTIONS = [
        "gaussian_noise", "shot_noise", "impulse_noise",
        "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
        "snow", "frost", "fog", "brightness", "contrast",
        "elastic_transform", "pixelate", "jpeg_compression"
    ]
    SEVERITY = 5
    data_root = PROJECT_ROOT / "data/raw/cifar10c"
    missing = missing_cifar10c_files(data_root, CORRUPTIONS)
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "CIFAR-10-C files are missing.\n"
            f"Missing:\n{missing_text}\n"
            "Place the CIFAR-10-C .npy files under data/raw/cifar10c/CIFAR-10-C."
        )

    print(f"\n{'Corruption':<22} {'Baseline':>10} {'TTT':>10} {'Improvement':>10}")
    print("-" * 56)
    gains = []
    rows = []
    # evaluate each corruption
    for corruption in CORRUPTIONS:
        loader = build_cifar10c_loader(
            root=str(data_root),
            corruption=corruption,
            severity=SEVERITY,
            batch_size=16,
            num_workers=0,
        )
        base = evaluate_ttt(adapter, loader, device, use_ttt=False)
        ttt  = evaluate_ttt(adapter, loader, device, use_ttt=True)
        improvement = (ttt["accuracy"] - base["accuracy"]) * 100
        gains.append(improvement)
        sign = "+" if improvement >= 0 else ""
        print(f"{corruption:<22} {base['accuracy']*100:>9.2f}% {ttt['accuracy']*100:>9.2f}% {sign}{improvement:>9.2f}%")
        rows.append(
            {
                "corruption": corruption,
                "severity": SEVERITY,
                "baseline_accuracy_percent": round(base["accuracy"] * 100, 4),
                "actmad_accuracy_percent": round(ttt["accuracy"] * 100, 4),
                "improvement_percentage_points": round(improvement, 4),
                "baseline_correct": base["correct"],
                "baseline_total": base["total"],
                "actmad_correct": ttt["correct"],
                "actmad_total": ttt["total"],
            }
        )

    print("-" * 56)
    avg = sum(gains) / len(gains)
    avg_sign = "+" if avg >= 0 else ""
    print(f"{'Average':<22} {'':>10} {'':>10} {avg_sign}{avg:>8.2f}%")
    print(f"\nTTT helps on {sum(1 for g in gains if g > 0)}/15 corruptions")
    print(f"TTT hurts on {sum(1 for g in gains if g < 0)}/15 corruptions")

    output_path = PROJECT_ROOT / "results/test_time_training/all_corruptions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nresults saved to {output_path}")
