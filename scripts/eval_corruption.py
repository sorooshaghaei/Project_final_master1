"""Evaluate baseline and ActMAD TTT on all CIFAR-10-C corruptions."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(".")

from src.datasets import CIFAR10C_CORRUPTIONS, build_cifar10c_loader, missing_cifar10c_files
from src.metrics import evaluate_ttt
from src.self_supervised import SSLConfig, SimCLRModel
from src.test_time_training import TTTAdapter, TTTConfig


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    simclr = SimCLRModel(SSLConfig())
    simclr.backbone.load_state_dict(
        torch.load("results/self_supervised/best_backbone.pt", map_location=device)
    )

    clf = nn.Linear(simclr.feat_dim, 10)
    clf.load_state_dict(torch.load("results/self_supervised/classifier.pt", map_location=device))
    model = nn.Sequential(simclr.backbone, clf).to(device)

    source_stats = torch.load("results/self_supervised/source_stats.pt", map_location=device)
    adapter = TTTAdapter(model, TTTConfig(lr=1e-4, steps_per_batch=10), source_stats)

    root = "data/raw/cifar10c"
    severity = 5
    allow_synthetic_fallback = True
    rows = []
    if missing_cifar10c_files(root, CIFAR10C_CORRUPTIONS):
        print(
            "[DATA] CIFAR-10-C files not found. "
            "Using locally generated CIFAR-10 test corruptions for this evaluation."
        )
    print(f"\n{'Corruption':<22} {'Baseline':>10} {'TTT':>10} {'Gain':>10}")
    print("-" * 58)

    for corruption in CIFAR10C_CORRUPTIONS:
        loader = build_cifar10c_loader(
            root=root,
            corruption=corruption,
            severity=severity,
            batch_size=16,
            num_workers=0,
            allow_synthetic_fallback=allow_synthetic_fallback,
        )
        base = evaluate_ttt(adapter, loader, device, use_ttt=False)
        ttt = evaluate_ttt(adapter, loader, device, use_ttt=True)
        gain = ttt["accuracy"] - base["accuracy"]
        rows.append(
            {
                "corruption": corruption,
                "severity": severity,
                "baseline_accuracy": base["accuracy"],
                "ttt_accuracy": ttt["accuracy"],
                "improvement": gain,
                "baseline_correct": base["correct"],
                "ttt_correct": ttt["correct"],
                "total": base["total"],
            }
        )
        sign = "+" if gain >= 0 else ""
        print(
            f"{corruption:<22} {base['accuracy']*100:>9.2f}% "
            f"{ttt['accuracy']*100:>9.2f}% {sign}{gain*100:>9.2f}%"
        )

    output_path = Path("results/test_time_training/results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    gains = [row["improvement"] for row in rows]
    avg_gain = sum(gains) / len(gains)
    avg_sign = "+" if avg_gain >= 0 else ""
    print("-" * 58)
    print(f"{'Average gain':<22} {'':>10} {'':>10} {avg_sign}{avg_gain*100:>9.2f}%")
    print(f"TTT improves {sum(1 for g in gains if g > 0)}/{len(gains)} corruptions")
    print(f"TTT degrades {sum(1 for g in gains if g < 0)}/{len(gains)} corruptions")
    print(f"Results saved to {output_path}")
