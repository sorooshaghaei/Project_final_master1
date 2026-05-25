"""Evaluate baseline and ActMAD TTT on all CIFAR-10-C corruptions."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[1]))

from run import get_device, project_path, require_ttt_artifacts
from src.datasets import CIFAR10C_CORRUPTIONS, build_cifar10c_loader, missing_cifar10c_files
from src.metrics import evaluate_ttt
from src.self_supervised import SSLConfig, SimCLRModel
from src.test_time_training import TTTAdapter, TTTConfig


if __name__ == "__main__":
    device = get_device()

    simclr = SimCLRModel(SSLConfig())
    backbone_path = project_path("results/self_supervised/best_backbone.pt")
    classifier_path = project_path("results/self_supervised/classifier.pt")
    source_stats_path = project_path("results/self_supervised/source_stats.pt")
    require_ttt_artifacts(
        {
            "backbone": backbone_path,
            "classifier": classifier_path,
            "source stats": source_stats_path,
        }
    )
    simclr.backbone.load_state_dict(torch.load(backbone_path, map_location=device))

    clf = nn.Linear(simclr.feat_dim, 10)
    clf.load_state_dict(torch.load(classifier_path, map_location=device))
    model = nn.Sequential(simclr.backbone, clf).to(device)

    source_stats = torch.load(source_stats_path, map_location=device)
    steps_per_batch = 5
    adaptation_lr = 1e-4
    use_safe_ttt = True
    max_allowed_confidence_drop = 0.05
    batch_size = 32
    adapter = TTTAdapter(
        model,
        TTTConfig(lr=adaptation_lr, steps_per_batch=steps_per_batch),
        source_stats,
    )

    root = project_path("data/raw")
    severity = 5
    allow_synthetic_fallback = True
    synthetic_fallback_used = bool(missing_cifar10c_files(root, CIFAR10C_CORRUPTIONS))
    rows = []
    if synthetic_fallback_used:
        print(
            "[DATA] CIFAR-10-C files not found. "
            "Using locally generated CIFAR-10 test corruptions for this evaluation."
        )
    print(f"\n{'Corruption':<22} {'Baseline':>10} {'TTT':>10} {'Gain':>10}")
    print("-" * 58)

    for corruption in CIFAR10C_CORRUPTIONS:
        corruption_fallback_used = bool(missing_cifar10c_files(root, [corruption]))
        loader = build_cifar10c_loader(
            root="data/raw",
            corruption=corruption,
            severity=severity,
            batch_size=batch_size,
            num_workers=0,
            allow_synthetic_fallback=allow_synthetic_fallback,
        )
        base = evaluate_ttt(adapter, loader, device, use_ttt=False)
        ttt = evaluate_ttt(
            adapter,
            loader,
            device,
            use_ttt=True,
            use_safe_ttt=use_safe_ttt,
            max_allowed_confidence_drop=max_allowed_confidence_drop,
        )
        gain = ttt["accuracy"] - base["accuracy"]
        rows.append(
            {
                "corruption": corruption,
                "severity": severity,
                "baseline_accuracy": base["accuracy"],
                "ttt_accuracy": ttt["accuracy"],
                "gain": gain,
                "improvement": gain,
                "baseline_correct": base["correct"],
                "ttt_correct": ttt["correct"],
                "total": base["total"],
                "batch_size": batch_size,
                "steps_per_batch": steps_per_batch,
                "adaptation_lr": adaptation_lr,
                "use_safe_ttt": use_safe_ttt,
                "safe_ttt_rejected_batches": ttt["safe_ttt_rejected_batches"],
                "safe_ttt_total_batches": ttt["safe_ttt_total_batches"],
                "synthetic_fallback_used": corruption_fallback_used,
            }
        )
        sign = "+" if gain >= 0 else ""
        print(
            f"{corruption:<22} {base['accuracy']*100:>9.2f}% "
            f"{ttt['accuracy']*100:>9.2f}% {sign}{gain*100:>9.2f}%"
        )

    output_path = project_path("results/test_time_training/results.csv")
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
