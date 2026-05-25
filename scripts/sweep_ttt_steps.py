"""Run a small sweep over the number of TTT adaptation steps."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from run import get_device, project_path, require_ttt_artifacts, set_seed
from src.datasets import build_cifar10c_loader, missing_cifar10c_files
from src.metrics import evaluate_ttt
from src.self_supervised import SSLConfig, SimCLRModel
from src.test_time_training import TTTAdapter, TTTConfig


SWEEP_STEPS = (3, 5, 10)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a small TTT adaptation-step sweep")
    parser.add_argument("--config", default="configs/test_time_training.yaml")
    args = parser.parse_args()

    config_path = project_path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = get_device()
    print(f"[SWEEP] device={device} | seed={seed}")

    ssl_cfg = SSLConfig(backbone=cfg.get("model", {}).get("backbone", "resnet18"))
    simclr = SimCLRModel(ssl_cfg)

    ckpt_path = project_path(cfg["model"]["checkpoint"])
    clf_path = project_path("results/self_supervised/classifier.pt")
    source_stats_path = project_path(cfg["adaptation"]["source_stats_path"])
    require_ttt_artifacts(
        {
            "backbone": ckpt_path,
            "classifier": clf_path,
            "source stats": source_stats_path,
        }
    )

    simclr.backbone.load_state_dict(torch.load(ckpt_path, map_location=device))
    clf = nn.Linear(simclr.feat_dim, cfg["model"]["num_classes"])
    clf.load_state_dict(torch.load(clf_path, map_location=device))
    model = nn.Sequential(simclr.backbone, clf).to(device)
    source_stats = torch.load(source_stats_path, map_location=device)

    dataset_cfg = cfg["dataset"]
    corruptions = dataset_cfg.get("corruptions") or [dataset_cfg["corruption"]]
    severity = dataset_cfg["severity"]
    batch_size = dataset_cfg["batch_size"]
    data_root = project_path(dataset_cfg["root"])
    allow_synthetic_fallback = dataset_cfg.get("allow_synthetic_fallback", True)
    missing_cifar10c = missing_cifar10c_files(data_root, corruptions)

    if missing_cifar10c:
        if not allow_synthetic_fallback:
            missing_text = "\n".join(f"- {path}" for path in missing_cifar10c)
            raise FileNotFoundError(
                "CIFAR-10-C files are missing.\n"
                f"Missing:\n{missing_text}\n"
                "Place the CIFAR-10-C .npy files under data/raw/CIFAR-10-C "
                "or set allow_synthetic_fallback: true."
            )
        if dataset_cfg.get("print_fallback_warning_once", True):
            print(
                "[DATA] CIFAR-10-C files not found. "
                "Using locally generated CIFAR-10 test corruptions for this evaluation."
            )

    adaptation_cfg = cfg["adaptation"]
    adaptation_lr = adaptation_cfg["lr"]
    use_safe_ttt = adaptation_cfg.get("use_safe_ttt", False)
    max_allowed_confidence_drop = adaptation_cfg.get("max_allowed_confidence_drop", 0.05)
    rows = []

    for steps_per_batch in SWEEP_STEPS:
        adapter = TTTAdapter(
            model,
            TTTConfig(
                lr=adaptation_lr,
                steps_per_batch=steps_per_batch,
                adaptation_task=adaptation_cfg.get("method", "actmad"),
            ),
            source_stats,
        )
        print(f"[SWEEP] steps_per_batch={steps_per_batch}")

        for corruption in corruptions:
            loader = build_cifar10c_loader(
                root=str(data_root),
                corruption=corruption,
                severity=severity,
                batch_size=batch_size,
                num_workers=dataset_cfg.get("num_workers", 0),
                allow_synthetic_fallback=allow_synthetic_fallback,
            )
            baseline = evaluate_ttt(adapter, loader, device, use_ttt=False)
            ttt = evaluate_ttt(
                adapter,
                loader,
                device,
                use_ttt=True,
                use_safe_ttt=use_safe_ttt,
                max_allowed_confidence_drop=max_allowed_confidence_drop,
            )
            gain = ttt["accuracy"] - baseline["accuracy"]
            rows.append(
                {
                    "steps_per_batch": steps_per_batch,
                    "corruption": corruption,
                    "severity": severity,
                    "baseline_accuracy": baseline["accuracy"],
                    "ttt_accuracy": ttt["accuracy"],
                    "gain": gain,
                    "safe_ttt_rejected_batches": ttt["safe_ttt_rejected_batches"],
                    "safe_ttt_total_batches": ttt["safe_ttt_total_batches"],
                }
            )
            sign = "+" if gain >= 0 else ""
            print(
                f"  {corruption:<18} baseline={baseline['accuracy']*100:6.2f}% "
                f"ttt={ttt['accuracy']*100:6.2f}% gain={sign}{gain*100:.2f}%"
            )

    output_path = project_path("results/test_time_training/ttt_steps_sweep.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "steps_per_batch",
        "corruption",
        "severity",
        "baseline_accuracy",
        "ttt_accuracy",
        "gain",
        "safe_ttt_rejected_batches",
        "safe_ttt_total_batches",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[SWEEP] Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
