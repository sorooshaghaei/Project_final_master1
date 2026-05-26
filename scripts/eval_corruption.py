import torch
import torch.nn as nn
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from run import get_device, project_path, require_ttt_artifacts
from src.self_supervised import SimCLRModel, SSLConfig
from src.test_time_training import TTTAdapter, TTTConfig
from src.datasets import build_cifar10c_loader
from src.metrics import evaluate_ttt

if __name__ == "__main__":
    device = get_device()

    # Load the pretrained model and classifier
    backbone_path = project_path("results/self_supervised/simclr_backbone.pt")
    clf_path = project_path("results/self_supervised/classifier.pt")
    source_stats_path = project_path("results/self_supervised/source_stats.pt")
    require_ttt_artifacts(
        {
            "backbone": backbone_path,
            "classifier": clf_path,
            "source stats": source_stats_path,
        }
    )

    ssl_cfg = SSLConfig()
    simclr = SimCLRModel(ssl_cfg)
    simclr.backbone.load_state_dict(torch.load(backbone_path, map_location=device))
    clf = nn.Linear(simclr.feat_dim, 10)
    clf.load_state_dict(torch.load(clf_path, map_location=device))
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

    print(f"\n{'Corruption':<22} {'Baseline':>10} {'TTT':>10} {'Improvement':>10}")
    print("-" * 56)
    gains = []
    # evaluate each corruption
    for corruption in CORRUPTIONS:
        loader = build_cifar10c_loader(
            root=str(project_path("data/raw/cifar10c")),
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

    print("-" * 56)
    avg = sum(gains) / len(gains)
    avg_sign = "+" if avg >= 0 else ""
    print(f"{'Average':<22} {'':>10} {'':>10} {avg_sign}{avg:>8.2f}%")
    print(f"\nTTT helps on {sum(1 for g in gains if g > 0)}/15 corruptions")
    print(f"TTT hurts on {sum(1 for g in gains if g < 0)}/15 corruptions")
