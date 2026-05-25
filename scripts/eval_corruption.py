import torch
import torch.nn as nn
from pathlib import Path
import sys
sys.path.append(".") # pour importer 

from src.self_supervised import SimCLRModel, SSLConfig
from src.test_time_training import TTTAdapter, TTTConfig
from src.datasets import build_cifar10c_loader
from src.metrics import evaluate_ttt

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu" # choix du device (GPU si disponible, sinon CPU)

    # Charger le modèle pré-entraîné et le classifier linéaire
    ssl_cfg = SSLConfig()
    simclr = SimCLRModel(ssl_cfg)
    simclr.backbone.load_state_dict(torch.load("results/self_supervised/simclr_backbone.pt", map_location=device))
    clf = nn.Linear(simclr.feat_dim, 10)
    clf.load_state_dict(torch.load("results/self_supervised/classifier.pt", map_location=device))
    model = nn.Sequential(simclr.backbone, clf).to(device)

    # Charger les statistiques source pour ACTMAD
    source_stats = torch.load("results/self_supervised/source_stats.pt", map_location=device)
    # Configurer et exécuter l'adaptation TTT
    ttt_cfg = TTTConfig(lr=1e-4, steps_per_batch=10)
    adapter = TTTAdapter(model, ttt_cfg, source_stats)

    # Évaluation sur une corruption spécifique
    CORRUPTIONS = [
        "gaussian_noise", "shot_noise", "impulse_noise",
        "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
        "snow", "frost", "fog", "brightness", "contrast",
        "elastic_transform", "pixelate", "jpeg_compression"
    ]
    SEVERITY = 5

    print(f"\n{'Corruption':<22} {'Baseline':>10} {'TTT':>10} {'Amélioration':>10}")
    print("-" * 56)
    gains = []
    # Évaluer chaque corruption
    for corruption in CORRUPTIONS:
        loader = build_cifar10c_loader(
            root="data/raw",
            corruption=corruption,
            severity=SEVERITY,
            batch_size=16,
            num_workers=0,
        )
        base = evaluate_ttt(adapter, loader, device, use_ttt=False)
        ttt  = evaluate_ttt(adapter, loader, device, use_ttt=True)
        amelioration = (ttt["accuracy"] - base["accuracy"]) * 100
        gains.append(amelioration)
        sign = "+" if amelioration >= 0 else ""
        print(f"{corruption:<22} {base['accuracy']*100:>9.2f}% {ttt['accuracy']*100:>9.2f}% {sign}{amelioration:>9.2f}%")

    print("-" * 56)
    avg = sum(gains) / len(gains)
    print(f"{'Moyenne':<22} {'':>10} {'':>10} {sign}{avg:>8.2f}%")
    print(f"\nTTT aide sur {sum(1 for g in gains if g > 0)}/15 corruptions")
    print(f"TTT aggrave sur {sum(1 for g in gains if g < 0)}/15 corruptions")
