# unified entry point for TER experiments
import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path


def main() -> None:
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Run TER experiments")
    parser.add_argument(
        "--task",
        choices=["test_time_training", "self_supervised", "ttt", "ssl"],
        required=True,
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # check config path early
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    task = args.task
    if task == "ttt": task = "test_time_training"
    elif task == "ssl": task = "self_supervised"

    cfg = yaml.safe_load(config_path.read_text())
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[TASK] {task} | [DEVICE] {device}")

    # SSL pretraining 
    if task == "self_supervised":
        from src.self_supervised import SelfSupervisedTrainer, SSLConfig
        from src.datasets import build_cifar10_loaders, build_simclr_loader

        ssl_cfg = SSLConfig(
            method=cfg["training"]["method"],
            epochs=cfg["training"]["epochs"],
            batch_size=cfg["training"]["batch_size"],
            lr=cfg["training"]["lr"],
        )

        trainer = SelfSupervisedTrainer(ssl_cfg, device=device)

        simclr_loader = build_simclr_loader(
            root=cfg["dataset"]["root"],
            batch_size=cfg["training"]["batch_size"],
            download=cfg["dataset"].get("download", False),
        )

        train_loader, val_loader = build_cifar10_loaders(
            root=cfg["dataset"]["root"],
            batch_size=cfg["training"]["batch_size"],
            download=cfg["dataset"].get("download", False),
        )

        trainer.pretrain(simclr_loader) # 2 vues augmentées
        clf, _ = trainer.linear_eval(train_loader, val_loader) # évaluation linéaire sur les features brutes
        clf_path = "results/self_supervised/classifier.pt"
        torch.save(clf.state_dict(), clf_path)
        print(f"[SSL] Classifier linéaire sauvegardé dans {clf_path}")
        

        # sauvegardes
        log_cfg = cfg.get("logging", {})
        if log_cfg.get("save_backbone"):
            Path(log_cfg["backbone_path"]).parent.mkdir(parents=True, exist_ok=True)
            trainer.save_backbone(log_cfg["backbone_path"])

        stats_cfg = cfg.get("source_stats", {})
        if stats_cfg.get("compute"):
            Path(stats_cfg["save_path"]).parent.mkdir(parents=True, exist_ok=True)
            trainer.compute_and_save_source_stats(val_loader, stats_cfg['save_path'])

    # TTT ActMAD adaptation
    elif task == "test_time_training":
        from src.test_time_training import TTTAdapter, TTTConfig
        from src.datasets import build_cifar10c_loader
        from src.metrics import evaluate_ttt
        import torchvision.models as tvm

        # charger backbone 
        from src.self_supervised import SimCLRModel, SSLConfig
        ssl_cfg = SSLConfig()
        simclr = SimCLRModel(ssl_cfg)
        ckpt = cfg["model"].get("checkpoint")
        if ckpt and Path(ckpt).exists():
            simclr.backbone.load_state_dict(torch.load(ckpt, map_location=device))
            print(f"[TTT] Backbone chargé depuis {ckpt}")

        clf = nn.Linear(simclr.feat_dim, cfg["model"]["num_classes"])
        clf_path = "results/self_supervised/classifier.pt"
        if Path(clf_path).exists():
            clf.load_state_dict(torch.load(clf_path, map_location=device))
            print(f"[TTT] Classifier linéaire chargé depuis {clf_path}")

        model = nn.Sequential(simclr.backbone, clf).to(device)

        source_stats = torch.load(cfg["adaptation"]["source_stats_path"], map_location=device)
        ttt_cfg = TTTConfig(
            lr=cfg["adaptation"]["lr"],
            steps_per_batch=cfg["adaptation"]["steps_per_batch"],
            adaptation_task=cfg["adaptation"]["method"],
        )

        adapter = TTTAdapter(model, ttt_cfg, source_stats)

        # évaluation sur une corruption 
        loader = build_cifar10c_loader(
            root=cfg["dataset"]["root"],
            corruption=cfg["dataset"]["corruption"],
            severity=cfg["dataset"]["severity"],
            batch_size=cfg["dataset"]["batch_size"],
        )

        print("\n--- Baseline (sans TTT) ---")
        res_base = evaluate_ttt(adapter, loader, device, use_ttt=False)
        print(f"Accuracy: {res_base['accuracy']*100:.2f}%")

        print("\n--- ActMAD TTT ---")
        res_ttt = evaluate_ttt(adapter, loader, device, use_ttt=True)
        print(f"Accuracy TTT: {res_ttt['accuracy']*100:.2f}%")
        print(f"Amélioration: {(res_ttt['accuracy'] - res_base['accuracy'])*100:.2f}%")

if __name__ == "__main__":
    main()