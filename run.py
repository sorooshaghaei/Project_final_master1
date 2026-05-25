# unified entry point for TER experiments
import argparse
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    set_seed(cfg.get("seed", 42))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[TASK] {task} | [DEVICE] {device}")

    # SSL pretraining 
    if task == "self_supervised":
        from src.self_supervised import SelfSupervisedTrainer, SSLConfig
        from src.datasets import build_cifar10_loaders, build_simclr_loader

        dataset_cfg = cfg["dataset"]
        model_cfg = cfg.get("model", {})
        training_cfg = cfg["training"]
        log_cfg = cfg.get("logging", {})

        ssl_cfg = SSLConfig(
            method=training_cfg["method"],
            epochs=training_cfg["epochs"],
            batch_size=training_cfg["batch_size"],
            lr=training_cfg["lr"],
            projection_dim=model_cfg.get("embedding_dim", 128),
            backbone=model_cfg.get("backbone", "resnet18"),
            linear_eval_epochs=training_cfg.get("linear_eval_epochs", 30),
        )

        trainer = SelfSupervisedTrainer(ssl_cfg, device=device)

        simclr_loader = build_simclr_loader(
            root=dataset_cfg["root"],
            batch_size=training_cfg["batch_size"],
            num_workers=dataset_cfg.get("num_workers", 2),
            download=dataset_cfg.get("download", False),
            max_samples=dataset_cfg.get("max_simclr_samples", dataset_cfg.get("max_train_samples")),
        )

        train_loader, val_loader = build_cifar10_loaders(
            root=dataset_cfg["root"],
            batch_size=training_cfg["batch_size"],
            num_workers=dataset_cfg.get("num_workers", 2),
            download=dataset_cfg.get("download", False),
            max_train_samples=dataset_cfg.get("max_train_samples"),
            max_test_samples=dataset_cfg.get("max_test_samples"),
        )

        trainer.pretrain(simclr_loader) # 2 vues augmentées
        clf, _ = trainer.linear_eval(train_loader, val_loader) # évaluation linéaire sur les features brutes
        clf_path = log_cfg.get("classifier_path", "results/self_supervised/classifier.pt")
        Path(clf_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(clf.state_dict(), clf_path)
        print(f"[SSL] Classifier linéaire sauvegardé dans {clf_path}")
        

        # sauvegardes
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

        dataset_cfg = cfg["dataset"]
        model_cfg = cfg["model"]
        adaptation_cfg = cfg["adaptation"]

        # charger backbone 
        from src.self_supervised import SimCLRModel, SSLConfig
        ssl_cfg = SSLConfig(backbone=model_cfg.get("backbone", "resnet18"))
        simclr = SimCLRModel(ssl_cfg)
        ckpt = model_cfg.get("checkpoint")
        if ckpt and Path(ckpt).exists():
            simclr.backbone.load_state_dict(torch.load(ckpt, map_location=device))
            print(f"[TTT] Backbone chargé depuis {ckpt}")

        clf = nn.Linear(simclr.feat_dim, model_cfg["num_classes"])
        clf_path = model_cfg.get("classifier_path", "results/self_supervised/classifier.pt")
        if Path(clf_path).exists():
            clf.load_state_dict(torch.load(clf_path, map_location=device))
            print(f"[TTT] Classifier linéaire chargé depuis {clf_path}")

        model = nn.Sequential(simclr.backbone, clf).to(device)

        source_stats = torch.load(adaptation_cfg["source_stats_path"], map_location=device)
        ttt_cfg = TTTConfig(
            lr=adaptation_cfg["lr"],
            steps_per_batch=adaptation_cfg["steps_per_batch"],
            adaptation_task=adaptation_cfg["method"],
        )

        adapter = TTTAdapter(model, ttt_cfg, source_stats)

        # évaluation sur une corruption 
        loader = build_cifar10c_loader(
            root=dataset_cfg["root"],
            corruption=dataset_cfg["corruption"],
            severity=dataset_cfg["severity"],
            batch_size=dataset_cfg["batch_size"],
            num_workers=dataset_cfg.get("num_workers", 2),
            max_samples=dataset_cfg.get("max_samples"),
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
