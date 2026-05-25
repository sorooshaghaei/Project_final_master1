"""Self-supervised SimCLR training utilities."""
from __future__ import annotations

import csv
import copy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


@dataclass
class SSLConfig:
    """Configuration for SimCLR pretraining."""
    method: str = "simclr"
    max_epochs: int = 50
    batch_size: int = 128
    lr: float = 3e-4
    temperature: float = 0.5
    projection_dim: int = 128
    backbone: str = "resnet18"
    patience: int = 7
    min_delta: float = 1e-3


class ProjectionHead(nn.Module):
    """Projection head used for the SimCLR contrastive objective."""

    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Compute the NT-Xent loss for two augmented views of the same batch."""
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = (z @ z.T) / temperature
    sim.fill_diagonal_(float("-inf"))

    labels = torch.cat(
        [
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size),
        ]
    ).to(z.device)

    return F.cross_entropy(sim, labels)


class SimCLRModel(nn.Module):
    """ResNet backbone with a SimCLR projection head."""

    def __init__(self, config: SSLConfig):
        super().__init__()
        base = getattr(models, config.backbone)(weights=None)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.projector = ProjectionHead(feat_dim, config.projection_dim)
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.projector(h)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone features without the projection head."""
        return self.backbone(x)


class SelfSupervisedTrainer:
    """Train SimCLR and provide linear evaluation / source-stat utilities."""

    def __init__(self, config: SSLConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.model = SimCLRModel(config).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_epochs,
        )

    @staticmethod
    def _save_backbone_state(model: SimCLRModel, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.backbone.state_dict(), path)

    def pretrain(
        self,
        train_loader,
        log_path: str | Path | None = None,
        best_backbone_path: str | Path | None = None,
        last_backbone_path: str | Path | None = None,
    ):
        """Run SimCLR pretraining with early stopping and checkpointing.

        The loop uses max_epochs as a safety limit. Training stops earlier if the
        average SSL loss does not improve by at least min_delta for patience
        consecutive epochs.
        """
        self.model.train()
        best_loss = float("inf")
        best_epoch = 0
        wait = 0
        best_state = None
        rows = []

        for epoch in range(1, self.config.max_epochs + 1):
            total_loss = 0.0
            num_batches = 0

            for (x1, x2), _ in train_loader:
                x1, x2 = x1.to(self.device), x2.to(self.device)
                z1, z2 = self.model(x1), self.model(x2)
                loss = nt_xent_loss(z1, z2, self.config.temperature)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            improved = avg_loss < (best_loss - self.config.min_delta)
            if improved:
                best_loss = avg_loss
                best_epoch = epoch
                wait = 0
                best_state = copy.deepcopy(self.model.state_dict())
                if best_backbone_path:
                    self._save_backbone_state(self.model, best_backbone_path)
            else:
                wait += 1

            if last_backbone_path:
                self._save_backbone_state(self.model, last_backbone_path)

            stopped_early = wait >= self.config.patience
            rows.append(
                {
                    "epoch": epoch,
                    "average_ssl_loss": avg_loss,
                    "learning_rate": lr,
                    "best_epoch": best_epoch,
                    "stopped_early": stopped_early,
                }
            )
            print(
                f"Epoch {epoch}/{self.config.max_epochs} | "
                f"loss={avg_loss:.4f} | best_epoch={best_epoch} | lr={lr:.6f}"
            )

            if stopped_early:
                print(
                    f"[SSL] Early stopping at epoch {epoch}. "
                    f"Best epoch: {best_epoch} (loss={best_loss:.4f})."
                )
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        if log_path:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "epoch",
                        "average_ssl_loss",
                        "learning_rate",
                        "best_epoch",
                        "stopped_early",
                    ],
                )
                writer.writeheader()
                writer.writerows(rows)
            print(f"[SSL] Training log saved to {log_path}")

        return train_loader

    def linear_eval(self, train_loader, val_loader, num_classes: int = 10):
        """Train a linear classifier on frozen backbone features."""
        self.model.eval()
        for p in self.model.backbone.parameters():
            p.requires_grad = False

        clf = nn.Linear(self.model.feat_dim, num_classes).to(self.device)
        opt = torch.optim.Adam(clf.parameters(), lr=1e-3)

        for _ in range(30):
            clf.train()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                with torch.no_grad():
                    h = self.model.encode(x)
                loss = F.cross_entropy(clf(h), y)
                opt.zero_grad()
                loss.backward()
                opt.step()

        clf.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                preds = clf(self.model.encode(x)).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        accuracy = correct / total if total else 0.0
        print(f"Linear evaluation accuracy: {accuracy * 100:.2f}%")
        return clf, val_loader

    def save_backbone(self, path: str | Path):
        """Save the current backbone state dict."""
        self._save_backbone_state(self.model, path)
        print(f"[SSL] Backbone saved to {path}")

    def compute_and_save_source_stats(self, val_loader, save_path: str | Path):
        """Compute source activation statistics for ActMAD."""
        from src.test_time_training import compute_source_stats

        compute_source_stats(self.model.backbone, val_loader, self.device, str(save_path))
