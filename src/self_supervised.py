#SSL methods
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from torchvision import models


@dataclass
class SSLConfig:
    method: str = "simclr"
    epochs: int = 100
    batch_size: int = 128
    lr: float = 3e-4
    temperature: float = 0.5
    projection_dim: int = 128
    backbone: str = "resnet18" 


# Projection head SimCLR
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)

# NT-Xent loss for SimCLR
def nt_xent_loss(z1, z2, temperature: float = 0.5) -> torch.Tensor:
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0) 
    sim = (z @ z.T) / temperature
    sim.fill_diagonal_(float("-inf")) # masque diagonale

    labels = torch.cat([
        torch.arange(B, 2 * B),
        torch.arange(0, B)
    ]).to(z.device)

    return F.cross_entropy(sim, labels)

# Backbone + projection head pour SSL
class SimCLRModel(nn.Module):
    def __init__(self, config: SSLConfig):
        super().__init__()
        base = getattr(models, config.backbone)(weights=None)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # adapté pour CIFAR-10 32x32
        base.maxpool = nn.Identity() # supprimer le maxpool pour conserver plus de détails
        feat_dim = base.fc.in_features
        base.fc = nn.Identity() # On enlève la tête de classification
        self.backbone = base
        self.projector = ProjectionHead(feat_dim, config.projection_dim)
        self.feat_dim = feat_dim
    
    def forward(self, x):
        h = self.backbone(x)
        return self.projector(h) # retourne les features projetés pour le calcul de la perte SSL
    
    def encode(self, x):
        """Features brutes sans projection, pour évaluation linéaire."""
        return self.backbone(x)


class SelfSupervisedTrainer:
    def __init__(self, config: SSLConfig, device: str="mps" if torch.backends.mps.is_available() else "cpu"):
        # store config for SSL settings
        self.config = config
        self.device = device
        self.model = SimCLRModel(config).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        # scheduler pour ajuster le taux d'apprentissage pendant l'entraînement
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )


    def pretrain(self, train_loader):
        # run self-supervised pretraining
        self.model.train()
        for epoch in range(self.config.epochs):
            total_loss = 0.0
            for (x1, x2), _ in train_loader: 
                x1, x2 = x1.to(self.device), x2.to(self.device)
                z1, z2 = self.model(x1), self.model(x2)
                loss = nt_xent_loss(z1, z2, self.config.temperature)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.config.epochs}, Loss: {total_loss/len(train_loader):.4f}")
        return train_loader

    def linear_eval(self, train_loader, val_loader, num_classes: int = 10):
        # run linear evaluation on frozen features
        self.model.eval()
        for p in self.model.backbone.parameters():
            p.requires_grad = False
        
        clf = nn.Linear(self.model.feat_dim, num_classes).to(self.device) # num_classes doit être défini selon le dataset
        opt = torch.optim.Adam(clf.parameters(), lr=1e-3)

        for epoch in range(30):
            clf.train()
            for x, y in train_loader: 
                x, y = x.to(self.device), y.to(self.device)
                with torch.no_grad():
                    h = self.model.encode(x)
                loss = F.cross_entropy(clf(h), y)
                opt.zero_grad(); loss.backward(); opt.step()

        clf.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                preds = clf(self.model.encode(x)).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        print(f"Linear evaluation accuracy: {correct/total*100:.2f}%")
        return clf, val_loader # garder pour TTT ou d'autres usages futurs
    
    # sauvegarde du backbone pour réutilisation en TTT
    def save_backbone(self, path: str):
        import os 
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.backbone.state_dict(), path)
        print(f"[SSL] Backbone sauvegardé dans {path}")

    # calcul et sauvegarde des statistiques d'activation pour ActMAD
    def compute_and_save_source_stats(self, val_loader, save_path: str):
        from src.test_time_training import compute_source_stats
        compute_source_stats(self.model.backbone, val_loader, self.device, save_path)
