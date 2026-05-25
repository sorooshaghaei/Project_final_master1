# base interface for ssl methods
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torchvision import models


@dataclass
class SSLConfig:
    """settings for ssl training."""
    method: str = "simclr"
    epochs: int = 200
    batch_size: int = 128
    lr: float = 3e-4
    temperature: float = 0.5
    projection_dim: int = 128
    backbone: str = "resnet18" # change the backbone here


# simclr projection head
class ProjectionHead(nn.Module):
    """map backbone features to the contrastive space."""
    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)

# nt-xent loss for simclr
def nt_xent_loss(z1, z2, temperature: float = 0.5) -> torch.Tensor:
    """compute the simclr contrastive loss."""
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0) 
    sim = (z @ z.T) / temperature
    sim.fill_diagonal_(float("-inf")) # mask self matches

    labels = torch.cat([
        torch.arange(B, 2 * B),
        torch.arange(0, B)
    ]).to(z.device)

    return F.cross_entropy(sim, labels)

# backbone and projection head
class SimCLRModel(nn.Module):
    """combine a backbone with a projection head."""
    def __init__(self, config: SSLConfig):
        super().__init__()
        base = getattr(models, config.backbone)(weights=None)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # fit cifar-10 images
        base.maxpool = nn.Identity() # keep more spatial detail
        feat_dim = base.fc.in_features
        base.fc = nn.Identity() # remove the classifier
        self.backbone = base
        self.projector = ProjectionHead(feat_dim, config.projection_dim)
        self.feat_dim = feat_dim
    
    def forward(self, x):
        h = self.backbone(x)
        return self.projector(h) # return projected features
    
    def encode(self, x):
        """return raw backbone features."""
        return self.backbone(x)


class SelfSupervisedTrainer:
    """train simclr and export its artifacts."""
    def __init__(self, config: SSLConfig, device: str="mps" if torch.backends.mps.is_available() else "cpu"):
        # keep the ssl settings
        self.config = config
        self.device = device
        self.model = SimCLRModel(config).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        # decay the learning rate
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )


    def pretrain(self, train_loader):
        # run simclr pretraining
        self.model.train()
        for epoch in range(self.config.epochs):
            total_loss = 0.0
            for (x1, x2), _ in train_loader: # loader returns augmented pairs
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
        """train a linear classifier on frozen features."""
        # freeze the backbone
        self.model.eval()
        for p in self.model.backbone.parameters():
            p.requires_grad = False
        
        clf = nn.Linear(self.model.feat_dim, num_classes).to(self.device) # set classes from the dataset
        opt = torch.optim.Adam(clf.parameters(), lr=1e-3)

        for epoch in range(30):
            clf.train()
            for x, y in train_loader: # loader returns labeled images
                x, y = x.to(self.device), y.to(self.device)
                with torch.no_grad():
                    h = self.model.encode(x)
                loss = F.cross_entropy(clf(h), y)
                opt.zero_grad(); loss.backward(); opt.step()

        # evaluate on the validation set
        clf.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                preds = clf(self.model.encode(x)).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        print(f"Linear evaluation accuracy: {correct/total*100:.2f}%")
        return clf, val_loader # reuse for ttt later
    
    # save the backbone for ttt
    def save_backbone(self, path: str):
        import os 
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.backbone.state_dict(), path)
        print(f"[SSL] Backbone saved to {path}")

    # save source activation stats
    def compute_and_save_source_stats(self, val_loader, save_path: str):
        from src.test_time_training import compute_source_stats
        compute_source_stats(self.model.backbone, val_loader, self.device, save_path)
