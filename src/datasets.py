"""Dataset registry used by TTT and SSL pipelines."""
# dataset registry and builders for TTT and SSL
from __future__ import annotations

from typing import Any

from dataclasses import dataclass
import random
from typing import Optional, Sequence, Tuple

from torch.utils.data import DataLoader, Dataset
from pathlib import Path

try:
    # optional torchvision import
    from torchvision import datasets, transforms
    from torchvision.transforms import functional as TF

    _TORCHVISION_AVAILABLE = True
except ImportError:  # no cover
    datasets = None
    transforms = None
    TF = None
    _TORCHVISION_AVAILABLE = False


@dataclass
class DatasetSpec:
    name: str
    root: str
    num_classes: int


@dataclass
class RotationShiftConfig:
    enabled: bool = False
    # mode can be "random" or "fixed"
    mode: str = "random"
    angles: Sequence[int] = (0, 90, 180, 270)
    fixed_angle: int = 90
    seed: int = 123


# dataset wrapper that rotates each sample to simulate distribution shift
class RotationShiftDataset(Dataset):
    """Dataset wrapper qui applique une rotation à chaque image pour simuler un changement de distribution au test."""
    def __init__(self, base: Dataset, config: RotationShiftConfig):
        if not _TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for RotationShiftDataset.")
        if config.mode not in {"random", "fixed"}:
            raise ValueError(f"Unsupported rotation mode: {config.mode}")

        self.base = base
        self.config = config
        # cache angle list for fast indexing
        self._angles = list(config.angles)
        if not self._angles:
            raise ValueError("Rotation angles list cannot be empty.")

        if config.mode == "fixed":
            # every sample uses the same rotation angle
            self._index_angles = [config.fixed_angle] * len(base)
        else:
            # assign a deterministic random angle per sample
            rng = random.Random(config.seed)
            self._index_angles = [rng.choice(self._angles) for _ in range(len(base))]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, label = self.base[index]
        # rotate the image to create a shifted test distribution
        angle = self._index_angles[index]
        image = TF.rotate(image, angle)
        return image, label


def get_dataset_spec(name: str) -> DatasetSpec:
    """Donne le nom d'un dataset, retourne un DatasetSpec avec les métadonnées nécessaires pour le construire."""
    # return dataset metadata
    registry = {
        # CIFAR-10 root should be data/raw/cifar-10-batches-py
        "cifar10": DatasetSpec(name="cifar10", root="data/raw", num_classes=10),
        "cifar10c": DatasetSpec(name="cifar10c", root="data/raw/cifar10c", num_classes=10),
        "imagenet_c": DatasetSpec(name="imagenet_c", root="data/raw/imagenet_c", num_classes=1000),
    }
    if name not in registry:
        raise KeyError(f"Unknown dataset: {name}")
    return registry[name]


def _require_torchvision() -> None:
    """Aide à vérifier que torchvision est disponible avant de construire des datasets qui en dépendent."""
    # guard for optional torchvision dependency
    if not _TORCHVISION_AVAILABLE:
        raise ImportError("torchvision is required to build CIFAR-10 datasets.")


def build_cifar10_transforms(train: bool) ->Any:
    """Retourne une composition de transformations pour CIFAR-10. Si train=True, inclut des augmentations de données."""
    # standard CIFAR-10 normalization
    _require_torchvision()
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    if train:
        # augment only during training
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
# Transformateur pour SSL avec deux vues augmentées d'une même image
class TwoViewsTransform:
    """Transformateur qui prend une image et retourne deux versions augmentées différentes pour l'apprentissage auto-supervisé."""
    def __init__(self):
        _require_torchvision()
        self.t = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])

    def __call__(self, x):
        return self.t(x), self.t(x)
    
def build_simclr_loader(root:str, batch_size: int = 128, num_workers: int = 2, download: bool = False,) -> DataLoader:
    """Loader pour le pré-entraînement SimCLR, qui retourne des paires augmentées d'une même image."""
    _require_torchvision()
    dataset = datasets.CIFAR10(
        root=root,
        train=True,
        download=download,
        transform=TwoViewsTransform(),
    )
    return DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
        

def rotation_shift_from_config(config: Optional[dict]) -> RotationShiftConfig:
    # build RotationShiftConfig from a config dict or return disabled config
    """Construit une RotationShiftConfig à partir d'un dictionnaire de configuration. 
        Si le dictionnaire est None ou vide, retourne une config désactivée."""
    if not config:
        return RotationShiftConfig(enabled=False)

    return RotationShiftConfig(
        enabled=bool(config.get("enabled", False)),
        mode=str(config.get("mode", "random")),
        angles=tuple(config.get("angles", (0, 90, 180, 270))),
        fixed_angle=int(config.get("fixed_angle", 90)),
        seed=int(config.get("seed", 123)),
    )


def build_cifar10_datasets(
    root: str,
    download: bool = False,
    rotation_shift: Optional[RotationShiftConfig] = None,
) -> Tuple[Dataset, Dataset]:
    """Retourne les datasets d'entraînement et de test CIFAR-10 avec une option de rotation 
        pour simuler un changement de distribution au test."""
    # return CIFAR-10 train/test datasets with optional rotation shift on test
    _require_torchvision()
    # training data uses standard augmentations
    train_dataset = datasets.CIFAR10(
        root=root,
        train=True,
        download=download,
        transform=build_cifar10_transforms(train=True),
    )
    # test data uses only normalization
    test_dataset = datasets.CIFAR10(
        root=root,
        train=False,
        download=download,
        transform=build_cifar10_transforms(train=False),
    )

    if rotation_shift and rotation_shift.enabled:
        # wrap test set to apply rotation shift
        test_dataset = RotationShiftDataset(test_dataset, rotation_shift)

    return train_dataset, test_dataset


def build_cifar10_loaders(
    root: str,
    batch_size: int = 128,
    num_workers: int = 2,
    download: bool = False,
    rotation_shift: Optional[RotationShiftConfig] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Retourne les dataloaders d'entraînement et de test CIFAR-10 avec une option de rotation
        pour simuler un changement de distribution au test."""
    # return CIFAR-10 train/test dataloaders
    train_dataset, test_dataset = build_cifar10_datasets(
        root=root,
        download=download,
        rotation_shift=rotation_shift,
    )

    # shuffle training data, keep test order fixed
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader

# CIFAR-10-C dataset builder for TTT evaluation
import numpy as np 
from torch.utils.data import TensorDataset
import torch

CIFAR10C_CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]

def build_cifar10c_loader(root: str, corruption: str, severity: int, batch_size: int = 1, num_workers: int = 2) -> DataLoader:
    """Charge un fichier .npy CIFAR-10-C pour une corrution et une sévérité données.
    Structure du fichier attendu : root/CIFAR-10-C/<corruption>.npy + root/CIFAR-10-C/labels.npy
    Chaque fichier .npy contient 50 000 images (10 000 par sévérité; dans l'ordre)."""
    if corruption not in CIFAR10C_CORRUPTIONS:
        raise ValueError(f"Corruption inconnue: {corruption}")
    if not 1 <= severity <= 5:
        raise ValueError(f"Sévérité doit être entre 1 et 5")
    
    # charger les données et labels
    data_path = Path(root) / "CIFAR-10-C" / f"{corruption}.npy"
    labels_path = Path(root) / "CIFAR-10-C" / "labels.npy"
    missing = [path for path in (data_path, labels_path) if not path.exists()]
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "CIFAR-10-C files are missing.\n"
            f"Missing:\n{missing_text}\n"
            "Place the CIFAR-10-C .npy files under data/raw/cifar10c/CIFAR-10-C/."
        )

    data = np.load(data_path) # shape (50000, 32, 32, 3), uint8
    labels = np.load(labels_path) # shape (50000,), int64

    # sélectionner la tranche correspondant à la sévérité demandée
    start = (severity - 1) * 10_000
    end = severity * 10_000
    data = data[start:end]
    labels = labels[start:end]

    # normalisation standard CIFAR-10
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    data = data.astype(np.float32) / 255.0
    data = (data - mean) / std
    data = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2) # shape (N, 3, 32, 32)
    labels = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
