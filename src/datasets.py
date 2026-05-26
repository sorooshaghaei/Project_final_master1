# dataset registry and loaders for ttt and ssl
from __future__ import annotations

from typing import Any

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

try:
    # optional torchvision import
    from torchvision import datasets, transforms

    _TORCHVISION_AVAILABLE = True
except ImportError:  # no cover
    datasets = None
    transforms = None
    _TORCHVISION_AVAILABLE = False


@dataclass
class DatasetSpec:
    name: str
    root: str
    num_classes: int


def get_dataset_spec(name: str) -> DatasetSpec:
    # return dataset metadata
    registry = {
        # cifar-10 is stored under data/raw
        "cifar10": DatasetSpec(name="cifar10", root="data/raw", num_classes=10),
        "cifar10c": DatasetSpec(name="cifar10c", root="data/raw/cifar10c", num_classes=10),
        "imagenet_c": DatasetSpec(name="imagenet_c", root="data/raw/imagenet_c", num_classes=1000),
    }
    if name not in registry:
        raise KeyError(f"Unknown dataset: {name}")
    return registry[name]


def _require_torchvision() -> None:
    # guard for optional torchvision dependency
    if not _TORCHVISION_AVAILABLE:
        raise ImportError("torchvision is required to build CIFAR-10 datasets.")


def _pin_memory_enabled() -> bool:
    return torch.cuda.is_available()


def build_cifar10_transforms(train: bool) ->Any:
    # standard cifar-10 normalization
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
# two augmented views for ssl
class TwoViewsTransform:
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
        pin_memory=_pin_memory_enabled(),
    )
        

def build_cifar10_datasets(
    root: str,
    download: bool = False,
) -> Tuple[Dataset, Dataset]:
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

    return train_dataset, test_dataset


def build_cifar10_loaders(
    root: str,
    batch_size: int = 128,
    num_workers: int = 2,
    download: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    # build train and test datasets first
    train_dataset, test_dataset = build_cifar10_datasets(
        root=root,
        download=download,
    )

    # shuffle training data, keep test order fixed
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=_pin_memory_enabled(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin_memory_enabled(),
    )

    return train_loader, test_loader

# cifar-10-c loader for ttt evaluation
import numpy as np 
from torch.utils.data import TensorDataset

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

def missing_cifar10c_files(root: str | Path, corruptions: Sequence[str]) -> list[Path]:
    data_root = Path(root) / "CIFAR-10-C"
    expected = [data_root / "labels.npy"]
    expected.extend(data_root / f"{corruption}.npy" for corruption in corruptions)
    return [path for path in expected if not path.exists()]

def build_cifar10c_loader(
    root: str,
    corruption: str,
    severity: int,
    batch_size: int = 1,
    num_workers: int = 2,
    max_samples: int | None = None,
) -> DataLoader:
    if corruption not in CIFAR10C_CORRUPTIONS:
        raise ValueError(f"unknown corruption: {corruption}")
    if not 1 <= severity <= 5:
        raise ValueError("severity must be between 1 and 5")
    
    # load data and labels
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

    # select the requested severity slice
    start = (severity - 1) * 10_000
    end = severity * 10_000
    data = data[start:end]
    labels = labels[start:end]
    if max_samples is not None:
        max_samples = min(int(max_samples), len(data))
        data = data[:max_samples]
        labels = labels[:max_samples]

    # standard cifar-10 normalization
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    data = data.astype(np.float32) / 255.0
    data = (data - mean) / std
    data = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2) # shape (n, 3, 32, 32)
    labels = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(data, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin_memory_enabled(),
    )
