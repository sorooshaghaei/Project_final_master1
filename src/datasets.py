"""Dataset registry used by TTT and SSL pipelines."""
# dataset registry and builders for ttt and ssl
from __future__ import annotations

from typing import Any

from dataclasses import dataclass
import pickle
import random
import warnings
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
    """Dataset wrapper that applies a rotation to each image to simulate a test-time distribution shift."""
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
    """Return a DatasetSpec with the metadata needed to build the dataset."""
    # return dataset metadata
    registry = {
        # cifar-10 root should be data/raw/cifar-10-batches-py
        "cifar10": DatasetSpec(name="cifar10", root="data/raw", num_classes=10),
        "cifar10c": DatasetSpec(name="cifar10c", root="data/raw/cifar10c", num_classes=10),
        "imagenet_c": DatasetSpec(name="imagenet_c", root="data/raw/imagenet_c", num_classes=1000),
    }
    if name not in registry:
        raise KeyError(f"Unknown dataset: {name}")
    return registry[name]


def _require_torchvision() -> None:
    """Check that torchvision is available before building datasets that depend on it."""
    # guard for optional torchvision dependency
    if not _TORCHVISION_AVAILABLE:
        raise ImportError("torchvision is required to build CIFAR-10 datasets.")


def build_cifar10_transforms(train: bool) ->Any:
    """Return a transform composition for CIFAR-10, with data augmentation when train=True."""
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
# two-view transform for ssl from the same image
class TwoViewsTransform:
    """Transform that returns two different augmented views of the same image for self-supervised learning."""
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
    """Loader for SimCLR pretraining that returns augmented pairs from the same image."""
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
    # build rotation shift config from a config dict or return disabled config
    """Build a RotationShiftConfig from a config dictionary, or return a disabled config."""
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
    """Return CIFAR-10 train and test datasets with an optional test-time rotation shift."""
    # return cifar-10 train/test datasets with optional rotation shift on test
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
    """Return CIFAR-10 train and test dataloaders with an optional test-time rotation shift."""
    # return cifar-10 train/test dataloaders
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

# cifar-10-c dataset builder for ttt evaluation
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


def missing_cifar10c_files(root: str | Path, corruptions: Sequence[str]) -> list[Path]:
    root = Path(root)
    if isinstance(corruptions, str):
        corruptions = [corruptions]
    required = [root / "CIFAR-10-C" / "labels.npy"]
    required.extend(root / "CIFAR-10-C" / f"{corruption}.npy" for corruption in corruptions)
    return [path for path in required if not path.exists()]


def _find_cifar10_batch_dir(root: Path) -> Path:
    candidates = [
        root / "cifar-10-batches-py",
        root.parent / "cifar-10-batches-py",
        root.parent.parent / "cifar-10-batches-py",
    ]
    for candidate in candidates:
        if (candidate / "test_batch").exists():
            return candidate
    raise FileNotFoundError(
        "CIFAR-10-C files are missing and the local CIFAR-10 test batch was not found.\n"
        "Run: python scripts/download_cifar10.py\n"
        "Then rerun the TTT command."
    )


def _load_cifar10_test_batch(root: Path) -> tuple[np.ndarray, np.ndarray]:
    batch_dir = _find_cifar10_batch_dir(root)
    with (batch_dir / "test_batch").open("rb") as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            batch = pickle.load(f, encoding="bytes")

    data = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(batch[b"labels"], dtype=np.int64)
    return data.astype(np.uint8), labels


def _box_blur(images: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return images
    padded = np.pad(images.astype(np.float32), ((0, 0), (radius, radius), (radius, radius), (0, 0)), mode="reflect")
    out = np.zeros_like(images, dtype=np.float32)
    count = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out += padded[:, radius + dy:radius + dy + 32, radius + dx:radius + dx + 32, :]
            count += 1
    return np.clip(out / count, 0, 255).astype(np.uint8)


def _pixelate(images: np.ndarray, block: int) -> np.ndarray:
    small = images[:, ::block, ::block, :]
    up = np.repeat(np.repeat(small, block, axis=1), block, axis=2)
    return up[:, :32, :32, :].astype(np.uint8)


def _wave_shift(images: np.ndarray, amount: int) -> np.ndarray:
    shifted = images.copy()
    for row in range(32):
        shift = int(round(np.sin(row / 3.0) * amount))
        shifted[:, row, :, :] = np.roll(images[:, row, :, :], shift, axis=1)
    return shifted


def _local_corruption(images: np.ndarray, corruption: str, severity: int) -> np.ndarray:
    seed = severity * 1009 + sum((i + 1) * ord(ch) for i, ch in enumerate(corruption))
    rng = np.random.default_rng(seed)
    x = images.astype(np.float32) / 255.0
    level = severity - 1

    if corruption == "gaussian_noise":
        out = x + rng.normal(0, [0.08, 0.12, 0.18, 0.26, 0.35][level], x.shape)
    elif corruption == "shot_noise":
        vals = [60, 25, 12, 5, 3][level]
        out = rng.poisson(np.clip(x, 0, 1) * vals) / vals
    elif corruption == "impulse_noise":
        out = x.copy()
        mask = rng.random(x.shape[:3]) < [0.03, 0.06, 0.09, 0.14, 0.20][level]
        salt = rng.random(x.shape[:3]) < 0.5
        out[mask & salt] = 1.0
        out[mask & ~salt] = 0.0
    elif corruption == "brightness":
        out = x + [0.08, 0.12, 0.18, 0.24, 0.32][level]
    elif corruption == "contrast":
        factor = [0.85, 0.70, 0.55, 0.40, 0.30][level]
        mean = x.mean(axis=(1, 2), keepdims=True)
        out = (x - mean) * factor + mean
    elif corruption == "fog":
        fog = rng.normal(0.75, 0.12, x.shape[:3])[..., None]
        fog = _box_blur(np.clip(fog * 255, 0, 255).astype(np.uint8), 2 + level).astype(np.float32) / 255.0
        out = x * (1 - [0.18, 0.26, 0.34, 0.42, 0.50][level]) + fog * [0.18, 0.26, 0.34, 0.42, 0.50][level]
    elif corruption == "snow":
        snow = rng.random(x.shape[:3]) < [0.02, 0.04, 0.07, 0.10, 0.14][level]
        out = x.copy()
        out[snow] = 1.0
        out = np.clip(out + snow[..., None] * 0.35, 0, 1)
    elif corruption == "frost":
        frost = rng.normal(0.65, 0.18, x.shape)
        out = x * [0.85, 0.78, 0.70, 0.62, 0.55][level] + frost * [0.15, 0.22, 0.30, 0.38, 0.45][level]
    elif corruption in {"defocus_blur", "glass_blur", "zoom_blur"}:
        return _box_blur(images, [1, 1, 2, 2, 3][level])
    elif corruption == "motion_blur":
        radius = [1, 2, 3, 4, 5][level]
        out_img = np.zeros_like(images, dtype=np.float32)
        count = 0
        for shift in range(-radius, radius + 1):
            out_img += np.roll(images, shift, axis=2)
            count += 1
        return np.clip(out_img / count, 0, 255).astype(np.uint8)
    elif corruption == "elastic_transform":
        return _wave_shift(images, [1, 2, 3, 4, 5][level])
    elif corruption == "pixelate":
        return _pixelate(images, [2, 2, 4, 4, 8][level])
    elif corruption == "jpeg_compression":
        levels = [48, 36, 28, 20, 14]
        return (np.round(images.astype(np.float32) / levels[level]) * levels[level]).clip(0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown corruption: {corruption}")

    return np.clip(out * 255, 0, 255).astype(np.uint8)


def _build_local_corruption_loader(
    root: Path,
    corruption: str,
    severity: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    data, labels = _load_cifar10_test_batch(root)
    data = _local_corruption(data, corruption, severity)
    return _normalised_tensor_loader(data, labels, batch_size=batch_size, num_workers=num_workers)


def _normalised_tensor_loader(
    data: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
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
        pin_memory=torch.cuda.is_available(),
    )


def build_cifar10c_loader(
    root: str,
    corruption: str,
    severity: int,
    batch_size: int = 1,
    num_workers: int = 2,
    allow_synthetic_fallback: bool = False,
) -> DataLoader:
    """Load one CIFAR-10-C .npy file for a given corruption and severity.
    Expected file structure: root/CIFAR-10-C/<corruption>.npy + root/CIFAR-10-C/labels.npy.
    Each .npy file contains 50,000 images, with 10,000 per severity level in order."""
    if corruption not in CIFAR10C_CORRUPTIONS:
        raise ValueError(f"Unknown corruption: {corruption}")
    if not 1 <= severity <= 5:
        raise ValueError("Severity must be between 1 and 5")
    
    # load data and labels
    root_path = Path(root)
    data_path = root_path / "CIFAR-10-C" / f"{corruption}.npy"
    labels_path = root_path / "CIFAR-10-C" / "labels.npy"
    missing = missing_cifar10c_files(root_path, [corruption])
    if missing:
        if not allow_synthetic_fallback:
            missing_text = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(
                "CIFAR-10-C files are missing.\n"
                f"Missing:\n{missing_text}\n"
                "Place the CIFAR-10-C .npy files under data/raw/cifar10c/CIFAR-10-C "
                "or set allow_synthetic_fallback: true."
            )
        return _build_local_corruption_loader(
            root=root_path,
            corruption=corruption,
            severity=severity,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    data = np.load(data_path) # shape (50000, 32, 32, 3), uint8
    labels = np.load(labels_path) # shape (50000,), int64

    # select the slice for the requested severity
    start = (severity - 1) * 10_000
    end = severity * 10_000
    data = data[start:end]
    labels = labels[start:end]

    return _normalised_tensor_loader(data, labels, batch_size=batch_size, num_workers=num_workers)
