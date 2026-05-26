# dataset and loaders for ttt and ssl
from __future__ import annotations

from typing import Any

from dataclasses import dataclass
import random
from typing import Optional, Sequence, Tuple

from torch.utils.data import DataLoader, Dataset
from pathlib import Path

try:
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
        pin_memory=True,
    )
        

def rotation_shift_from_config(config: Optional[dict]) -> RotationShiftConfig:
    # build rotation settings or disable them
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
    # add optional rotation shift on test data
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
    # build train and test datasets first
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

# cifar-10-c loader for ttt evaluation
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
    cifar10c_root = Path(root) / "CIFAR-10-C"
    expected = [cifar10c_root / "labels.npy"]
    expected.extend(cifar10c_root / f"{corruption}.npy" for corruption in corruptions)
    return [path for path in expected if not path.exists()]


def _fallback_cifar10_root(root: Path) -> Path:
    candidates = [root, root.parent, root.parent.parent]
    for candidate in candidates:
        if (candidate / "cifar-10-batches-py").exists():
            return candidate
    return root


def _apply_synthetic_corruption(
    x: torch.Tensor,
    corruption: str,
    severity: int,
    generator: torch.Generator,
) -> torch.Tensor:
    amount = severity / 5.0

    if corruption == "gaussian_noise":
        x = x + torch.randn(x.shape, generator=generator) * (0.04 + 0.14 * amount)
    elif corruption == "shot_noise":
        noise = torch.randn(x.shape, generator=generator) * torch.sqrt(x.clamp(0, 1) + 0.01)
        x = x + noise * (0.03 + 0.10 * amount)
    elif corruption == "impulse_noise":
        mask = torch.rand((1, *x.shape[1:]), generator=generator) < (0.01 + 0.05 * amount)
        salt = (torch.rand((1, *x.shape[1:]), generator=generator) > 0.5).float()
        x = torch.where(mask.expand_as(x), salt.expand_as(x), x)
    elif corruption in {"defocus_blur", "glass_blur", "motion_blur", "zoom_blur"}:
        kernel_size = 3 if severity <= 3 else 5
        sigma = 0.4 + 1.2 * amount
        x = TF.gaussian_blur(x, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
        if corruption == "glass_blur":
            x = x + torch.randn(x.shape, generator=generator) * 0.02 * amount
    elif corruption == "snow":
        snow = torch.rand(x.shape, generator=generator) * (0.15 + 0.25 * amount)
        x = x * (1.0 - 0.15 * amount) + snow
    elif corruption in {"frost", "fog"}:
        haze = torch.full_like(x, 0.65 if corruption == "fog" else 0.8)
        x = x * (1.0 - 0.20 * amount) + haze * (0.20 * amount)
    elif corruption == "brightness":
        x = TF.adjust_brightness(x, 1.0 + 0.20 * severity)
    elif corruption == "contrast":
        x = TF.adjust_contrast(x, max(0.15, 1.0 - 0.15 * severity))
    elif corruption == "elastic_transform":
        x = x + torch.randn(x.shape, generator=generator) * 0.03 * amount
    elif corruption == "pixelate":
        small_size = max(4, 32 // (severity + 1))
        x = TF.resize(x, [small_size, small_size], interpolation=transforms.InterpolationMode.NEAREST)
        x = TF.resize(x, [32, 32], interpolation=transforms.InterpolationMode.NEAREST)
    elif corruption == "jpeg_compression":
        levels = max(8, 64 - 10 * severity)
        x = torch.round(x * levels) / levels

    return x.clamp(0.0, 1.0)


def _build_synthetic_cifar10c_loader(
    root: str | Path,
    corruption: str,
    severity: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    _require_torchvision()
    cifar_root = _fallback_cifar10_root(Path(root))
    dataset = datasets.CIFAR10(root=str(cifar_root), train=False, download=False)
    generator = torch.Generator().manual_seed(10_000 + severity * 101 + CIFAR10C_CORRUPTIONS.index(corruption))
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

    images = []
    labels = []
    for image, label in dataset:
        x = TF.to_tensor(image)
        x = _apply_synthetic_corruption(x, corruption, severity, generator)
        images.append((x - mean) / std)
        labels.append(label)

    tensor_dataset = TensorDataset(torch.stack(images), torch.tensor(labels, dtype=torch.long))
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def build_cifar10c_loader(
    root: str,
    corruption: str,
    severity: int,
    batch_size: int = 1,
    num_workers: int = 2,
    allow_synthetic_fallback: bool = False,
) -> DataLoader:
    """load one cifar-10-c corruption and severity."""
    if corruption not in CIFAR10C_CORRUPTIONS:
        raise ValueError(f"Corruption inconnue: {corruption}")
    if not 1 <= severity <= 5:
        raise ValueError("severity must be between 1 and 5")
    
    # load data and labels
    data_path = Path(root) / "CIFAR-10-C" / f"{corruption}.npy"
    labels_path = Path(root) / "CIFAR-10-C" / "labels.npy"
    missing = [path for path in (data_path, labels_path) if not path.exists()]
    if missing:
        if allow_synthetic_fallback:
            return _build_synthetic_cifar10c_loader(root, corruption, severity, batch_size, num_workers)
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

    # standard cifar-10 normalization
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    data = data.astype(np.float32) / 255.0
    data = (data - mean) / std
    data = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2) # shape (n, 3, 32, 32)
    labels = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
