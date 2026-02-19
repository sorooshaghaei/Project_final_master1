"""Dataset registry used by TTT and SSL pipelines."""

from dataclasses import dataclass


@dataclass
class DatasetSpec:
    name: str
    root: str
    num_classes: int


def get_dataset_spec(name: str) -> DatasetSpec:
    """Return dataset metadata. Extend this registry as needed."""
    registry = {
        "cifar10c": DatasetSpec(name="cifar10c", root="data/raw/cifar10c", num_classes=10),
        "imagenet_c": DatasetSpec(name="imagenet_c", root="data/raw/imagenet_c", num_classes=1000),
    }
    if name not in registry:
        raise KeyError(f"Unknown dataset: {name}")
    return registry[name]
