# download cifar-10 into data/raw so everyone shares the same path

from __future__ import annotations

from pathlib import Path
import sys


def main() -> int:
    # find repo root and standard data folder
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data" / "raw"
    data_root.mkdir(parents=True, exist_ok=True)

    try:
        # import torchvision only when needed
        from torchvision import datasets
    except ImportError:
        print(
            "torchvision is not installed. Install it and rerun:\n"
            "  pip install torchvision",
            file=sys.stderr,
        )
        return 1

    # download both train and test splits
    datasets.CIFAR10(root=str(data_root), train=True, download=True)
    datasets.CIFAR10(root=str(data_root), train=False, download=True)

    # check expected folder name
    expected = data_root / "cifar-10-batches-py"
    if expected.exists():
        print(f"CIFAR-10 ready at: {expected}")
    else:
        print(
            "Download completed, but the expected folder was not found.\n"
            f"Check: {expected}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
