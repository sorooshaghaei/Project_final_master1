"""Download CIFAR-10 into the repository data folder."""

from __future__ import annotations

from pathlib import Path
import sys


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data" / "raw"
    data_root.mkdir(parents=True, exist_ok=True)

    try:
        from torchvision import datasets
    except ImportError:
        print(
            "torchvision is not installed. Install it and rerun:\n"
            "  pip install torchvision",
            file=sys.stderr,
        )
        return 1

    datasets.CIFAR10(root=str(data_root), train=True, download=True)
    datasets.CIFAR10(root=str(data_root), train=False, download=True)

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
