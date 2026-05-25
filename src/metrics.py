"""Evaluation metrics for baseline and test-time training experiments."""
from __future__ import annotations

from typing import Dict

import torch


def top1_accuracy(num_correct: int, num_total: int) -> float:
    """Return top-1 accuracy from the number of correct predictions."""
    if num_total <= 0:
        raise ValueError("num_total must be positive")
    return float(num_correct) / float(num_total)


def evaluate_ttt(adapter, test_loader, device: str, use_ttt: bool = True) -> Dict[str, float]:
    """Evaluate the model with or without TTT adaptation."""
    correct = 0
    total = 0

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        context = torch.enable_grad() if use_ttt else torch.no_grad()
        with context:
            logits = adapter.adapt_on_batch(x) if use_ttt else adapter.predict(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = top1_accuracy(correct, total)
    return {"accuracy": acc, "correct": correct, "total": total}


def mean_corruption_error(
    adapter,
    root: str,
    device: str,
    severity: int = 3,
    batch_size: int = 16,
    use_ttt: bool = True,
) -> float:
    """Compute mean corruption error on CIFAR-10-C for one severity level."""
    from src.datasets import CIFAR10C_CORRUPTIONS, build_cifar10c_loader

    errors = []
    for corruption in CIFAR10C_CORRUPTIONS:
        loader = build_cifar10c_loader(
            root=root,
            corruption=corruption,
            severity=severity,
            batch_size=batch_size,
        )
        result = evaluate_ttt(adapter, loader, device, use_ttt=use_ttt)
        error = 1.0 - result["accuracy"]
        errors.append(error)
        print(f"[{corruption}] error={error * 100:.2f}%")

    mce = sum(errors) / len(errors)
    print(f"[mCE] severity={severity}, use_ttt={use_ttt} -> {mce * 100:.2f}%")
    return mce
