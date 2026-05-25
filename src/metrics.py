"""Evaluation metrics for baseline and test-time training experiments."""
from __future__ import annotations

import torch


def top1_accuracy(num_correct: int, num_total: int) -> float:
    """Return top-1 accuracy from the number of correct predictions."""
    if num_total <= 0:
        raise ValueError("num_total must be positive")
    return float(num_correct) / float(num_total)


def evaluate_ttt(
    adapter,
    test_loader,
    device: str,
    use_ttt: bool = True,
    use_safe_ttt: bool = False,
    max_allowed_confidence_drop: float = 0.05,
) -> dict[str, float | int]:
    """Evaluate the model with or without TTT adaptation."""
    correct = 0
    total = 0
    safe_rejected = 0
    safe_total = 0

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        if not use_ttt:
            logits = adapter.predict(x)
        elif use_safe_ttt:
            safe_total += 1
            base_logits = adapter.predict(x)
            with torch.enable_grad():
                adapted_logits = adapter.adapt_on_batch(x)

            base_conf = torch.softmax(base_logits, dim=1).max(dim=1).values.mean()
            adapted_conf = torch.softmax(adapted_logits, dim=1).max(dim=1).values.mean()
            if adapted_conf + max_allowed_confidence_drop < base_conf:
                logits = base_logits
                safe_rejected += 1
            else:
                logits = adapted_logits
        else:
            with torch.enable_grad():
                logits = adapter.adapt_on_batch(x)

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = top1_accuracy(correct, total)
    return {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "safe_ttt_rejected_batches": safe_rejected,
        "safe_ttt_total_batches": safe_total,
    }


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
