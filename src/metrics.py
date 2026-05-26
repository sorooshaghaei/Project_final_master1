# evaluation metrics
from typing import Dict
import torch

def top1_accuracy(num_correct: int, num_total: int) -> float:
    #division by zero
    if num_total <= 0:
        raise ValueError("num_total must be positive")
    return float(num_correct) / float(num_total)

# evaluate ttt against baseline

def _mean_confidence(logits: torch.Tensor) -> float:
    return logits.softmax(dim=1).max(dim=1).values.mean().item()


def evaluate_ttt(
    adapter,
    test_loader,
    device: str,
    use_ttt: bool = True,
    use_safe_ttt: bool = False,
    max_allowed_confidence_drop: float = 0.05,
) -> Dict[str, float]:
    correct = total = 0
    safe_ttt_rejected_batches = 0
    safe_ttt_total_batches = 0

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        if not use_ttt:
            logits = adapter.predict(x)
        elif use_safe_ttt:
            safe_ttt_total_batches += 1
            baseline_logits = adapter.predict(x)
            adapted_logits = adapter.adapt_on_batch(x)

            baseline_confidence = _mean_confidence(baseline_logits)
            adapted_confidence = _mean_confidence(adapted_logits)
            if adapted_confidence < baseline_confidence - max_allowed_confidence_drop:
                safe_ttt_rejected_batches += 1
                logits = baseline_logits
            else:
                logits = adapted_logits
        else:
            logits = adapter.adapt_on_batch(x)

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = top1_accuracy(correct, total)
    return {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "safe_ttt_rejected_batches": safe_ttt_rejected_batches,
        "safe_ttt_total_batches": safe_ttt_total_batches,
    }

def mean_corruption_error(adapter, root: str, device: str, severity: int = 3, batch_size: int = 16, use_ttt: bool = True) -> float:
    from src.datasets import build_cifar10c_loader, CIFAR10C_CORRUPTIONS

    errors = []
    for corruption in CIFAR10C_CORRUPTIONS:
        loader = build_cifar10c_loader(
            root=root,
            corruption=corruption,
            severity=severity,
            batch_size=batch_size,
        )
        result = evaluate_ttt(adapter, loader, device, use_ttt=use_ttt)
        errors.append(1.0 - result["accuracy"])
        print(f" [{corruption}] err={errors[-1]*100:.2f}%")
        
    mce = sum(errors) / len(errors)
    print(f"[mCE] severity={severity}, use_ttt={use_ttt} -> {mce*100:.2f}%")
    return mce
