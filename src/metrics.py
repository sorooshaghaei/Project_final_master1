# common evaluation metrics
from typing import Dict
import torch

def top1_accuracy(num_correct: int, num_total: int) -> float:
    # guard against division by zero
    if num_total <= 0:
        raise ValueError("num_total must be positive")
    return float(num_correct) / float(num_total)

# evaluate ttt against baseline

def evaluate_ttt(adapter, test_loader, device: str, use_ttt: bool = True) -> Dict[str, float]:
    correct = total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad() if not use_ttt else torch.enable_grad():
            logits = adapter.adapt_on_batch(x) if use_ttt else adapter.predict(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = top1_accuracy(correct, total)
    return {"accuracy": acc, "correct": correct, "total": total}

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
