"""Common evaluation metrics."""


def top1_accuracy(num_correct: int, num_total: int) -> float:
    if num_total <= 0:
        raise ValueError("num_total must be positive")
    return float(num_correct) / float(num_total)
