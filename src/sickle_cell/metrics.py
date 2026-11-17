"""Classification metrics computed from a confusion matrix.

Pure Python on purpose: these functions run without PyTorch installed, so the
numbers in the paper can be re-checked from the published figures anywhere.
"""
from typing import Dict, List, Sequence


def _check_square(cm: Sequence[Sequence[int]]) -> int:
    n = len(cm)
    if n == 0 or any(len(row) != n for row in cm):
        raise ValueError("confusion matrix must be square and non-empty")
    return n


def accuracy(cm: Sequence[Sequence[int]]) -> float:
    n = _check_square(cm)
    total = sum(sum(row) for row in cm)
    correct = sum(cm[i][i] for i in range(n))
    return correct / total if total else 0.0


def per_class_metrics(cm: Sequence[Sequence[int]], class_names: Sequence[str]) -> Dict[str, Dict[str, float]]:
    n = _check_square(cm)
    if len(class_names) != n:
        raise ValueError("class_names length must match the matrix size")
    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(class_names):
        tp = cm[i][i]
        fn = sum(cm[i]) - tp
        fp = sum(cm[r][i] for r in range(n)) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(cm[i]),
        }
    return out


def macro_f1(cm: Sequence[Sequence[int]], class_names: Sequence[str]) -> float:
    per = per_class_metrics(cm, class_names)
    return sum(v["f1"] for v in per.values()) / len(per)
