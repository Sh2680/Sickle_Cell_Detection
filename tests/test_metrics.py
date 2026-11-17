import json
from pathlib import Path

import pytest

from sickle_cell.metrics import accuracy, macro_f1, per_class_metrics

CLASSES = ["AIN", "non_sickle", "sickle"]
RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_accuracy_perfect():
    assert accuracy([[5, 0, 0], [0, 5, 0], [0, 0, 5]]) == 1.0


def test_per_class_known_values():
    cm = [[2, 1], [1, 2]]
    m = per_class_metrics(cm, ["a", "b"])
    assert m["a"]["precision"] == pytest.approx(2 / 3)
    assert m["a"]["recall"] == pytest.approx(2 / 3)
    assert m["a"]["support"] == 3


def test_rejects_non_square():
    with pytest.raises(ValueError):
        accuracy([[1, 2, 3], [4, 5, 6]])


def test_rejects_wrong_class_count():
    with pytest.raises(ValueError):
        per_class_metrics([[1, 0], [0, 1]], ["only_one"])


def test_paper_matrices_reproduce_reported_accuracy():
    """Three of the four matrices in the paper reproduce the reported accuracy exactly."""
    data = json.loads((RESULTS / "confusion_matrices.json").read_text())
    expected = {"InceptionV3": 83.04, "MobileNetV2": 83.71, "ShuffleNetV2": 85.94}
    for name, pct in expected.items():
        assert round(accuracy(data["matrices"][name]) * 100, 2) == pct


def test_resnet50_matrix_does_not_match_reported_accuracy():
    """Documents a known discrepancy: the ResNet50 figure implies ~50.9%, not 87.28%."""
    data = json.loads((RESULTS / "confusion_matrices.json").read_text())
    acc = accuracy(data["matrices"]["ResNet50"]) * 100
    assert round(acc, 2) == 50.89
    assert macro_f1(data["matrices"]["ResNet50"], CLASSES) < 0.5
