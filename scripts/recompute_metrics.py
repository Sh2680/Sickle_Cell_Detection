#!/usr/bin/env python3
"""Recompute accuracy and per-class metrics from the paper's confusion matrices and
compare them with the values the paper reports.

Usage:
    python scripts/recompute_metrics.py
Writes results/recomputed_metrics.json and prints a comparison table.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sickle_cell.metrics import accuracy, macro_f1, per_class_metrics  # noqa: E402

RESULTS = ROOT / "results"


def main() -> None:
    cm_data = json.loads((RESULTS / "confusion_matrices.json").read_text())
    reported = json.loads((RESULTS / "reported_metrics.json").read_text())["models"]
    classes = cm_data["classes"]

    recomputed = {}
    print(f"{'Model':<14}{'n':>5}{'Acc (paper)':>13}{'Acc (recomp.)':>15}{'Macro-F1 (recomp.)':>21}{'Sickle recall':>15}")
    for name, cm in cm_data["matrices"].items():
        acc = accuracy(cm)
        mf1 = macro_f1(cm, classes)
        per = per_class_metrics(cm, classes)
        recomputed[name] = {
            "n": sum(map(sum, cm)),
            "accuracy": round(acc * 100, 2),
            "macro_f1": round(mf1, 3),
            "per_class": {k: {m: round(v, 3) for m, v in d.items()} for k, d in per.items()},
        }
        paper_acc = reported[name]["accuracy"]
        match = "match" if abs(paper_acc - acc * 100) < 0.01 else "MISMATCH"
        print(f"{name:<14}{sum(map(sum, cm)):>5}{paper_acc:>12.2f}%{acc * 100:>14.2f}%"
              f"{mf1:>21.3f}{per['sickle']['recall']:>15.3f}   {match}")

    out = RESULTS / "recomputed_metrics.json"
    out.write_text(json.dumps(recomputed, indent=2))
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
