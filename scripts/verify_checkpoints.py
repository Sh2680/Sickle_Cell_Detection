#!/usr/bin/env python3
"""Verify checkpoint integrity (SHA-256) and that each one loads into its architecture.

Usage:
    python scripts/verify_checkpoints.py            # checksums only
    python scripts/verify_checkpoints.py --load     # also build and load each model (needs torch, timm)
"""
import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sickle_cell.config import MODEL_SPECS, MODELS_DIR  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true", help="also load each checkpoint")
    args = parser.parse_args()

    expected = {}
    for line in (MODELS_DIR / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name.strip()] = digest

    failures = 0
    for name, spec in MODEL_SPECS.items():
        path = MODELS_DIR / spec["file"]
        if not path.exists():
            print(f"[MISSING] {path.name}")
            failures += 1
            continue
        ok = sha256(path) == expected.get(path.name)
        print(f"[{'OK' if ok else 'FAIL'}] checksum {path.name}")
        failures += 0 if ok else 1

        if args.load:
            try:
                from sickle_cell.models import load_model
                load_model(name)
                print(f"[OK]   loads into {spec['display_name']}")
            except Exception as exc:  # report and continue with other models
                print(f"[FAIL] {spec['display_name']}: {exc}")
                failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
