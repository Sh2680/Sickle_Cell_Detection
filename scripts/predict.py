#!/usr/bin/env python3
"""Command-line prediction with optional Grad-CAM overlay.

Usage:
    python scripts/predict.py --model resnet50 --image smear.png
    python scripts/predict.py --model resnet50 --image smear.png --gradcam out.png
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sickle_cell.config import MODEL_SPECS  # noqa: E402
from sickle_cell.predict import SickleCellClassifier  # noqa: E402


def save_gradcam(classifier: SickleCellClassifier, image_path: str, out_path: str) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from sickle_cell.gradcam import GradCAM

    layer = {
        "resnet50": lambda m: m.layer4[-1],
        "shufflenet": lambda m: m.conv5,
        "mobilenet": lambda m: m.conv_head,
        "inception": lambda m: m.Mixed_7c,
    }[classifier.model_name](classifier.model)

    img = Image.open(image_path).convert("RGB")
    x = classifier.transform(img).unsqueeze(0).to(classifier.device)
    cam, idx = GradCAM(classifier.model, layer)(x)

    base = np.array(img.resize((classifier.input_size, classifier.input_size)))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(base)
    ax.imshow(cam, cmap="jet", alpha=0.4)
    ax.set_title(f"Grad-CAM: {['AIN', 'non_sickle', 'sickle'][idx]}")
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return idx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="resnet50")
    parser.add_argument("--image", required=True, help="path to a blood smear image")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gradcam", metavar="OUT_PNG", help="save a Grad-CAM overlay to this path")
    args = parser.parse_args()

    clf = SickleCellClassifier(args.model, device=args.device)
    probs = clf.predict(args.image)
    top = max(probs, key=probs.get)
    print(json.dumps({"model": args.model, "image": args.image, "prediction": top,
                      "probabilities": {k: round(v, 4) for k, v in probs.items()}}, indent=2))
    if args.gradcam:
        save_gradcam(clf, args.image, args.gradcam)
        print(f"Grad-CAM saved to {args.gradcam}")


if __name__ == "__main__":
    main()
