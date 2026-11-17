"""Central configuration: class labels, input sizes, and checkpoint locations."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# Alphabetical order, matching torchvision's ImageFolder convention and the
# row/column order of the paper's confusion matrices.
CLASS_NAMES = ("AIN", "non_sickle", "sickle")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODEL_SPECS = {
    "resnet50": {
        "file": "resnet50_sickle_dropout.pth",
        "input_size": 224,
        "display_name": "ResNet50",
    },
    "shufflenet": {
        "file": "shufflenet_sickle.pth",
        "input_size": 224,
        "display_name": "ShuffleNetV2",
    },
    "mobilenet": {
        "file": "mobilenet_sickle.pth",
        "input_size": 224,
        "display_name": "MobileNetV2",
    },
    "inception": {
        "file": "inception_v3_full_model.pth",
        "input_size": 299,
        "display_name": "InceptionV3",
    },
}
