"""Model construction and checkpoint loading for the four architectures."""
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models as tv_models

from .config import CLASS_NAMES, MODEL_SPECS, MODELS_DIR


def _build_resnet50(num_classes: int, dropout: float = 0.5) -> nn.Module:
    # Checkpoint keys are fc.1.* -> head is Sequential(Dropout, Linear).
    model = tv_models.resnet50(weights=None)
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    return model


def _build_shufflenet(num_classes: int) -> nn.Module:
    model = tv_models.shufflenet_v2_x1_0(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_mobilenet(num_classes: int) -> nn.Module:
    import timm
    return timm.create_model("mobilenetv2_100", pretrained=False, num_classes=num_classes)


_BUILDERS = {
    "resnet50": _build_resnet50,
    "shufflenet": _build_shufflenet,
    "mobilenet": _build_mobilenet,
}


def load_model(name: str, device: str = "cpu", weights_path: Optional[Path] = None) -> nn.Module:
    """Load a trained checkpoint ready for inference.

    The InceptionV3 file is a pickled full model (requires timm); the other
    three are state dicts loaded into the architectures above with strict=True,
    so any key mismatch fails loudly instead of silently.
    """
    if name not in MODEL_SPECS:
        raise KeyError(f"unknown model '{name}'. Choose from {sorted(MODEL_SPECS)}")
    path = Path(weights_path) if weights_path else MODELS_DIR / MODEL_SPECS[name]["file"]
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    if name == "inception":
        obj = torch.load(path, map_location=device, weights_only=False)
        if isinstance(obj, nn.Module):
            model = obj
        else:  # state dict fallback
            import timm
            model = timm.create_model("inception_v3", pretrained=False, num_classes=len(CLASS_NAMES))
            model.load_state_dict(obj, strict=True)
    else:
        model = _BUILDERS[name](len(CLASS_NAMES))
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)

    model.to(device).eval()
    return model
