"""High-level inference API."""
from pathlib import Path
from typing import Dict, Union

import torch
from PIL import Image

from .config import CLASS_NAMES, MODEL_SPECS
from .models import load_model
from .preprocess import build_eval_transform

ImageInput = Union[str, Path, Image.Image]


class SickleCellClassifier:
    """Wraps a trained checkpoint for single-image prediction.

    Example:
        clf = SickleCellClassifier("resnet50")
        print(clf.predict("smear.png"))
    """

    def __init__(self, model_name: str = "resnet50", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.input_size = MODEL_SPECS[model_name]["input_size"]
        self.transform = build_eval_transform(self.input_size)
        self.model = load_model(model_name, device=device)

    def _to_tensor(self, image: ImageInput) -> torch.Tensor:
        img = image if isinstance(image, Image.Image) else Image.open(image)
        img = img.convert("RGB")
        return self.transform(img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image: ImageInput) -> Dict[str, float]:
        """Return class probabilities keyed by class name."""
        logits = self.model(self._to_tensor(image))
        if isinstance(logits, tuple):  # defensive: some heads return aux outputs
            logits = logits[0]
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
        if len(probs) != len(CLASS_NAMES):
            raise RuntimeError(f"model produced {len(probs)} outputs, expected {len(CLASS_NAMES)}")
        return dict(zip(CLASS_NAMES, probs))
