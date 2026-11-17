"""Grad-CAM: class-discriminative heatmaps from a chosen convolutional layer."""
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, output):
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Return a (H, W) heatmap in [0, 1] matched to the input size, and the class used."""
        if x.dim() != 4 or x.shape[0] != 1:
            raise ValueError("GradCAM expects a single image batch of shape (1, C, H, W)")
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), class_idx
