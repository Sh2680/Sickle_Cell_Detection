"""Image transforms matching the paper's preprocessing description.

Training: resize, random horizontal flip, rotation of +/-15 degrees, colour
jitter, random resized crop (80-100% of the area), ImageNet normalisation.
Evaluation: resize to the model input size, ImageNet normalisation.

The paper does not give the colour-jitter strengths; the values below are
assumptions and can be changed without affecting checkpoint loading.
"""
from torchvision import transforms

from .config import IMAGENET_MEAN, IMAGENET_STD


def build_train_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_eval_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
