# Model card: sickle cell classifiers

Four ImageNet-pretrained architectures fine-tuned for three-way classification of blood smear cell images: **AIN**, **non_sickle**, **sickle**.

| File | Architecture | Input | Params (approx.) | Loader | Size |
|---|---|---|---:|---|---:|
| `resnet50_sickle_dropout.pth` | torchvision ResNet-50, head `Dropout → Linear(2048, 3)` | 224×224 | 23.6 M | state dict | 94 MB |
| `shufflenet_sickle.pth` | torchvision ShuffleNetV2 ×1.0, head `Linear(1024, 3)` | 224×224 | 1.27 M | state dict | 5 MB |
| `mobilenet_sickle.pth` | timm `mobilenetv2_100`, head `Linear(1280, 3)` | 224×224 | 2.26 M | state dict | 9 MB |
| `inception_v3_full_model.pth` | timm `inception_v3` (pickled full model) | 299×299 | — | `torch.load(..., weights_only=False)` | 88 MB |

Parameter counts for the state-dict models were approximated by summing the checkpoint tensors (this includes BatchNorm buffers). The InceptionV3 file is a pickled full module, so its count is not listed here.

## Intended use
Research on automated screening support for sickle cell disease from microscopic blood smear images.

## Not intended for
- Clinical diagnosis, treatment decisions or any use without review by a qualified clinician or laboratory.
- Populations, staining protocols or microscopes that differ substantially from the training data.

## Training data
Merged Kaggle (Uganda, Field/Leishman stains) and Roboflow blood-smear datasets. See `data/README.md` and the paper.

## Reported performance (test set, n = 448)

| Model | Accuracy (paper) | Accuracy (recomputed from figure) |
|---|---:|---:|
| ResNet50 | 87.28% | **50.89%** (does not reproduce) |
| ShuffleNetV2 | 85.94% | 85.94% |
| MobileNetV2 | 83.71% | 83.71% |
| InceptionV3 | 83.04% | 83.04% |

The ResNet50 discrepancy is documented in `docs/RESULTS.md`. Until it is resolved, treat the ResNet50 checkpoint as unverified. The ShuffleNetV2 checkpoint has the best reproduced accuracy in this release.

## Known limitations
- Single merged dataset; no external validation; class imbalance between AIN and the other classes.
- For InceptionV3, MobileNetV2 and ShuffleNetV2, most errors are between AIN and non_sickle. The ResNet50 matrix has a different error pattern (see `docs/RESULTS.md`).
- Input preprocessing (224/299 resize and ImageNet normalisation) must match training; images are assumed to be RGB smears at comparable magnification.

## Integrity
SHA-256 checksums are listed in `models/SHA256SUMS`. Verify with `python scripts/verify_checkpoints.py`.

## Licence
Model weights are released with the repository under the MIT licence (see `LICENSE`). The training data carry their own licences, which are not covered by this one.
