# Results and verification

This document records the reported results from the paper, an independent recomputation from the paper's own confusion-matrix figures, and the discrepancies found. The recomputation is reproducible with `python scripts/recompute_metrics.py`.

## 1. Test-set confusion matrices (recomputed)

Each matrix is read from the paper's figure. Rows are the actual class, columns the predicted class. Every matrix totals **448**, matching the stated test-set size.

**InceptionV3** (figure 3)

| Actual \ Predicted | AIN | non_sickle | sickle |
|---|---:|---:|---:|
| AIN (140) | 114 | 26 | 0 |
| non_sickle (135) | 34 | 101 | 0 |
| sickle (173) | 6 | 10 | 157 |

**MobileNetV2** (figure 4)

| Actual \ Predicted | AIN | non_sickle | sickle |
|---|---:|---:|---:|
| AIN (140) | 120 | 20 | 0 |
| non_sickle (135) | 38 | 97 | 0 |
| sickle (173) | 6 | 9 | 158 |

**ShuffleNetV2** (figure 5)

| Actual \ Predicted | AIN | non_sickle | sickle |
|---|---:|---:|---:|
| AIN (140) | 112 | 28 | 0 |
| non_sickle (135) | 18 | 117 | 0 |
| sickle (173) | 6 | 11 | 156 |

**ResNet50** (figure 6, attributed by figure order; the image is unlabeled)

| Actual \ Predicted | AIN | non_sickle | sickle |
|---|---:|---:|---:|
| AIN (140) | 70 | 48 | 22 |
| non_sickle (135) | 75 | 30 | 30 |
| sickle (173) | 40 | 5 | 128 |

## 2. Reported vs. recomputed

| Model | Accuracy (paper) | Accuracy (recomputed) | Macro-F1 (paper, Table 2) | Macro-F1 (recomputed) | Sickle recall (paper) | Sickle recall (recomputed) | Reproduces? |
|---|---:|---:|---:|---:|---:|---:|:--:|
| ResNet50 | 87.28% | **50.89%** | 0.86 (text: 0.94) | **0.477** | 0.97 | **0.740** | **No** |
| ShuffleNetV2 | 85.94% | 85.94% | 0.85 | 0.855 | 0.90 | 0.902 | Yes |
| MobileNetV2 | 83.71% | 83.71% | 0.83 | 0.829 | 0.91 | 0.913 | Yes |
| InceptionV3 | 83.04% | 83.04% | 0.82 | 0.823 | 0.91 | 0.908 | Yes |

Sickle-class F1 is 0.95 for the three reproducible models, matching the paper's F1 row.

## 3. Discrepancies to resolve before citing the results

1. **ResNet50 is not supported by its figure.** The paper's headline model reports 87.28% accuracy. The confusion matrix attributed to it implies 50.89%. The paper also says AIN images were predicted "with high reliability", whereas the matrix gives 50% AIN recall, and it reports macro-F1 0.94 in the text and 0.86 in Table 2. Possible explanations include a figure from a different training run, a different checkpoint, or an error in the figure or table. The checkpoint `resnet50_sickle_dropout.pth` could not be evaluated here because the dataset is not available offline. The authors should re-evaluate that checkpoint on the 448-image test set and report the result.
2. **Validation vs. test.** The text describes 83.04%, 83.71% and 85.94% as validation accuracy. These values reproduce exactly from matrices that total 448, the stated test size. Either the text mislabels the split or the validation set has the same size, and the wording should be made consistent.
3. **Split proportions.** The paper states 70 / 20 / 10. The counts (2,789 / 459 / 448 of 3,696) give 75.4 / 12.4 / 12.1.
4. **Table 2 labelling.** The F1 row (0.95) matches sickle-class F1, not macro-F1. "Support 173" is the sickle count, not the test size. The unlabelled "Macro-avg" row matches either macro-F1 or macro-recall to two decimals, so it cannot be interpreted without clarification.
5. **Class names.** The text uses "nonsickle", the figures use "non_sickle". This repository uses `non_sickle`, following the matrix labels and alphabetical ImageFolder order.
6. **Bibliography.** Reference [5] is 2023 in the text but 2021 in the list. Reference [9] does not match the study attributed to Jiebin Liang et al.
7. **Data provenance and ethics.** Kaggle and Roboflow each have their own licence terms. The paper does not state whether patient images were de-identified or whether ethics approval was obtained.

## 4. Scope of the results

- A single merged dataset of about 3,700 images; no external or multi-site validation.
- Three classes; the "AIN" class is a catch-all for unclear images, not a clinical category.
- Explainability (Grad-CAM, LIME) was assessed visually on examples. No quantitative explanation metric was reported.
- These results are research findings and are not validated for clinical use.
