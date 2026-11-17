# Data

The dataset is **not** included in this repository.

Source: [Sanika2903/sickle-cell-classification-dataset](https://huggingface.co/datasets/Sanika2903/sickle-cell-classification-dataset) on Hugging Face.
It merges two sources described in the paper: a Kaggle blood-smear collection (Eastern Uganda, Field and Leishman stains) and a Roboflow multi-class hematology set. Check each source's license before redistributing or commercial use.

Expected layout for training or evaluation (ImageFolder convention, class folder names must match `CLASS_NAMES` in `src/sickle_cell/config.py`):

```
data/
  train/
    AIN/
    non_sickle/
    sickle/
  val/
    AIN/  non_sickle/  sickle/
  test/
    AIN/  non_sickle/  sickle/
```

Paper split sizes: train 2,789 · validation 459 · test 448 (3,696 total).
