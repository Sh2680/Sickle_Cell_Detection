from sickle_cell.config import CLASS_NAMES, MODEL_SPECS, MODELS_DIR


def test_three_classes_in_alphabetical_order():
    assert CLASS_NAMES == ("AIN", "non_sickle", "sickle")
    assert list(CLASS_NAMES) == sorted(CLASS_NAMES)


def test_every_checkpoint_file_is_declared():
    names = {p.name for p in MODELS_DIR.glob("*.pth")}
    declared = {spec["file"] for spec in MODEL_SPECS.values()}
    assert names == declared


def test_input_sizes():
    assert MODEL_SPECS["inception"]["input_size"] == 299
    assert all(MODEL_SPECS[m]["input_size"] == 224 for m in ("resnet50", "shufflenet", "mobilenet"))
