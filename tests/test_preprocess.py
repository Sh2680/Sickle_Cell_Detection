import pytest

torch = pytest.importorskip("torch")
PIL_Image = pytest.importorskip("PIL.Image")

from sickle_cell.preprocess import build_eval_transform, build_train_transform  # noqa: E402


@pytest.mark.parametrize("size", [224, 299])
def test_eval_transform_shape(size):
    img = PIL_Image.new("RGB", (640, 480), color=(200, 100, 100))
    out = build_eval_transform(size)(img)
    assert tuple(out.shape) == (3, size, size)


def test_train_transform_shape():
    img = PIL_Image.new("RGB", (300, 300))
    assert tuple(build_train_transform(224)(img).shape) == (3, 224, 224)
