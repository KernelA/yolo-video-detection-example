import pytest

from yolo_models.processing.color_utils import hex_to_rgb, rgb_to_hex


@pytest.mark.parametrize("r", [0, 128, 255])
@pytest.mark.parametrize("g", [0, 128, 255])
@pytest.mark.parametrize("b", [0, 128, 255])
def test_rgb_seq(r, g, b):
    color = (r, g, b)
    assert hex_to_rgb(rgb_to_hex(color)) == color
