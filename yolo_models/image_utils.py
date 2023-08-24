from typing import NamedTuple, Tuple, Optional

import cv2
import numpy as np


class PadInfo(NamedTuple):
    pad_left: int
    pad_top: int


class ScaleInfo(NamedTuple):
    to_orig_scale_width: float
    to_orig_scale_height: float


def pad_image(
        image: np.ndarray,
        required_width: int,
        required_height: int,
        padded_image: Optional[np.ndarray] = None) -> Tuple[np.ndarray, PadInfo]:
    image_height = image.shape[0]
    image_width = image.shape[1]

    if image_height == required_height and image_width == required_width:
        if padded_image is None:
            return image, PadInfo(0, 0)

        np.copyto(padded_image, image)
        return padded_image, PadInfo(0, 0)

    assert image_height <= required_height
    assert image_width <= required_width

    pad_width = required_width - image_width
    pad_height = required_height - image_height

    pad_width_left = pad_width // 2
    pad_width_right = required_width - pad_width_left - image_width
    pad_height_top = pad_height // 2
    pad_height_bottom = required_height - pad_height_top - image_height

    if padded_image is None:
        padded_image = np.pad(image, ((pad_height_top, pad_height_bottom),
                              (pad_width_left, pad_width_right), (0, 0)), constant_values=0)
    else:
        padded_image.fill(0)
        height_last_index = padded_image.shape[0] - pad_height_bottom
        width_last_index = padded_image.shape[1] - pad_width_right
        padded_image[pad_height_top:height_last_index, pad_width_left:width_last_index] = image

    return padded_image, PadInfo(pad_width_left, pad_height_top)


def resize_to_required_size_keep_aspect_ratio(image: np.ndarray, max_size: int) -> Tuple[np.ndarray, ScaleInfo]:
    if max(image.shape) == max_size:
        return image, ScaleInfo(1, 1)

    height, width = image.shape[:2]
    aspect_ratio = width / height

    if aspect_ratio > 1:
        # Landscape orientation
        new_width = max_size
        new_height = min(round(new_width / aspect_ratio), max_size)
    else:
        # Portrait orientation
        new_height = max_size
        new_width = min(round(new_height * aspect_ratio), max_size)

    scale_info = ScaleInfo(width / new_width, height / new_height)

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4), scale_info
