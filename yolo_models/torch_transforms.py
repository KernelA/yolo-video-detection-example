from typing import Tuple

import torch
from torch import nn
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

from .image_utils import ScaleInfo, PadInfo


class PadToRequiredSize(nn.Module):
    def __init__(self, required_width: int, required_height: int):
        super().__init__()
        self.required_width = required_width
        self.required_height = required_height

    def forward(self, input: Tuple[torch.Tensor, ScaleInfo]) -> Tuple[torch.Tensor, ScaleInfo, PadInfo]:
        image, scale_info = input
        image_height = image.shape[1]
        image_width = image.shape[2]

        assert image_height <= self.required_height
        assert image_width <= self.required_width

        if image_height == self.required_height and image_width == self.required_width:
            return image, scale_info, PadInfo(0, 0)

        pad_width = self.required_width - image_width
        pad_height = self.required_height - image_height

        pad_width_left = pad_width // 2
        pad_width_right = self.required_width - pad_width_left - image_width
        pad_height_top = pad_height // 2
        pad_height_bottom = self.required_height - pad_height_top - image_height

        return F.pad(
            image, [pad_width_left, pad_height_top, pad_width_right, pad_height_bottom]), scale_info, PadInfo(pad_width_left, pad_height_top)

class ResizeToRequiredSize(nn.Module):
    def __init__(self, max_size: int):
        super().__init__()
        self.max_size = max_size

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, ScaleInfo]:
        if max(image.shape) == self.max_size:
            return image, ScaleInfo(1.0, 1.0)

        height, width = image.shape[1:3]
        aspect_ratio = width / height

        if aspect_ratio > 1:
            # Landscape orientation
            new_width = self.max_size
            new_height = int(min(round(new_width / aspect_ratio), self.max_size))
        else:
            # Portrait orientation
            new_height = self.max_size
            new_width = int(min(round(new_height * aspect_ratio), self.max_size))


        return F.resize(image, [new_height, new_width], interpolation=InterpolationMode.BILINEAR, antialias=True), ScaleInfo(width / new_width, height / new_height)
