import torch
import numpy as np

from torchvision import transforms
from yolo_models.torch_transforms import ResizeToRequiredSize, PadToRequiredSize


def test_jit():
    seq_transforms = torch.nn.Sequential(
        ResizeToRequiredSize(10),
        PadToRequiredSize(20, 20)
    )

    to_tensor = transforms.ToTensor()

    seq_transforms = torch.jit.script(seq_transforms)
    image = np.random.randint(0, 256, size=(20, 20, 3), dtype=np.uint8)

    tensor_image, _, _ = seq_transforms(to_tensor(image))
