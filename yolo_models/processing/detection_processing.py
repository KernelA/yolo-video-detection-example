from typing import Optional

import numpy as np
import torch
import cv2

from .info import ParamsIndex
from .base_processing import BaseProcessServer
from ..detection import PyTorchYoloDetector

class DetectorProcess(BaseProcessServer):
    def __init__(self,
                 path_to_model: str,
                 update_shared_mem_name: str,
                 params_shared_mem_name: str,
                 array_shared_mem_name: str,
                 image_with: int,
                 image_height: int,
                 num_channels: int,
                image_dtype,
                device: Optional[str] = None):
        super().__init__(update_shared_mem_name, params_shared_mem_name, array_shared_mem_name, image_with, image_height, num_channels, image_dtype)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cpu":
            self._logging.warning("GPU is not available. Select CPU. Can be slow")

        self._detector = PyTorchYoloDetector(path_to_model,
                                             device=torch.device(device),
                                             trace=True,
                                             numpy_post_process=True)

    def process(self, image: np.ndarray, params: list):
        det_info = self._detector.predict(
            image,
            score_threshold=params[ParamsIndex.SCORE_THRESH],
            nms_threshold=params[ParamsIndex.IOU_THRESH],
            max_k=params[ParamsIndex.TOP_K],
            eta=params[ParamsIndex.ETA])

        for xyxy in det_info.xyxy_boxes:
            cv2.rectangle(image, xyxy[:2], xyxy[2:], (255, 0, 0), thickness=1)
