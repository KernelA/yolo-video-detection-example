from typing import Optional, Dict, Tuple
import json

import numpy as np
import torch
import cv2

from .info import ParamsIndex, DrawInfo
from .base_processing import BaseProcessServer
from ..detection import PyTorchYoloDetector
from .color_utils import check_regex_color, hex_to_rgb, rgb_to_hex


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
        super().__init__(update_shared_mem_name, params_shared_mem_name,
                         array_shared_mem_name, image_with, image_height, num_channels, image_dtype)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cpu":
            self._logging.warning("GPU is not available. Select CPU. Can be slow")

        self._detector = PyTorchYoloDetector(path_to_model,
                                             device=torch.device(device),
                                             trace=True,
                                             numpy_post_process=True)

        self._color_mapping = self.generate_class_colormap()

    @property
    def color_mapping(self):
        return self._color_mapping.copy()

    def generate_class_colormap(self) -> Dict[str, Tuple[int]]:
        return {class_name: (255, 0, 0) for class_name in self._detector.class_mapping.values()}

    def save_class_colormap(self, path_to_file: str):
        with open(path_to_file, "w", encoding="utf-8") as f:
            json.dump({class_name: rgb_to_hex(color)
                      for class_name, color in self.color_mapping.items()}, f, indent=2)

    def load_class_colormap(self, path_to_file: str):
        with open(path_to_file, "rb") as f:
            new_color_mapping = json.load(f)

        for key in self._color_mapping:
            if key not in new_color_mapping:
                raise KeyError(f"Cannot find: '{key}' in the new colormap")

            try:
                self._color_mapping[key] = hex_to_rgb(new_color_mapping[key])
            except ValueError as exc:
                raise KeyError(f"Invalid hex color by '{key}'") from exc

    def process(self,
                image: np.ndarray,
                params: list,
                ):
        det_info = self._detector.predict(
            image,
            score_threshold=params[ParamsIndex.SCORE_THRESH],
            nms_threshold=params[ParamsIndex.IOU_THRESH],
            max_k=params[ParamsIndex.TOP_K],
            eta=params[ParamsIndex.ETA])

        draw_info = DrawInfo(params[ParamsIndex.DRAW_INFO])

        for class_label, score, xyxy in zip(det_info.classes, det_info.scores, det_info.xyxy_boxes):
            color = self._color_mapping[class_label]

            cv2.rectangle(image, xyxy[:2], xyxy[2:], color, thickness=1)

            if DrawInfo.DRAW_TEXT in draw_info:
                (_, text_height), _ = cv2.getTextSize(
                    class_label, cv2.FONT_HERSHEY_DUPLEX, 1, 1)
                new_y = xyxy[1] + text_height
                cv2.putText(image, class_label, (xyxy[0], new_y),
                            cv2.FONT_HERSHEY_DUPLEX, 1, color, 1)

            if DrawInfo.DRAW_CONF in draw_info:
                cv2.putText(image, f"{score:.2f}", xyxy[:2], cv2.FONT_HERSHEY_DUPLEX, 1, color, 1)
