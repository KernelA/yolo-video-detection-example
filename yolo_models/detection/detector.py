from typing import List, NamedTuple, Optional, Sequence

import cv2
import numpy as np
import onnxruntime as ort

from ..bbox_utils import (clip_boxes_inplace,
                          restore_original_coordinates_inplace,
                          scale_bbox_inplace, xyhw2xyxy_inplace,
                          yolo_bbox2xywh_inplace)
from ..classes import parse_class_mapping_from_str
from ..image_utils import (PadInfo, ScaleInfo, pad_image,
                           resize_to_required_size_keep_aspect_ratio)


class DetectionInfo(NamedTuple):
    scores: np.ndarray
    xyxy_boxes: np.ndarray
    classes: Sequence[str]


class ONNXYoloV8Detector:
    def __init__(self, path_to_model: str, providers: Optional[List[str]] = None):
        self.session = ort.InferenceSession(
            path_to_model,
            providers=providers
        )
        inputs = self.session.get_inputs()
        self.session.get_modelmeta()
        outputs = self.session.get_outputs()

        assert len(inputs) == 1, f"Detection model expected only one input, but found {len(inputs)}"
        assert len(
            outputs) == 1, f"Detection model expected only one input, but found {len(outputs)}"

        self._class_mapping = parse_class_mapping_from_str(
            self.session.get_modelmeta().custom_metadata_map["names"])
        input_shape = inputs[0].shape
        self._max_image_size = max(input_shape[2:])
        self.input_name = inputs[0].name

    def postpocess(self,
                   pred: np.ndarray,
                   score_threshold: float,
                   nms_threshold: float,
                   pad_info: PadInfo,
                   scale_info: ScaleInfo,
                   original_width: int,
                   original_height: int) -> DetectionInfo:
        """pred: [1 x 84 x 8400]

            1 - bath size
            84 - 0,1,2,3 is x,y,width,height, 4,5,6,7,8,9.... probability for each class
            8400 - number of possible detected objects

            https://github.com/ultralytics/ultralytics/issues/2670#issuecomment-1551453142
        """
        xywh_yolo = pred[0, :4, :].T
        raw_scores = pred[0, 4:, :]
        scores = raw_scores.max(axis=0)

        xywh = yolo_bbox2xywh_inplace(xywh_yolo)
        bbox_indices = cv2.dnn.NMSBoxes(xywh, scores, score_threshold, nms_threshold)

        det_xywh = xywh[bbox_indices]
        class_indices = raw_scores[:, bbox_indices].argmax(axis=0)

        restore_original_coordinates_inplace(det_xywh, pad_info)
        scale_bbox_inplace(det_xywh, scale_info)
        det_xyxy = xyhw2xyxy_inplace(det_xywh).round().astype(int)
        clip_boxes_inplace(det_xyxy, original_width, original_height)

        return DetectionInfo(
            scores[bbox_indices],
            det_xyxy,
            [self._class_mapping[int(class_index)] for class_index in class_indices]
        )

    def preprocess_image(self, image: np.ndarray):
        image, scale_info = resize_to_required_size_keep_aspect_ratio(image, self._max_image_size)
        image, pad_info = pad_image(image, self._max_image_size, self._max_image_size)
        image = image.transpose((2, 0, 1)).astype(np.float32)
        image /= 255
        return image[np.newaxis, ...], scale_info, pad_info

    def _raw_predict(self, image: np.ndarray):
        """image is RGB image
        """
        return self.session.run(None, {self.input_name: image})[0]

    def predict(self, image: np.ndarray, score_threshold: float, nms_threshold: float) -> DetectionInfo:
        original_height, original_width = image.shape[:2]
        image, scale_info, pad_info = self.preprocess_image(image)
        raw_pred = self._raw_predict(image)
        return self.postpocess(raw_pred, score_threshold, nms_threshold, pad_info, scale_info, original_width=original_width, original_height=original_height)
