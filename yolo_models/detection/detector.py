from typing import List, NamedTuple, Optional, Sequence, Dict
from abc import ABC, abstractmethod, abstractproperty

import numpy as np
import cv2
import onnxruntime as ort
from ultralytics.nn.autobackend import check_class_names
from ultralytics.nn.tasks import attempt_load_one_weight
import torch
from torchvision import ops
from torchvision.transforms import functional

from yolo_models.image_utils import PadInfo, ScaleInfo

from ..torch_transforms import ResizeToRequiredSize, PadToRequiredSize
from ..bbox_utils import (clip_boxes_inplace,
                          restore_original_coordinates_inplace,
                          scale_bbox_inplace, xyhw2xyxy_inplace,
                          yolo_bbox2xywh_inplace)
from ..classes import parse_class_mapping_from_str
from ..image_utils import (PadInfo, ScaleInfo, pad_image,
                           resize_to_required_size_keep_aspect_ratio)
from ..nms import boxes_nms


class DetectionInfo(NamedTuple):
    scores: np.ndarray
    xyxy_boxes: np.ndarray
    classes: Sequence[str]


class Detector(ABC):
    def __init__(self):
        self._padded_image_buffer = None

    @abstractmethod
    def _init_padded_image_buffer(self):
        raise NotImplementedError()

    @abstractproperty
    def max_image_size(self):
        raise NotImplementedError()

    @abstractproperty
    def class_mapping(self) -> Dict[int, str]:
        raise NotImplementedError()

    def preprocess_image(self, image: np.ndarray):
        if self._padded_image_buffer is None:
            self._init_padded_image_buffer()

        image, scale_info = resize_to_required_size_keep_aspect_ratio(image, self.max_image_size)
        image, pad_info = pad_image(image, self.max_image_size,
                                    self.max_image_size, padded_image=self._padded_image_buffer)
        image = image.transpose((2, 0, 1))

        if image.dtype != np.float32:
            image = image.astype(np.float32)
            image /= 255

        return image[np.newaxis, ...], scale_info, pad_info

    def postpocess(self,
                   pred: np.ndarray,
                   score_threshold: float,
                   nms_threshold: float,
                   pad_info: PadInfo,
                   scale_info: ScaleInfo,
                   original_width: int,
                   original_height: int,
                   max_k: int,
                   eta: float) -> DetectionInfo:
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

        # in some version Touch designer NMSBoxes does not work
        # bbox_indices = cv2.dnn.NMSBoxes(xywh, scores, score_threshold, nms_threshold, eta, max_k)
        bbox_indices = boxes_nms(xywh, scores, score_threshold, nms_threshold, eta, max_k)

        det_xywh = xywh[bbox_indices]
        class_indices = raw_scores[:, bbox_indices].argmax(axis=0)

        restore_original_coordinates_inplace(det_xywh, pad_info)
        scale_bbox_inplace(det_xywh, scale_info)
        det_xyxy = xyhw2xyxy_inplace(det_xywh).round().astype(int)
        clip_boxes_inplace(det_xyxy, original_width, original_height)
        class_mapping = self.class_mapping

        return DetectionInfo(
            scores[bbox_indices],
            det_xyxy,
            [class_mapping[int(class_index)] for class_index in class_indices]
        )

    @abstractmethod
    def _raw_predict(self, image: np.ndarray):
        raise NotImplementedError()

    def predict(self,
                image: np.ndarray,
                score_threshold: float,
                nms_threshold: float,
                max_k: int = 0,
                eta: float = 1.0) -> DetectionInfo:

        original_height, original_width = image.shape[:2]
        image, scale_info, pad_info = self.preprocess_image(image)
        raw_pred = self._raw_predict(image)
        return self.postpocess(
            raw_pred,
            score_threshold,
            nms_threshold,
            pad_info,
            scale_info,
            original_width=original_width,
            original_height=original_height,
            max_k=max_k,
            eta=eta)


class ONNXYoloDetector(Detector):
    def __init__(self,
                 path_to_model: str,
                 providers: Optional[List[str]] = None,
                 provider_options: Optional[List[dict]] = None,
                 session_options: Optional[ort.SessionOptions] = None):
        super().__init__()

        self.session = ort.InferenceSession(
            path_to_model,
            session_options=session_options,
            provider_options=provider_options,
            providers=providers
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()

        assert len(inputs) == 1, f"Detection model expected only one input, but found {len(inputs)}"
        assert len(
            outputs) == 1, f"Detection model expected only one input, but found {len(outputs)}"

        self._class_mapping = parse_class_mapping_from_str(
            self.session.get_modelmeta().custom_metadata_map["names"])
        input_shape = inputs[0].shape
        self._num_channels = input_shape[1]
        self._max_image_size = max(input_shape[2:])
        self._input_name = inputs[0].name

    def _init_padded_image_buffer(self):
        if self._padded_image_buffer is None:
            self._padded_image_buffer = np.empty(
                (self._max_image_size, self._max_image_size, self._num_channels), dtype=np.uint8)

    @property
    def class_mapping(self) -> Dict[int, str]:
        return self._class_mapping

    @property
    def max_image_size(self):
        return self._max_image_size

    def _raw_predict(self, image: np.ndarray):
        """image is RGB image
        """
        return self.session.run(None, {self._input_name: image})[0]

    def predict(self,
                image: np.ndarray,
                score_threshold: float,
                nms_threshold: float,
                max_k: int = 0,
                eta: float = 1.0) -> DetectionInfo:
        original_height, original_width = image.shape[:2]
        image, scale_info, pad_info = self.preprocess_image(image)
        raw_pred = self._raw_predict(image)
        return self.postpocess(
            raw_pred,
            score_threshold,
            nms_threshold,
            pad_info,
            scale_info,
            original_width=original_width,
            original_height=original_height,
            max_k=max_k,
            eta=eta)


class PyTorchYoloDetector(Detector):
    def __init__(self,
                 path_to_model: str,
                 device: torch.device,
                 half_precision: bool = False,
                 trace: bool = False,
                 numpy_post_process: bool = True):
        super().__init__()
        self._device = device
        self._half_precision = half_precision
        self._model, _ = attempt_load_one_weight(path_to_model,
                                                device=device,
                                                fuse=True)
        self._model = self._model.half() if half_precision else self._model.float()
        self._model.eval()
        self._class_mapping = self._model.module.names if hasattr(self._model, 'module') else self._model.names  # get class names
        self._class_mapping = check_class_names(self._class_mapping)
        self._num_channels = self._model.yaml["ch"]
        self._max_image_size = self._model.args["imgsz"]
        self._numpy_post_process = numpy_post_process

        gen = torch.Generator(self._device).manual_seed(0)
        warmup_image = torch.rand((1, self._num_channels, self.max_image_size, self.max_image_size), generator=gen, device=self._device)

        if trace:
            self._model(warmup_image)
            self._model = torch.jit.trace(self._model, warmup_image)

        self._image_transforms = torch.jit.script(torch.nn.Sequential(
            ResizeToRequiredSize(self.max_image_size),
            PadToRequiredSize(self.max_image_size, self.max_image_size)
        ))

    def _init_padded_image_buffer(self):
        pass

    @torch.no_grad()
    def preprocess_image(self, image: np.ndarray):
        torch_image = functional.to_tensor(image).to(self._device)
        torch_image = functional.convert_image_dtype(torch_image, dtype=torch.float16 if self._half_precision else torch.float32)
        torch_image, scale_info, pad_info = self._image_transforms(torch_image)
        return torch_image[None, ...], scale_info, pad_info

    @property
    def max_image_size(self):
        return self._max_image_size

    @property
    def class_mapping(self) -> Dict[int, str]:
        return self._class_mapping

    @torch.no_grad()
    def _raw_predict(self, image: torch.Tensor):
        out = self._model(image)[0]

        if self._numpy_post_process:
            return out.cpu().numpy()

        return out

    def postpocess(self, pred: np.ndarray,
                   score_threshold: float,
                   nms_threshold: float,
                   pad_info: PadInfo,
                   scale_info: ScaleInfo,
                   original_width: int,
                   original_height: int,
                   max_k: int,
                   eta: float) -> DetectionInfo:

        if self._numpy_post_process:
            return super().postpocess(pred, score_threshold, nms_threshold, pad_info, scale_info, original_width, original_height, max_k, eta)

        return self._torch_postpocess(pred, score_threshold, nms_threshold, pad_info, scale_info, original_width, original_height, max_k, eta)


    def _torch_postpocess(self,
                   pred: torch.Tensor,
                   score_threshold: float,
                   nms_threshold: float,
                   pad_info: PadInfo,
                   scale_info: ScaleInfo,
                   original_width: int,
                   original_height: int,
                   max_k: int,
                   eta: float) -> DetectionInfo:
        """pred: [1 x 84 x 8400]

            1 - bath size
            84 - 0,1,2,3 is x,y,width,height, 4,5,6,7,8,9.... probability for each class
            8400 - number of possible detected objects

            https://github.com/ultralytics/ultralytics/issues/2670#issuecomment-1551453142
        """
        xywh_yolo = pred[0, :4, :].T
        raw_scores = pred[0, 4:, :]
        scores, _ = raw_scores.max(dim=0)

        xywh = yolo_bbox2xywh_inplace(xywh_yolo)
        xyxy = ops.box_convert(xywh, "xywh", "xyxy")
        bbox_indices = ops.nms(xyxy, scores, nms_threshold)

        if max_k > 0:
            bbox_indices = bbox_indices[:max_k]

        bbox_indices = bbox_indices[scores[bbox_indices] > score_threshold]
        det_xywh = xywh[bbox_indices]
        class_indices = raw_scores[:, bbox_indices].argmax(axis=0).cpu()

        restore_original_coordinates_inplace(det_xywh, pad_info)
        scale_bbox_inplace(det_xywh, scale_info)
        det_xyxy = xyhw2xyxy_inplace(det_xywh)
        det_xyxy.round_()
        det_xyxy = det_xyxy.to(torch.int32)
        clip_boxes_inplace(det_xyxy, original_width, original_height)
        class_mapping = self.class_mapping

        return DetectionInfo(
            scores[bbox_indices].cpu().numpy(),
            det_xyxy.cpu().numpy(),
            [class_mapping[int(class_index)] for class_index in class_indices]
        )
