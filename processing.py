import argparse
import enum
from multiprocessing import shared_memory
import time
import sys
import pathlib
import os
import logging

for i, path in enumerate(sys.path):
    if pathlib.Path(path).name == "site-packages":
        index = i
        break
else:
    raise RuntimeError("Cannot find required dir")

sys.path.insert(0, sys.path[index])
del index

from matplotlib import cm
import numpy as np
import torch
import cv2

from yolo_models.detection import PyTorchYoloDetector
from yolo_models.log_set import init_logging

# Sync with touch designer code
@enum.unique
class BufferStates(enum.IntEnum):
    SERVER = 0
    CLIENT = 1
    SERVER_ALIVE = 2

@enum.unique
class States(bytes, enum.Enum):
    NULL_STATE = b'0'
    READY_SERVER_MESSAGE = b'1'
    READY_CLIENT_MESSAGE = b'2'
    IS_SERVER_ALIVE = b'3'

@enum.unique
class ParamsIndex(enum.IntEnum):
    IOU_THRESH = 0
    SCORE_THRESH = 1
    TOP_K = 2
    ETA = 3
    IMAGE_WIDTH = 4
    IMAGE_HEIGHT = 5
    IMAGE_CHANNELS = 6
    SHARED_ARRAY_MEM_NAME = 7
    SHARD_STATE_MEM_NAME = 8
    IMAGE_DTYPE = 9


class ProcessingServer:
    def __init__(self,
                 update_shared_mem_name: str,
                 params_shared_mem_name: str,
                 array_shared_mem_name: str,
                 image_with: int,
                 image_height: int,
                 num_channels: int,
                 image_dtype):
        self._logging  = logging.getLogger("server_processing")
        dtype_size = np.dtype(image_dtype).itemsize
        self._image_size_bytes = image_height * image_with * num_channels * dtype_size
        self._image_width = image_with
        self._image_height = image_height
        self._num_channels = num_channels
        self._image_dtype = image_dtype
        self._sh_mem_update = None
        self._sh_mem_params = None
        self._sh_mem_array = None
        self._shared_array = None
        self._update_shared_mem_name = update_shared_mem_name
        self._params_shared_mem_name = params_shared_mem_name
        self._array_shared_mem_name = array_shared_mem_name

    def init_mem(self):
        assert self._sh_mem_update is None,  "Memory already initialized"
        self._sh_mem_update = shared_memory.SharedMemory(name=self._update_shared_mem_name, create=True, size=len(BufferStates))

        params = [None] * len(ParamsIndex)
        params[ParamsIndex.ETA] = 1.0
        params[ParamsIndex.IOU_THRESH] = 0.5
        params[ParamsIndex.SCORE_THRESH] = 0.5
        params[ParamsIndex.TOP_K] = 0
        params[ParamsIndex.IMAGE_WIDTH] = self._image_width
        params[ParamsIndex.IMAGE_HEIGHT] = self._image_height
        params[ParamsIndex.IMAGE_CHANNELS] = self._num_channels
        params[ParamsIndex.SHARED_ARRAY_MEM_NAME] = self._array_shared_mem_name
        params[ParamsIndex.SHARD_STATE_MEM_NAME] = self._update_shared_mem_name
        params[ParamsIndex.IMAGE_DTYPE] = self._image_dtype

        self._sh_mem_params = shared_memory.ShareableList(
            name=self._params_shared_mem_name, sequence=params
        )
        self._sh_mem_array = shared_memory.SharedMemory(name=self._array_shared_mem_name, create=True, size=self._image_size_bytes)
        self._shared_array = np.ndarray(
            (self._image_height, self._image_width, self._num_channels),
            dtype=self._image_dtype, buffer=self._sh_mem_array.buf)

    def __enter__(self):
        self.init_mem()

        return self

    def __exit__(self, type, value, traceback):
        self._logging.info("Stop processing")
        self.dispose()

    def dispose(self):
        if self._sh_mem_update is not None:
            self._logging.info("Free update shared memory")
            self._sh_mem_update.buf[BufferStates.SERVER_ALIVE] = States.NULL_STATE.value[0]
            self._sh_mem_update.close()
            self._sh_mem_update.unlink()

        del self._shared_array

        if self._sh_mem_array is not None:
            self._logging.info("Free array shared memory")
            self._sh_mem_array.close()
            self._sh_mem_array.unlink()

        if self._sh_mem_params is not None:
            self._logging.info("Free params shared memory")
            self._sh_mem_params.shm.close()
            self._sh_mem_params.shm.unlink()

    def start_processing(self):
        self._sh_mem_update.buf[BufferStates.SERVER] = States.NULL_STATE.value[0]
        self._sh_mem_update.buf[BufferStates.CLIENT] = States.NULL_STATE.value[0]
        self._sh_mem_update.buf[BufferStates.SERVER_ALIVE] = States.IS_SERVER_ALIVE.value[0]

        self._logging.info("Awaiting message")

        while True:
            while self._sh_mem_update.buf[BufferStates.SERVER] != States.READY_SERVER_MESSAGE.value[0]:
                time.sleep(1e-3)

            self._sh_mem_update.buf[BufferStates.SERVER] = States.NULL_STATE.value[0]

            self.process(self._shared_array, self._sh_mem_params)
            self._sh_mem_update.buf[BufferStates.CLIENT] = States.READY_CLIENT_MESSAGE.value[0]

    def process(self, image: np.ndarray, params: list):
        pass


class DetectorSever(ProcessingServer):
    def __init__(self,
                 path_to_model: str,
                 update_shared_mem_name: str,
                 params_shared_mem_name: str,
                 array_shared_mem_name: str,
                 image_with: int,
                 image_height: int,
                 num_channels: int,
                image_dtype):
        super().__init__(update_shared_mem_name, params_shared_mem_name, array_shared_mem_name, image_with, image_height, num_channels, image_dtype)
        self._detector = PyTorchYoloDetector(path_to_model,
                                             device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
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

def main(args):
    with DetectorSever(args.checkpoint_path,
                  args.shared_update_mem_name,
                  args.shared_params_mem_name,
                  args.shared_array_mem_name,
                  args.image_width,
                  args.image_height,
                  args.num_channels,
                  args.image_type) as det:
        det.start_processing()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-iw", "--image_width", type=int, default=640)
    parser.add_argument("-ih", "--image_height", type=int, default=640)
    parser.add_argument("-c", "--num_channels", type=int, default=3)
    parser.add_argument("--image_type", type=str, default="float32")
    parser.add_argument("--shared_array_mem_name", type=str, default="array")
    parser.add_argument("--shared_update_mem_name", type=str, default="update_info")
    parser.add_argument("--shared_params_mem_name", type=str, default="params")
    parser.add_argument("-p", "--checkpoint_path", type=str, required=True)
    parser.add_argument("--log_config", type=str, default="log_settings.yaml")

    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint_path):
        raise FileNotFoundError(f"Cannot find: '{args.checkpoint_path}'")

    init_logging(log_config=args.log_config)
    main(args)

