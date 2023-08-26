from multiprocessing import shared_memory
import logging

import numpy as np

from .info import ParamsIndex, BufferStates, States

class BaseProcessServer:
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
