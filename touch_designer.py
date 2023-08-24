from multiprocessing import shared_memory
import enum

import numpy as np
import cv2

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

SHARED_MEM_PARAMS_LIST = shared_memory.ShareableList(name="params")

WIDTH = SHARED_MEM_PARAMS_LIST[ParamsIndex.IMAGE_WIDTH]
HEIGHT = SHARED_MEM_PARAMS_LIST[ParamsIndex.IMAGE_HEIGHT]
DTYPE = SHARED_MEM_PARAMS_LIST[ParamsIndex.IMAGE_DTYPE]
NUM_CHANNELS = SHARED_MEM_PARAMS_LIST[ParamsIndex.IMAGE_CHANNELS]
EXIT = False

COLOR_CONVERSION = cv2.COLOR_RGBA2RGB if NUM_CHANNELS == 3 else None

SHARED_MEM_UPDATE_STATES = shared_memory.SharedMemory(name=SHARED_MEM_PARAMS_LIST[ParamsIndex.SHARD_STATE_MEM_NAME], create=False)
SHARED_MEM_ARRAY = shared_memory.SharedMemory(name=SHARED_MEM_PARAMS_LIST[ParamsIndex.SHARED_ARRAY_MEM_NAME], create=False)

ARRAY = np.ndarray((WIDTH, HEIGHT, NUM_CHANNELS), dtype=DTYPE, buffer=SHARED_MEM_ARRAY.buf)


def onSetupParameters(scriptOp):
    page = scriptOp.appendCustomPage("Detection")
    p = page.appendFloat("Nms", label="Intersection ratio between bboxes")
    p.default = 0.5
    p.min = 0
    p.max = 1
    p = page.appendFloat("Score", label="Minimum score for object")
    p.default = 0.5
    p.min = 0.1
    p.max = 1
    p = page.appendInt("Maxk", label="Maximum number of objects to detect based on score")
    p.default = 5
    p.min = 0
    p.max = 1000
    p = page.appendFloat("Eta", label="Filtering")
    p.default = 1.0
    p.min = 0
    p.clampMin = True
    p.max = 1000
    return


def onDestroy():
    global EXIT
    global SHARED_MEM_UPDATE_STATES
    global SHARED_MEM_ARRAY
    global SHARED_MEM_PARAMS_LIST
    EXIT = True
    SHARED_MEM_UPDATE_STATES.close()
    SHARED_MEM_ARRAY.close()
    SHARED_MEM_PARAMS_LIST.shm.close()
    debug("Free all shared memory")
    return


def onCook(scriptOp):
    global SHARED_MEM_UPDATE_STATES
    global SHARED_MEM_PARAMS_LIST
    global SHARED_MEM_ARRAY
    global EXIT
    global COLOR_CONVERSION

    if not scriptOp.inputs:
        return

    video_in = scriptOp.inputs[0]
    # By default, the image is flipped up. We flip it early
    frame = video_in.numpyArray(delayed=True, writable=False)

    if frame is None:
        return

    if COLOR_CONVERSION is not None:
        image = cv2.cvtColor(frame, COLOR_CONVERSION)

    image = cv2.resize(image, (WIDTH, HEIGHT))

    SHARED_MEM_PARAMS_LIST[ParamsIndex.SCORE_THRESH] = scriptOp.par.Score.eval()
    SHARED_MEM_PARAMS_LIST[ParamsIndex.IOU_THRESH] = scriptOp.par.Nms.eval()
    SHARED_MEM_PARAMS_LIST[ParamsIndex.TOP_K] = scriptOp.par.Maxk.eval()
    SHARED_MEM_PARAMS_LIST[ParamsIndex.ETA] = scriptOp.par.Eta.eval()
    np.copyto(ARRAY, image)

    SHARED_MEM_UPDATE_STATES.buf[BufferStates.SERVER] = States.READY_SERVER_MESSAGE.value[0]

    while not EXIT and SHARED_MEM_UPDATE_STATES.buf[BufferStates.SERVER_ALIVE] == States.IS_SERVER_ALIVE.value[0] and SHARED_MEM_UPDATE_STATES.buf[BufferStates.CLIENT] != States.READY_CLIENT_MESSAGE.value[0]:
        pass

    if SHARED_MEM_UPDATE_STATES.buf[BufferStates.SERVER_ALIVE] != States.IS_SERVER_ALIVE.value[0]:
        raise ValueError("Server process died")

    scriptOp.copyNumpyArray(ARRAY)
    SHARED_MEM_UPDATE_STATES.buf[BufferStates.CLIENT] = States.NULL_STATE.value[0]

    return
