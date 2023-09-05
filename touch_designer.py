import time
from multiprocessing import shared_memory

import cv2
import numpy as np
from yolo_models.processing.info import (BufferStates, DrawInfo, ParamsIndex,
                                         States)

# Sync with external process
SHARED_MEM_PARAMS_LIST = shared_memory.ShareableList(name="params")

MAX_IMAGE_WIDTH = SHARED_MEM_PARAMS_LIST[ParamsIndex.IMAGE_WIDTH]
MAX_IMAGE_HEIGHT = SHARED_MEM_PARAMS_LIST[ParamsIndex.IMAGE_HEIGHT]
DTYPE = SHARED_MEM_PARAMS_LIST[ParamsIndex.IMAGE_DTYPE]
NUM_CHANNELS = SHARED_MEM_PARAMS_LIST[ParamsIndex.IMAGE_CHANNELS]
EXIT = False

COLOR_CONVERSION = cv2.COLOR_RGBA2RGB if NUM_CHANNELS == 3 else None

SHARED_MEM_UPDATE_STATES = shared_memory.SharedMemory(
    name=SHARED_MEM_PARAMS_LIST[ParamsIndex.SHARD_STATE_MEM_NAME], create=False)
SHARED_MEM_ARRAY = shared_memory.SharedMemory(
    name=SHARED_MEM_PARAMS_LIST[ParamsIndex.SHARED_ARRAY_MEM_NAME], create=False)

ARRAY = np.ndarray((MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT, NUM_CHANNELS),
                   dtype=DTYPE, buffer=SHARED_MEM_ARRAY.buf)


def onSetupParameters(scriptOp):
    page = scriptOp.appendCustomPage("Detection")
    p = page.appendFloat("Nms", label="Intersection ratio between bboxes")
    p.min = 0
    p.max = 1
    p.default = 0.5
    p = page.appendFloat("Score", label="Minimum score for object")
    p.min = 0.1
    p.max = 1
    p.default = 0.5
    p = page.appendInt("Maxk", label="Maximum number of objects to detect based on score")
    p.min = 0
    p.max = 1000
    p.default = 5
    p = page.appendFloat("Eta", label="Filtering")
    p.min = 0
    p.clampMin = True
    p.max = 1000
    p.default = 1.0
    p = page.appendToggle("Drawtext", label="Draw text labels")
    p.default = False
    p = page.appendToggle("Drawscore", label="Draw score labels")
    p.default = False
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

    if image.shape[0] > MAX_IMAGE_HEIGHT:
        debug("Too large image height")
        scriptOp.clear()
        return

    if image.shape[1] > MAX_IMAGE_WIDTH:
        debug("Too large image width")
        scriptOp.clear()
        return

    SHARED_MEM_PARAMS_LIST[ParamsIndex.IMAGE_HEIGHT] = image.shape[0]
    SHARED_MEM_PARAMS_LIST[ParamsIndex.IMAGE_WIDTH] = image.shape[1]
    SHARED_MEM_PARAMS_LIST[ParamsIndex.SCORE_THRESH] = scriptOp.par.Score.eval()
    SHARED_MEM_PARAMS_LIST[ParamsIndex.IOU_THRESH] = scriptOp.par.Nms.eval()
    SHARED_MEM_PARAMS_LIST[ParamsIndex.TOP_K] = scriptOp.par.Maxk.eval()
    SHARED_MEM_PARAMS_LIST[ParamsIndex.ETA] = scriptOp.par.Eta.eval()
    draw_info = DrawInfo.DRAW_BBOX

    if scriptOp.par.Drawtext.eval():
        draw_info |= DrawInfo.DRAW_TEXT

    if scriptOp.par.Drawscore.eval():
        draw_info |= DrawInfo.DRAW_CONF

    SHARED_MEM_PARAMS_LIST[ParamsIndex.DRAW_INFO] = int(draw_info)

    ARRAY[:image.shape[0], :image.shape[1]] = image
    SHARED_MEM_UPDATE_STATES.buf[BufferStates.SERVER] = States.READY_SERVER_MESSAGE.value[0]

    start_time = time.monotonic()

    while not EXIT and SHARED_MEM_UPDATE_STATES.buf[BufferStates.SERVER_ALIVE] == States.IS_SERVER_ALIVE.value[0] and SHARED_MEM_UPDATE_STATES.buf[BufferStates.CLIENT] != States.READY_CLIENT_MESSAGE.value[0]:
        time.sleep(1e-3)
        elapsed = time.monotonic() - start_time

        if elapsed > 1:
            debug("Too long processing copy frame as is")
            scriptOp.clear()
            break

    if SHARED_MEM_UPDATE_STATES.buf[BufferStates.SERVER_ALIVE] != States.IS_SERVER_ALIVE.value[0]:
        raise ValueError("Server process died")

    scriptOp.copyNumpyArray(ARRAY[:image.shape[0], :image.shape[1]])
    SHARED_MEM_UPDATE_STATES.buf[BufferStates.CLIENT] = States.NULL_STATE.value[0]
    return
