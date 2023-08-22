import cv2
import numpy as np
import os

from yolo_models.detection import ONNXYoloV8Detector
import onnxruntime as ort

DETECTOR = None

os.environ["CUDA_PATH"] = ""


def get_model(path_to_model):
    sess_opt = ort.SessionOptions()
    sess_opt.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    return ONNXYoloV8Detector(path_to_model, providers=["CUDAExecutionProvider"], session_options=sess_opt)


def onSetupParameters(scriptOp):
    page = scriptOp.appendCustomPage("Detection")
    page.appendFile("Modelpath", label="Model path")
    p = page.appendFloat("Nms", label="Intersection ratio between bboxes")
    p.default = 0.5
    p.min = 0
    p.max = 1
    p = page.appendFloat("Score", label="Minimum score for object")
    p.default = 0.5
    p.min = 0
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


def onPulse(par):
    global DETECTOR
    if par.name == "Modelpath":
        DETECTOR = None
    return


def onCook(scriptOp):
    global DETECTOR

    if scriptOp.par.Modelpath and DETECTOR is None:
        debug("Load model from: ", scriptOp.par.Modelpath.eval())
        DETECTOR = get_model(scriptOp.par.Modelpath.eval())

    if DETECTOR is None:
        debug("Model is not loaded")
        return

    if not scriptOp.inputs:
        return

    video_in = scriptOp.inputs[0]

    frame = video_in.numpyArray(delayed=True, writable=False)

    if frame is None:
        return

    image = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGBA2RGB)

    detection = DETECTOR.predict(
        image,
        score_threshold=scriptOp.par.Score.eval(),
        nms_threshold=scriptOp.par.Nms.eval(),
        max_k=scriptOp.par.Maxk.eval(),
        eta=scriptOp.par.Eta.eval())

    for xyxy in detection.xyxy_boxes:
        image = cv2.rectangle(image, xyxy[:2], xyxy[2:], (255, 0, 0), 3)

    scriptOp.copyNumpyArray(image)

    return
