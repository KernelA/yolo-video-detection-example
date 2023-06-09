import argparse
import os
import pathlib

import av
import torch
from tqdm.auto import tqdm
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results

from video_detection.log_set import init_logging


def main(args):
    device = torch.device(args.device)

    if not os.path.isfile(args.input_video):
        raise FileNotFoundError(args.input_video)

    model_path = pathlib.Path(args.checkpoint_path)
    model = YOLO(str(model_path), args.task_type)
    model.info(verbose=False)
    model.to(device)

    in_video_path = args.input_video
    out_video_path = pathlib.Path(args.output_video)
    out_video_path.parent.mkdir(parents=True, exist_ok=True)

    with av.open(str(in_video_path), mode="r") as in_container:
        input_video_stream = in_container.streams.video[0]
        in_frames = input_video_stream.frames

        if in_frames == 0:
            for frame in in_container.decode(input_video_stream):
                in_frames += 1
                del frame

    with av.open(str(in_video_path), "r") as container:
        input_video_stream = container.streams.video[0]
        input_video_stream.thread_type = "AUTO"

        with av.open(str(out_video_path), mode="w") as out_container:
            out_video_stream = out_container.add_stream("hevc", rate=input_video_stream.average_rate, options={
                                                        "preset": "slow"})
            out_video_stream.width = input_video_stream.width
            out_video_stream.height = input_video_stream.height
            out_video_stream.pix_fmt = "yuv420p"

        for frame in tqdm(container.decode(input_video_stream), total=in_frames, mininterval=5):
            img = frame.to_rgb().to_ndarray()
            predictions: Results = model.predict(img, verbose=False)

            for pred in predictions:
                img = pred.plot(img=img)

            out_frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in out_video_stream.encode(out_frame):
                out_container.mux(packet)

        for packet in out_video_stream.encode():
            out_container.mux(packet)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("-c", "--checkpoint_path", type=str,
                        required=True, help="A path to YOLO model")
    parser.add_argument("-i", "--input_video", type=str,
                        required=True, help="A path to input video")
    parser.add_argument("-o", "--output_video", type=str,
                        required=True, help="A path to input video")
    parser.add_argument("--task_type", type=str,
                        choices=["classify", "detect", "segment", "pose"], required=True)

    args = parser.parse_args()
    main(args)
