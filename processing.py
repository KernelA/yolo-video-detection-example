import argparse
import os

from yolo_models.processing.detection_processing import DetectorProcess
from yolo_models.log_set import init_logging


def main(args):
    with DetectorProcess(args.checkpoint_path,
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

    parser.add_argument("-iw", "--image_width", type=int, default=640,
                        help="Width of image to transfer between processes")
    parser.add_argument("-ih", "--image_height", type=int, default=640,
                        help="Height of image to transfer between processes")
    parser.add_argument("-c", "--num_channels", type=int, default=3,
                        help="A number of channels in image. 3 for RGB")
    parser.add_argument("--image_type", type=str,
                        choices=["uint8", "float32"], default="float32", help="Image dtype")
    parser.add_argument("--shared_array_mem_name", type=str, default="array",
                        help="Name of shared memory for array")
    parser.add_argument("--shared_update_mem_name", type=str, default="update_info",
                        help="Name of shared memory for transfering information about updates")
    parser.add_argument("--shared_params_mem_name", type=str, default="params",
                        help="Name of shared memory for transfering of parameters")
    parser.add_argument("-p", "--checkpoint_path", type=str,
                        required=True, help="A path to model .pt")
    parser.add_argument("--log_config", type=str, default="log_settings.yaml",
                        help="A path to settings for logging")

    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint_path):
        raise FileNotFoundError(f"Cannot find: '{args.checkpoint_path}'")

    init_logging(log_config=args.log_config)
    main(args)
