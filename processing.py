import argparse
import os
import logging

from yolo_models.processing.detection_processing import DetectorProcess
from yolo_models.log_set import init_logging


def main(args):
    logger = logging.getLogger()

    with DetectorProcess(args.checkpoint_path,
                         args.shared_update_mem_name,
                         args.shared_params_mem_name,
                         args.shared_array_mem_name,
                         args.image_width,
                         args.image_height,
                         args.num_channels,
                         args.image_type) as det:

        if os.path.exists(args.class_colormap):
            logger.info("Load class colors from '%s'", args.class_colormap)
            det.load_class_colormap(args.class_colormap)
        else:
            logger.info("Cannot find a file with class colors. Create it with defaults colors.")
            det.save_class_colormap(args.class_colormap)

        det.start_processing()


def check_file(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cannot find: '{path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-iw", "--image_width", type=int, default=1280,
                        help="A maximum width of image to transfer between processes")
    parser.add_argument("-ih", "--image_height", type=int, default=1280,
                        help="A maximum height of image to transfer between processes")
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
    parser.add_argument("--log_config", type=str, default=None,
                        help="A path to settings for logging")
    parser.add_argument("--class_colormap", type=str, default="class_colors.json",
                        help="A path to json with colors for each class")

    args = parser.parse_args()
    check_file(args.checkpoint_path)
    init_logging(log_config=args.log_config)
    main(args)
