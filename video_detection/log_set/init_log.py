import os
from typing import Optional
import logging.config

import yaml


def init_logging(log_config: Optional[str] = None, log_env_var: str = "LOG_CONFIG"):
    path_to_config = log_config if log_config is not None else os.environ[log_env_var]

    with open(path_to_config, "rb") as file:
        log_config = yaml.safe_load(file)

    logging.config.dictConfig(log_config)
