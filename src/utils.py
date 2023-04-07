import functools
from omegaconf import OmegaConf
import logging
from typing import Optional, Dict
from colorama import Fore, Back, Style
import sys
import datetime
import numpy as np
from pathlib import Path
from os import path
import re
import pandas as pd
import pickle
import json


def conf(params_file: str, as_default: bool = False) -> callable:
    @functools.wraps(conf)
    def _decorator(f: callable) -> callable:
        @functools.wraps(f)
        def _wrapper(*args, **kwargs) -> None:
            cfg_params = OmegaConf.load(params_file)
            if as_default:
                cfg_params.update(kwargs)
                kwargs = cfg_params
            else:
                kwargs.update(cfg_params)
            return f(*args, **kwargs)

        return _wrapper

    return _decorator


class ColoredFormatter(logging.Formatter):
    """Colored log formatter."""

    def __init__(self, *args, colors: Optional[Dict[str, str]] = None, **kwargs) -> None:
        """Initialize the formatter with specified format strings."""

        super().__init__(*args, **kwargs)

        self.colors = colors if colors else {}

    def format(self, record) -> str:
        """Format the specified record as text."""

        record.color = self.colors.get(record.levelname, '')
        record.reset = Style.RESET_ALL

        return super().format(record)


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


dict_data_folder = {
    '2': {'data_file': 'Data/dataset.json', 'class_label': 'Data/classes_two.npy'},
    '3': {'data_file': 'Data/dataset.json', 'class_label': 'Data/classes.npy'}
}


@conf("params.yml", as_default=True)
def set_output_dir(**params):
    if not params.get("output_dir", 0):
        default_output_dir = params["model"] + "_" + str(params["dataset"]['max_length'])
        data_params = params["dataset"]
        if params["training"]["remove_layers"]:
            default_output_dir += "_removed_layers_"
            default_output_dir += "_".join(params["training"]["remove_layers"].split(","))
        if params["training"]["freeze_layers"]:
            default_output_dir += "_frozen_layers_"
            default_output_dir += "_".join(params["training"]["freeze_layers"].split(","))
        if params["training"]["freeze_embeddings"]:
            default_output_dir += "_frozen_embeddings"
        if params["training"]["train_att"]:
            default_output_dir += "_attn"
        default_output_dir += '_' + data_params['type_attention'] + '_' + str(data_params['variance'])
        if data_params['decay']:
            default_output_dir += '_' + data_params['method'] + '_' + str(data_params['window']) + '_' + str(
                data_params['alpha']) + '_' + str(
                data_params['p_value'])
        loc_path = Path().parent
        params["output_dir"] = loc_path / default_output_dir
    Path(params["output_dir"]).mkdir(parents=True, exist_ok=True)
    return params


def set_logger(level=logging.INFO):
    formatter = ColoredFormatter(
        '{color}[{levelname:.1s}] {message}{reset}',
        style='{', datefmt='%Y-%m-%d %H:%M:%S',
        colors={
            'DEBUG': Fore.CYAN,
            'INFO': Fore.GREEN,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT,
        }
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.handlers[:] = []
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
