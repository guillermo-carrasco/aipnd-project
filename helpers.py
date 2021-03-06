""" Generic helpers
"""
import argparse
import os

import torch


def is_dir(dirname):
    """
    Checks if a directory exists and is accessible
    :param dirname: Path to check
    :return: True if directory exists and is accessible, false otherwise
    """
    return os.path.exists(dirname) and os.access(dirname, os.R_OK)


def bounded(x, min_value, max_value, val_type=float):
    x = val_type(x)
    if x < min_value or x > max_value:
        raise argparse.ArgumentTypeError("{} not in range [{}, {}]".format(str(x), str(min_value), str(max_value)))
    return x


def get_device(gpu=True):
    """ Returns the device to run the model training on
    :param gpu: If available, train on GPU
    """
    return torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")


def summarize_params(params):
    """
    Constructs a string with a summary of the CLI arguments
    :param params: CLI arguments
    :return: A summary of CLI arguments
    """
    return 'Running with following parameters: \n\n{}'.format(
              '\n'.join(['{}: {}'.format(k, v) for k, v in params.items()]))