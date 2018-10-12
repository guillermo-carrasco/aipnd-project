""" Generic helpers
"""
import torch


def get_device():
    """ Returns the device to run the model training on
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")