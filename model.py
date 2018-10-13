""" Operate with PyTorch models
"""
import logging

from collections import OrderedDict

import torch
from torch import nn, optim
from torchvision import models

log = logging.getLogger()


def build_model(architecture, dropout=0.5, lr=0.001):
    """
    Builds a deep CNN to classify images of flowers
    :param architecture: Architecture to use for the CNN
    :param dropout: Probability of dropout
    :param lr: Learning rate
    :return: mode, criterion, optimizer
    """
    arch = getattr(models, architecture)
    model = arch(pretrained=True)

    log.info('Building model with architecture: {}'.format(architecture))

    # Save the model architecture for future loading
    model.architecture = architecture

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('dropout', nn.Dropout(dropout)),
        ('fc1', nn.Linear(1024, 120)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(120, 90)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(90, 80)),
        ('relu3', nn.ReLU()),
        ('fc4', nn.Linear(80, 102)),
        ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    return model, criterion, optimizer


def save_model(model, class_to_idx, dest='checkpoint.pth'):
    """
    Saves a model in disk
    :param model: PyTorch model
    :param class_to_idx: Map of indexes to classes
    """
    model.class_to_idx = class_to_idx
    torch.save({'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx},
                dest)