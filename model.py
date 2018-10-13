""" Operate with PyTorch models
"""
import logging
import os

from collections import OrderedDict

import torch
from torch import nn, optim
from torchvision import models

from helpers import get_device


def build_model(architecture, hidden_units=1024, dropout=0.5, lr=0.001):
    """
    Builds a deep CNN to classify images of flowers
    :param architecture: Architecture to use for the CNN
    :param hidden_units: Number of hidden units in the first FC layer
    :param dropout: Probability of dropout
    :param lr: Learning rate
    :return: mode, criterion, optimizer
    """
    arch = getattr(models, architecture)
    model = arch(pretrained=True)

    logging.info('Building model with architecture: {}'.format(architecture))

    # Save the model architecture for future loading
    model.architecture = architecture

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('dropout', nn.Dropout(dropout)),
        ('fc1', nn.Linear(hidden_units, 120)),
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


def check_accuracy(model, test_data):
    """
    Checks accuracy of a model
    """
    correct = 0
    total = 0
    device = get_device()
    with torch.no_grad():
        model.eval()
        model.to(get_device())
        for data in test_data:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100*correct / total


def save_model(model, class_to_idx, dest='.'):
    """
    Saves a model in disk
    :param model: PyTorch model
    :param class_to_idx: Map of indexes to classes
    :param dest: where to save the model
    """
    dest = os.path.join(dest, 'checkpoint.pth')
    model.class_to_idx = class_to_idx
    torch.save({'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx,
                'architecture': model.architecture},
                dest)


def load_model(path):
    """
    Loads a model from disc
    :param path: Path to the model
    :return:
    """
    checkpoint = torch.load(path)
    m, c, o = build_model(checkpoint['architecture'])
    m.class_to_idx = checkpoint['class_to_idx']
    m.load_state_dict(checkpoint['state_dict'])

    return m, c, o
