""" Common data processing and manipulation tasks
"""
import os

import numpy as np
import torch

from PIL import Image
from torchvision import datasets, transforms


def get_datasets_and_loaders(data_dir):
    """ Returns a dictionary containing trining, test and validation datasets and data loaders
    """
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    data_transforms = {
        'train': transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])]),
        'test_validation': transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test_validation']),
        'validation': datasets.ImageFolder(valid_dir, transform=data_transforms['test_validation'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32, shuffle=True),
        'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size=32, shuffle=True)
    }

    return image_datasets, dataloaders


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_pil = Image.open(image)

    # Resize
    size = 256
    width, height = img_pil.size
    shortest_side = min(width, height)
    img_pil = img_pil.resize((int((width / shortest_side) * size), int((height / shortest_side) * size)))

    # Center crop to 224 x 224
    left = (256 - 224) / 2
    top = (256 - 224) / 2
    right = (256 + 224) / 2
    bottom = (256 + 224) / 2
    img_pil = img_pil.crop((left, top, right, bottom))

    # Convert image to a numpy array
    img = (np.array(img_pil)) / 255

    # Normalize the pixel values
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    img = img.transpose((2, 0, 1))

    return img
