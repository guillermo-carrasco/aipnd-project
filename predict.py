import argparse
import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from helpers import get_device
from model import load_model
from process import process_image

logging.getLogger().setLevel(logging.INFO)


def predict(image_path, model, topk=1, gpu=True, cat_to_name=None):
    """ Predict the class (or classes) of an image using a trained deep learning model.
    """
    model.to(get_device(gpu))
    img = process_image(image_path)
    img_torch = torch.from_numpy(img)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    with torch.no_grad():
        output = model.forward(img_torch.cuda())

    probabilities = F.softmax(output.data, dim=1).topk(topk)
    index_to_class = {val: key for key, val in model.class_to_idx.items()}

    probs = np.array(probabilities[0][0])
    classes = [cat_to_name[index_to_class[i]] for i in np.array(probabilities[1][0])] if cat_to_name is not None \
        else [index_to_class[i] for i in np.array(probabilities[1][0])]

    # Log the prediction results for the user to see them
    logging.info('\n'.join(['Class: {} with probability {}'.format(c, p) for c, p in zip(classes, probs)]))

    return probs, classes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict the flass of the flower in an image')
    parser.add_argument('img_path', type=str, help='Path to the flower image')
    parser.add_argument('checkpoint', help='Model checkpoint to load')
    parser.add_argument('--category-names', help='Dictionary containing real category names')
    parser.add_argument('--gpu', action='store_true', help='Train the model on a GPU (if available)')
    args = parser.parse_args()

    # Minimal error checking
    if not os.path.exists(args.img_path):
        raise argparse.ArgumentTypeError('Input image does not exist or is not readable')
    if not os.path.exists(args.checkpoint):
        raise argparse.ArgumentTypeError('Path to model checkpoint is does not exist or is not accessible')
    if args.category_names and not os.path.exists(args.category_names):
        raise argparse.ArgumentTypeError('Provided path to category names file does not exist or is not accessible')

    cat_to_name = None
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)

    model = load_model(args.checkpoint)
    probs, classes = predict(model, args.img_path, topk=args.topk, gpu=args.gpu, cat_to_name=cat_to_name)