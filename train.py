import argparse
import logging
import os

import torch

from helpers import get_device
from model import build_model, save_model
from process import get_datasets_and_loaders

log = logging.getLogger()


def train(epochs=15, print_every=40, save=True):
    """
    Trains a deep CNN to classify images of flowers

    :param epochs: Number of epochs to run the training
    :param print_every: Print progress every print_every steps
    :param save: Save the trained model on disk or not
    :return:
    """
    device = get_device()
    model, criterion, optimizer = build_model()
    image_datasets, dataloaders = get_datasets_and_loaders('flowers')

    steps = 0
    model.to(device)

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(dataloaders['train']):
            steps += 1
            model.train()

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                val_loss = 0
                accuracy = 0

                for ii, (inputs_val, labels_val) in enumerate(dataloaders['validation']):
                    optimizer.zero_grad()

                    inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
                    model.to(device)
                    with torch.no_grad():
                        outputs = model.forward(inputs_val)
                        val_loss = criterion(outputs, labels_val)
                        ps = torch.exp(outputs).data
                        equality = (labels_val.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

                val_loss = val_loss / len(dataloaders['validation'])
                accuracy = accuracy / len(dataloaders['validation'])

                print("Epoch: {}/{}... ".format(e + 1, epochs),
                      "Loss: {:.4f}".format(running_loss / print_every),
                      "Validation Loss {:.4f}".format(val_loss),
                      "Accuracy: {:.4f}".format(accuracy))

                running_loss = 0
    if save:
        save_model(model, image_datasets['train'].class_to_idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model to classify flowers')
    parser.add_argument('data_dir', type=str, required=True, help='Directory with training and test data')
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        log.error('ERROR: Data directory does not exist or is not accessible')

    train()
