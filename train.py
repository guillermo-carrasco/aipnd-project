import argparse
import logging

import torch

from helpers import get_device, is_dir, bounded, summarize_params
from model import build_model, save_model, check_accuracy
from process import get_datasets_and_loaders

logging.getLogger().setLevel(logging.INFO)


def train(model, criterion, optimizer, data_dir, epochs=15, print_every=40, gpu=True, save=True, save_dir='.'):
    """
    Trains a deep CNN to classify images of flowers

    :param model: Model to train
    :param criterion: Loss funtion
    :param optimizer: Loss function optimizer
    :param data_dir: Data directory for training, test and validation sets
    :param epochs: Number of epochs to run the training
    :param print_every: Print progress every print_every steps
    :param gpu: Train on a GPU (if available)
    :param save: Save the trained model on disk or not
    :param save_dir: Directory where to save the checkpoint file
    :return:
    """
    device = get_device(gpu)
    image_datasets, dataloaders = get_datasets_and_loaders(data_dir)

    steps = 0
    model.to(device)

    # Save model hyperparameters
    model.epochs = epochs

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
        save_model(model, image_datasets['train'].class_to_idx, dest=save_dir)

    # Check accuracy after training
    accuracy = check_accuracy(model, dataloaders['test'])
    logging.info('Accuracy of the network on the test images: %d %%' % accuracy)

    return model


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a model to classify flowers')
    parser.add_argument('data_dir', type=str, help='Directory with training and test data')
    parser.add_argument('--save-dir', type=str, default='.', help='Directory where to save the model checkpoint')
    parser.add_argument('--arch', default='densenet121', choices=['densenet121', 'vgg16', 'densenet161'],
                        help='CNN Architecture to use for feature detection')
    parser.add_argument('--learning-rate', default=0.001, type=float, help='Learning rate to use while training')
    parser.add_argument('--hidden-units', default=1024, type=int, help='NUmber of hidden units on the first FC layer')
    parser.add_argument('--dropout', default=0.5, type=float, help='Probability of dropout')
    parser.add_argument('--epochs', default=15, type=int, help='Number of epochs to run the training')
    parser.add_argument('--gpu', action='store_true', help='Train the model on a GPU (if available)')
    args = parser.parse_args()

    # *Minimal* error validation
    if not is_dir(args.data_dir):
        raise argparse.ArgumentTypeError('Data directory does not exist or is not accessible')
    if not is_dir(args.save_dir):
        raise argparse.ArgumentTypeError('Save directory does not exist or is not accessible')
    bounded(args.dropout, 0, 1, float)

    # Summarize parameters
    params = {
        'Data directory': args.data_dir,
        'Save directory': args.save_dir,
        'Architecture': args.arch,
        'Learning rate': args.learning_rate,
        'Hidden units': args.hidden_units,
        'Dropout probability': args.dropout,
        'Epochs': args.epochs,
        'GPU': 'true' if args.gpu else 'false'
    }
    logging.info(summarize_params(params))

    model, criterion, optimizer = build_model(architecture=args.arch,
                                              hidden_units=args.hidden_units,
                                              dropout=args.dropout,
                                              lr=args.learning_rate)
    train(model, criterion, optimizer, args.data_dir)
