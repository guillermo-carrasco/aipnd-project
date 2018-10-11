import argparse
import logging
import os

log = logging.getLogger()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model to classify flowers')
    parser.add_argument('data_dir', type=str, required=True, help='Directory with training and test data')
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        log.error('ERROR: Data directory does not exist or is not accessible')
