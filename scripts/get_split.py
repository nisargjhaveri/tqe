import argparse

import sys
import numpy as np
from sklearn.model_selection import ShuffleSplit


def loadFile(inFile):
    return np.array(
                map(lambda l: l.decode('utf-8'), list(inFile)),
                dtype=object
            )


def printFile(lines):
    lines = map(lambda l: l.encode('utf-8'), lines)
    sys.stdout.writelines(lines)


def split(inFile, split):
    lines = loadFile(inFile)

    splitter = ShuffleSplit(n_splits=1, test_size=.2, random_state=42)
    train_index, dev_index = splitter.split(lines).next()

    dev_len = len(dev_index) / 2

    if split == "train":
        printFile(lines[train_index])
    elif split == "dev":
        printFile(lines[dev_index[:dev_len]])
    elif split == "test":
        printFile(lines[dev_index[dev_len:]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Train Translation Quality Estimation')

    parser.add_argument('split', type=str, choices=['train', 'dev', 'test'],
                        help='Which split to generate')

    parser.add_argument('input_file', type=file,
                        help='File to split')

    args = parser.parse_args()

    split(args.input_file, args.split)
