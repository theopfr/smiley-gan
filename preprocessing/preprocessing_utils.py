
import numpy as np
import random
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt


""" loads raw dataset from numpy files """
def load_raw_dataset(path, amount: int=10000):
    if amount == "all":
        return np.load(path)
    return np.load(path)[:amount]

""" normalizes the matrix to a 0-1 range """
def normalize(batch: list) -> list:
    for idx in tqdm(range(len(batch)), desc="normalizing"):
        sample = np.array(batch[idx]) / 255

        max_x = max(sample)
        min_x = min(sample)

        for pixel in range(len(sample)):
            sample[pixel] = 2 * ((sample[pixel] - min_x) / max_x - min_x) - 1

        sample = np.array(sample).reshape((28, 28))

        batch[idx] = sample

    return batch

""" splits dataset into train, test and validation batch """
def split(dataset, testing_size: float=0.1, validation_size: float=0.1) -> list:
    test_size = int(np.round(len(dataset)*testing_size))
    val_size = int(np.round(len(dataset)*validation_size))
    train_set, test_set, validation_set = dataset[(test_size+val_size):], dataset[:test_size], dataset[test_size:(test_size+val_size)]

    return train_set, test_set, validation_set

""" save dataset to numpy file """
def save_dataset(dataset_path: str, dataset):
    np.save(dataset_path, dataset)

""" load preprocessed dataset """
def load_dataset(dataset_path: str):
    return np.load(dataset_path, allow_pickle=True) 
