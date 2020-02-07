
import numpy as np
from preprocessing_utils import *
import argparse


class DoodleDataset:
    def __init__(self, raw_dataset_paths: list=[""], preprocessed_dataset_path: str=""):
        self.raw_dataset_paths = raw_dataset_paths
        self.preprocessed_dataset_path = preprocessed_dataset_path

    def preprocess(self, amount: int=10000):
        """ loading dataset subsets (ie.: cat, bus, umbrella, ...) from .npy files """
        print("[*] loading raw datasets and concatenating")
        dataset = []
        for path in self.raw_dataset_paths:
            subset = load_raw_dataset(path, amount=amount)
            dataset.extend(subset)

        """ normalizing samples to [-1, 1] """
        print("[*] normalizing samples")
        dataset = normalize(dataset)

        """ shuffeling dataset """
        print("[*] shuffeling dataset")
        np.random.shuffle(dataset)

        """ splitting dataset into training, testing and validation batch """
        print("[*] splitting into train, test, validation")
        train_batch, test_batch, validation_batch = split(dataset, testing_size=0.1, validation_size=0.001)
        dataset = np.array([train_batch, test_batch, validation_batch])

        print("[*] finished dataset preprocessing")

        return dataset

    """ saving the dataset to .npy file """
    def save(self, dataset):
        print("[*] saving preprocessed dataset")
        save_dataset(self.preprocessed_dataset_path, np.array(dataset))


if __name__ == "__main__":
    doodleDataset = DoodleDataset(raw_dataset_paths=["dataset/doodles/full_numpy_bitmap_face.npy"],
                                  preprocessed_dataset_path="dataset/dataset.npy")

    dataset = doodleDataset.preprocess(amount="all")
    doodleDataset.save(dataset)

    # other datasets: "dataset/doodles/full_numpy_bitmap_cat.npy", "dataset/doodles/full_numpy_bitmap_bear.npy"