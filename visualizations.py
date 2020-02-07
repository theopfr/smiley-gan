
import matplotlib.pyplot as plt
import numpy as np

""" loads preprocessed dataset """
def load_dataset(dataset_path):
    return np.load(dataset_path, allow_pickle=True)

""" shows image matrices of certain class """
def show(dataset, amount=10):
    for sample in dataset[:amount]:
        plt.matshow(sample)
        plt.show()


if __name__ == "__main__":
    """ loading dataset from .npy file """
    dataset = load_dataset("dataset/dataset.npy")

    """ concatenating train, test, validation batches """
    dataset = np.concatenate((dataset[0], dataset[1], dataset[0]))

    """ show samples """
    show(dataset, amount=10)