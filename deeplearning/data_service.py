"""
Data Service module with the configuration of data
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Constant for classes
CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot",
)


class DataManager:
    """
    DataManager class to handle the loading and preprocessing of datasets.
    """

    def __init__(self):
        """
        Initialize the DataManager class with batch size and data root path.
        """
        self.batch_size = 64
        self.data_root_path = "data"

    def get_fashion_mnist_training_data(self):
        """
        Get the training data for Fashion MNIST dataset.

        Returns:
            train_dataloader (DataLoader): A DataLoader instance for the Fashion MNIST training dataset.
        """
        training_data = datasets.FashionMNIST(
            root=self.data_root_path,
            train=True,
            download=True,
            transform=ToTensor(),
        )
        train_dataloader = DataLoader(training_data, batch_size=self.batch_size)
        for X, y in train_dataloader:
            print("LOADED FASHION MNIST TRAINING DATASET")
            print(f" | Shape of X [Num, Layers, Height, Width]: {X.shape}")
            print(f" | Shape of y: {y.shape} {y.dtype}")
            break
        return train_dataloader

    def get_fashion_mnist_test_data(self):
        """
        Get the test data for Fashion MNIST dataset.

        Returns:
            test_dataloader (DataLoader): A DataLoader instance for the Fashion MNIST test dataset.
        """
        test_data = datasets.FashionMNIST(
            root=self.data_root_path,
            train=False,
            download=True,
            transform=ToTensor(),
        )
        test_dataloader = DataLoader(test_data, batch_size=self.batch_size)
        for X, y in test_dataloader:
            print("LOADED FASHION MNIST TEST DATASET")
            print(f" | Shape of X [Num, Layers, Height, Width]: {X.shape}")
            print(f" | Shape of y: {y.shape} {y.dtype}")
            break
        return test_dataloader

    @staticmethod
    def select_n_random(data, labels, n=100):
        """
        Selects n random datapoints and their corresponding labels from a dataset
        """
        assert len(data) == len(labels)

        perm = torch.randperm(len(data))
        return data[perm][:n], labels[perm][:n]
