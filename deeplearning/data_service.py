import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


class DataManager:

    def __init__(self):
        self.batch_size = 64
        self.data_root_path = "data"

    def get_fashion_mnist_training_data(self):
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
