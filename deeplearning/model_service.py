from pprint import pprint

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")


class ModelManager:

    def __init__(
        self, training_data: DataLoader, test_data: DataLoader, nn_config: dict
    ):
        self.training_data: DataLoader = training_data
        self.test_data: DataLoader = test_data
        self.nn_config: dict = nn_config
        self.my_model = MyModel(self.nn_config).to(device=device)
        print(self.my_model)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.my_model.parameters(), lr=0.01)

    def train(self):
        size = len(self.training_data.dataset)
        self.my_model.train()

        for batch, (X, y) in enumerate(self.training_data):
            print("TARGET SHAPE:", y.shape)
            break

        for batch, (X, y) in enumerate(self.training_data):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = self.my_model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self):
        size = len(self.test_data.dataset)
        num_batches = len(self.test_data)
        self.my_model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in self.test_data:
                X, y = X.to(device), y.to(device)
                pred = self.my_model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )

    def train_model(self, epochs=5):
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train()
            self.test()
        print("Done!")


class MyModel(nn.Module):
    def __init__(self, nn_config: dict):
        self.nn_config: dict = nn_config
        pprint(self.nn_config)
        super().__init__()
        self.layers = nn.ModuleList()
        for layer_config in self.nn_config:
            print(layer_config)
            layer = self.get_layer(layer_config)
            self.layers.append(layer)
        self.print_bool = True

    def get_layer(self, layer_config: dict):
        layer_type = layer_config["type"]

        if layer_type == "linear":
            return nn.Linear(layer_config["in_size"], layer_config["out_size"])

        if layer_type == "flatten":
            return nn.Flatten()

        if layer_type == "relu":
            return nn.ReLU()

        if layer_type == "conv2d":
            return nn.Conv2d(
                layer_config["in_size"],
                layer_config["out_size"],
                kernel_size=layer_config.get("kernel_size", 3),
                stride=layer_config.get("stride", 2),
                padding=layer_config.get("padding", 1),
            )

        if layer_type == "maxpool2d":
            return nn.MaxPool2d(kernel_size=layer_config.get("kernel_size", 3))

        if layer_type == "dropout":
            return nn.Dropout(p=layer_config.get("proba", 0.5))

        if layer_type == "log_softmax":
            return nn.LogSoftmax(dim=layer_config.get("dim", 1))

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if self.print_bool:
                print(f"Layer {idx + 1} output shape: {x.shape}")
        self.print_bool = False
        return x
