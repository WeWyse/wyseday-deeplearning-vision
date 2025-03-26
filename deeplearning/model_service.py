"""
Model Service module with the configuration of the NN model
"""

import random
from pprint import pprint
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from deeplearning import matplot_helper


class ModelManager:
    """
    ModelManager class to manage the training and testing of the model

    Args:
        training_data (DataLoader): DataLoader for training data
        test_data (DataLoader): DataLoader for test data
        nn_config (dict): configuration for the neural network
        device (str): device to run the model on (e.g. 'cuda' or 'cpu')
        tensorboard_writer (SummaryWriter, optional): SummaryWriter for TensorBoard logging. Defaults to None.
    """

    def __init__(
        self,
        training_data: DataLoader,
        test_data: DataLoader,
        nn_config: Dict[str, Any],
        device: str,
        tensorboard_writer: SummaryWriter = None,
    ):
        self.training_data: DataLoader = training_data
        self.test_data: DataLoader = test_data
        self.nn_config: Dict[str, Any] = nn_config
        self.device: str = device
        self.tensorboard_writer: SummaryWriter = tensorboard_writer
        self.graph_bool = True
        self.my_model = MyModel(self.nn_config).to(device=self.device)
        print(self.my_model)
        self.loss_fn = nn.CrossEntropyLoss()
        #self.optimizer = torch.optim.SGD(self.my_model.parameters(), lr=0.01)
        self.optimizer = torch.optim.Adam(self.my_model.parameters(), lr=0.001)

    def train(self, epoch: int):
        """
        Method to train the model
        """
        running_loss = 0.0
        size = len(self.training_data.dataset)
        self.my_model.train()

        for batch, (X, y) in enumerate(self.training_data):
            print("BATCH SIZE:", y.shape)
            break

        for batch, (X, y) in enumerate(self.training_data):
            X, y = X.to(self.device), y.to(self.device)

            if self.tensorboard_writer and self.graph_bool:
                try:
                    self.tensorboard_writer.add_graph(self.my_model, X)
                    self.tensorboard_writer.flush()
                    self.graph_bool = False
                except:
                    pass

            # Compute prediction error
            pred = self.my_model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss_value, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

            running_loss += loss.item()
            if batch % 100 == 99 and self.tensorboard_writer:
                # ...log the running loss
                self.tensorboard_writer.add_scalar(
                    "training loss",
                    running_loss / 1000,
                    epoch * len(self.training_data) + batch,
                )
                self.tensorboard_writer.flush()

                # ...log a Matplotlib Figure showing the model's predictions on a
                # random mini-batch
                # images, labels = DataManager.select_n_random(
                #     self.training_data.dataset.data, self.training_data.dataset.targets
                # )

                # preds, probs = self.output_to_prob(images=images)

                # self.tensorboard_writer.add_figure(
                #     "predictions vs. actuals",
                #     matplot_helper.plot_classes_preds(images, labels, preds, probs),
                #     global_step=epoch * len(self.training_data) + batch,
                # )
                # self.tensorboard_writer.flush()
                running_loss = 0.0

    def test(self, epoch: int):
        """
        Method to test the model
        """
        size = len(self.test_data.dataset)
        num_batches = len(self.test_data)
        self.my_model.eval()
        test_loss, correct = 0, 0
        random_idx_to_print = random.randint(0, len(self.test_data) - 1)
        with torch.no_grad():
            for idx, (X, y) in enumerate(self.test_data):
                X, y = X.to(self.device), y.to(self.device)
                pred = self.my_model(X)
                if idx == random_idx_to_print:
                    pred_for_plot, prob = self.output_to_prob(pred)
                    print("Test Batch:", idx)
                    self.tensorboard_writer.add_figure(
                        "predictions vs. actuals",
                        matplot_helper.plot_classes_preds(
                            X.cpu(), y.cpu(), pred_for_plot, prob
                        ),
                        global_step=epoch,
                    )
                    self.tensorboard_writer.flush()
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar(
                "Test Accuracy",
                100 * correct,
                epoch,
            )
            self.tensorboard_writer.flush()

    def train_model(self, epochs: int = 5):
        """
        Method to train the model for a given number of epochs

        Args:
            epochs (int, optional): Number of epochs. Defaults to 5.
        """
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            self.train(epoch)
            self.test(epoch)
        print("Done!")

    def output_to_prob(self, output: torch.Tensor) -> Tuple[np.ndarray, List[float]]:
        """
        Generates predictions and corresponding probabilities from a trained
        network and a list of images
        Args:
            output (torch.Tensor): output tensor from the model
        Returns:
            Tuple[np.ndarray, List[float]]: predictions and probabilities
        """
        # convert output probabilities to predicted class
        _, preds_tensor = torch.max(output, 1)
        preds = np.squeeze(preds_tensor.cpu().numpy())
        return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


class MyModel(nn.Module):
    """
    MyModel class to create a custom neural network architecture
    Args:
        nn_config (dict): configuration for the neural network
    """

    def __init__(self, nn_config: Dict[str, Any]):
        self.nn_config: Dict[str, Any] = nn_config
        pprint(self.nn_config)
        super().__init__()
        self.layers = nn.ModuleList()
        self.layer_names = []
        count = 0
        for layer_config in self.nn_config:
            print(layer_config)
            layer_name = str(count)+" : "+layer_config.get("name", layer_config["type"])  # Use name if provided
            self.layer_names.append(layer_name) 
            layer = self.get_layer(layer_config)
            self.layers.append(layer)
            count += 1
        self.print_bool = True

    def get_layer(self, layer_config: Dict[str, Any]) -> nn.Module:
        """
        Method to create a layer based on the layer configuration

        Args:
            layer_config (dict): configuration for the layer
        Returns:
            nn.Module: layer
        """
        layer_type = layer_config["type"]

        if layer_type == "linear":
            return nn.Linear(layer_config["in_size"], layer_config["out_size"])

        if layer_type == "flatten":
            return nn.Flatten()

        if layer_type == "relu":
            return nn.ReLU()

        if layer_type == "conv2d":
            return nn.Conv2d(
                in_channels=layer_config["in_size"],
                out_channels=layer_config["out_size"],
                kernel_size=layer_config.get("kernel_size", 3),
                stride=layer_config.get("stride", 2),            
                padding=layer_config.get("padding", 1),          
                dilation=layer_config.get("dilation", 1),       
                groups=layer_config.get("groups", 1),            
                bias=layer_config.get("bias", True),          
                padding_mode=layer_config.get("padding_mode", "zeros") 
            )

        if layer_type == "maxpool2d":
            return nn.MaxPool2d(
                kernel_size=layer_config.get("kernel_size", 3),
                stride=layer_config.get("stride", None),  
                padding=layer_config.get("padding", 0),
                dilation=layer_config.get("dilation", 1),
                ceil_mode=layer_config.get("ceil_mode", False)
            )

        if layer_type == "dropout":
            return nn.Dropout(p=layer_config.get("proba", 0.5))

        if layer_type == "log_softmax":
            return nn.LogSoftmax(dim=layer_config.get("dim", 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method to forward the input through the layers of the network.

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: output
        """
        for name, layer in zip(self.layer_names, self.layers):  # Iterate using names
            x = layer(x)
            if self.print_bool:
                print(f"Layer {name} output shape: {x.shape}")

        self.print_bool = False
        return x
