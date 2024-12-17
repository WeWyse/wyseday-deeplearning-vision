"""
Main file for executing the Deep Learning code
"""

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from deeplearning.config_service import ConfigManager
from deeplearning.data_service import DataManager
from deeplearning.model_service import ModelManager


def main():

    ## LOAD CONFIG
    config_manager = ConfigManager()
    nn_configs = config_manager.load()
    nn_config = nn_configs["my_model_cnn"]

    ### Getting the datasets
    data_manager = DataManager()
    training_data = data_manager.get_fashion_mnist_training_data()
    test_data = data_manager.get_fashion_mnist_test_data()

    ### Creating the Model Class
    model_manager = ModelManager(
        training_data=training_data, test_data=test_data, nn_config=nn_config
    )
    model_manager.train_model(epochs=100)


if __name__ == "__main__":
    main()
