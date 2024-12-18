"""
Configuration Service module with the configuration manager
"""

import torch
import yaml


class ConfigManager:
    """
    This class is responsible for managing the configuration file.
    It provides a method to load the configuration from a YAML file.
    """

    def __init__(self, yaml_path="my_neural_network.yml"):
        """
        Initializes the ConfigManager class with a default YAML file path.

        Parameters:
            yaml_path (str): The path to the YAML configuration file. Defaults to "my_neural_network.yml".
        """
        self.yaml_path = yaml_path

    def load(self):
        """
        Loads the configuration from the YAML file.

        Returns:
            dict: A dictionary containing the configuration settings.
        """
        with open(self.yaml_path) as file:
            config = yaml.safe_load(file)
        return config

    def obtain_device(self):
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"Using {device} device")
        return device
