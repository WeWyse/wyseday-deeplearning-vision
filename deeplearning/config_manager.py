import yaml


class ConfigManager:

    def __init__(self, yaml_path="my_neural_network.yml"):
        self.yaml_path = yaml_path

    def load(self):
        with open(self.yaml_path) as file:
            config = yaml.safe_load(file)
        return config
