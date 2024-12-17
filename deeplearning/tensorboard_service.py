import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from deeplearning import matplot_helper
from deeplearning.data_service import CLASSES, DataManager


class TensorboardManager:

    def __init__(self, data: DataLoader):
        self.writer = SummaryWriter("runs/fashion_mnist_experiment_1")
        self.data = data

    def sample_data(self):
        dataiter = iter(self.data)
        images, labels = next(dataiter)
        img_grid = torchvision.utils.make_grid(images)
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
        matplot_helper.imshow(img_grid, one_channel=True)
        self.writer.add_figure("four_fashion_mnist_images", fig)
        self.writer.flush()

    def project_data(self):

        # select random images and their target indices
        images, labels = DataManager.select_n_random(
            self.data.dataset.data, self.data.dataset.targets
        )

        # get the class labels for each image
        class_labels = [CLASSES[lab] for lab in labels]

        # log embeddings
        features = images.view(-1, 28 * 28)
        self.writer.add_embedding(
            features, metadata=class_labels, label_img=images.unsqueeze(1)
        )
        self.writer.flush()
