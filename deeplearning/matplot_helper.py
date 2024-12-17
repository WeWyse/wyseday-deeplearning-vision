"""
Helper module for matplotlib manipulations
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from deeplearning.data_service import CLASSES


def imshow(img: torch.Tensor, one_channel: bool = False):
    """
    Helper function to display an image using matplotlib.

    Args:
        img (torch.Tensor): The image tensor to display.
        one_channel (bool): Whether the image has one color channel (grayscale) or three (RGB). Default is False.

    Returns:
        None

    The function first checks if the image has one color channel. If so, it calculates the mean of the image.
    Then, it unnormalizes the image by dividing it by 2 and adding 0.5. After that, it converts the image to a numpy array.
    Depending on the number of color channels, it displays the image using matplotlib's imshow function,
    either in grayscale or in RGB. For grayscale images, it also adjusts the extent of the image based on its aspect ratio.
    """
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    scale_factor = 3
    aspect_ratio = npimg.shape[0] / npimg.shape[1]

    if one_channel:
        plt.imshow(
            npimg,
            cmap="Greys",
            extent=[
                0,
                scale_factor * npimg.shape[1],
                0,
                scale_factor * npimg.shape[0] * aspect_ratio,
            ],
        )
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_classes_preds(
    images: torch.Tensor, labels: torch.Tensor, preds: torch.Tensor, probs: torch.Tensor
):
    """
    Generates a matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.

    Args:
        images (torch.Tensor): The batch of images to be plotted.
        labels (torch.Tensor): The true labels of the images.
        preds (torch.Tensor): The predicted labels of the images.
        probs (torch.Tensor): The probabilities of the predicted labels.

    Returns:
        fig (Figure): The matplotlib Figure containing the subplots of the images and their labels.

    The function first creates a matplotlib Figure with a size of 5x15.
    Then, it iterates over the first four images in the batch and for each image,
    it creates a subplot, displays the image using the imshow function,
    and sets the title of the subplot to the predicted label, the probability of the prediction,
    and the true label. The color of the title is green if the prediction is correct and red otherwise.
    """
    fig = plt.figure(figsize=(5, 15))
    for idx in np.arange(4):
        ax = fig.add_subplot(4, 1, idx + 1, xticks=[], yticks=[])
        imshow(images[idx], one_channel=True)
        ax.set_title(
            "{0}, {1:.1f}%\n(label: {2})".format(
                CLASSES[preds[idx]], probs[idx] * 100.0, CLASSES[labels[idx]]
            ),
            color=("green" if preds[idx] == labels[idx].item() else "red"),
        )
    return fig
