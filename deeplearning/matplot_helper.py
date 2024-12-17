"""
Helper module for matplotlib manipulations
"""

import matplotlib.pyplot as plt
import numpy as np

from deeplearning.data_service import CLASSES


def imshow(img, one_channel=False):
    """
    Helper function to display an image using matplotlib.

    Args:
        img (Tensor): The image tensor to display.
        one_channel (bool): Whether the image has one color channel (grayscale) or three (RGB). Default: False.

    Returns:
        None
    """
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    scale_factor = 3
    aspect_ratio = npimg.shape[0] / npimg.shape[1]

    if one_channel:
        plt.imshow(npimg, cmap="Greys",
            extent=[0, scale_factor*npimg.shape[1], 0, scale_factor*npimg.shape[0]*aspect_ratio] 
        )
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_classes_preds(images, labels, preds, probs):
    """
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    """
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(5, 15))
    for idx in np.arange(4):
        ax = fig.add_subplot(4, 1, idx + 1, xticks=[], yticks=[])
        imshow(images[idx], one_channel=True)
        ax.set_title(
            "{0}, {1:.1f}%\n(label: {2})".format(
                CLASSES[preds[idx]], 
                probs[idx] * 100.0, 
                CLASSES[labels[idx]]
            ),
            color=("green" if preds[idx] == labels[idx].item() else "red"),
        )
    return fig
