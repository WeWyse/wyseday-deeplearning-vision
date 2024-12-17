"""
Helper module for matplotlib manipulations
"""

import matplotlib.pyplot as plt
import numpy as np

# Constant for classes
CLASSES = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

def matplotlib_imshow(img, one_channel=False):
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
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))