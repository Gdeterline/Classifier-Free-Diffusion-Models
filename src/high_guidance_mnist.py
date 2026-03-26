# When trained on the MNIST dataset, Classifier & Classifier-Free Guidance DDPMs seem to sample, for high values of the guidance
# scale, a "global" shape of the digit (class), that satisfies most representations (with some exceptions coming from very
# different distributions for that class)

# This script aims at providing a visualisation of the explanation of such a phenomenon

import torch
from utils import dataset_loader
import matplotlib.pyplot as plt
import numpy as np


def compute_mean_image(mnist_class: int) -> torch.Tensor:
    """
    Compute the mean image of a given class in the MNIST dataset.

    Parameters
    ----------
    mnist_class : int
        The class for which to compute the mean image.

    Returns
    -------
    None
    """
    # Plot the samples of the class mnist_class
    loader = dataset_loader("MNIST", batch_size=1000)
    for images, labels in loader:
        class_images = images[labels == mnist_class]
        break

    # Compute the mean image of the class
    mean_image = torch.mean(class_images, dim=0)

    return mean_image

def plot_mean_images(save: bool = False) -> None:
    """
    Plot the mean images of all classes in the MNIST dataset, in a 2x5 grid.
    """
    plt.figure(figsize=(10, 5))
    for i in range(10):
        mean_image = compute_mean_image(i)
        plt.subplot(2, 5, i + 1)
        plt.imshow(mean_image.squeeze())
        plt.title(f"Class {i}")
        plt.axis("off")
    if save:
        plt.savefig("report/images/mnist_mean_images.png")
    plt.show()
    
if __name__ == "__main__":
    plot_mean_images(save=True)