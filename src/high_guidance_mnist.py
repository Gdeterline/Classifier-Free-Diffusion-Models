# When trained on the MNIST dataset, Classifier & Classifier-Free Guidance DDPMs seem to sample, for high values of the guidance
# scale, a "global" shape of the digit (class), that satisfies most representations (with some exceptions coming from very
# different distributions for that class)

# This script aims at providing a visualisation of the explanation of such a phenomenon

import torch
from utils import dataset_loader
import matplotlib.pyplot as plt
import numpy as np

def compute_mean_image(mnist_class: int, increase: float) -> torch.Tensor:
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
    
    if increase > 0:
        mean_image = torch.sigmoid(increase * (mean_image - 0.5))
        
    return mean_image

def plot_mean_image(mnist_class: int, increase: float = 0, save: bool = False) -> None:
    """
    Plot the mean image of a given class in the MNIST dataset.

    Parameters
    ----------
    mnist_class : int
        The class for which to plot the mean image.
    increase : float
        The amount by which to increase the mean image values. Default is 0.
    save : bool
        Whether to save the plot as an image file. Default is False.
    """
    mean_image = compute_mean_image(mnist_class, increase=increase)
    plt.imshow(mean_image.squeeze())
    plt.title(f"Mean Image of Class {mnist_class}")
    plt.axis("off")
    if save:
        plt.savefig(f"report/images/mnist_mean_image_class_{mnist_class}.png")
    plt.show()

def plot_mean_images_grid(save: bool = False) -> None:
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
    digits = [2, 4, 5, 6]
    for digit in digits:
        plot_mean_image(digit, increase=2, save=True)