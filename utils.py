import torch
from torchvision import datasets, transforms

def dataset_loader(dataset_name: str = "MNIST", batch_size: int = 128) -> torch.utils.data.DataLoader:
    """
    Check if the dataset is available in torchvision.datasets, if not, raise an error. 
    If it is available, load the dataset and return a DataLoader object.
    
    Parameters
    -----------
    dataset_name (str): 
        The name of the dataset to load. 
        Currently supports "MNIST", "CIFAR10", and "CIFAR100". 
        Default is "MNIST".
        
    batch_size (int): 
        The number of samples in each batch. 
        Default is 128.

    Returns
    --------
    DataLoader:
        A DataLoader object containing the loaded dataset.
    """
    
    # Check if the dataset is available in torchvision.datasets
    if dataset_name not in ["MNIST", "CIFAR10", "CIFAR100"]:
        raise ValueError(f"Dataset {dataset_name} is not supported. Please choose from 'MNIST', 'CIFAR10', or 'CIFAR100'.")
    
    # Load the dataset and return a DataLoader object
    if dataset_name == "MNIST":
        dataset = datasets.MNIST(
            "./data",
            train=True,
            download=True,
            transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))
    elif dataset_name == "CIFAR10":
        dataset = datasets.CIFAR10(
            "./data",
            train=True,
            download=True,
            transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))
    else:
        dataset = datasets.CIFAR100(
            "./data",
            train=True,
            download=True,
            transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))
        
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def get_device() -> torch.device:
    """
    Get the available device (GPU or CPU) for PyTorch operations.

    Returns
    --------
    torch.device:
        The available device (GPU or CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def display_samples(loader: torch.utils.data.DataLoader, num_samples: int = 5) -> None:
    """
    Display a few randomly selected samples from the dataset.

    Parameters
    -----------
    loader (DataLoader): 
        The DataLoader object containing the dataset.
    num_samples (int): 
        The number of samples to display. Default is 5.
    """
    import matplotlib.pyplot as plt
    import random as rd
    
    samples = rd.sample(list(loader.dataset), num_samples)
    plt.figure(figsize=(10, 2))
    for i, (image, label) in enumerate(samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(image.permute(1, 2, 0))
        plt.title(f"Label: {label}")
        plt.axis('off')
    plt.show()