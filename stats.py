from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

def load_dataset(dataset_name: str, root: str = './data'):
    if dataset_name.lower() == "mnist":
        dataset = datasets.MNIST(
            root,
            train=True,
            download=True,
            transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ]))
    elif dataset_name.lower() == "cifar10":
        dataset = datasets.CIFAR10(
            root,
            train=True,
            download=True,
            transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))
    elif dataset_name.lower() == "cifar100":
        dataset = datasets.CIFAR100(
            root,
            train=True,
            download=True,
            transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    return dataset

def plot_image_grid(dataset: str = 'mnist', num_row: int = 10, num_col: int = 10, save: bool = False) -> None:
    tdataset = load_dataset(dataset)
    fig, axes = plt.subplots(num_row, num_col, figsize=(5,5))
    for i in range(num_row*num_col):
        image, _ = tdataset[i]
        row = i//num_col
        col = i%num_col
        ax = axes[row, col]
        ax.set_xticks([])
        ax.set_yticks([])
        image = image.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
        if image.shape[2] == 1:
            image = image.squeeze(2)  # (H, W, 1) -> (H, W) pour les images en niveaux de gris
        ax.imshow(image)
    #plt.suptitle(f"Example of Images from the Dataset {tdataset.__class__.__name__}")
    if save:
        plt.savefig(f"report/images/{tdataset.__class__.__name__}_image_grid.png")
    plt.show()

def plot_class_distribution(dataset: str = 'mnist', save: bool = False) -> None:
    tdataset = load_dataset(dataset)
    labels = [label for _, label in tdataset]
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # if dataset is cifar10, put the class names instead of the label numbers
    if dataset.lower() == "cifar10":
        class_names = tdataset.classes
        unique_labels = [class_names[label] for label in unique_labels]
    
    
    plt.figure(figsize=(8, 5))
    plt.bar(unique_labels, counts, color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Nombre d\'échantillons')
    plt.title(f"Class Distribution in {tdataset.__class__.__name__}")
    plt.xticks(unique_labels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    if save:
        plt.savefig(f"report/images/{tdataset.__class__.__name__}_class_distribution.png")
    plt.show()
    
    
if __name__ == "__main__":
    plot_class_distribution(dataset='cifar10', save=True)