import torch
from src.utils import dataset_loader, display_samples


def main():
    
    # Input for verbose mode
    verbose_input = input("Do you want to enable verbose mode? (y/n, default is n): ")
    verbose = verbose_input.strip().lower() == 'y'
    
    # Prompt the user to choose a dataset
    dataset_name = input("Please choose a dataset to load (MNIST, CIFAR10, CIFAR100, defaults to MNIST): ")
    
    # Use the provided dataset name or default to MNIST
    if dataset_name.strip() == "":
        dataset_name = "MNIST"
    else:
        dataset_name = dataset_name.strip().upper()

    # Prompt the user to enter a batch size
    batch_size_input = input("Please enter the batch size (default is 128): ")
    
    # Use the provided batch size or default to 128
    if batch_size_input.strip() == "":
        batch_size = 128
    else:
        batch_size = int(batch_size_input)
    
    # Load the dataset using the dataset_loader function
    try:
        data_loader = dataset_loader(dataset_name, batch_size)
        
        print(f"{dataset_name} dataset loaded successfully with batch size {batch_size}.")
        
        if verbose:
            print(f"Number of samples in the dataset: {len(data_loader.dataset)}")
            print(f"Number of batches: {len(data_loader)}")
            print(f"Sample shape: {data_loader.dataset[0][0].shape}")
    
    except ValueError as e:
        print(e)
        # Retry the input if the dataset name is invalid
        main()
        
    # Prompt the user to display samples from the dataset
    display_samples_input = input("Do you want to display samples from the dataset? (y/n, default is n): ")
    if display_samples_input.strip().lower() == 'y':
        
        num_samples_input = input("How many samples do you want to display? (default is 5): ")
        
        if num_samples_input.strip() == "":
            num_samples = 5
        else:            
            num_samples = int(num_samples_input)
            
        display_samples(data_loader, num_samples=num_samples)
        
    
if __name__ == "__main__":
    main()
    
    
    
        


