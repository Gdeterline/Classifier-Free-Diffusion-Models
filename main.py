import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import dataset_loader, display_samples
from src.embeddings import embed_classes, PositionalEncoding
from src.models.unet import UNet

def load_unet(source_channel: int, unet_base_channel: int, num_classes: int):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    emb = embed_classes(num_classes, unet_base_channel).to(device)
    unet = UNet(
        source_channel=source_channel,
        unet_base_channel=unet_base_channel,
        num_norm_groups=32,
    ).to(device)
    
    return unet, emb

def train(
    model: UNet,
    emb: nn.Embedding,
    lr: float = 2e-4,
    num_epochs: int = 200,
    p_uncond: float = 0.2,
    loader: torch.utils.data.DataLoader = None,
    dataset_name: str = "MNIST",
    verbose: bool = False
):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    emb.to(device)
    
    if verbose:
        print(f"Training on {device} with learning rate {lr} for {num_epochs} epochs.")
        print(f"Using dataset: {dataset_name} with batch size {loader.batch_size}.")
        print(f"Unconditional probability: {p_uncond}")
    
    # initialize optimizer and scheduler
    opt = torch.optim.Adam(list(emb.parameters()) + list(model.parameters()), lr=lr, eps=1e-08)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        opt,
        start_factor=1.0/5000,
        end_factor=1.0,
        total_iters=5000
    )

    # 1. Initialize T and alpha
    #   (See above note for precision.)
    T = 1000
    alphas = torch.linspace(start=0.9999, end=0.98, steps=T, dtype=torch.float64).to(device)
    alpha_bars = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bars_t = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars_t = torch.sqrt(1.0 - alpha_bars)

    # remove log file if exists
    log_file = f".logs/buffer/guided_unet_train_loss_{dataset_name}.log"
    if os.path.exists(log_file):
        os.remove(log_file)

    epoch_times = []

    # loop
    for epoch_idx in range(num_epochs):
        
        # Keep track of time for each epoch to print duration and ETA based on average epoch time
        start_time = time.time()
        
        if verbose:
            print(f"Starting epoch {epoch_idx+1}/{num_epochs}...")
        epoch_loss = []
        for batch_idx, (data, y) in enumerate(loader):
            model.train()
            opt.zero_grad()

            # Pick up x_0 (shape: [64, 3, 32, 32])
            x_0 = data.to(device)
            y_ = y.to(device)

            # Pick up random timestep, t .
            #    Instead of picking up t=1,2, ... ,T ,
            #    here we pick up t=0,1, ... ,T-1 .
            #   (i.e, t == 0 means diffused for 1 step)
            b = x_0.size(dim=0)
            t = torch.randint(T, (b,)).to(device)

            # Generate the seed of noise, epsilon .
            #    We just pick up from 1D standard normal distribution with the same shape,
            #    because off-diagonal elements in covariance is all zero.
            eps = torch.randn_like(x_0).to(device)

            # Compute x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) epsilon
            #    (t == 0 means diffused for 1 step)
            x_t = sqrt_alpha_bars_t[t][:,None,None,None].float() * x_0 + sqrt_one_minus_alpha_bars_t[t][:,None,None,None].float() * eps

            # Get class embedding
            y_emb = emb(y_)

            # Set empty in class embedding with probability p_uncond (See above)
            rnd = torch.rand(b).to(device)
            mul = torch.where(rnd < p_uncond, 0.0, 1.0)
            y_emb = y_emb * mul[:,None]

            # Get loss and apply gradient (update)
            model_out = model(x_t, t, y_emb)
            loss = F.mse_loss(model_out, eps, reduction="mean")
            loss.backward()
            opt.step()
            scheduler.step()

            # log
            epoch_loss.append(loss.item())
            if verbose:
                print("epoch{} (iter{}) - loss {:5.4f}".format(epoch_idx+1, batch_idx+1, loss), end="\r")

        end_time = time.time()
        epoch_duration = end_time - start_time
        epoch_times.append(epoch_duration)
        avg_time = sum(epoch_times) / len(epoch_times)
        remaining_time = avg_time * (num_epochs - epoch_idx - 1)
        remaining_time_format = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
        
        # finalize epoch (save log and checkpoint)
        epoch_average_loss = sum(epoch_loss)/len(epoch_loss)
        if verbose:
            print("epoch{} (iter{}) - loss {:5.4f}".format(epoch_idx+1, batch_idx+1, epoch_average_loss))
            print(f"Epoch {epoch_idx+1} completed in {epoch_duration:.2f} seconds.")
            print(f"Estimated remaining time: {remaining_time_format}")
            
        with open(log_file, "a") as f:
            for l in epoch_loss:
                f.write("%s\n" %l)
        torch.save(model.state_dict(), f"./src/models/saved/guided_unet_{epoch_idx}.pt")
        torch.save(emb.state_dict(), f"./src/models/saved/guided_embedding_{epoch_idx}.pt")
        
    # Training completed
    if verbose:
        print("Training completed in {:.2f} minutes.".format(sum(epoch_times)/60))
    else:    
        print("Training completed.")


def main():
    
    # Input for verbose mode
    verbose_input = input("Do you want to enable verbose mode? (y/n, default is y): ")
    verbose = verbose_input.strip().lower() != 'n'
    
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
        source_channel = data_loader.dataset[0][0].shape[0]
        
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
        
    # Prompt the user to enter the model chosen [1] for UNet, etc.
    model_choice = input("Please select a model:\n[1] tsmatz's UNet\n[2] Not implemented yet\nEnter the number corresponding to your choice (default is 1): ")
    
    if model_choice.strip() == "" or model_choice.strip() == "1":
        unet_base_channel = 128
        num_classes = len(data_loader.dataset.classes)
        model, emb = load_unet(source_channel, unet_base_channel, num_classes)
        print("UNet model loaded successfully.")
    else:
        print("Invalid choice. Please select a valid model.")
        main()
        
    # Prompt the user to train or sample
    train_or_sample = input("Do you want to train the model or sample from it? (train/sample, default is sample): ")
    
    if train_or_sample.strip().lower() == "train":
        lr_input = input("Please enter the learning rate (default is 2e-4): ")
        num_epochs_input = input("Please enter the number of epochs (default is 200): ")
        p_uncond_input = input("Please enter the unconditional probability (default is 0.2): ")
        
        lr = float(lr_input) if lr_input.strip() != "" else 2e-4
        num_epochs = int(num_epochs_input) if num_epochs_input.strip() != "" else 200
        p_uncond = float(p_uncond_input) if p_uncond_input.strip() != "" else 0.2
        
        train(model, emb, lr=lr, num_epochs=num_epochs, p_uncond=p_uncond, loader=data_loader, dataset_name=dataset_name, verbose=verbose)
        
    else:
        print("Sampling functionality not implemented yet. Please run the training to generate samples.")
    
    
    
if __name__ == "__main__":
    main()
    
    
    
        


