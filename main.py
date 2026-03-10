import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from src.utils import dataset_loader, display_samples
from src.embeddings import embed_classes, PositionalEncoding
from src.models.unet import UNet
import tqdm
import matplotlib.pyplot as plt

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

    # ensure directory exists
    if not os.path.exists("./src/models/saved"):
        os.makedirs("./src/models/saved")
        
    if not os.path.exists("./.logs/buffer"):
        os.makedirs("./.logs/buffer")

    # remove log file if exists
    log_file = f".logs/buffer/guided_unet_train_loss_{dataset_name}.log"
    if os.path.exists(log_file):
        os.remove(log_file)


    with open(log_file, "a") as f:
            # first line of the log file contains hyperparameters and epochs info
            f.write(f"Training log for guided UNet on {dataset_name} dataset\n")
            f.write(f"Hyperparameters: learning_rate={lr}, num_epochs={num_epochs}, p_uncond={p_uncond}\n")
            
            # second line of the log file contains column names for the training log
            f.write("timestamp,epoch,average_loss,epoch_duration_seconds\n")

    epoch_times = []

    # loop
    for epoch_idx in range(num_epochs):
        
        # Keep track of time for each epoch to print duration and ETA based on average epoch time
        start_time = time.time()
        
        if verbose:
            print(f"Starting epoch {epoch_idx+1}/{num_epochs}...")
        epoch_loss = []
        for batch_idx, (data, y) in enumerate(tqdm.tqdm(loader, total=len(loader), desc=f"Epoch {epoch_idx+1}/{num_epochs}", leave=False)):
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
            #if verbose:
                #print("epoch{} (iter{}) - loss {:5.4f}".format(epoch_idx+1, batch_idx+1, loss), end="\r")
                # add a progression bar for each epoch with tqdm, and print the epoch duration and estimated remaining time at the end of each epoch                
                #print(f"Epoch {epoch_idx+1}/{num_epochs}, Batch {batch_idx+1}/{len(loader)}, Loss: {loss.item():.4f}", end="\r")
                    

        end_time = time.time()
        epoch_duration = end_time - start_time
        epoch_times.append(epoch_duration)
        avg_time = sum(epoch_times) / len(epoch_times)
        remaining_time = avg_time * (num_epochs - epoch_idx - 1)
        remaining_time_format = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
        
        # finalize epoch (save log and checkpoint)
        epoch_average_loss = sum(epoch_loss)/len(epoch_loss)
        if verbose:
            #print("epoch{} (iter{}) - loss {:5.4f}".format(epoch_idx+1, batch_idx+1, epoch_average_loss))
            print(f"Epoch {epoch_idx+1} completed in {epoch_duration:.2f} seconds - Average Loss: {epoch_average_loss:.4f}")
            print(f"Estimated remaining time: {remaining_time_format}")
            
        with open(log_file, "a") as f:
            # for each epoch, save the timestamp, epoch index, average loss, and epoch duration in seconds
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{epoch_idx+1},{epoch_average_loss},{epoch_duration}\n")
        
        # save checkpoint every 10 epochs
        if (epoch_idx + 1) % 10 == 0:
            torch.save(model.state_dict(), f"./src/models/saved/guided_unet_{dataset_name}_{epoch_idx}.pt")
            torch.save(emb.state_dict(), f"./src/models/saved/guided_embedding_{dataset_name}_{epoch_idx}.pt")
            
            # keep only the last 5 checkpoints (embedding and unet) to save disk space
            checkpoint_files = sorted([f for f in os.listdir("./src/models/saved") if f.startswith(f"guided_unet_{dataset_name}_") and f.endswith(".pt")], key=lambda x: int(x.split(f"guided_unet_{dataset_name}_")[1].split(".pt")[0]))
            embedding_checkpoint_files = sorted([f for f in os.listdir("./src/models/saved") if f.startswith(f"guided_embedding_{dataset_name}_") and f.endswith(".pt")], key=lambda x: int(x.split(f"guided_embedding_{dataset_name}_")[1].split(".pt")[0]))
            if len(checkpoint_files) > 5:
                os.remove(os.path.join("./src/models/saved", checkpoint_files[0]))
            if len(embedding_checkpoint_files) > 5:
                os.remove(os.path.join("./src/models/saved", embedding_checkpoint_files[0]))
                
            
    # save final model and embedding
    torch.save(model.state_dict(), f"./src/models/saved/guided_unet_{dataset_name}_final.pt")
    torch.save(emb.state_dict(), f"./src/models/saved/guided_embedding_{dataset_name}_final.pt")
        
    # Training completed
    if verbose:
        print("Training completed in {:.2f} minutes. Saving log file...".format(sum(epoch_times)/60))
    else:    
        print("Training completed. Saving log file...")
        
    # Move log file from .logs/buffer to .logs with a timestamp
    final_log_file = f".logs/guided_unet_train_loss_{dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    os.rename(log_file, final_log_file)
    if verbose:
        print(f"Log file saved to {final_log_file}")
        
    print(f"Final model and embedding saved to ./src/models/saved/guided_unet_{dataset_name}_final.pt   \
        and ./src/models/saved/guided_embedding_{dataset_name}_final.pt")
    
    print("You can find the training log in the .logs directory with the name format guided_unet_train_loss_{dataset_name}_YYYYMMDD_HHMMSS.log")
    
    print("You can load the final model and embedding for inference or further training.")



def load_weights(model: UNet, emb: nn.Embedding, dataset_name: str = "MNIST", epoch_idx: str = "final", verbose: bool = False):
    """
    Load the weights of the trained model and embedding from the final saved checkpoint files.
    Return the model and embedding with the loaded weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if epoch_idx is None or epoch_idx == "final":
        if verbose:
            print("Loading final model and embedding for dataset {}...".format(dataset_name))
        try:
            model.load_state_dict(torch.load(f"./src/models/saved/guided_unet_{dataset_name}_final.pt", map_location=device))
            emb.load_state_dict(torch.load(f"./src/models/saved/guided_embedding_{dataset_name}_final.pt", map_location=device))
        except FileNotFoundError:
            raise FileNotFoundError("Final model or embedding checkpoint not found. Please ensure that the training has been completed and the checkpoint files are saved.")
        except (EOFError, RuntimeError) as e:
            raise RuntimeError(f"Final checkpoint file for dataset {dataset_name} is corrupted or empty. Try loading a specific epoch checkpoint instead. Original error: {e}")
    else:
        integer_epoch_idx = None
        try:
            integer_epoch_idx = int(epoch_idx)
        except ValueError:
            raise ValueError("epoch_idx should be an integer or 'final'")
        if verbose:
            print(f"Loading model and embedding from epoch {integer_epoch_idx}, for dataset {dataset_name}...")
        try:
            model.load_state_dict(torch.load(f"./src/models/saved/guided_unet_{dataset_name}_{integer_epoch_idx}.pt", map_location=device))
            emb.load_state_dict(torch.load(f"./src/models/saved/guided_embedding_{dataset_name}_{integer_epoch_idx}.pt", map_location=device))
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint for epoch {integer_epoch_idx} on dataset {dataset_name} not found. Please ensure the checkpoint file exists.")
        except (EOFError, RuntimeError) as e:
            raise RuntimeError(f"Checkpoint file for epoch {integer_epoch_idx} on dataset {dataset_name} is corrupted or empty. Original error: {e}")
    return model, emb


def run_inference(
    model: UNet,
    emb: nn.Embedding, 
    class_name: str, 
    class_list: list, 
    s: float, 
    source_channel: int = 3,
    img_size: int = 32,
    num_row: int = 10, 
    num_col: int = 10

):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    T = 1000
    alphas = torch.linspace(start=0.9999, end=0.98, steps=T, dtype=torch.float64).to(device)
    alpha_bars = torch.cumprod(alphas, dim=0)

    ##########
    # generate sigma_t
    ##########
    alpha_bars_prev = torch.cat((torch.ones(1).to(device), alpha_bars[:-1]))
    sigma_t_squared = (1.0 - alphas) * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
    sigma_t = torch.sqrt(sigma_t_squared)

    ##########
    # make white noise
    ##########
    x = torch.randn(num_row*num_col, source_channel, img_size, img_size).to(device)

    ##########
    # generate images
    ##########
    with torch.no_grad():
        # generate class embedding
        # (the first half is for epsilon(x, y), the second half is for epsilon(x, empty))
        class_id_list = [i for i,v in enumerate(class_list) if v==class_name]
        if len(class_id_list) == 0:
            raise Exception("class name doesn't exist")
        y = class_id_list[0]
        y_batch = (torch.tensor(y).to(device)).repeat(num_row*num_col)
        y_batch = torch.cat((y_batch, y_batch), dim=0)
        y_emb_batch = emb(y_batch)
        mask = torch.cat((
            torch.ones(num_row*num_col).to(device),
            torch.zeros(num_row*num_col).to(device)))
        y_emb_batch = y_emb_batch * mask[:,None]
        # loop T-1, T-2, ... ,0
        for t in tqdm.tqdm(reversed(range(T)), total=T):
            # generate t
            # (the first half is for epsilon(x, y), the second half is for epsilon(x, empty))
            t_batch = (torch.tensor(t).to(device)).repeat(num_row*num_col)
            t_batch = torch.cat((t_batch, t_batch), dim=0)
            # compute epsilon
            # (the first half is for epsilon(x, y), the second half is for epsilon(x, empty))
            x_batch = torch.cat((x, x), dim=0)
            eps_batch = model(x_batch, t_batch, y_emb_batch)
            eps_cond, eps_uncond = torch.split(eps_batch, len(eps_batch)//2, dim=0)
            eps = (1.0 + s) * eps_cond - s * eps_uncond
            # update x
            if t > 0:
                z = torch.randn(num_row*num_col, source_channel, img_size, img_size).to(device)
            else:
                z = torch.zeros(num_row*num_col, source_channel, img_size, img_size).to(device)
            x = (1.0 / torch.sqrt(alphas[t])).float() * (x - ((1.0 - alphas[t]) / torch.sqrt(1.0 - alpha_bars[t])).float() * eps) + \
                sigma_t[t].float() * z

    ##########
    # output
    ##########

    # reshape to channels-last : (N,C,H,W) --> (N,H,W,C)    
    x = x.permute(0, 2, 3, 1)
    # clip
    x = torch.clamp(x, min=0.0, max=1.0)
    # draw
    fig, axes = plt.subplots(num_row, num_col, figsize=(5,5))
    for i in range(num_row*num_col):
        image = x[i].cpu().numpy()
        row = i//num_col
        col = i%num_col
        ax = axes[row, col]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(image)
    plt.suptitle(f"Generated images for class '{class_name}' with s={s}")
    
    # ensure directory exists
    if not os.path.exists("./report/images"):
        os.makedirs("./report/images")
    
    title_class_name = class_name.replace(" ", "_").replace("-", "_")
    
    plt.savefig(f"./report/images/guided_unet_{title_class_name}_s{s}.png")
    plt.tight_layout()
    plt.show()
    
    


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
        
        # list available checkpoints for the user to choose from
        checkpoint_dir = "./src/models/saved"
        available_checkpoints_unet = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"guided_unet_{dataset_name}_") and f.endswith(".pt")]
        available_checkpoints_embedding = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"guided_embedding_{dataset_name}_") and f.endswith(".pt")]
        if verbose:
            print("All available checkpoints for dataset {}:".format(dataset_name))
            for ckpt in available_checkpoints_unet:
                # extract epoch index from checkpoint name
                epoch_idx = ckpt.split(f"guided_unet_{dataset_name}_")[1].split(".pt")[0]
                print(epoch_idx)
        
        epoch_idx = input("Please enter the epoch index to load (integer or 'final', default is last appeared): ")
        
        # ensure the user input is valid (and in the available checkpoints)
        if epoch_idx.strip() == "" or epoch_idx.strip() == "final":
            
            if len(available_checkpoints_unet) == 0 or len(available_checkpoints_embedding) == 0:
                print("No checkpoints found for dataset {}. Please train the model first.".format(dataset_name))
                main()
            else:
        
                # check if final checkpoint exists, if yes load final checkpoint, if not load the last appeared checkpoint
                final_unet_checkpoint = f"guided_unet_{dataset_name}_final.pt"
                final_embedding_checkpoint = f"guided_embedding_{dataset_name}_final.pt"
                if final_unet_checkpoint in available_checkpoints_unet and final_embedding_checkpoint in available_checkpoints_embedding:
                    epoch_idx = "final"
                else:
                    # get the last appeared checkpoint (the one with the largest epoch index)
                    if len(available_checkpoints_unet) == 0 or len(available_checkpoints_embedding) == 0:
                        print("No checkpoints found for dataset {}. Please train the model first.".format(dataset_name))
                        main()
                    else:
                        last_unet_checkpoint = sorted(available_checkpoints_unet, key=lambda x: int(x.split(f"guided_unet_{dataset_name}_")[1].split(".pt")[0]))[-1]
                        last_embedding_checkpoint = sorted(available_checkpoints_embedding, key=lambda x: int(x.split(f"guided_embedding_{dataset_name}_")[1].split(".pt")[0]))[-1]
                        if last_unet_checkpoint.split(f"guided_unet_{dataset_name}_")[1].split(".pt")[0] != last_embedding_checkpoint.split(f"guided_embedding_{dataset_name}_")[1].split(".pt")[0]:
                            print("The last appeared UNet checkpoint and embedding checkpoint do not match. Please check the available checkpoints.")
                            main()
                        else:
                            epoch_idx = last_unet_checkpoint.split(f"guided_unet_{dataset_name}_")[1].split(".pt")[0]
                    
        elif epoch_idx.strip() != "final" and epoch_idx.strip() != "":
            try:
                int_epoch_idx = int(epoch_idx.strip())
                unet_checkpoint_name = f"guided_unet_{dataset_name}_{int_epoch_idx}.pt"
                embedding_checkpoint_name = f"guided_embedding_{dataset_name}_{int_epoch_idx}.pt"
                if unet_checkpoint_name not in available_checkpoints_unet or embedding_checkpoint_name not in available_checkpoints_embedding:
                    print(f"Checkpoint for epoch {int_epoch_idx} not found. Please choose from the available checkpoints.")
                    main()
            except ValueError:
                print("Invalid input for epoch index. Please enter an integer or 'final'.")
                main()
        
        model, emb = load_weights(model, emb, dataset_name=dataset_name, epoch_idx=epoch_idx, verbose=verbose)
        if verbose:
            print("Model weights loaded successfully.")
        
        class_name = input("Please enter the class name to generate (default is '0 - zero' for MNIST): ")
        if class_name.strip() == "":
            class_name = "0 - zero"
        s_input = input("Please enter the guidance scale s (default is 0 (unconditional)): ")
        s = float(s_input) if s_input.strip() != "" else 0.0
         
        run_inference(model, emb, class_name, data_loader.dataset.classes, s=s, source_channel=source_channel, img_size=data_loader.dataset[0][0].shape[1])
        print(f"Generated images for class '{class_name}' with guidance scale s={s}.")
    
    
if __name__ == "__main__":
    main()
    
    
    
        


