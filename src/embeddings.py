import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import get_device

# ----------------------------------------------- Class Embedding Function ----------------------------------------------- #

def embed_classes(num_classes: int, unet_base_channel: int) -> nn.Embedding:
    """
    Create an embedding layer for class labels.
    
    Parameters
    -----------
    num_classes (int):
        The number of classes in the dataset.
        
    unet_base_channel (int):
        The base number of channels in the UNet model. The embedding dimension will be unet_base_channel * 4.
        
    Returns
    --------
    nn.Embedding:
        An embedding layer that maps class labels to embeddings of dimension unet_base_channel * 4
    """
    device = get_device()
    emb = nn.Embedding(num_classes, unet_base_channel*4).to(device)
    return emb

# ----------------------------------------------- Positional Encoding Class ----------------------------------------------- #

class PositionalEncoding(nn.Module):
    """
    Positional encoding module for diffusion models.
    This module generates sinusoidal positional encodings for the given timesteps and applies
    a feedforward network to produce the final positional encoding.
    
    Parameters
    -----------
    base_dim (int): 
        The dimension of the input timesteps. This should be an even number.
        
    hidden_dim (int): 
        The dimension of the hidden layer in the feedforward network.
    
    output_dim (int): 
        The dimension of the output positional encoding.
    """
    def __init__(
        self,
        base_dim, # 128
        hidden_dim, # 256
        output_dim, # 512
    ):
        super().__init__()

        # In this example, we assume that the number of embedding dimension is always even.
        # (If not, please pad the result.)
        assert(base_dim % 2 == 0)
        self.timestep_dim = base_dim

        self.hidden1 = nn.Linear(
            base_dim,
            hidden_dim)
        self.hidden2 = nn.Linear(
            hidden_dim,
            output_dim)

    def forward(self, picked_up_timesteps):
        """
        Generate sinusoidal positional encodings for the given timesteps and apply a feedforward network.
        
        Parameters
        -----------
        picked_up_timesteps (torch.Tensor):
            A tensor of shape (batch_size,) containing the timesteps for which to generate positional encodings.
            
        Returns
        --------
        torch.Tensor:
            A tensor of shape (batch_size, output_dim) containing the positional encodings for the given timesteps.
        """
        device = get_device()    
        
        # Generate 1 / 10000^{2i / d_e}
        # shape : (timestep_dim / 2, )
        interval = 1.0 / (10000**(torch.arange(0, self.timestep_dim, 2.0).to(device) / self.timestep_dim))
        
        # Generate t / 10000^{2i / d_e}
        # shape : (batch_size, timestep_dim / 2)
        position = picked_up_timesteps.type(torch.get_default_dtype())
        
        # Compute the radian values for the sinusoidal functions
        radian = position[:, None] * interval[None, :]
        
        # Get sin(t / 10000^{2i / d_e}) and unsqueeze
        # shape : (batch_size, timestep_dim / 2, 1)
        sin = torch.sin(radian).unsqueeze(dim=-1)
        
        # Get cos(t / 10000^{2i / d_e}) and unsqueeze
        # shape : (batch_size, timestep_dim / 2, 1)
        cos = torch.cos(radian).unsqueeze(dim=-1)
        
        # Get sinusoidal positional encoding
        # shape : (batch_size, timestep_dim)
        positional_encoding_tmp = torch.concat((sin, cos), dim=-1)   # shape : (num_timestep, timestep_dim / 2, 2)
        
        d = positional_encoding_tmp.size()[1]
        pe = positional_encoding_tmp.view(-1, d * 2)                 # shape : (num_timestep, timestep_dim)
        
        # Apply feedforward network to the positional encoding
        # shape : (batch_size, timestep_dim * 4)
        out = self.hidden1(pe)
        out = F.silu(out)
        out = self.hidden2(out)

        return out