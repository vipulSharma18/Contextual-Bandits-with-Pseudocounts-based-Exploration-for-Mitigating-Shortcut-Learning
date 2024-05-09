#The encoder architecture details are taken from 
# “CURL: Contrastive Learning for RL.” Available: https://mishalaskin.github.io/curl/

import torch
from torch import nn
from torch.nn import functional as F

class ConvNetEncoder(nn.Module):
    def __init__(self, z_dim=16, num_layers=4):
        super(ConvNetEncoder, self).__init__()
        # Initial convolution layer
        self.initial_conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2)
        # Additional layers
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1) for _ in range(num_layers - 1)
        ])
        # Layer normalization
        self.layer_norm = nn.LayerNorm(z_dim)
        self.size_after_convs = self._calculate_size_after_convs(224)  # input image size is 224x224
        # Output MLP to get the latent vector
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * (self.size_after_convs ** 2), 1024),
            nn.ReLU(),
            nn.Linear(1024, z_dim),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Forward pass of the encoder.
        """
        x = x / 255.0  # Normalize input
        z = F.relu(self.initial_conv(x))
        for conv in self.conv_layers:
            z = F.relu(conv(z))
        # Pass through MLP
        z = self.mlp(z)
        # Normalize and activate
        z = self.layer_norm(z)
        z = torch.tanh(z)
        return z

    def _calculate_size_after_convs(self, input_size):
        """
        Utility to calculate the size of the image after all convolutions.
        Assume square input.
        """
        size = input_size
        size = (size - 2) // 2 + 1  # Initial conv: kernel_size=3, stride=2
        for _ in range(len(self.conv_layers)):  # Other convs: kernel_size=3, stride=1
            size = size - 2 + 1  # Kernel size 3, stride 1
        return size

'''
# Example of using the encoder
encoder = ConvNetEncoder(z_dim=16, num_layers=3)
# Example input tensor (batch size, channels, height, width)
x = torch.rand(4, 3, 224, 224)
z = encoder(x)
print(z.shape)  # Should show (4, 16) if input size and output linear layer are correctly calculated
'''

def save_encoder(model, path): 
    torch.save(model.state_dict(), path)

def load_encoder(path, z_dim=16, num_layers=4): 
    encoder = ConvNetEncoder(z_dim=z_dim, num_layers=num_layers)
    encoder.load_state_dict(torch.load(path))
    return encoder