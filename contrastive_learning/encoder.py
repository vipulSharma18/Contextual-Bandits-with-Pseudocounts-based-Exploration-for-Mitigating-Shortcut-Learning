#The encoder architecture details are taken from 
# “CURL: Contrastive Learning for RL.” Available: https://mishalaskin.github.io/curl/

import torch
import torchvision.models as models

def ConvNetEncoder(z_dim=16, num_layers=4, patch_size=56):
        resnet18 = models.resnet18(weights=None)
        num_ftrs = resnet18.fc.in_features
        resnet18.fc = torch.nn.Linear(num_ftrs, z_dim)
        return resnet18

def save_encoder(model, path): 
    torch.save(model.state_dict(), path)