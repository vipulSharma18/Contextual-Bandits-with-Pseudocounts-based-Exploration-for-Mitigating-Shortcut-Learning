#The encoder architecture details are taken from 
# “CURL: Contrastive Learning for RL.” Available: https://mishalaskin.github.io/curl/

import torch
import torchvision.models as models

def ConvNetEncoder(z_dim=256, patch_size=56):
        resnet18 = models.resnet18(weights=None)
        num_ftrs = resnet18.fc.in_features
        resnet18.fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, z_dim),
            torch.nn.LayerNorm(z_dim)
        )
        return resnet18

def save_encoder(model, path): 
    torch.save(model.state_dict(), path)
    
def load_context_gen(z_dim=256, patch_size=56, path = None): 
    # Load the pretrained ResNet18 model
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, z_dim),
            torch.nn.LayerNorm(z_dim)
        )
    model.load_state_dict(torch.load(path))
    return model