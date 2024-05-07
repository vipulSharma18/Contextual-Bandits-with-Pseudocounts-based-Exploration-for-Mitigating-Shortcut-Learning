import torch
import torchvision.models as models
from torch import nn

def set_seed(seed): 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_model(): 
    resnet18 = models.resnet18(weights=None)
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 1)
    return resnet18