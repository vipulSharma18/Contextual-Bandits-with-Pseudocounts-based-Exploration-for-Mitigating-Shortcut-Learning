import torch
import torchvision.models as models
from torch import nn

def set_seed(seed): 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_model(): 
    resnet18 = models.resnet18(weights=None)
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 1) #prob of class 1, 1-prob = prob of class 0
    return resnet18

def load_model(model_path):
    # Load the pretrained ResNet18 model
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model.load_state_dict(torch.load(model_path))
    return model
