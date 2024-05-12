import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import torch 
import torch.nn as nn
import torch.optim as optim 
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
from PIL import Image
import numpy as np
import pandas as pd
import time

from supervised_learning import set_seed, load_model
from contrastive_learning import load_context_gen
from image_augmenter import PatchOperations
from bandit.neural_bandit import load_bandit
import wandb 


#pick an image, save it, save its patches, save the embeddings for the patches and show it
#save the outputs of the bandits and the patches for those outputs

def save_tensor_to_image(patches, setting):
    return None
def sample(setting, seed=42): 
    path = '../data/'+setting  #for red vs green task
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(root=f'{path}/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    z_dim, patch_size = 256, 224//4
    patchOps = PatchOperations(patch_size=patch_size, image_size=(224,224))
    encoder_path = '../contrastive_learning/model_weights/256_56_resnet18_'+str(setting)+'_'+str(seed)+'.pth'
    context_generator = load_context_gen(z_dim=256, patch_size=56, path = encoder_path).to(device)
    context_generator.eval()
    num_coins, hidden_size, exploration_factor = 64, 512, 0.8
    #load bandit model
    bandit_path = 'model_weights/bandit_' + setting + '_' + str(seed) + '.pth'
    bandit = load_bandit(bandit_path, context_length=z_dim+2, hidden_size=hidden_size,\
        num_coins=num_coins, lambda_explore=exploration_factor).to(device)
    #load supervised model
    resnet_weights = 'model_weights/'
    resnet_template = 'supervised_' #0.6_1_1.pth p_s a_s seed
    sup_model = load_model(resnet_weights+resnet_template+setting+'_'+str(seed)+'.pth').to(device)
    bandit.exploration_rate_gen.exploit()
    K = patchOps.num_patches #patches per image
    bandit.eval()
    sup_model.eval()
    bandit.pseudocount_gen.eval()
    output_path = 'qualitative_check_data/'+setting
    os.makedirs(output_path, exist_ok=True)
    with torch.no_grad():
        for images, cls_label in train_loader:
            cls_label = cls_label.to(device)
            images = images.to(device)
            patches = patchOps.extract_patches(images)
            context = context_generator(patches.reshape(-1,3,patchOps.patch_size, patchOps.patch_size))
            context = torch.cat((context, one_hot(cls_label.repeat_interleave(patchOps.num_patches), num_classes=2)), dim=-1)
            torch.set_printoptions(threshold=10_000)
            print('context generated + clas label:', torch.sum(context, dim=-1))
            bandit_output, _, _ = bandit(context)
            bandit_output = torch.reshape(bandit_output, (-1, K)) #batch_dim, K num of patches
            print('bandit output:', bandit_output)
            bandit_logits, patch_idx = torch.max(bandit_output, dim=-1)
            #print('max bandit logit/action and patch idx', bandit_logits, patch_idx)
            transformed_images = patchOps.mask_images(images, patch_idx) #only pick image patch given by patch_idx, rest are cyan
            logits_bandit = sup_model(transformed_images)
            logit_vanilla = sup_model(images)
            print('output of supervised model/actual reward:', logits_bandit, 'total image reward:', logit_vanilla, 'actual class:', cls_label)
            break

if __name__=='__main__': 
    settings = ['0.6_1', '0.6_2', '0.6_3', '0.6_4', '0.6_5', '0.7_1', '0.7_2', '0.7_3', '0.7_4', '0.7_5', '0.8_1', '0.8_2', '0.8_3', '0.8_4', '0.8_5', '0.9_1', '0.9_2', '0.9_3', '0.9_4', '0.9_5']
    #settings = ['0.6_1']
    for setting in settings:
        print("Running for setting:", setting)
        sample(setting) 
        print("------------------------------------------------")