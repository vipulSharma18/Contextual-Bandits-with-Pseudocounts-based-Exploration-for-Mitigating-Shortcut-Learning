'''
anchor/query is a patch from crop. 
key/positives and negatives are augmented patches 
'''
import sys
import os
# Determine the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to sys.path
sys.path.append(parent_dir)

import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch # type: ignore
from torchvision import datasets, transforms # type: ignore
from torch.utils.data import DataLoader # type: ignore
import pandas as pd # type: ignore
from supervised_learning import set_seed
import wandb # type: ignore
import numpy as np # type: ignore

from image_augmenter import PatchOperations
from encoder import ConvNetEncoder

@torch.no_grad()
def momentum_update(k_enc, q_enc, beta=0.999): 
    for param_k, param_q in zip(k_enc.parameters(), q_enc.parameters()): 
        param_k.data = beta*param_k.data + (1.0 - beta) * param_q.data

def experiment(setting='0.9_3', seed=42): 
    set_seed(seed)
    
    path = '../data/'+setting  #for red vs green task
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #load data
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(root=f'{path}/train', transform=transform)
    val_dataset = datasets.ImageFolder(root=f'{path}/val', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    patchOps = PatchOperations(patch_size=56, image_size=(224,224))
    
    z_dim, num_layers = 16, 4
    q_enc = ConvNetEncoder(z_dim, num_layers)
    k_enc = ConvNetEncoder(z_dim, num_layers)
    q_enc.to(device)
    k_enc.to(device)
    #Bilinear product similarity in the latent space as done in CURL
    W = torch.rand(z_dim, z_dim, requires_grad=True).to(device)
    W = torch.nn.Parameter(W)
    #loss and optim setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(q_enc.parameters(), lr=0.001)
    optimizer_W = optim.SGD([W], lr=0.001)
    k_enc.eval()
    
    #wandb setup
    wandb.init(
        project="RL_Project_CSCI2951F", 
        config={
            'architecture': 'ConvEncoder_'+str(z_dim)+'_'+str(num_layers),
            'setting': setting, 
            'seed': seed, 
            'task': 'red vs green'
        })
    for epoch in range(100): 
        #train loop
        q_enc.train()
        k_enc.load_state_dict(q_enc.state_dict())
        train_loss = 0
        for i, (images, labels) in enumerate(train_loader): 
            images = images.to(device)
            queries, keys = patchOps.query_key(images)
            queries.to(device)
            keys.to(device)
            optimizer.zero_grad()
            optimizer_W.zero_grad()
            z_q = q_enc(queries)
            with torch.no_grad():
                z_k = k_enc(keys)
                z_k = z_k.detach()
            K = patchOps.num_patches #sub-batch/patches per image
            B = images.size(0) #total num of images
            proj_k = torch.matmul(W, z_k.T) #(z_dim,z_dim) x z_dim,(B*K) -> (z_dim, B*K)
            logits = torch.matmul(z_q, proj_k) #(B*K),z_dim x z_dim,(B*K) -> (B*K, B*K)
            labels = torch.arange(K).repeat(B).to(device) #class indices, i.e., each query should only match with its corresponding key
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer_W.step() 
            momentum_update(k_enc, q_enc, beta=0.999)
            train_loss += loss.item()
            if i%10 == 0: 
                print(f"Batch {i}, train loss:{loss.item()}")
            wandb.log({'train_batch':i, 'train_batch_loss': loss.item()})
        train_loss /= len(train_loader)
        #validation
        val_loss= 0
        q_enc.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader): 
                images = images.to(device)
                queries, keys = patchOps.query_key(images)
                queries.to(device)
                keys.to(device)
                z_q = q_enc(queries)
                z_k = k_enc(keys)
                z_k = z_k.detach()
                K = patchOps.num_patches #sub-batch/patches per image
                B = images.size(0) #total num of images
                proj_k = torch.matmul(W, z_k.T) #(z_dim,z_dim) x z_dim,(B*K)
                logits = torch.matmul(z_q, proj_k) #(B*K),z_dim x z_dim,(B*K)
                labels = torch.arange(K).repeat(B).to(device) #class indices, i.e., each query should only match with its corresponding key
                loss = criterion(logits, labels)
                val_loss += loss.item()
                if i%10 == 0: 
                    print(f"Batch {i}, val loss:{loss.item()}")
                wandb.log({'val_batch':i, 'val_batch_loss': loss.item()})
        val_loss /= len(val_loader)
        print(f"Epoch {epoch}, Train Loss:{train_loss}, Val loss:{val_loss}")
        wandb.log({'Epoch': epoch, 'Train Loss':train_loss, 'Val Loss':val_loss})
    torch.save(q_enc.state_dict(), 'model_weights/16_4/'+'contrastive_encoder_'+setting+'_'+str(seed)+'.pth')
    torch.save(W, 'model_weights/'+str(z_dim)+'_'+str(num_layers)+'/'+'W_'+setting+'_'+str(seed)+'.pth')
    wandb.finish()
    del q_enc, k_enc


if __name__=='__main__': 
    experiment()
    '''
    settings = ['0.6_1', '0.6_2', '0.6_3', '0.6_4', '0.6_5', '0.7_1', '0.7_2', '0.7_3', '0.7_4', '0.7_5', '0.8_1', '0.8_2', '0.8_3', '0.8_4', '0.8_5', '0.9_1', '0.9_2', '0.9_3', '0.9_4', '0.9_5']
    for setting in settings: 
        print("==========================================================================")
        print("Running experiment for setting", setting)
        print("==========================================================================")
        for seed in [1,42,89,23,113]:
            print("Running for seed", seed, "of experiment", setting) 
            experiment(setting, seed)
            print("Seed completed execution!", seed, setting)
            print("------------------------------------------------------------------")
        print("Experiment complete", setting)
    '''