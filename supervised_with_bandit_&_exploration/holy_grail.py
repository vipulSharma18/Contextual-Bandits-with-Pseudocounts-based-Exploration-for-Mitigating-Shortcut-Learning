import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import torch 
import torch.nn as nn
import torch.optim as optim 
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import pandas as pd
import time

from supervised_learning import set_seed, create_model
from contrastive_learning import load_context_gen
from image_augmenter import PatchOperations
from bandit.neural_bandit import ContextualBanditWithPseudoCounts
import wandb 


def calculate_accuracy(outputs, labels, threshold=0.5):
    binary_predictions = torch.where(outputs >= threshold, torch.tensor(1.0), torch.tensor(0.0))
    correct = torch.sum(binary_predictions == labels)
    total = len(labels)
    return correct.item() / total * 100

def experiment(setting='0.9_5', seed=1): 
    set_seed(42)
    path = '../data/'+setting  #for red vs green task
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(root=f'{path}/train', transform=transform)
    val_dataset = datasets.ImageFolder(root=f'{path}/val', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=28, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=28, shuffle=False)
    z_dim, patch_size = 256, 224//4
    patchOps = PatchOperations(patch_size=patch_size, image_size=(224,224))
    #encoder paths
    encoder_path = '../contrastive_learning/model_weights/255_56_resnet18_'+str(setting)+'_'+str(seed)+'.pth'
    context_generator = load_context_gen(z_dim=256, patch_size=56, path = encoder_path).to(device)
    context_generator.eval()
    #bandit and cfn model
    num_coins, hidden_size, exploration_factor = 64, 512, 0.8
    bandit = ContextualBanditWithPseudoCounts(context_length=z_dim, hidden_size=hidden_size,\
        num_coins=num_coins, lambda_explore=exploration_factor).to(device)
    #supervised model
    sup_model = create_model().to(device)
    #loss and optim setup
    criterion_sup = nn.MSELoss()
    criterion_bandit = nn.MSELoss()
    criterion_cfn = nn.MSELoss() #bandit.pseudocount_gen
    
    lr_bandit, lr_sup, lr_cfn = 1e-02, 1e-02, 1e-02
    optim_sup = optim.Adam(sup_model.parameters(), lr=lr_sup)
    optim_bandit = optim.Adam(bandit.parameters(), lr=lr_bandit)
    optim_cfn = optim.Adam(bandit.pseudocount_gen.parameters(), lr=lr_cfn)
    
    #wandb setup
    epochs=5
    wandb.init(
        project="RL_Project_CSCI2951F", 
        config={
            'architecture': 'CombinedFrameworkResnetBackbone',
            'context_dim': z_dim, 
            'num_coins': num_coins, 
            'bandit_size': hidden_size,
            'exploration_factor': exploration_factor,
            'patch_size': patch_size,
            'setting': setting, 
            'task': 'red vs green',
            'lr_bandit': lr_bandit,
            'lr_sup': lr_sup,
            'lr_cfn': lr_cfn, 
            'seed': seed, 
            'epochs': epochs
        })
    for epoch in range(epochs): 
        sup_train_loss, bandit_train_loss, cfn_train_loss = 0, 0, 0
        bandit.exploration_rate_gen.reset()
        #times = {'patch': [], 'forward': [], 'logit_cal': [], 'back_q': [], 'back_k': []}
        batch_running_time = time.time()
        for i, (images, cls_label) in enumerate(train_loader): 
            start = time.time()
            optim_sup.zero_grad()
            optim_bandit.zero_grad()
            optim_cfn.zero_grad()
            K = patchOps.num_patches #patches per image
            #get patches
            
            #class conditioned bandits. concatenating 1-hot class label to state context.
            
            #send context to bandit.
            context = None
            bandit_output, pseudocount_output, coin_label = bandit(context)
            #pick patch with max probability -> 
            
            #reconstruct images based on action
            transformed_images = None
            #supervised forward-backward
            logits = sup_model(transformed_images)
            loss_sup = criterion_sup(logits, labels)
            loss_sup.backward()
            optim_sup.step()
            sup_train_loss += loss_sup.item()
            #bandit backward
            logits, labels = [], []
            loss_bandit = criterion_bandit(logits, labels)
            loss_bandit.backward()
            optim_bandit.step()
            bandit.exploration_rate_gen.step()
            train_loss_sup += loss.item()
            cfn_train_loss
            bandit_train_loss
            sup_train_loss
            if i%10 == 0: 
                print(f"Epoch {epoch}, Batch {i}, train loss:{loss.item()}, Elapsed time for epoch : {(time.time() - batch_running_time)/60}")
                #print("Batch Time statistics", end=": ")
                #for k in times: 
                #    print(k, ":", np.mean(times[k]), end=",", sep="")
                #print()
                wandb.log({'train_batch':i, 'train_batch_loss': loss.item()})
        train_loss /= len(train_loader)
        #validation
        val_loss= 0
        q_enc.eval()
        k_enc.eval()
        with torch.no_grad():
            for i, (images, cls_label) in enumerate(val_loader): 
                queries, keys = patchOps.query_key(images.to(device)) #batch_dim, num_patches, 3, h, w
                del images
                z_q = q_enc(queries) #B*K, z_dim
                z_k = k_enc(keys) #B*K, z_dim
                K = patchOps.num_patches #patches per image
                logits, labels = [], []
                for j in range(len(cls_label)): 
                    logits.append(torch.mm(z_q[j*K:j*K+K], z_k[j*K:j*K+K].t())) #K,K -> diagonals are positive pairs
                    labels.append(torch.arange(K, dtype=torch.long, device=device))
                logits = torch.cat(logits, dim=0)
                labels = torch.cat(labels, dim=0)
                #print(keys.size(), logits.size(), labels.size(), K)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                if i%10 == 0: 
                    print(f"Batch {i}, val loss:{loss.item()}")
                    wandb.log({'val_batch':i, 'val_batch_loss': loss.item()})
            val_loss /= len(val_loader)
        print(f"Epoch {epoch}, Train Loss:{train_loss}, Val loss:{val_loss}")
        wandb.log({'Epoch': epoch, 'Train Loss':train_loss, 'Val Loss':val_loss})
        if val_loss<best_val_loss: 
            best_val_loss = val_loss
            torch.save(q_enc.state_dict(), 'model_weights/'+str(z_dim)+'_'+str(patch_size)+'_resnet18_'+setting+'_'+str(seed)+'.pth')
    wandb.finish()
    del q_enc, k_enc



if __name__=='__main__': 
    experiment('0.9_5', 1)
    '''
    settings = ['0.6_1', '0.6_2', '0.6_3', '0.6_4', '0.6_5', '0.7_1', '0.7_2', '0.7_3', '0.7_4', '0.7_5', '0.8_1', '0.8_2', '0.8_3', '0.8_4', '0.8_5', '0.9_1', '0.9_2', '0.9_3', '0.9_4', '0.9_5']
    #settings = ['0.8_1', '0.8_2', '0.8_3', '0.8_4', '0.8_5', '0.9_1', '0.9_2', '0.9_3', '0.9_4', '0.9_5']
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