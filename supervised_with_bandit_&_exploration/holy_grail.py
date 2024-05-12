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

from supervised_learning import set_seed, create_model
from contrastive_learning import load_context_gen
from image_augmenter import PatchOperations
from bandit.neural_bandit import ContextualBanditWithPseudoCounts
import wandb 


def calculate_accuracy(outputs, labels, threshold=0.5):
    binary_predictions = torch.where(outputs >= threshold, torch.tensor(1.0), torch.tensor(0.0))
    correct = torch.sum(binary_predictions == labels)
    total = len(labels)
    #print(outputs.size(), labels.size(), (binary_predictions==labels).size(), correct, total)
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
    encoder_path = '../contrastive_learning/model_weights/256_56_resnet18_'+str(setting)+'_'+str(seed)+'.pth'
    context_generator = load_context_gen(z_dim=256, patch_size=56, path = encoder_path).to(device)
    context_generator.eval()
    #bandit and cfn model
    #+2 in context length for the 1-hot vector of class
    num_coins, hidden_size, exploration_factor = 64, 512, 0.8
    bandit = ContextualBanditWithPseudoCounts(context_length=z_dim+2, hidden_size=hidden_size,\
        num_coins=num_coins, lambda_explore=exploration_factor).to(device)
    #supervised model
    sup_model = create_model().to(device)
    #loss and optim setup
    criterion_sup = nn.MSELoss()
    criterion_bandit = nn.MSELoss()
    criterion_cfn = nn.MSELoss() #bandit.pseudocount_gen
    
    lr_bandit, lr_sup, lr_cfn = 0.05, 1e-02, 0.05
    optim_sup = optim.Adam(sup_model.parameters(), lr=lr_sup)
    optim_bandit = optim.Adam(bandit.parameters(), lr=lr_bandit)
    optim_cfn = optim.Adam(bandit.pseudocount_gen.parameters(), lr=lr_cfn)
    
    #wandb setup
    epochs=30
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
        cumulative_bandit_reward = 0
        sup_train_loss, bandit_train_loss, cfn_train_loss, train_acc = 0, 0, 0, 0
        bandit.exploration_rate_gen.reset()
        K = patchOps.num_patches #patches per image
        bandit.train()
        sup_model.train()
        bandit.pseudocount_gen.train()
        batch_running_time = time.time()
        for i, (images, cls_label) in enumerate(train_loader): 
            cls_label = cls_label.to(device)
            images = images.to(device)
            optim_sup.zero_grad()
            optim_bandit.zero_grad()
            optim_cfn.zero_grad()
            #get patches -> done
            patches = patchOps.extract_patches(images)
            #get context for each patch -> done
            with torch.no_grad(): 
                context = context_generator(patches.reshape(-1,3,patchOps.patch_size, patchOps.patch_size))
                #class conditioned bandits. concatenating 1-hot class label to state context. -> done
                context = torch.cat((context, one_hot(cls_label.repeat_interleave(patchOps.num_patches), num_classes=2)), dim=-1)
            #send context to bandit. -> done
            bandit_output, pseudocount_output, coin_label = bandit(context)
            #pick patch with max probability -> done
            bandit_output = torch.reshape(bandit_output, (-1, K)) #batch_dim, K num of patches
            bandit_logits, patch_idx = torch.max(bandit_output, dim=-1)
            #reconstruct images based on action -> done
            transformed_images = patchOps.mask_images(images, patch_idx) #only pick image patch given by patch_idx, rest are cyan
            #supervised forward-backward -> done
            logits = sup_model(transformed_images)
            loss_sup = criterion_sup(logits, cls_label.unsqueeze(-1).type(torch.float))
            loss_sup.backward()
            optim_sup.step()
            b_acc = calculate_accuracy(logits, cls_label.unsqueeze(-1))
            train_acc += b_acc
            sup_train_loss += loss_sup.item()
            #bandit backward -> done
            sup_logits = logits.detach().squeeze(-1)
            bandit_labels = torch.where(cls_label==0, 1-sup_logits, sup_logits) #prob of the correct class
            cumulative_bandit_reward += torch.sum(bandit_labels).item()
            loss_bandit = criterion_bandit(bandit_logits, bandit_labels)
            loss_bandit.backward()
            optim_bandit.step()
            bandit_train_loss += loss_bandit.item()
            #cfn backward -> done
            #print(coin_label.size())
            loss_cfn = criterion_cfn(pseudocount_output, coin_label.to(device).type(torch.float))
            loss_cfn.backward()
            optim_cfn.step()
            cfn_train_loss += loss_cfn.item()
            #exploration rate exponential decay-> done
            bandit.exploration_rate_gen.step()
            #logging -> done
            if i%10 == 0: 
                print(f"Epoch {epoch}, Train: Batch {i}, train acc:{b_acc}, sup loss:{loss_sup.item()}, bdit loss: {loss_bandit.item()}, cfn loss: {loss_cfn.item()}, cumulative bdit reward: {cumulative_bandit_reward}, Elapsed time for epoch : {(time.time() - batch_running_time)/60}")
                #print("Batch Time statistics", end=": ")
                #for k in times: 
                #    print(k, ":", np.mean(times[k]), end=",", sep="")
                #print()
                wandb.log({'train_batch':i, 'b train sup acc': b_acc, 'b train sup loss':loss_sup.item(), 'b train bdit loss':loss_bandit.item(), \
                    'b train cfn loss':loss_cfn.item(), 'b train cumulative bdit reward':cumulative_bandit_reward})
        sup_train_loss /= len(train_loader)
        bandit_train_loss /= len(train_loader) #1 image = 1 context/state
        cfn_train_loss /= (len(train_loader)*K) #1 image = 16 or K states/patches to explore
        train_acc /= len(train_loader)
        #validation
        cumulative_bandit_reward = 0
        sup_val_loss, bandit_val_loss, cfn_val_loss, val_acc = 0, 0, 0, 0
        bandit.exploration_rate_gen.exploit() #make exploration factor =0, i.e., pure exploitation
        bandit.eval()
        sup_model.eval()
        bandit.pseudocount_gen.eval()
        with torch.no_grad():
            for i, (images, cls_label) in enumerate(val_loader): 
                cls_label = cls_label.to(device)
                images = images.to(device)
                patches = patchOps.extract_patches(images)
                with torch.no_grad(): 
                    context = context_generator(patches.reshape(-1,3,patchOps.patch_size, patchOps.patch_size))
                    #class conditioned bandits. concatenating 1-hot class label to state context. -> done
                    context = torch.cat((context, one_hot(cls_label.repeat_interleave(patchOps.num_patches), num_classes=2)), dim=-1)
                bandit_output, pseudocount_output, coin_label = bandit(context)
                #pick patch with max probability -> done
                bandit_output = torch.reshape(bandit_output, (-1, K)) #batch_dim, K num of patches
                bandit_logits, patch_idx = torch.max(bandit_output, dim=-1)
                #reconstruct images based on action -> done
                transformed_images = patchOps.mask_images(images, patch_idx) #only pick image patch given by patch_idx, rest are cyan
                #supervised forward-backward -> done
                logits = sup_model(transformed_images)
                loss_sup = criterion_sup(logits, cls_label.unsqueeze(-1).type(torch.float))
                sup_val_loss += loss_sup.item()
                b_acc = calculate_accuracy(logits, cls_label.unsqueeze(-1))
                val_acc += b_acc
                #bandit backward -> done
                sup_logits = logits.detach().squeeze(-1)
                bandit_labels = torch.where(cls_label==0, 1-sup_logits, sup_logits) #prob of the correct class
                cumulative_bandit_reward += torch.sum(bandit_labels).item()
                loss_bandit = criterion_bandit(bandit_logits, bandit_labels)
                bandit_val_loss += loss_bandit.item()
                #cfn backward -> done
                loss_cfn = criterion_cfn(pseudocount_output, coin_label.to(device).type(torch.float))
                cfn_val_loss += loss_cfn.item()
                if i%10 == 0: 
                    print(f"Epoch {epoch}, Val: Batch {i}, acc:{b_acc},sup loss:{loss_sup.item()}, bdit loss: {loss_bandit.item()}, cfn loss: {loss_cfn.item()}, cumulative bdit reward: {cumulative_bandit_reward}, Elapsed time for epoch : {(time.time() - batch_running_time)/60}")
                    wandb.log({'val_batch':i,'b val sup acc': b_acc, 'b val sup loss':loss_sup.item(), 'b val bdit loss':loss_bandit.item(), \
                    'b val cfn loss':loss_cfn.item(), 'b val cumulative bdit reward':cumulative_bandit_reward})
            sup_val_loss /= len(val_loader)
            bandit_val_loss /= len(val_loader) #1 image = 1 context/state
            cfn_val_loss /= (len(val_loader)*K) #1 image = 16 or K states/patches to explore
            val_acc /= len(val_loader)
        print(f"Epoch {epoch}, Sup Train Loss:{sup_train_loss}, Sup Val loss:{sup_val_loss}", end=" ")
        print(f"Sup Train Acc:{train_acc}, Sup Val Acc:{val_acc}", end=" ")
        print(f"Bandit Train Loss:{bandit_train_loss}, Bandit Val loss:{bandit_val_loss}", end=" ")
        print(f"CFN Train Loss:{cfn_train_loss}, CFN Val loss:{cfn_val_loss}")
        wandb.log({'Epoch': epoch, 'Sup Train Loss':sup_train_loss, 'Sup Val Loss':sup_val_loss,\
            "Sup Train Acc": train_acc, "Sup Val Acc": val_acc, \
                "Bandit Train Loss":bandit_train_loss, "Bandit Val loss":bandit_val_loss, \
                    "CFN Train Loss":cfn_train_loss, "CFN Val loss":cfn_val_loss})
        torch.save(bandit.state_dict(), 'model_weights/bandit_'+setting+'_'+str(seed)+'.pth')
        torch.save(sup_model.state_dict(), 'model_weights/supervised_'+setting+'_'+str(seed)+'.pth')
    wandb.finish()


#0.25 mins per epoch, 10 epochs per model -> 1 model = 2.5 mins
# 20 settings*5 seeds -> 100 models -> 250 mins -> 4.16 hrs -> to be safe 5 hrs reserved
#splitting into 2 -> 2.5 hrs 
#*3 as 30 epochs -> 8 hrs worst case. best case 6 hrs.

if __name__=='__main__': 
    #experiment('0.9_5', seed=1)
    settings = ['0.6_1', '0.6_2', '0.6_3', '0.6_4', '0.6_5', '0.7_1', '0.7_2', '0.7_3', '0.7_4', '0.7_5', '0.8_1', '0.8_2', '0.8_3', '0.8_4', '0.8_5', '0.9_1', '0.9_2', '0.9_3', '0.9_4', '0.9_5']
    #settings = ['0.8_1', '0.8_2', '0.8_3', '0.8_4', '0.8_5', '0.9_1', '0.9_2', '0.9_3', '0.9_4', '0.9_5']
    print("Running for settings:", settings)
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