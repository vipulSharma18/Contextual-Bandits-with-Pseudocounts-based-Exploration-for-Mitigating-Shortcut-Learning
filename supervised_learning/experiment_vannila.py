import sys
import os

# Determine the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

import torch.nn as nn
import torch.optim as optim
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
from .supervised_model import create_model, set_seed
import wandb

def calculate_accuracy(outputs, labels, threshold=0.5):
    binary_predictions = torch.where(outputs >= threshold, torch.tensor(1.0), torch.tensor(0.0))
    correct = torch.sum(binary_predictions == labels)
    total = len(labels)
    return correct.item() / total * 100

def experiment(setting='0.9_3', seed=42, additional_transform=None): 
    set_seed(seed)
    #path = '../data/greyscale/'+setting  #for greyscale task
    path = '../data/'+setting  #for red vs green task
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model()
    model.to(device)
    transform_list = [transforms.ToTensor()]
    if additional_transform is not None: 
        transform_list.insert(0, additional_transform) 
    transform = transforms.Compose(transform_list)
    #load data
    train_dataset = datasets.ImageFolder(root=f'{path}/train', transform=transform)
    val_dataset = datasets.ImageFolder(root=f'{path}/val', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    
    #loss and optim setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-02)
    
    best_acc = 0
    #wandb setup
    wandb.init(
        project="RL_Project_CSCI2951F", 
        config={
            'architecture': 'ResNet18',
            'setting': setting, 
            'seed': seed,
            'task': 'red vs green'
        })
    for epoch in range(100): 
        #train loop
        model.train()
        train_loss, train_accuracy = 0, 0
        for i, (images, labels) in enumerate(train_loader): 
            labels = labels.to(device).float() #either 0 or 1 based on the folder names, can be converted to -1 and 1 to increase loss and learning
            #labels = torch.where(labels > 0, torch.tensor(1.0), torch.tensor(-1.0))
            optimizer.zero_grad()
            outputs = model(images.to(device)).squeeze() #logits of a linear layer without any activation
            # the outputs are thresholded at 0.5 to get 1 or 0 label prediction
            #print(labels.size(), outputs.size())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            accuracy = calculate_accuracy(outputs, labels)
            train_accuracy += accuracy
            if i%10 == 0: 
                print(f"Batch {i}, train loss:{loss.item()}, train accuracy:{accuracy}")
            wandb.log({'train_batch':i, 'train_batch_loss': loss.item(), 'train_batch_accuracy': accuracy})
        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        #validation
        val_loss, val_accuracy = 0, 0
        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader): 
                labels = labels.to(device).float()
                #labels = torch.where(labels > 0, torch.tensor(1.0), torch.tensor(-1.0))
                outputs = model(images.to(device)).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                accuracy = calculate_accuracy(outputs, labels)
                val_accuracy += accuracy
                if i%10 == 0: 
                    print(f"Batch {i}, val loss:{loss.item()}, val accuracy:{accuracy}")
                wandb.log({'val_batch':i, 'val_batch_loss': loss.item(), 'val_batch_accuracy': accuracy})
        #pick best model and save it
        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        print(f"Epoch {epoch}, Train Loss:{train_loss}, Train Acc:{train_accuracy}, Val loss:{val_loss}, Val acc:{val_accuracy}")
        wandb.log({'Epoch': epoch, 'Train Loss':train_loss, 'Train Acc':train_accuracy, 'Val Loss':val_loss, 'Val Acc':val_accuracy})
        if val_accuracy>best_acc: 
            torch.save(model.state_dict(), 'results_vanilla_'+setting+'_'+str(seed)+'.pth')
            best_acc=val_accuracy
        if best_acc>=99.75 and train_accuracy>99: 
            print("Validation Accuracy above 99.8, no further training required")
            break
    wandb.finish()
    del model


if __name__=='__main__': 
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