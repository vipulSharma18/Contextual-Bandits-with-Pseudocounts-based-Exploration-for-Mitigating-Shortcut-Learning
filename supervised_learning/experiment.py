import torch.nn as nn
import torch.optim as optim
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
from supervised_model import create_model, set_seed
import wandb

def calculate_accuracy(outputs, labels, threshold=0.5):
    binary_predictions = torch.where(outputs >= threshold, torch.tensor(1.0), torch.tensor(0.0))
    correct = torch.sum(binary_predictions == labels)
    total = len(labels)
    return correct.item() / total * 100

def experiment(setting='0.9_3', seed=42): 
    set_seed(seed)
    path = '../data/'+setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model()
    model.to(device)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    #load data
    train_dataset = datasets.ImageFolder(root=f'{path}/train', transform=transform)
    val_dataset = datasets.ImageFolder(root=f'{path}/val', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    
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
            'seed': seed
        })
    for epoch in range(100): 
        #train loop
        model.train()
        train_loss, train_accuracy = 0, 0
        for i, (images, labels) in enumerate(train_loader): 
            labels = labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(images.to(device)).squeeze()
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
                optimizer.zero_grad()
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
        if best_acc==100: 
            print("100% Validation Accuracy reached, no further training required")
            break
    wandb.finish()


if __name__=='__main__': 
    settings = ['0.6_1', '0.6_2', '0.6_3', '0.6_4', '0.6_5', '0.7_1', '0.7_2', '0.7_3', '0.7_4', '0.7_5', '0.8_1', '0.8_2', '0.8_3', '0.8_4', '0.8_5', '0.9_1', '0.9_2', '0.9_3', '0.9_4', '0.9_5']
    for setting in settings: 
        print("==========================================================================")
        print("Running experiment for setting", setting)
        print("==========================================================================")
        for seed in [1,42,89,23,113]:
            print("Running for seed", seed) 
            experiment(setting, seed)
            print("Seed completed execution!")
            print("------------------------------------------------------------------")
        print("Experiment complete")