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
import torch 
import torch.optim as optim 
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader
from PIL import Image
import wandb # type: ignore
import gc
import numpy as np

from supervised_learning import set_seed
from image_augmenter import PatchOperations, RemoveBackgroundTransform
from encoder import ConvNetEncoder
import time

@torch.no_grad()
def momentum_update(k_enc, q_enc, beta=0.999): 
    for (name_k,param_k), (name_q,param_q) in zip(k_enc.named_parameters(), q_enc.named_parameters()): 
        param_k.data = beta*param_k.data + (1.0 - beta) * param_q.data
        #wandb.log({f"query_grad/{name_q}": wandb.Histogram(param_q.grad.cpu().detach().numpy())})
        
def experiment(setting='0.9_1', seed=42): 
    path = '../data/'+setting  #for red vs green task
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(42)
    gc.collect()
    #load data
    #1 possible data augmentation
        #remove background cyan, make it (0,0,0) to remove excessive noise in the data.
        #background (0, 255, 255), data (x,0,0) class 0 or (0,x,0) class 1
    transform = transforms.Compose([RemoveBackgroundTransform(),transforms.ToTensor()])
    #transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(root=f'{path}/train', transform=transform)
    val_dataset = datasets.ImageFolder(root=f'{path}/val', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    z_dim, patch_size = 256, 224//4
    patchOps = PatchOperations(patch_size=patch_size, image_size=(224,224))
    q_enc = ConvNetEncoder(z_dim, patch_size=patch_size)
    k_enc = ConvNetEncoder(z_dim, patch_size=patch_size)
    q_enc.to(device)
    k_enc.to(device)
    #loss and optim setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(q_enc.parameters(), lr=0.01)
    #wandb setup
    wandb.init(
        project="RL_Project_CSCI2951F", 
        config={
            'architecture': 'ConvEncoder_'+str(z_dim),
            'setting': setting, 
            'task': 'red vs green'
        })
    for epoch in range(10): 
        #train loop
        q_enc.train()
        k_enc.train()
        train_loss = 0
        times = {'patch': [], 'forward': [], 'logit_cal': [], 'back_q': [], 'back_k': []}
        batch_running_time = time.time()
        for i, (images, cls_label) in enumerate(train_loader): 
            start = time.time()
            optimizer.zero_grad()
            queries, keys = patchOps.query_key(images.to(device)) #batch_dim, num_patches, 3, h, w
            times['patch'].append(time.time()-start)
            start = time.time()
            z_q = q_enc(queries) #B*K, z_dim
            z_k = k_enc(keys) #B*K, z_dim
            z_k = z_k.detach()
            K = patchOps.num_patches #patches per image
            times['forward'].append(time.time()-start)
            start = time.time()
            logits, labels = [], []
            #oprint(keys.size(), K, len(cls_label), z_k.size(), z_q.size())
            for j in range(len(cls_label)): 
                logits.append(torch.mm(z_q[j*K:j*K+K], z_k[j*K:j*K+K].t())) #K,K -> diagonals are positive pairs
                labels.append(torch.arange(K, dtype=torch.long, device=device))
            logits = torch.cat(logits, dim=0)
            labels = torch.cat(labels, dim=0)
            #print(keys.size(), logits.size(), labels.size(), K)
            times['logit_cal'].append(time.time()-start)
            start = time.time()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            times['back_q'].append(time.time()-start)
            start = time.time()
            momentum_update(k_enc, q_enc, beta=0.999)
            times['back_k'].append(time.time()-start)
            train_loss += loss.item()
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
                z_q = q_enc(queries) #B*K, z_dim
                z_k = k_enc(keys) #B*K, z_dim
                z_k = z_k.detach()
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
    torch.save(q_enc.state_dict(), 'model_weights/16_4/'+'contrastive_encoder_'+setting+'.pth')
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