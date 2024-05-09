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
import wandb # type: ignore

from supervised_learning import set_seed
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
    patchOps = PatchOperations(patch_size=224//2, image_size=(224,224))
    z_dim, num_layers = 16, 4
    q_enc = ConvNetEncoder(z_dim, num_layers, patch_size=224//2)
    k_enc = ConvNetEncoder(z_dim, num_layers, patch_size=224//2)
    q_enc.to(device)
    k_enc.to(device)
    #loss and optim setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(q_enc.parameters(), lr=0.9)
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
    for epoch in range(20): 
        #train loop
        q_enc.train()
        train_loss = 0
        for i, (images, labels) in enumerate(train_loader): 
            images = images.to(device)
            queries, keys = patchOps.query_key(images)  #for each image, return augmented list of patches
            queries.to(device)
            keys.to(device)
            optimizer.zero_grad()
            z_q = q_enc(queries) #B*K, z_dim
            with torch.no_grad():
                z_k = k_enc(keys) #B*K, z_dim
            K = patchOps.num_patches #sub-batch/patches per image
            B = images.size(0) #total num of images
            #distance measure is L2 norm calculated by matrix multiplication
            l_pos = torch.bmm(z_q.view(B*K, 1, z_dim), z_k.view(B*K, z_dim, 1)).squeeze(-1)
            l_neg = torch.mm(z_q, z_k.t()) 
            mask = torch.eye(B*K, device=device).bool()
            l_neg[mask] = float('-inf') #diagnoal elements are positive pair already accounted for in l_pos
            logits = torch.cat([l_pos, l_neg], dim=1)
            labels = torch.zeros(B*K, dtype=torch.long, device=device)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            momentum_update(k_enc, q_enc, beta=0.9)
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
                K = patchOps.num_patches #sub-batch/patches per image
                B = images.size(0) #total num of images
                #distance measure is L2 norm calculated by matrix multiplication
                l_pos = torch.bmm(z_q.view(B*K, 1, z_dim), z_k.view(B*K, z_dim, 1)).squeeze(-1) 
                l_neg = torch.mm(z_q, z_k.t()) 
                mask = torch.eye(B*K, device=device).bool()
                l_neg[mask] = float('-inf') #diagnoal elements are positive pair already accounted for in l_pos
                logits = torch.cat([l_pos, l_neg], dim=1)
                labels = torch.zeros(B*K, dtype=torch.long, device=device)
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