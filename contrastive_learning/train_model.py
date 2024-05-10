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
    for (name_k,param_k), (name_q,param_q) in zip(k_enc.named_parameters(), q_enc.named_parameters()): 
        param_k.data = beta*param_k.data + (1.0 - beta) * param_q.data
        #wandb.log({f"query_grad/{name_q}": wandb.Histogram(param_q.grad.cpu().detach().numpy())})
        
def experiment(setting='0.9_5'): 
    path = '../data/'+setting  #for red vs green task
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #load data
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(root=f'{path}/train', transform=transform)
    val_dataset = datasets.ImageFolder(root=f'{path}/val', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    patchOps = PatchOperations(patch_size=56, image_size=(224,224))
    z_dim, num_layers = 256, 4
    q_enc = ConvNetEncoder(z_dim, num_layers, patch_size=56)
    k_enc = ConvNetEncoder(z_dim, num_layers, patch_size=56)
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
            'task': 'red vs green'
        })
    for epoch in range(10): 
        #train loop
        q_enc.train()
        train_loss = 0
        for i, (images, labels) in enumerate(train_loader): 
            batch_loss = 0
            for img_idx, image in enumerate(images): 
                queries, keys = patchOps.query_key(image.to(device)) #num_patches, 3, height, width
                optimizer.zero_grad()
                z_q = q_enc(queries) #K, z_dim
                z_k = k_enc(keys) #K, z_dim
                z_k = z_k.detach()
                K = patchOps.num_patches #sub-batch/patches per image
                #distance measure is L2 norm calculated by matrix multiplication
                logits = torch.mm(z_q, z_k.t()) #K,K -> diagonals are positive pairs
                labels = torch.arange(K, dtype=torch.long, device=device)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                momentum_update(k_enc, q_enc, beta=0.9)
                batch_loss += loss.item()
            batch_loss /= len(images)
            if i%10 == 0: 
                print(f"Batch {i}, train loss:{batch_loss}")
            #wandb.log({'train_batch':i, 'train_batch_loss': batch_loss})
            train_loss += batch_loss
        train_loss /= len(train_loader)
        #validation
        val_loss= 0
        q_enc.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader): 
                batch_loss = 0
                for img_idx, image in enumerate(images): 
                    queries, keys = patchOps.query_key(image.to(device)) 
                    z_q = q_enc(queries) #K, z_dim
                    z_k = k_enc(keys) #K, z_dim
                    z_k = z_k.detach()
                    K = patchOps.num_patches #sub-batch/patches per image
                    #distance measure is L2 norm calculated by matrix multiplication
                    logits = torch.mm(z_q, z_k.t()) #K,K -> diagonals are positive pairs
                    labels = torch.arange(K, dtype=torch.long, device=device)
                    loss = criterion(logits, labels)
                    batch_loss += loss.item()
                batch_loss /= len(images)
                if i%10 == 0: 
                    print(f"Batch {i}, val loss:{batch_loss}")
                #wandb.log({'val_batch':i, 'val_batch_loss': batch_loss})
                val_loss += batch_loss
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