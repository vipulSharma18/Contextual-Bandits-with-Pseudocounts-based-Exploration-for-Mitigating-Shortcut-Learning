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

from supervised_learning import set_seed
from image_augmenter import PatchOperations, RemoveBackgroundTransform
from encoder import ConvNetEncoder

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

@torch.no_grad()
def momentum_update(k_enc, q_enc, beta=0.999): 
    for (name_k,param_k), (name_q,param_q) in zip(k_enc.named_parameters(), q_enc.named_parameters()): 
        param_k.data = beta*param_k.data + (1.0 - beta) * param_q.data
        #wandb.log({f"query_grad/{name_q}": wandb.Histogram(param_q.grad.cpu().detach().numpy())})

def experiment(rank, world_size, setting='0.9_1', seed=42): 
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    path = '../data/'+setting  #for red vs green task
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', rank)
    set_seed(42)
    transform = transforms.Compose([RemoveBackgroundTransform(),transforms.ToTensor()])
    
    train_dataset = datasets.ImageFolder(root=f'{path}/train', transform=transform)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=20, sampler=train_sampler, num_workers = 4)
    #validation will be checked only on 1 process
    val_dataset = datasets.ImageFolder(root=f'{path}/val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False, num_workers=4)
    
    patchOps = PatchOperations(patch_size=224//4, image_size=(224,224))
    z_dim = 256
    q_enc = ConvNetEncoder(z_dim, patch_size=224//4)
    k_enc = ConvNetEncoder(z_dim, patch_size=224//4)
    q_enc.to(device)
    q_enc = DDP(q_enc, device_ids=[rank])
    k_enc.to(device)
    #loss and optim setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(q_enc.parameters(), lr=0.1)
    #wandb setup
    #do only from 1 process: 
    if rank==0: 
        wandb.init(
            project="RL_Project_CSCI2951F", 
            config={
                'architecture': 'ConvEncoder_'+str(z_dim),
                'setting': setting, 
                'task': 'red vs green'
            })
    dist.barrier()
    for epoch in range(10): 
        #train loop
        q_enc.train()
        k_enc.train()
        train_loss = 0
        for i, (images, cls_label) in enumerate(train_loader): 
            optimizer.zero_grad()
            queries, keys = [], []
            for img_idx, image in enumerate(images): 
                q, k = patchOps.query_key(image) #num_patches, 3, height, width
                queries.append(q)
                keys.append(k)
            keys = torch.cat(keys, dim=0).to(device)
            queries = torch.cat(queries, dim=0).to(device)
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
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            #block rank 1 and only do for rank 0. query encoder gradients should be in sync by now after optimizer.step
            if rank==0: 
                momentum_update(k_enc, q_enc, beta=0.99)
                if i%10 == 0: 
                    print(f"Batch {i}, train loss:{loss.item()}")
                wandb.log({'train_batch':i, 'train_batch_loss': loss.item()})
            dist.barrier() #to make sure k_enc is updated in both the processes
        #Do validation data checks only for 1 process: 
        if rank==0: 
            train_loss /= len(train_loader)
            #validation
            val_loss= 0
            q_enc.eval()
            k_enc.eval()
            with torch.no_grad():
                for i, (images, cls_label) in enumerate(val_loader): 
                    queries, keys = [], []
                    for img_idx, image in enumerate(images): 
                        q, k = patchOps.query_key(image) #num_patches, 3, height, width
                        queries.append(q)
                        keys.append(k)
                    keys = torch.cat(keys, dim=0).to(device)
                    queries = torch.cat(queries, dim=0).to(device)
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
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
                    if i%10 == 0: 
                        print(f"Batch {i}, val loss:{loss.item()}")
                    wandb.log({'val_batch':i, 'val_batch_loss': loss.item()})
                val_loss /= len(val_loader)
            print(f"Epoch {epoch}, Train Loss:{train_loss}, Val loss:{val_loss}")
            wandb.log({'Epoch': epoch, 'Train Loss':train_loss, 'Val Loss':val_loss})
    #do only from rank0 proc
    if rank==0: 
        torch.save(q_enc.state_dict(), 'model_weights/16_4/'+'contrastive_encoder_'+setting+'.pth')
        wandb.finish()
    del q_enc, k_enc
    cleanup()

def setup(rank, world_size): 
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

def cleanup(): 
    dist.destroy_process_group()

if __name__=='__main__': 
    world_size = torch.cuda.device_count()
    mp.spawn(experiment, args=(world_size,), nprocs=world_size)

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