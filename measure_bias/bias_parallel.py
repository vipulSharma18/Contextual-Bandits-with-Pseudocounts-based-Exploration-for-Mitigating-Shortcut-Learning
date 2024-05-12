import sys
import os

# Determine the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from optimal_classifier import load_lda
from supervised_learning import load_model
from supervised_learning import set_seed

import wandb
import pandas as pd
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def reliance_tabular(model=None, data=None): 
    X,y = data[0]
    z_s, z_c = data[1]
    y_pred = model.predict(X) #either 1 or -1
    #print(y_pred.shape)
    reliance = np.mean(y_pred*(np.where(z_s>=0, 1, -1) - np.where(z_c>=0, 1, -1)))
    #print(reliance)
    return reliance

def reliance_image(model=None, data=None): 
    test_loader = data[0]
    z_s, z_c = data[1]
    y_pred = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad(): 
        for images, labels in test_loader: 
            outputs = model(images.to(device)).squeeze()
            y_pred.extend(outputs.tolist())
    y_pred = np.array(y_pred) #logit values, optimized 
    y_pred = np.where(y_pred>=0.5, 1, -1)  #thresholding the same as the training setup, i.e., at >=0.5, but the labels now 1 or -1
    #probably want to redo all the experiments with consistent thresholding and labels for classes. don't have the time as of now.
    #print(y_pred.shape)
    reliance = np.mean(y_pred*(np.where(z_s>=0, 1, -1) - np.where(z_c>=0, 1, -1)))
    #print(reliance)
    return reliance

def reliance(model=None, data=None, type=None): 
    '''
    reliance_M = E_z[ y_pred_M(z) (sign(z_s) - sign(z_c))]
    '''
    if type=='image': 
        return reliance_image(model, data)
    elif type=='table': 
        return reliance_tabular(model, data)

if __name__=='__main__': 
    optimal_weights = '../optimal_classifier/model_weights/'
    optimal_template = 'lda_model_' #0.6_2_1.joblib p_s a_s seed
    
    #weights for vanilla
    resnet_weights = '../supervised_learning/model_weights/'
    #template for vanilla
    resnet_template = 'results_vanilla_' #0.6_1_1.pth p_s a_s seed
    '''
    #weights for bandit based exploration augmented resent
    resnet_weights = '../supervised_with_bandit_&_exploration/model_weights/'
    #template for bandit
    resnet_template = 'supervised_' #0.6_1_1.pth p_s a_s seed
    '''
    print("Configs:", resnet_weights, resnet_template)
    data_path = '../data/'
    test = 'synthetic_test_data.csv'
    df = pd.read_csv(data_path+test)
    df['x'] = df['x'].apply(lambda x: [float(i) for i in x[1:-1].split(',')])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    settings = ['0.6_1', '0.6_2', '0.6_3', '0.6_4', '0.6_5', '0.7_1', '0.7_2', '0.7_3', '0.7_4', '0.7_5', '0.8_1', '0.8_2', '0.8_3', '0.8_4', '0.8_5', '0.9_1', '0.9_2', '0.9_3', '0.9_4', '0.9_5']
    #settings = ['0.8_5']
    #settings = ['0.9_1', '0.9_2', '0.9_3', '0.9_4', '0.9_5']
    
    for setting in settings: 
        print("==========================================================================")
        print("Measuring bias for setting", setting)
        print("==========================================================================")
        seeds = [1,42,89,23,113]
        for seed in seeds:
            print("Running for seed", seed, "of experiment", setting) 
            #model
            set_seed(seed)
            wandb.init(
                project="RL_Project_CSCI2951F", 
                config={
                'architecture': 'Bias Measurement with Vanilla Supervised Learning',
                #'architecture': 'Bias Measurement with bandits',
                'task': 'red vs green',
                'class_threshold': 0.5,
                'setting': setting, 
                'seed': seed
                })
            resnet = load_model(resnet_weights+resnet_template+setting+'_'+str(seed)+'.pth').to(device)
            resnet.eval()
            lda = load_lda(optimal_weights+optimal_template+setting+'_'+str(seed)+'.joblib')
            
            # tabular data
            s = setting.split('_')
            p_s, a_s = float(s[0]), int(s[1])
            curr_df = df[(df['p_s']==p_s) & (df['a_s']==a_s)]
            X_tab, y_tab = np.stack(curr_df['x']), curr_df['class_label']
            #print(X_tab.shape, y_tab.shape) sanity check
            
            #image data
            test_dataset = datasets.ImageFolder(root=f'{data_path+setting}/test', transform=transform)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            #print(len(test_loader)) sanity check
            
            z_s, z_c = curr_df['z_s'], curr_df['z_c']
            
            #bias calculation
            resnet_reliance = reliance(model=resnet, data=(test_loader, (z_s, z_c)), type='image')
            lda_reliance = reliance(model=lda, data=((X_tab,y_tab), (z_s, z_c)), type='table')
            bias = resnet_reliance - lda_reliance
            
            #logging
            print({'p_s': p_s, 'a_s': a_s, 'seed': seed, 'resnet_reliance': resnet_reliance, 'lda_reliance': lda_reliance, 'bias': bias})
            wandb.log({'p_s': p_s, 'a_s': a_s, 'seeds': seed, 'resnet_reliance': resnet_reliance, 'lda_reliance': lda_reliance, 'bias': bias})
            print("Seed completed execution!", seed, setting)
            print("------------------------------------------------------------------")
            wandb.finish()

        print("Experiment complete", setting)