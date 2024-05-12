# coin flipping network for pseudo-counts
# takes the context vector (output of ResNet18) as input state



import numpy as np
import torch 
import torch.nn as nn
import math 

class ExplorationBonus(nn.Module): 
    def __init__(self, input_size=256, hidden_size=512, output_size=64): 
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh() #used in the linear model in the CFN code
    
    def forward(self,x): 
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        output = self.tanh(x)
        return output

class ExponentialDecayExploration(): 
    def __init__(self, init_value, exp_decay_rate=0.01): 
        self.init_value = init_value
        self.exp_decay_rate = exp_decay_rate
        self.current_step = 0
        self.exploit_on = False
    def __call__(self): 
        if not self.exploit_on: 
            value = self.init_value*math.exp(-self.exp_decay_rate*self.current_step)
        elif self.exploit_on: 
            value = 0
        return value
    def step(self): 
        self.current_step+=1
    def reset(self): 
        self.current_step = 0
        self.exploit_on = False
    def exploit(self): 
        self.exploit_on = True
    
#code taken from this
#https://github.com/samlobel/CFN/blob/main/bonus_based_exploration/intrinsic_motivation/intrinsic_rewards.py

class CoinFlipMaker(object):
    """Sam's thang"""
    def __init__(self, num_coins, p_replace=1):
        self.p_replace = p_replace
        self.num_coins = num_coins
    def __call__(self, num_samples):
        return torch.tensor(2 * np.random.binomial(1, 0.5, size=(num_samples, self.num_coins)) - 1)