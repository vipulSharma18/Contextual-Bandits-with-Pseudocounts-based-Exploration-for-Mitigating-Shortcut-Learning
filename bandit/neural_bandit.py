#Take patch embedding as context
#use Coin flip networks to calculate exploration bonus/intrinsic reward
#use MLP to estimate extrinsic reward

import sys
import os
# Determine the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to sys.path
sys.path.append(parent_dir)

import torch
import torch.nn as nn
from pseudocounts.cfn import ExplorationBonus, CoinFlipMaker, ExponentialDecayExploration

#total pixels: 50176 (224*224)
#pixels of core feature(40*40): 1600
#pixels of a patch(56*56): 3136 -> Much more than the core feature size
#at maximum 4 neighboring patches, in the worst case object split across 4 patches

class ContextualBanditWithPseudoCounts(nn.Module): 
    def __init__(self, context_length=256, hidden_size=512, num_coins=64, lambda_explore=0.8): 
        super().__init__()
        self.pseudocount_gen = ExplorationBonus(input_size=context_length, hidden_size=hidden_size, output_size=num_coins)
        self.coin_flip_maker = CoinFlipMaker(num_coins=num_coins)
        self.exploration_rate_gen = ExponentialDecayExploration(init_value=lambda_explore, exp_decay_rate=0.02) #start at 0.8, reach 0.1 at 100th step/batch
        self.reward_model = nn.Sequential(
            nn.Linear(context_length, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, context): 
        pred_reward = self.reward_model(context)
        
        #f(n)**2 = d/N, f(n) is [-1,1] d dim (tanh)
        pseudocount_output = self.pseudocount_gen(context) #batch_sz, d dim
        coin_label = self.coin_flip_maker(context.size(0)) #batch_sz, d dim
        inverse_pseudocount = pseudocount_output.detach()
        exploration_bonus = inverse_pseudocount.pow(2).mean(dim=-1).unsqueeze(-1) #batch_sz,1
        total_reward = pred_reward + self.exploration_rate_gen()*exploration_bonus
        
        bandit_output = nn.Sigmoid(total_reward) #probability of including context/patch/state in input to supervisory model
        #actual reward will be the softmax probability of true class
        return bandit_output, pseudocount_output, coin_label