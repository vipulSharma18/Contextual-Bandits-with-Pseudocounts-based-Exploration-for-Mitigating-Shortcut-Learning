import sys
import os
# Determine the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from bandit.neural_bandit import ContextualBanditWithPseudoCounts


ContextualBanditWithPseudoCounts(context_length=256, hidden_size=512, num_coins=64, lambda_explore=0.8)
    
    
    
