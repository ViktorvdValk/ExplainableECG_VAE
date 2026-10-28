"""
Helper utilities for ECG VAE - contains functions used by Trainer_12.
"""

import random
import os
import torch
from torch.nn import init
import numpy as np
from datetime import datetime
import git as git_module

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def git_autocommit(version):
    """
    Auto-commit current git state with version timestamp.
    
    Args:
        version: Version string to include in commit message
    """
    print(f'-- Commit and Push to git {version} --')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_message = f"{version} run at {timestamp}"
    repo = git_module.Repo(os.getcwd())
    repo.git.add(update=True)
    repo.index.commit(commit_message)
    origin = repo.remote(name='origin')
    origin.push()


def init_weights(net, init_type='normal', gain=0.02, a=0):
    """
    Initialize neural network weights with specified method.
    
    Args:
        net: Neural network module
        init_type: Initialization method ('normal', 'xavier', 'kaiming', 'orthogonal')
        gain: Gain factor for initialization
        a: Negative slope for leaky relu (used in kaiming)
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=a, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func)


def set_all_seeds(seed):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LambdaScheduler:
    """
    Scheduler for dynamic lambda adjustment in loss weighting.
    
    Used by Trainer_contrastive class but NOT by Trainer_12.
    Should only be used if other trainers need it.
    """
    def __init__(self, init_lambda=0.0, max_lambda=2.0, gamma=0.1, step_size=200):
        self.current_lambda = init_lambda
        self.max_lambda = max_lambda
        self.gamma = gamma
        self.step_size = step_size
        self.step_count = 0
    
    def step(self):
        """Gradually increase lambda using a schedule."""
        self.step_count += 1
        if self.step_count % self.step_size == 0:
            self.current_lambda = min(
                self.max_lambda,
                self.current_lambda + self.gamma
            )
    
    def get_lambda(self):
        return self.current_lambda


def flatten(l):
    """Flatten a list of lists into a single list."""
    return [item for sublist in l for item in sublist]
