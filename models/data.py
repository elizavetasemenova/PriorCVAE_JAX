# Data loading with PyTorch Dataloader

# see https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/guide4/Research_Projects_with_JAX.html


import numpy as np 
import random as rnd
from jax import random

# type checkers 
from typing import Sequence, Union

import torch.utils.data as data

from numpyro.infer import Predictive

# PyTorch
import torch
import torch.utils.data as data

# import priors 
from models.priors import *

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
    

def create_data_loaders(*datasets : Sequence[data.Dataset], 
                        train : Union[bool, Sequence[bool]] = True, 
                        batch_size : int = 128, 
                        num_workers : int = 4, 
                        seed : int = 42):
    """
    Creates data loaders used in JAX for a set of datasets.
    
    Args:
      datasets: Datasets for which data loaders are created.
      train: Sequence indicating which datasets are used for 
        training and which not. If single bool, the same value
        is used for all datasets.
      batch_size: Batch size to use in the data loaders.
      num_workers: Number of workers for each dataset.
      seed: Seed to initialize the workers and shuffling with.
    """
    loaders = []
    if not isinstance(train, (list, tuple)):
        train = [train for _ in datasets]
    for dataset, is_train in zip(datasets, train):
        loader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=is_train,
                                 drop_last=is_train,
                                 collate_fn=numpy_collate,
                                 num_workers=num_workers,
                                 persistent_workers=is_train,
                                 generator=torch.Generator().manual_seed(seed))
        loaders.append(loader)
    return loaders


class Dataset_GP1d(data.Dataset):
    
    # generate GP draws over the regular grid in the interval (x_lim_low, x_lim_high) with n_dataPoints points
    def __init__(self, 
                 n_dataPoints=400, # number of data points in the interval
                 n_samples=10000,  # number of samples
                 x_lim_low = 0,    # lower limit if the interval
                 x_lim_high = 1,   # upper limit if the interval
                 seed = 42,        # seed
                 ls = None         # lengthscale to generate from. If None, draw from prior
                 ): 

        self.n_dataPoints = n_dataPoints
        self.n_samples = n_samples
        self.x_lim_low = x_lim_low
        self.x_lim_high = x_lim_high
        self.ls = ls
        self.x, self.y, self.ls = self.__simulatedata__()
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx=0):
        return(self.x[idx], self.y[idx], self.ls[idx])

    def __simulatedata__(self):
        rng_key = rnd.randint(0, 912393890428)    # use seed?
        rng_key, _ = random.split(random.PRNGKey(rng_key))
        x = np.linspace(self.x_lim_low, self.x_lim_high, self.n_dataPoints)   
        gp_predictive = Predictive(GP, num_samples=self.n_samples)  
        all_draws = gp_predictive(rng_key, x=x, gp_kernel = exp_sq_kernel, jitter=1e-6, length=self.ls)

        ls_draws = np.array(all_draws['ls'])
        gp_draws = np.array(all_draws['y'])
        
        return (x.repeat(self.n_samples).reshape(x.shape[0], self.n_samples).transpose(), gp_draws, ls_draws)
    
