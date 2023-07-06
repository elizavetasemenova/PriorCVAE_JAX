"""
File contains utility functions used throughout the package
"""

from typing import Sequence, Union, List

import jax.numpy as jnp
import numpy as np
import torch
import torch.utils.data as data


def numpy_collate(batch):
    """
    Used while creating a dataloader.

    Details: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/guide4/Research_Projects_with_JAX.html
    """
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def create_data_loaders(*datasets: Sequence[data.Dataset], train: Union[bool, Sequence[bool]] = True,
                        batch_size: int = 128, num_workers: int = 4, seed: int = 42) -> List[data.DataLoader]:
    """
    Creates data loaders used in JAX for a set of datasets.
    
    :param datasets: Datasets for which data loaders are created.
    :param train: Sequence indicating which datasets are used for training and which not. If single bool, the same value
                  is used for all datasets.
    :param batch_size: Batch size to use in the data loaders.
    :param num_workers: Number of workers for each dataset.
    :param seed: Seed to initialize the workers and shuffling with.

    Details: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/guide4/Research_Projects_with_JAX.html

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
                                 persistent_workers=True,
                                 generator=torch.Generator().manual_seed(seed))
        loaders.append(loader)
    return loaders


def euclidean_dist(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate Euclidean distance between the two vectors, x and y.

    # FIXME: What is the correct formula here?
    d(x, y) = ...

    :param x: Jax ndarray of the shape, (N_1, D).
    :param y: Jax ndarray of the shape, (N_2, D).

    :returns: the Euclidean distance value.
    """
    assert isinstance(x, jnp.ndarray) and isinstance(y, jnp.ndarray)
    if len(x.shape) == 1:
        x = x.reshape(x.shape[0], 1)
    if len(y.shape) == 1:
        y = y.reshape(y.shape[0], 1)
    n_x, m_x = x.shape
    n_y, m_y = y.shape
    assert m_x == m_y

    # FIXME: Why is for loop required?
    delta = jnp.zeros((n_x, n_y))
    for d in jnp.arange(m_x):
        x_d = x[:, d]
        y_d = y[:, d]
        delta += (x_d[:, jnp.newaxis] - y_d) ** 2
    return jnp.sqrt(delta)
