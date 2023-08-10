"""
Offline dataset.
"""
import random

import jax
import jax.numpy as jnp


class OfflineDataset:
    """Use offline dataset, and randomly generate a batch from it."""

    def __init__(self, dataset: jnp.ndarray, x: jnp.ndarray = None, c: jnp.ndarray = None):
        """
        Initialize the OfflineDataset class.

        :param dataset: jax ndarray of the dataset.
        :param x: jax ndarray of the x values, defaults to None.
        :param c: jax ndarray of the conditional value, defaults to None.
        """
        self.dataset = dataset
        self.x = x
        self.c = c

        if self.x is not None:
            assert self.dataset.shape[0] == self.x.shape[0]
        if self.c is not None:
            assert self.dataset.shape[0] == self.c.shape[0]

    def simulatedata(self, n_samples: int = 10000, batch_idx: jnp.ndarray = None) -> [jnp.ndarray, jnp.ndarray,
                                                                                      jnp.ndarray]:
        """
        Make a batch of data from the dataset array.

        :param n_samples: number of samples.
        :param batch_idx: an array of elements to return in a batch

        :returns: A tuple
            - x
            - samples
            - conditional value
        """

        if batch_idx is None:
            rng_key, _ = jax.random.split(jax.random.PRNGKey(random.randint(0, 9999)))
            batch_idx = jax.random.randint(rng_key, [n_samples], 0, self.dataset.shape[0])

        batch_data = self.dataset[batch_idx]
        batch_x = self.x[batch_idx] if self.x is not None else None
        batch_c = self.c[batch_idx] if self.c is not None else None

        return batch_x, batch_data, batch_c
