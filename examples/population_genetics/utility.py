"""
Utility functions for population genetics experiment.
"""

import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt


def split_data_into_time_batches(data: jnp.ndarray, time_slice: int = 6) -> jnp.ndarray:
    """
    Split the data into time slices.

    :param data: the data of shape [N, D]
    :param time_slice: an integer representing the total number of time slice

    :output a jax.np.ndarray of shape [N/time_slice, time_slice, D]
    """
    assert data.shape[0] % time_slice == 0
    return data.reshape((-1, time_slice, data.shape[-1]))


def read_csv_data(file_path: str) -> jnp.ndarray:
    """
    Read CSV file and return data as jax.np.ndarray.
    """
    return jnp.array(pd.read_csv(file_path).values)


def plot_data(data: jnp.ndarray):
    """
    Plot a time-sliced data.

    :param data: jax.np.ndarray of shape [time_slice, D]

    Note: Currently the code only works with time_slice=6.
    """
    assert data.shape[0] == 6
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    for i, arr in enumerate(data):
        row = int(i / 3)
        col = int(i % 3)

        img_shape = int(jnp.sqrt(arr.shape[0]))
        axs[row][col].imshow(arr.reshape((img_shape, img_shape)))

    plt.suptitle("Sample data.")
    plt.tight_layout()
    plt.show()
