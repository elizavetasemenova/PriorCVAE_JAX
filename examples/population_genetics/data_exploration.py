"""Explore the input data"""

import jax.numpy as jnp

from utility import split_data_into_time_batches, read_csv_data, plot_data

if __name__ == '__main__':
    data_csv_file = r"data/samples_1.csv"
    n_plot = 5

    data = read_csv_data(data_csv_file)
    print(f"Total data is {data.shape[0]}")

    time_sliced_data = split_data_into_time_batches(data, time_slice=6)
    print(f"Data shape after time slicing: {time_sliced_data.shape}")

    hyperparams = time_sliced_data[:, :, :3]
    prior_data = time_sliced_data[:, :, 3:]
    print(f"Hyperparams data shape: {hyperparams.shape}")
    print(f"Prior data shape: {prior_data.shape}")

    for d in prior_data[:n_plot]:
        plot_data(d)

    print(f"Minimum value in the data {jnp.min(prior_data)}")
    print(f"Maximum value in the data {jnp.max(prior_data)}")


