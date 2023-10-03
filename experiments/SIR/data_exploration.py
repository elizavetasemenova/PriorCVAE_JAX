"""
Exploring SIR data
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from priorCVAE.datasets import SIRDataset

if __name__ == '__main__':
    num_days = 14
    z_init = jnp.array([762, 1.0, 0.0])

    sir_dataset = SIRDataset(z_init=z_init, num_days=num_days)
    _, z, c = sir_dataset.simulatedata(n_samples=10)

    for z_i in z:
        # plt.plot(jnp.arange(num_days), z_i[:, 0], c="tab:blue", alpha=0.4)
        plt.plot(jnp.arange(num_days), z_i, c="tab:red", alpha=0.4)
        # plt.plot(jnp.arange(num_days), z_i[:, 2], c="tab:green", alpha=0.4)

    # plt.plot(-9999, 9999, c="tab:blue", label="S")
    plt.plot(-9999, 9999, c="tab:red", label="I")
    # plt.plot(-9999, 9999, c="tab:green", label="R")

    plt.xlim([0, num_days - 1])
    plt.ylim(jnp.min(z), jnp.max(z))
    plt.legend(loc="upper right")
    plt.show()
