import time

import matplotlib.pyplot as plt
import jax.config as config
import jax.numpy as jnp

from priorCVAE.datasets import GPDataset
from priorCVAE.priors.kernels import SquaredExponential

config.update("jax_enable_x64", True)

if __name__ == '__main__':

    start_time = time.time()
    gp_dataset_generator = GPDataset(kernel=SquaredExponential(), n_data=400, x_lim_low=0, x_lim_high=1,
                                     sample_lengthscale=True)

    correct_samples = []
    correct_ls = []
    x_val = None  # As it is same for all the data
    while len(correct_samples) <= 1000:
        samples_x, samples_y, samples_ls = gp_dataset_generator.simulatedata()
        if x_val is None:
            x_val = samples_x[0]


        def log_constraint_val(x):
            return jnp.log(30 * x + 1)/3 + 0.1

        print("samples_generated!!!")
        total_samples = samples_y.shape[0]

        for s_x, s_y, s_ls in zip(samples_x, samples_y, samples_ls):
            if jnp.min(s_y) >= 0 and jnp.all(log_constraint_val(s_x) - s_y >= 0):
                correct_samples.append(s_y)
                correct_ls.append(s_ls)

    correct_samples = jnp.array(correct_samples).reshape((-1, samples_y.shape[-1]))
    correct_ls = jnp.array(correct_ls).reshape((-1, 1))

    end_time = time.time()
    print(f"Time taken to geneare {correct_samples.shape[0]} samples is {end_time - start_time} secs!")
    for s in correct_samples[: 100]:
        plt.plot(x_val, s,)

    plt.xlim([0, 1])
    plt.ylim([-3, 3])
    plt.show()

    jnp.savez("constrained_gp_samples.npz", x=x_val, y=correct_samples, ls=correct_ls)
