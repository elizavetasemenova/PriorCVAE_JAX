import jax.numpy as jnp

from model_validation.utils import generate_decoder_samples
from priorCVAE.diagnostics import mean_bootstrap_interval, frobenius_norm_of_kernel_diff



def mean_bootstrap_interval_contains_zero(samples: jnp.ndarray, **kwargs):
    ci_lower, ci_upper = mean_bootstrap_interval(samples)
    zero_in_interval = (ci_lower <= 0) & (0 <= ci_upper)
    num_valid = jnp.where(zero_in_interval)[0].shape[0]
    return num_valid == samples.shape[1]


def norm_of_kernel_diff(samples: jnp.ndarray, kernel: jnp.ndarray, **kwargs):
    norm = frobenius_norm_of_kernel_diff(samples, kernel)
    return norm