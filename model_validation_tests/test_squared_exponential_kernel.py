import pytest
import jax
import jax.numpy as jnp
from jax import random
import scipy.stats as stats
from flax.training import train_state
import optax


from priorCVAE.models import VAE, MLPEncoder, MLPDecoder
from priorCVAE.diagnostics import (
    compute_empirical_covariance,
    frobenius_norm_of_kernel_diff,
    mean_bootstrap_interval,
)
from priorCVAE.datasets import GPDataset
from priorCVAE.utility import create_data_loaders
from priorCVAE.losses import scaled_sum_squared_loss, kl_divergence
from priorCVAE.priors import SquaredExponential
from .utils import generate_vae_samples

# Kernel parameters
LENGTHSCALE = 1.0
KERNEL_VARIANCE = 1.0

# Model parameters
INPUT_DIM = 50
HIDDEN_DIM = 40
LATENT_DIM = 30
OUTPUT_DIM = INPUT_DIM
VAE_VARIANCE = 1.0

# Training parameters
NUM_DATA = 4000
BATCH_SIZE = 100
NUM_EPOCHS = 200

# Test parameters
NUM_SAMPLES = 1000


@pytest.fixture(scope="module")
def datasets():
    train_set = GPDataset(
        n_data=INPUT_DIM, n_samples=NUM_DATA, lengthscale=LENGTHSCALE
    )
    train_loader, = create_data_loaders(
        train_set, batch_size=BATCH_SIZE
    )
    return train_loader

@pytest.fixture(scope="module")
def grid():
    return jnp.linspace(0, 1, INPUT_DIM)

@pytest.fixture(scope="module")
def kernel(grid):
    return SquaredExponential(LENGTHSCALE, KERNEL_VARIANCE)(grid, grid)

@pytest.fixture(scope="module")
def decoder():
    return MLPDecoder(HIDDEN_DIM, OUTPUT_DIM)

@pytest.fixture(scope="module")
def decoder_params(datasets):
    train_loader = datasets
    state = train_vae(train_loader)
    return state.params['decoder']

def test_mean_bootstrap_interval_contains_zero(decoder_params, decoder):
    samples = generate_vae_samples(decoder_params, decoder, NUM_SAMPLES, LATENT_DIM)
    ci_lower, ci_upper = mean_bootstrap_interval(samples)
    zero_in_interval = (ci_lower <= 0) & (0 <= ci_upper)
    num_valid = jnp.where(zero_in_interval)[0].shape[0]

    assert num_valid == INPUT_DIM


# Todo: use training methods in module
def train_vae(train_loader):
    encoder = MLPEncoder(HIDDEN_DIM, LATENT_DIM)
    decoder = MLPDecoder(HIDDEN_DIM, OUTPUT_DIM)
    model = VAE(encoder, decoder)

    @jax.jit
    def train_step(state, batch, z_rng):
        def loss_fn(params, z_rng):
            _, y, _ = batch 
            y_hat, z_mu, z_logvar = model.apply({'params': params}, y, z_rng) 
            rcl_loss = scaled_sum_squared_loss(y_hat, y, VAE_VARIANCE)
            kld_loss = kl_divergence(z_mu, z_logvar)
            loss = rcl_loss + kld_loss
            return loss
        
        grads = jax.grad(loss_fn)(state.params, z_rng)
        return state.apply_gradients(grads=grads), loss_fn(state.params, z_rng)

    @jax.jit
    def eval(state, batch, z_rng):
        def loss_fn(params, z_rng):
            _, y, _ = batch 
            y_hat, z_mu, z_logvar = model.apply({'params': params}, y, z_rng) 
            rcl_loss = scaled_sum_squared_loss(y_hat, y, VAE_VARIANCE)
            kld_loss = kl_divergence(z_mu, z_logvar)
            loss = rcl_loss + kld_loss
            return loss
        
        return loss_fn(state.params, z_rng)
    
    # initialise parameters
    key = jax.random.PRNGKey(0) 
    rng, z_key, eval_key, init_key = random.split(key, 4)

    batch_init = random.normal(init_key, (BATCH_SIZE, INPUT_DIM)) # Dummy input data
    y = batch_init
    params = model.init(rng, y, key)['params']

    # optimizer
    optimizer = optax.adam(learning_rate=0.001)

    # store training state
    state = train_state.TrainState.create(apply_fn=model, params=params, tx=optimizer)

    _loss_train = []

    for epoch in range(NUM_EPOCHS):
        for _, batch_train in enumerate(train_loader):
            z_key, key = random.split(z_key)
            state, loss_train = train_step(state, batch_train, key)
            _loss_train.append(loss_train)

    return state