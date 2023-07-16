import jax
import jax.numpy as jnp
from jax import random
import scipy.stats as stats
from flax.training import train_state
import optax
import mlflow

from priorCVAE.models import VAE, MLPEncoder, MLPDecoder
from priorCVAE.diagnostics import (
    compute_empirical_covariance,
    frobenius_norm_of_kernel_diff,
    plot_realizations,
    plot_covariance_matrix
)
from priorCVAE.datasets import GPDataset
from priorCVAE.utility import create_data_loaders
from priorCVAE.losses import scaled_sum_squared_loss, kl_divergence
from priorCVAE.priors import SquaredExponential
from model_validation.utils import generate_vae_samples
from model_validation.model_validation_tests import mean_bootstrap_interval_contains_zero


# Kernel parameters
kernel_name = "SquaredExponentialKernel"
lengthscale = 0.2
kernel_variance = 1.0

grid_size = 50
grid = jnp.linspace(0, 1, grid_size)

# Model parameters
input_dim = grid_size
hidden_dim = 40
latent_dim = 30
output_dim = input_dim
vae_variance = 1.0

# Training parameters
num_data = 4000
batch_size = 100
num_epochs = 200

# Test parameters
num_samples = 1000

def run():

    mlflow.start_run(run_name="SquaredExponentialKernel")
    mlflow.log_param("grid_size", grid_size)
    kernel = SquaredExponential(lengthscale, kernel_variance)(grid, grid)

    train_set = GPDataset(
        n_data=input_dim, n_samples=num_data, lengthscale=lengthscale
    )
    test_set = GPDataset(
        n_data=input_dim, n_samples=num_data, lengthscale=lengthscale
    )
    train_loader, = create_data_loaders(
        train_set, batch_size=batch_size
    )
    test_loader, = create_data_loaders(
        test_set, batch_size=batch_size
    )

    log_params()

    key = jax.random.PRNGKey(0) 
    key, train_key, test_key = random.split(key, 3)

    state = train_vae(train_key, train_loader, test_loader)
    decoder_params = state.params['decoder']

    run_tests(key, decoder_params, kernel)    
    

def run_tests(key, decoder_params, kernel):
    decoder = MLPDecoder(hidden_dim, output_dim)
    res = mean_bootstrap_interval_contains_zero(key, decoder_params, decoder, latent_dim)
    mlflow.log_metric("mean_bootstrap_interval_contains_zero", res)

    samples = generate_vae_samples(key, decoder_params, decoder, num_samples, latent_dim)
    norm = frobenius_norm_of_kernel_diff(samples, kernel)
    mlflow.log_metric("norm_of_kernel_diff", norm)

    fig, ax = plot_realizations(grid, samples, "VAE samples")
    mlflow.log_figure(fig, 'vae_realizations.png')

    cov_matrix = compute_empirical_covariance(samples)
    fig, ax = plot_covariance_matrix(cov_matrix, "Empirical covariance")
    mlflow.log_figure(fig, 'vae_covariance.png')

    fig, ax = plot_covariance_matrix(kernel, kernel_name)
    mlflow.log_figure(fig, 'kernel.png')


def log_params():
    mlflow.log_param("kernel_name", kernel_name)
    mlflow.log_param("grid_size", grid_size)
    mlflow.log_param("lengthscale", lengthscale)
    mlflow.log_param("kernel_variance", kernel_variance)
    mlflow.log_param("input_dim", input_dim)
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("latent_dim", latent_dim)
    mlflow.log_param("output_dim", output_dim)
    mlflow.log_param("vae_variance", vae_variance)
    mlflow.log_param("training_set_size", num_data)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("num_samples", num_samples)

# TODO: use training methods in module
def train_vae(key, train_loader, test_loader):
    encoder = MLPEncoder(hidden_dim, latent_dim)
    decoder = MLPDecoder(hidden_dim, output_dim)
    model = VAE(encoder, decoder)

    @jax.jit
    def train_step(state, batch, z_rng):
        def loss_fn(params, z_rng):
            _, y, _ = batch 
            y_hat, z_mu, z_logvar = model.apply({'params': params}, y, z_rng) 
            rcl_loss = scaled_sum_squared_loss(y_hat, y, vae_variance)
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
            rcl_loss = scaled_sum_squared_loss(y_hat, y, vae_variance)
            kld_loss = kl_divergence(z_mu, z_logvar)
            loss = rcl_loss + kld_loss
            return loss
        
        return loss_fn(state.params, z_rng)
    
    # initialise parameters
    rng, z_key, eval_key, init_key = random.split(key, 4)

    batch_init = random.normal(init_key, (batch_size, input_dim)) # Dummy input data
    y = batch_init
    params = model.init(rng, y, key)['params']

    # optimizer
    optimizer = optax.adam(learning_rate=0.001)

    # store training state
    state = train_state.TrainState.create(apply_fn=model, params=params, tx=optimizer)

    _loss_train = []
    _loss_test = []

    for epoch in range(num_epochs):
        for _, batch_train in enumerate(train_loader):
            z_key, key = random.split(z_key)
            state, loss_train = train_step(state, batch_train, key)
            _loss_train.append(loss_train)

    for _, batch_test in enumerate(test_loader):

        eval_key, key = random.split(eval_key)
        loss_test = eval(state, batch_test, key)
        _loss_test.append(loss_test)
    mlflow.log_metric("train_loss", loss_train)
    mlflow.log_metric("test_loss", loss_test)

    return state


if __name__ == "__main__":
    run()