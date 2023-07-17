import jax
import jax.numpy as jnp
from jax import random
import scipy.stats as stats
from flax.training import train_state
import optax
import wandb

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


config = {
    "kernel": {
        "name": "SquaredExponentialKernel",
        "lengthscale": 0.2,
        "variance": 1.0
    },
    "grid": {
        "size": 50,
        "x_min": 0,
        "x_max": 1
    },
    "model": {
        "input_dim": 50,
        "hidden_dim": 40,
        "latent_dim": 30,
        "output_dim": 50,
        "vae_variance": 1.0
    },
    "training": {
        "num_data": 4000,
        "batch_size": 100,
        "num_epochs": 20
    },
    "test": {
        "num_samples": 10000
    }
}

kernel_name = config["kernel"]["name"]
lengthscale = config["kernel"]["lengthscale"]
kernel_variance = config["kernel"]["variance"]

grid_size = config["grid"]["size"]
x_min = config["grid"]["x_min"]
x_max = config["grid"]["x_max"]
grid = jnp.linspace(x_min, x_max, grid_size)

input_dim = config["model"]["input_dim"]
hidden_dim = config["model"]["hidden_dim"]
latent_dim = config["model"]["latent_dim"]
output_dim = config["model"]["output_dim"]
vae_variance = config["model"]["vae_variance"]

# Training parameters
num_data = config["training"]["num_data"]
batch_size = config["training"]["batch_size"]
num_epochs = config["training"]["num_epochs"]

# Test parameters
num_samples = config["test"]["num_samples"]


def main():


    run = wandb.init(project="first-sweeps", config=config)
    kernel = SquaredExponential(lengthscale, kernel_variance)(grid, grid)

    train_set = GPDataset(
        n_data=input_dim, n_samples=num_data, lengthscale=lengthscale, x_lim_low=x_min, x_lim_high=x_max
    )
    val_set = GPDataset(
        n_data=input_dim, n_samples=num_data, lengthscale=lengthscale, x_lim_low=x_min, x_lim_high=x_max
    )
    train_loader, = create_data_loaders(
        train_set, batch_size=batch_size
    )
    val_loader, = create_data_loaders(
        val_set, batch_size=batch_size
    )

    key = jax.random.PRNGKey(0) 
    key, train_key, test_key = random.split(key, 3)

    state = train_vae(train_key, train_loader, val_loader)
    decoder_params = state.params['decoder']

    run_tests(key, decoder_params, kernel)    
    wandb.finish()

def run_tests(key, decoder_params, kernel):
    decoder = MLPDecoder(hidden_dim, output_dim)
    res = mean_bootstrap_interval_contains_zero(key, decoder_params, decoder, latent_dim)
    wandb.log({"mean_bootstrap_interval_contains_zero": res})

    samples = generate_vae_samples(key, decoder_params, decoder, num_samples, latent_dim)
    norm = frobenius_norm_of_kernel_diff(samples, kernel)
    wandb.log({"norm_of_kernel_diff": norm})

    # fig, ax = plot_realizations(grid, samples, "VAE samples")
    # wandb.log({"vae_realizations": wandb.Image(fig)})

    # cov_matrix = compute_empirical_covariance(samples)
    # fig, ax = plot_covariance_matrix(cov_matrix, "Empirical covariance")
    # wandb.log({"empirical_covariance": wandb.Image(fig)})

    # fig, ax = plot_covariance_matrix(kernel, kernel_name)
    # wandb.log({"kernel": wandb.Image(fig)})


# TODO: use training methods in module
def train_vae(key, train_loader, val_loader):
    encoder = MLPEncoder(hidden_dim, latent_dim)
    decoder = MLPDecoder(hidden_dim, output_dim)
    model = VAE(encoder, decoder)

    @jax.jit
    def train_step(state, batch, z_rng):
        def loss_fn(params, z_rng):
            _, y, _ = batch 
            y_hat, z_mu, z_logvar = model.apply({'params': params}, y, z_rng) 
            rcl_loss = scaled_sum_squared_loss(y_hat, y, wandb.config.model['vae_variance'])
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
            rcl_loss = scaled_sum_squared_loss(y_hat, y, wandb.config.model['vae_variance'])
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
    _loss_val = []

    for epoch in range(num_epochs):
        for _, batch in enumerate(train_loader):
            z_key, key = random.split(z_key)
            state, loss_train = train_step(state, batch, key)
            _loss_train.append(loss_train)

        for _, batch in enumerate(val_loader):

            eval_key, key = random.split(eval_key)
            loss_test = eval(state, batch, key)
            _loss_val.append(loss_test)

        wandb.log({"train_loss": loss_train, "val_loss": loss_test})

    return state


if __name__ == "__main__":
    # main()

    sweep_configuration = {
        'method': 'bayes',
        'metric': 
        {
            'goal': 'minimize', 
            'name': 'norm_of_kernel_diff'
            },
        'parameters': 
        {
            'model': {
                'parameters': { 
                    'vae_variance': {'max': 3., 'min': 1.}
                }
            }
        }
    }

    sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        project='my-first-sweep'
        )

    wandb.agent(sweep_id, function=main, count=8)