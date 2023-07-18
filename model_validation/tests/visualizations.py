import wandb
from priorCVAE.diagnostics import plot_realizations, plot_covariance_matrix, compute_empirical_covariance


def plot_vae_realizations(samples, x, **kwargs):
    fig, _ = plot_realizations(x, samples, "VAE samples")
    wandb.log({"vae_realizations": wandb.Image(fig)})

def plot_empirical_covariance(samples, **kwargs):
    cov_matrix = compute_empirical_covariance(samples)
    fig, ax = plot_covariance_matrix(cov_matrix, "Empirical covariance")
    wandb.log({"empirical_covariance": wandb.Image(fig)})

def plot_kernel(kernel, kernel_name, **kwargs):
    fig, ax = plot_covariance_matrix(kernel, kernel_name)
    wandb.log({"kernel": wandb.Image(fig)})
