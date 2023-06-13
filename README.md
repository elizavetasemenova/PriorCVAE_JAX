# PriorCVAE - JAX

## Environment
Create the environment `numpyro11_jax`: 
 
```
conda create -n numpyro11_jax python=3.10
conda activate numpyro11_jax
conda install -c conda-forge jax=0.4.2
conda install -c conda-forge numpyro=0.11.0
conda install -c conda-forge flax=0.6.1
pip install --upgrade jax jaxlib chex
conda install -c conda-forge optax=0.1.4
conda install pytorch=1.12.1 -c pytorch
conda install -c anaconda Jupyter
conda install -c conda-forge jraph
conda install -c conda-forge matplotlib
conda install -c conda-forge arviz
conda install -c conda-forge dill
conda install -c anaconda seaborn
conda install -c conda-forge optuna
conda install -c conda-forge wandb

conda install -c conda-forge geopandas=0.12.2
```


# PriorCVAE_JAX

- [x] create environment
- [ ] generate 1d GP data
- [ ] run example MCMC
- [ ] port CVAE code
