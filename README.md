# PriorCVAE - JAX

## Environment
Create the environment `numpyro10_torch`: 
 
```
conda create -n numpyro10_jax python=3.8.15
conda activate numpyro10_jax
conda install -c conda-forge mamba
conda install -c conda-forge jax=0.3.25
conda install -c conda-forge numpyro=0.10.1
conda install pytorch=1.12.1 -c pytorch
conda install -c conda-forge flax
conda install -c conda-forge optax
conda install -c conda-forge jraph
conda install -c conda-forge matplotlib
conda install -c anaconda Jupyter
conda install -c conda-forge arviz
conda install -c conda-forge dill
conda install -c anaconda seaborn

conda install -c conda-forge geopandas
mamba install -c conda-forge wandb
```


# PriorCVAE_JAX

- [ ] create environment
- [ ] generate 1d GP data
- [ ] run example MCMC
- [ ] port CVAE code
