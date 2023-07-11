# PriorCVAE in JAX

This repository is based on the following two papers:

1. Semenova, Elizaveta, et al. "PriorVAE: encoding spatial priors with variational autoencoders for small-area estimation." Journal of the Royal Society Interface 19.191 (2022): 20220094.
2. Semenova, Elizaveta, Max Cairney-Leeming, and Seth Flaxman. "PriorCVAE: scalable MCMC parameter inference with Bayesian deep generative modelling." arXiv preprint arXiv:2304.04307 (2023).

## Environment

We recommend setting up a [conda](https://docs.conda.io/projects/conda/en/latest/index.html) environment.
```shell
conda create -n prior_cvae python==3.10
conda activate prior_cvae
```

Within the virtual environment, install the dependencies by running
```shell
pip install -r requirements.txt
```

## To runs tests

First install the test-requirements by running the following command from within the conda environment:
```shell
pip install -r requirements-test.txt
```
Then, run the following command:
```shell
pytest -v tests/
```
