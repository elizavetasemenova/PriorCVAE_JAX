# Model Validation

This directory contains tests for validating the performance of our models. [Weights & Biases](https://docs.wandb.ai) is used to track the performance of models and the results of experiments. [Hydra](https://hydra.cc/docs/intro/) is used for configuration.

## Getting Started

To install additional requirements:

```bash
pip install -r model_validation/requirements.txt
```

## Running an Experiment

From the project's root directory run:

```bash
python -m model_validation.run
```
