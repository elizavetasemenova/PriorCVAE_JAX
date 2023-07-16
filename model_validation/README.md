# Model Validation

This directory contains tests for validating the performance of our models. MLflow is used to track the performance of models and the results of experiments.

## Getting Started

To install MLflow run:

```bash
pip install -r requirements.txt
```

## Running an Experiment

From the project's root directory run:

```bash
python -m model_validation.run
```

## Launching the MLflow UI

To start the MLflow UI run:
```bash
mlflow ui
```

The UI can be accessed at `http://127.0.0.1:5000`.
