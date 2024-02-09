#!/bin/bash

#SBATCH --time=20:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/zimbabwe_%j.log

module load Anaconda3/2022.10
source activate $DATA/newenv

cd ~/PriorCVAE_JAX/experiments/
python train_zimbabwe_gp.py data_shp_path="data/zwe2016phia.geojson" batch_size=2000 kernel@prior=$5 wandb.username=$1 hidden_dim=$3 latent_dim=$4 normalize=$6 distributions@prior_ls=uniform prior_ls.low=0.01 prior_ls.high=1.0 vae_var=1. iterations=20000 kl_scaling=1. reconstruction_scaling=$2
