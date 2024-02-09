#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/gp_inference_%j.log

module load Anaconda3/2022.10
source activate $DATA/newenv

cd ~/PriorCVAE_JAX/experiments/
python encode_gp_prior.py iterations=20000 loss=squared_sum_kl batch_size=2000 kernel@prior=matern52 wandb.username=$1 hidden_dim=64 latent_dim=64 x_dim=$2 n_data=8
