#!/bin/bash

#SBATCH --time=9:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=gp_inference_%j.log

module load Anaconda3/2022.10
source activate $DATA/newenv

cd ~/PriorCVAE_JAX/experiments/
python encode_gp_prior.py iterations=20000 loss=squared_sum_kl batch_size=2000 kernel@prior=matern12 wandb.username=$1 hidden_dim=200 latent_dim=80
