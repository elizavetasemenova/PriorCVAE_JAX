  #!/bin/sh
  wandbusername=${1:-null}
  reconstruction_scaling=$2  
  hidden_dim=$3
  latent_dim=$4    
  sbatch zimbabwe_matern52_outputs.slurm $wandbusername $reconstruction_scaling $hidden_dim $latent_dim 
