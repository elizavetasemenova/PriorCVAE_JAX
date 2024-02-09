  #!/bin/sh
  wandbusername=${1:-null}
  dim=${2:-1} 
  sbatch gp_matern52_2d_mmd.slurm $wandbusername $dim
