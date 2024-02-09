  #!/bin/sh
  wandbusername=${1:-null}
  sbatch gp_matern12.slurm $wandbusername
