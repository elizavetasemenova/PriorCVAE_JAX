  #!/bin/sh
  wandbusername=${1:-null}

  for kernel in squared_exponential; do
      for hidden_dim in 30 50 64; do
          for latent_dim in 30 50 64; do
	      for reconstruction_scaling in 0.1 0.5 1 2 5; do
                  sbatch zimbabwe_sweeps.slurm $wandbusername $reconstruction_scaling $hidden_dim $latent_dim $kernel False
		  sleep 1200
	      done
	      sleep 1200
	  done
	  sleep 1200
      done
      sleep 1200
  done
