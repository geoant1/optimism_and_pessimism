#!/bin/bash -l

#SBATCH -J replays                               
#SBATCH -o /u/gantonov/out/vary_params/job.out.%j            # standard out file
#SBATCH -e /u/gantonov/out/vary_params/job.err.%j            # standard err file
#SBATCH -D /u/gantonov/code/python/contour/    # work directory
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                      # launch job on a single core
#SBATCH --cpus-per-task=1                        # on a shared node
#SBATCH --mem=20000MB                     # memory limit for the job
#SBATCH --time=12:00:00                          # run time, up to 24h

module load miniconda/3/4.5.4
source activate fit

# launch
srun python ./analyse_replays.py 


