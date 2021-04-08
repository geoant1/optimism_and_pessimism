#!/bin/bash -l

#SBATCH -J fit_all                               
#SBATCH -o /u/gantonov/out/job.out.%j            # standard out file
#SBATCH -e /u/gantonov/out/job.err.%j            # standard err file
#SBATCH -D /u/gantonov/code/python/all_params/new    # work directory
#SBATCH --nodes=50
#SBATCH --ntasks-per-node=1                      # launch job on a single core
#SBATCH --cpus-per-task=40                       # on a shared node
#SBATCH --mem-per-cpu=1500MB                     # memory limit for the job
#SBATCH --time=24:00:00                          # run time, up to 24h

module load miniconda/3/4.5.4
source activate fit

# launch
srun python ./main6.py 


