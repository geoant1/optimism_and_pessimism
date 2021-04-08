#!/bin/bash -l

#SBATCH -J fit_mpi                               
#SBATCH -o /u/gantonov/out/job.out.%j            # standard out file
#SBATCH -e /u/gantonov/out/job.err.%j            # standard err file
#SBATCH -D /u/gantonov/code/python/all_params/HMCMC    # work directory
#SBATCH --ntasks=320                             # launch job on a single core
#SBATCH --mem-per-cpu=1000MB                     # memory limit for the job
#SBATCH --time=8:00:00                          # run time, up to 24h

module load miniconda/3/4.5.4
module load intel impi

source activate fit

# launch
mpirun -np $SLURM_NTASKS python ./main_mpi.py 1 


