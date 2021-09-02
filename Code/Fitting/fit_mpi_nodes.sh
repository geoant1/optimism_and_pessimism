#!/bin/bash -l

#SBATCH -J fit_mpi                               
#SBATCH -o /u/gantonov/out_test/job.out.%j     # standard out file
#SBATCH -e /u/gantonov/out_test/job.err.%j     # standard err file
#SBATCH -D /u/gantonov/code/python/fit/test    # work directory
#SBATCH --nodes=2
#SBATCH --ntasks=51
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000 
#SBATCH --time=5:00:00                    # run time, up to 24h

module purge
module load intel/19.1.1 impi/2019.7 mkl/2020.1
#module load intel impi
module load miniconda/3/4.5.4
export I_MPI_FABRICS=ofi
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SLURM_CPU_BIND=none

source activate fit

# launch
srun python main.py 8

