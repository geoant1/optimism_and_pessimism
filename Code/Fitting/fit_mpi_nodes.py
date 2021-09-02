#!/bin/bash -l
import os 
import numpy as np

work_dir = '/u/gantonov/code/python/fit/'
out_dir  = '/u/gantonov/out_test/'

num_av            = 5
num_abc           = 101
num_tasks         = (num_abc-1)*num_av + 1
num_cpus_per_node = 40
num_nodes         = np.ceil(num_tasks/num_cpus_per_node).astype(int)

num_subjects = 2

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

for sub in range(num_subjects):
    
    save_folder = os.path.join(out_dir, 'save_params_%u'%sub)
    
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    job_file = os.path.join(os.getcwd(), 'fit_%u.sh'%sub)

    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash -l\n")
        fh.writelines("#SBATCH -J fit_%u\n"%sub)
        fh.writelines("#SBATCH -o " + os.path.join(out_dir, 'job.out.%j') + "\n")
        fh.writelines("#SBATCH -e " + os.path.join(out_dir, 'job.err.%j') + "\n")
        fh.writelines("#SBATCH -D " + work_dir + "\n")
        fh.writelines("#SBATCH --nodes=%u\n"%num_nodes)
        fh.writelines("#SBATCH --ntasks=%u\n"%num_tasks)
        fh.writelines("#SBATCH --cpus-per-task=1\n")
        fh.writelines("#SBATCH --mem-per-cpu=2000\n")
        fh.writelines("#SBATCH --time=24:00:00\n")
        
        fh.writelines("module purge\n")
        fh.writelines("module load intel/19.1.1 impi/2019.7 mkl/2020.1\n")
        fh.writelines("module load miniconda/3/4.5.4\n")
        fh.writelines("export I_MPI_FABRICS=ofi\n")
        fh.writelines("export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n")
        fh.writelines("export SLURM_CPU_BIND=none\n")
        fh.writelines("source activate fit\n")
        fh.writelines("srun python main.py %u %u %u"%(sub, num_abc, num_av))
    
    os.system("sbatch %s"%job_file)