import numpy as np
import sys, os
from misc_fit import sample_prior, perturb, backup, convert_params
from agent import Agent
from astroabc.setup_mpi_mp import *

data_path      = '/u/gantonov/data/'

# load data
this_sub       = int(sys.argv[1]) # subject number
sub_data_path  = os.path.join(data_path, 'subject_data', str(this_sub)) # path to subject data

blocks_max_rwd = np.load(os.path.join(sub_data_path, 'blocks_max_rwd.npy'), allow_pickle=True)[7:] # max possible reward for this subject
blocks_rwd     = np.load(os.path.join(sub_data_path, 'blocks_obt_rwd.npy'), allow_pickle=True)[7:] # actual reward collected
blocks_sts     = np.load(os.path.join(sub_data_path, 'blocks_sts.npy'), allow_pickle=True)         # starting states

save_folder    = '/u/gantonov/out_test/save_params_%u'%this_sub # path to save the fitted parameters
out_file       = os.path.join(save_folder, 'backup.txt')

# ---------------- NB when fitting ---------------- #
# specify:                                          #
# -------- number of tasks = (num_abc * num_av) + 1 #
# ------------------------------------------------- #

# fitting necessities
num_abc        = 11 # number of parameter sampled at each iteration
num_av         = 5  # number of simulations run with each parameter
eps            = np.logspace(np.log10(0.6), np.log10(0.10), 55) # tolerance threshold for each ABC iteration
sigma          = np.logspace(np.log10(0.5), np.log10(0.01), 55) # perturbation variance for each ABC iteration

# function that simulates data
def main(params):

    # Initialise the agent
    p = convert_params(params)
    a = Agent(**p)

    training_states = blocks_sts[:7]
    task_states     = blocks_sts[7:]
    
    # Training
    a.train(training_states)

    # Task
    a.task(task_states)
    
    return np.array(a.r_blocks)

# function for running main(params) in parallel 
def simulation(params):
    
    r_data = parallel.sim_pool.map(main, [params]*num_av)
    
    return r_data

# main ABC sampling loop
def sample_loop(from_restart):
    if from_restart:
        z = np.atleast_2d(np.genfromtxt(out_file, skip_header=True))
        it_start      = int(z[-1, -1]) + 1
        theta_old     = z[-1, :-2]
        theta_old[15] = np.log10(theta_old[15])
    else:
        while True:
            priors = [sample_prior() for i in range(num_abc-1)]
            r      = parallel.pool.map(simulation, priors)
            e      = [np.mean([dist(i) for i in j]) for j in r]
            print(e, flush=True)
            if np.any(np.array(e) < eps[0]):
                break
        min_idx   = np.argmin(e)
        theta_old = priors[min_idx]
        print('\n eps: %.4f \n'%eps[0])
        print(theta_old, flush=True)
        backup(out_file, 0, eps[0], theta_old)
        it_start = 1
    
    it = it_start
    for it in range(it_start, len(eps)):
        while True:
            theta_new = [perturb(theta_old, sigma[it]) for i in range(num_abc-1)]
            r         = parallel.pool.map(simulation, theta_new)
            e         = [np.mean([dist(i) for i in j]) for j in r]
            if np.any(np.array(e) < eps[it]):
                break
        min_idx   = np.argmin(e)
        theta_old = priors[min_idx]
        print('\n eps: %.4f \n'%eps[it])
        print(theta_old, flush=True)
        backup(out_file, it, eps[it], theta_old)
        it += 1
        if it == len(eps):
            break
    
    return None

# ABC likelihood
def dist(a):
    perf_agent = []
    perf_true  = []
    for i in range(5):
        for j in range(0, 54, 18):
            perf_agent.append(np.sum(a[i][j:j+18])/np.sum(blocks_max_rwd[i][j:j+18]))
            perf_true.append(np.sum(blocks_rwd[i][j:j+18])/np.sum(blocks_max_rwd[i][j:j+18]))
    err = np.sqrt(np.sum(np.power((np.array(perf_true) - np.array(perf_agent)), 2)))
    return err

if os.path.isfile(out_file):
    from_restart = True
else:
    from_restart = False
    
# fire up MPI and begin fitting
parallel = Parallel(True, False, True, None, num_abc, True)
if parallel.rank in parallel.abc_ranks:
    if parallel.rank == 0:
        # check if directory exists
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        print('Fired up!', flush=True)
    sample_loop(from_restart)
else:
    parallel.sim_pool.worker()

# exit
if parallel.rank in parallel.abc_ranks:
    parallel.pool.close()
if parallel.rank != 0:
    parallel.sim_pool.close()