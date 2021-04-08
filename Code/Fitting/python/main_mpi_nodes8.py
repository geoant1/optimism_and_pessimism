import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat
from scipy.stats import *
import sys, os, glob, csv
from misc import idcs2state, convert_params, get_a_opp, replay_2moves, state2idcs, get_new_state
from agent import Agent
from astroabc.setup_mpi_mp import *

# World 1
world1 = np.array([[0, 9, 3, 5],
                   [8, 2, 1, 10]], dtype=int)

idcs1  = np.array([[0, 1, 2, 3], 
                   [4, 5, 6, 7]])

# World 2
world2 = np.array([[3, 1, 0, 9],
                   [5, 10, 8, 2]], dtype=int)

idcs2  = np.array([[5, 1, 2, 6],
                   [4, 0, 3, 7]])

# Load data
this_sub  = int(sys.argv[1])
data_path = os.path.join('/u/gantonov/data', 'Behavioral Data.mat')
data      = loadmat(data_path)

task_data      = data['TSK'][0, this_sub]
blocks_max_rwd = []
blocks_loc     = []
blocks_rwd     = []

# Convert squalid matlab into something humane
for i in range(7, 12):
    blocks_rwd.append(np.squeeze(task_data[0, i]['D']['rwd'][0][0]))
    blocks_max_rwd.append(np.squeeze(task_data[0, i]['maxRWD']))
    blocks_loc.append(np.squeeze(task_data[0, i]['loc']))

# for i in range(len(blocks_rwd)):
#     for j in range(len(blocks_rwd[i])):
#         if blocks_rwd[i][j] == -5:
#             blocks_rwd[i][j] - blocks_max_rwd[i][j]

for i in range(len(blocks_loc)):
    for j in range(len(blocks_loc[i])):
        blocks_loc[i][j] -= 1
        
# Now create a sequence of states for the agent to start in 
blocks_sts = []
for i in range(len(blocks_loc)):
    this_block = blocks_loc[i]
    tmp = []
    for j in range(len(this_block)):
        these_idcs = this_block[j]
        if j == len(this_block)-1:
            this_state = idcs2state(these_idcs, idcs2)
        else:
            this_state = idcs2state(these_idcs, idcs1)
        tmp.append(this_state)
    blocks_sts.append(tmp)

# Coolio almost done
# Some moves are nan what the heck. Will change to optimal moves instead because why not.
# Actually no I will just ignore those moves. Because this why not is better. 
# Fit

def main(params):

    rwd_all = []

    # Initialise the agent
    p = convert_params(params)
    a = Agent(p)

    # ----------------
    #  Start the task
    # ----------------
    # We have 2 blocks. Each block has 3 epochs with 6 1-move trials followed by 12 2-move trials
    # Every 6 consecutive trials have distinct starting locations. Except for first 24 2-move trials – 
    # in these each starting location is repeated once
    sts = blocks_sts.copy()
    rwd = []
    bl  = 0
    these_states = sts[bl]
    for i in range(3):
        m = a.explore_one_move(world1, idcs1, states=these_states[i*18:i*18+6], num_trials=6)
        for mi in m:
            rwd.append(mi)
        m = a.explore_two_moves(world1, idcs1, states=these_states[i*18+6:i*18+6+12], num_trials=12)
        for mi in m:
            rwd.append(mi)
    rwd_all.append(rwd)

    bl += 1
    these_states = sts[bl]
    rwd = []
    forget = False
    for i in range(3):
        if i == 0:
            m = a.explore_one_move(world1, idcs1, states=these_states[i*18:i*18+6], online_learning=False, num_trials=6)
            for mi in m:
                rwd.append(mi)
            m = a.explore_two_moves(world1, idcs1, states=these_states[i*18+6:i*18+6+6], online_learning=False, num_trials=6)
            for mi in m:
                rwd.append(mi)
            m = a.explore_two_moves(world1, idcs1, states=these_states[i*18+6+6:i*18+6+6+6], online_learning=True, num_trials=6)
            for mi in m:
                rwd.append(mi)
        else:
            m = a.explore_one_move(world1, idcs1, states=these_states[i*18:i*18+6], num_trials=6)
            for mi in m:
                rwd.append(mi)
            if i == 2:
                forget=False
            m = a.explore_two_moves(world1, idcs1, states=these_states[i*18+6:i*18+6+12], num_trials=12, T_forget=forget)
            for mi in m:
                rwd.append(mi)
    rwd_all.append(rwd)

    Q1 = a.Q1
    Q2 = a.Q2

    av_rew2 = np.mean(a.rew_history2)
    av_rew1 = np.mean(a.rew_history1)

    # for i in range(8):
    #      for j in range(4):
    dist = (Q2 - av_rew2)
    Q2   = Q2 - (1-p['tau_forget_block'])*dist

    dist = (Q1 - av_rew1)
    Q1   = Q1 - (1-p['tau_forget_block'])*dist

    T2 = a.T2
    T2 = p['rho'] * T2 + (1-p['rho']) * 1/7 
    for i in range(T2.shape[0]):
        for j in range(T2.shape[1]):
            T2[i, j, i]  = 0

            row = T2[i, j, :]
            tmp = np.sum(row)
            if tmp > 0:
                T2[i, j, :] = row / tmp
    a.T2 = T2
         
    Q1, Q2 = replay_2moves(Q1.copy(), Q2.copy(), T2, world2, p['evm_threshold'], p['beta2'], p['beta1'], p['alpha2r'], p['alpha1r'])
    
    a.Q1 = Q1
    a.Q2 = Q2

    bl += 1
    these_states = sts[bl]
    rwd = []
    for i in range(3):
        if i == 0:
            m = a.explore_one_move(world2, idcs1, states=these_states[i*18:i*18+6], online_learning=False, num_trials=6)
            for mi in m:
                rwd.append(mi)
            m = a.explore_two_moves(world2, idcs1, states=these_states[i*18+6:i*18+6+6], online_learning=False, num_trials=6)
            for mi in m:
                rwd.append(mi)
            m = a.explore_two_moves(world2, idcs1, states=these_states[i*18+6+6:i*18+6+6+6], online_learning=True, num_trials=6)
            for mi in m:
                rwd.append(mi)
        else:
            m = a.explore_one_move(world2, idcs1, states=these_states[i*18:i*18+6], num_trials=6)
            for mi in m:
                rwd.append(mi)
            m = a.explore_two_moves(world2, idcs1, states=these_states[i*18+6:i*18+6+12], num_trials=12)
            for mi in m:
                rwd.append(mi)
    rwd_all.append(rwd)

    bl += 1
    these_states = sts[bl]
    rwd = []
    forget = False
    for i in range(3):
        if i == 0:
            m = a.explore_one_move(world2, idcs1, states=these_states[i*18:i*18+6], online_learning=False, num_trials=6)
            for mi in m:
                rwd.append(mi)
            m = a.explore_two_moves(world2, idcs1, states=these_states[i*18+6:i*18+6+6], online_learning=False, num_trials=6)
            for mi in m:
                rwd.append(mi)
            m = a.explore_two_moves(world2, idcs1, states=these_states[i*18+6+6:i*18+6+6+6], online_learning=True, num_trials=6)
            for mi in m:
                rwd.append(mi)
        else:
            m = a.explore_one_move(world2, idcs1, states=these_states[i*18:i*18+6], num_trials=6)
            for mi in m:
                rwd.append(mi)
            if i == 2:
                forget = True
            m = a.explore_two_moves(world2, idcs1, states=these_states[i*18+6:i*18+6+12], num_trials=12, T_forget=forget)
            for mi in m:
                rwd.append(mi)
    rwd_all.append(rwd)

     # World change
    # Rearrange the model
    T2       = a.T2.copy()
    T2       = p['Block_forget']*T2 + (1-p['Block_forget'])*(1./7)
    for i in range(T2.shape[0]):
        for j in range(T2.shape[1]):
            T2[i, j, i]  = 0

            row = T2[i, j, :]
            tmp = np.sum(row)
            if tmp > 0:
                T2[i, j, :] = row / tmp

    T2_rearr = T2.copy()
    for s in range(8):
        for act in range(4):
            si = state2idcs(s, idcs1)
            s1i_b, _ = get_new_state(si, act, world2, idcs1)
            s1b = idcs2state(s1i_b, idcs1)
            si = state2idcs(s, idcs2)
            s1i_a, _ = get_new_state(si, act, world2, idcs2)
            s1a = idcs2state(s1i_a, idcs2)
            
            T2_rearr[s, act, s1a] = T2[s, act, s1b]
            T2_rearr[s, act, s1b] = T2[s, act, s1a]
            
    new_T2   = (1-p['T_forget_block']) * T2 + p['T_forget_block'] * T2_rearr
    for i in range(new_T2.shape[0]):
        for j in range(new_T2.shape[1]):
            new_T2[i, j, i]  = 0

            row = new_T2[i, j, :]
            tmp = np.sum(row)
            if tmp > 0:
                new_T2[i, j, :] = row / tmp
    a.T2 = new_T2
    
    # Forget MF Q values
    Q2  = a.Q2.copy()
    Q1  = a.Q1.copy()

    av_rew2 = np.mean(a.rew_history2)
    av_rew1 = np.mean(a.rew_history1)

    # for i in range(8):
    #      for j in range(4):
    dist = (Q2 - av_rew2)
    Q2   = Q2 - (1-p['tau_forget_block'])*dist

    dist = (Q1 - av_rew1)
    Q1   = Q1 - (1-p['tau_forget_block'])*dist
 
    Q1, Q2 = replay_2moves(Q1.copy(), Q2.copy(), new_T2, world2, p['evm_threshold'], p['beta2'], p['beta1'], p['alpha2r'], p['alpha1r'])
    
    a.Q2 = Q2
    a.Q1 = Q1

    bl += 1
    these_states = sts[bl]
    rwd = []
    for i in range(3):
        if i == 0:
            m = a.explore_one_move(world2, idcs2, states=these_states[i*18:i*18+6], online_learning=False, num_trials=6)
            for mi in m:
                rwd.append(mi)
            m = a.explore_two_moves(world2, idcs2, states=these_states[i*18+6:i*18+6+6], online_learning=False, num_trials=6)
            for mi in m:
                rwd.append(mi)
            m = a.explore_two_moves(world2, idcs2, states=these_states[i*18+6+6:i*18+6+6+6], online_learning=True, num_trials=6)
            for mi in m:
                rwd.append(mi)
        else:
            m = a.explore_one_move(world2, idcs2, states=these_states[i*18:i*18+6], num_trials=6)
            for mi in m:
                rwd.append(mi)
            m = a.explore_two_moves(world2, idcs2, states=these_states[i*18+6:i*18+6+12], num_trials=12)
            for mi in m:
                rwd.append(mi)
    rwd_all.append(rwd)
    
    return np.array(rwd_all)

# Run
#data_path = os.path.join('/u/gantonov/data', 'Parameter_fits.xlsx')
#df = pd.read_excel(data_path)
#s  = this_sub + 1
save_path = '/u/gantonov/out/save_params_%u'%this_sub

def sample_prior():
    
    p = [np.random.gamma(1, 2),
         np.random.gamma(1, 2),
         np.random.gamma(1, 2),
         np.random.uniform(0, 1),
         np.random.uniform(0, 1),
         np.random.normal(0, 1),
         np.random.beta(6, 2),
         np.random.beta(6, 2),
         np.random.beta(6, 2),
         np.random.beta(6, 2),
         np.random.normal(0, 1),
         np.random.normal(0, 1),
         np.random.uniform(0, 1),
         np.random.uniform(0, 1),
         np.random.uniform(0, 1),
         scipy.stats.loggamma.rvs(0.02, loc=-1, scale=1/100, size=1)[0],
         np.random.uniform(0, 1),
         np.random.uniform(0, 1)]
    
    return p

def perturb(theta, sig):
    
    c0 = [0, 1, 2]
    c1 = [3, 4, 6, 7, 8, 9, 12, 13, 14, 16, 17]
    c2 = 15
    p = np.zeros(18)
    for i in range(18):
        while True:
            p[i] = theta[i] + np.random.normal(0, sig)
            if i in c0:
                if p[i] < 0:
                    continue
                else: break
            elif i in c1:
                if p[i] < 0 or p[i] > 1:
                    continue
                else: break
            elif i == c2:
                if p[i] >= 0:
                    continue
                else: break
            else: break
    return p

def simulation(params):
    
    r_data = parallel.sim_pool.map(main, [params]*5)
    return r_data

def backup(out_file, it, thresh, params):
    if it == 0:
        f = open(out_file, 'w')
        for np in range(len(params)):
            f.write("param#%s \t "%np)
        f.write("dist \t")
        f.write('It \n')
        for p in range(len(params)):
            if p == 15:
                f.write("%.8f \t" % 10**params[p])
            else:
                f.write("%.8f \t" % params[p])
        f.write('%.4f \t'%thresh)
        f.write('%u \n'%it)
    else:
        f = open(out_file, 'a')
        for p in range(len(params)):
            if p == 15:
                f.write("%.8f \t" % 10**params[p])
            else:
                f.write("%.8f \t" % params[p])
        f.write('%.4f \t'%thresh)
        f.write('%u \n'%it)
    f.flush()
    f.close()

def sample_loop(from_restart):
    if from_restart:
        z = np.atleast_2d(np.genfromtxt(out_file, skip_header=True))
        it_start = int(z[-1, -1]) + 1
        theta_old = z[-1, :-2]
    else:
        while True:
            priors = [sample_prior() for i in range(num_abc-1)]
            r = parallel.pool.map(simulation, priors)
            e = [np.mean([dist(i) for i in j]) for j in r]
            if np.any(np.array(e) < eps[0]):
                break
        theta_old = priors[np.argmin(e)]
        print('\n eps: %.4f \n'%eps[0])
        print(theta_old, flush=True)
        backup(out_file, 0, eps[0], theta_old)
        it_start = 1

    for it in range(it_start, len(eps)):
        while True:
            theta_new = [perturb(theta_old, sigma[it]) for i in range(num_abc-1)]
            r = parallel.pool.map(simulation, theta_new)
            e = [np.mean([dist(i) for i in j]) for j in r]
            if np.any(np.array(e) < eps[it]):
                break
        theta_old = theta_new[np.argmin(e)]
        print('\n eps: %.4f \n'%eps[it])
        print(theta_old, flush=True)
        backup(out_file, it, eps[it], theta_old)

def dist(a):
    perf_agent = []
    perf_true  = []
    for i in range(5):
        for j in range(0, 54, 18):
            perf_agent.append(np.sum(a[i][j:j+18])/np.sum(blocks_max_rwd[i][j:j+18]))
            perf_true.append(np.sum(blocks_rwd[i][j:j+18])/np.sum(blocks_max_rwd[i][j:j+18]))
    err = np.sqrt(np.sum(np.power((np.array(perf_true) - np.array(perf_agent)), 2)))
    return err

num_abc = 1001
eps   = np.logspace(np.log10(0.4), np.log10(0.12), 50)
sigma = np.logspace(np.log10(1.0), np.log10(0.01), 50)
save_folder = '/u/gantonov/out/save_params_%u'%this_sub
out_file = os.path.join(save_folder, 'backup.txt')

if os.path.isfile(out_file):
    from_restart = True
else:
    from_restart = False

parallel = Parallel(True, False, True, None, num_abc, True)
if parallel.rank in parallel.abc_ranks:
    sample_loop(from_restart)
else:
    parallel.sim_pool.worker()

if parallel.rank in parallel.abc_ranks:
    parallel.pool.close()
if parallel.rank !=0:
    parallel.sim_pool.close()
