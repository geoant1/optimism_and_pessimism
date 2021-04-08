import numpy as np
import pandas as pd
from scipy.io import loadmat
import sys, os, glob, csv
from misc import idcs2state, convert_params, get_a_opp, replay_2moves, state2idcs, get_new_state
from agent import Agent
from multiprocessing import Pool

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
this_sub  = 0
data_path = os.path.join('/Users/GA/Documents/Dayan_lab/Data/Eran', 'Behavioral Data.mat')
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
    forget=False
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
                forget=True
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
    a.T2 = T2
         
    Q1, Q2 = replay_2moves(Q1.copy(), Q2.copy(), T2, world2, p['evm_threshold'], p['beta2'], p['beta1'], p['alpha2'], p['alpha1'])
    
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
 
    Q1, Q2 = replay_2moves(Q1.copy(), Q2.copy(), new_T2, world2, p['evm_threshold'], p['beta2'], p['beta1'], p['alpha2'], p['alpha1'])
    
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
         np.random.gamma(1, 0.01)]
    
    return p

def perturb(theta, sigma):
    
    while True:
        
        c = theta + np.random.normal(0, sigma, size=16)

        c1 = c[0] < 0
        c2 = c[1] < 0
        c3 = c[2] < 0
        c4 = c[3] < 0 or c[3] > 1
        c5 = c[4] < 0 or c[4] > 1
        c6 = c[6] < 0 or c[6] > 1
        c7 = c[7] < 0 or c[7] > 1
        c8 = c[8] < 0 or c[8] > 1
        c9 = c[9] < 0 or c[9] > 1
        c10 = c[12] < 0 or c[12] > 1
        c11 = c[13] < 0 or c[13] > 1
        c12 = c[14] < 0 or c[14] > 1
        c13 = c[15] < 0 or c[15] > 1
        
        if c1 or c2 or c3 or c4 or c5 or c6 or c7 or c8 or c9 or c10 or c11 or c12 or c13:
            pass
        else:
            return c

def dist(a):
    perf_agent = []
    perf_true  = []
    for i in range(5):
        for j in range(0, 54, 18):
            perf_agent.append(np.sum(a[i][j:j+18])/np.sum(blocks_max_rwd[i][j:j+18]))
            perf_true.append(np.sum(blocks_rwd[i][j:j+18])/np.sum(blocks_max_rwd[i][j:j+18]))
            
    err = np.sqrt(np.sum(np.power((np.array(perf_true) - np.array(perf_agent)), 2)))
    return err

eps   = np.logspace(np.log10(0.6), np.log10(0.2), 25)
sigma = np.logspace(np.log10(1), np.log10(0.01), 25)

with Pool(3) as p:
    
    it = 0
    while True:
        
        priors = [sample_prior()]*3
        r = p.map(main, priors)
        e = p.map(dist, r)
        e = np.mean(e)
        
        if e < eps[it]:
            break
    
    theta_old = priors[0]
    print(theta_old)
    
    for it in range(1, len(eps)):
        
        while True:
            
            theta_new = perturb(theta_old, sigma[it])
            
            theta_new = [theta_new]*3
            r = p.map(main, theta_new)
            e = p.map(dist, r)
            e = np.mean(e)
            
            if e < eps[it]:
                break
        
        print('\n eps: %.2f \n'%eps[it])
        theta_old = theta_new[0]
        print(theta_old)