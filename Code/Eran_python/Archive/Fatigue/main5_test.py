import numpy as np
import pandas as pd
from scipy.io import loadmat
import sys, os, glob, csv, shutil
from misc import idcs2state, get_new_state, idcs2state, state2idcs, replay_1move
from analysis_new import analyse_1move, analyse_2moves
from agent import Agent

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

worlds = [world1, world2]
idcs   = [idcs1, idcs2]

p = {'beta': 2, # ..................................... # Inverse temperature for 1-move trials
     'beta1': 0.16, # .......................................... # Inverse temperature for 1st move in 2-move trials
     'beta2': 0.27, # .......................................... # Inverse temperature for 2nd move in 2-move trials
     'alpha1': 0.16167969, # ................................... # Learning rate for 1st move in 2-move trials
     'alpha2': 0.5485062, # .................................... # Learning rate for 2nd move in 2-move trials
     'gamma': 0.7, # ........................................... # 2nd move discounting
     'Q_init': -2.7504501, # ................................... # Mean value for initialising Q values
     'T_learn_rate': 1, # ............................. # State transition learning rate
     'opp_T_learn_rate': 1, # ......................... # Opposite state transition learning rate
     'rho': 0.98, # ................................ # Model memory
     'tau': 0.5, # ................................. # Q-value remembrance parameter
     'biases': [0, 0, 0, 0] # ..... # Biases in favour of each action
     }

save_path = '/Users/GA/Documents/Dayan_lab/Data/Eran/fatigue/evm_gradient'

num_trials = 600
sts = np.random.choice(np.arange(8), size=num_trials, replace=True)

a = Agent(p)
evms = [0.00001, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

c = 0
for t in range(0, num_trials, 50):
    save_path0 = os.path.join(save_path, str(c))
    a.explore_one_move(worlds[c%2], idcs[c%2], states=sts[t:t+50], evm_thresh=evms[c], num_trials=50, save_folder=save_path0)
    analyse_1move(save_path0, worlds[c%2], idcs[c%2])
    # Off-task replay
    c+=1
    if c>= len(evms):
        break
    T2       = a.T2.copy()
    T2_rearr = T2.copy()
    for s in range(8):
        for act in range(4):
            si = state2idcs(s, idcs[(c-1)%2])
            s1i_b, _ = get_new_state(si, act, worlds[(c-1)%2], idcs[(c-1)%2])
            s1b = idcs2state(s1i_b, idcs[(c-1)%2])
            si = state2idcs(s, idcs[c%2])
            s1i_a, _ = get_new_state(si, act, worlds[c%2], idcs[c%2])
            s1a = idcs2state(s1i_a, idcs[c%2])
            
            T2_rearr[s, act, s1a] = T2[s, act, s1b]
            T2_rearr[s, act, s1b] = T2[s, act, s1a]
    a.T2 = T2_rearr
    Q2 = a.Q2.copy()
    
    replay_exp2 = np.empty((0, 4))
    for sr in range(8):
        for ar in range(4):
            this_action_probas = T2_rearr[sr, ar, :]
            s1r  = np.argmax(this_action_probas)
            rr = np.sum(worlds[c%2].ravel()*this_action_probas)
            
            this_replay = np.array([sr, ar, rr, s1r])
            replay_exp2 = np.vstack((replay_exp2, this_replay))
            
    Q2, replay_gain, replay_backups, Q_history = replay_1move(Q2, replay_exp2, evms[c], p['alpha2'], p['beta'])
    a.Q2 = Q2
    
    if os.path.isdir(os.path.join(save_path0, 'Offtask')):
        shutil.rmtree(os.path.join(save_path0, 'Offtask'))
    os.mkdir(os.path.join(save_path0, 'Offtask'))
    
    save_name = os.path.join(save_path0, 'Offtask', 'data.npz')
    np.savez(save_name, Q2=Q2, T2=T2_rearr, replay_gain=replay_gain, replay_backups=replay_backups)
    
save_path = '/Users/GA/Documents/Dayan_lab/Data/Eran/fatigue/unfatigued'

a = Agent(p)

c = 0
for t in range(0, num_trials, 50):
    save_path0 = os.path.join(save_path, str(c))
    a.explore_one_move(worlds[c%2], idcs[c%2], states=sts[t:t+50], evm_thresh=0.00001, num_trials=50, save_folder=save_path0)
    analyse_1move(save_path0, worlds[c%2], idcs[c%2])
    c+=1
    if c>= len(evms):
        break
    T2       = a.T2.copy()
    T2_rearr = T2.copy()
    for s in range(8):
        for act in range(4):
            si = state2idcs(s, idcs[(c-1)%2])
            s1i_b, _ = get_new_state(si, act, worlds[(c-1)%2], idcs[(c-1)%2])
            s1b = idcs2state(s1i_b, idcs[(c-1)%2])
            si = state2idcs(s, idcs[c%2])
            s1i_a, _ = get_new_state(si, act, worlds[c%2], idcs[c%2])
            s1a = idcs2state(s1i_a, idcs[c%2])
            
            T2_rearr[s, act, s1a] = T2[s, act, s1b]
            T2_rearr[s, act, s1b] = T2[s, act, s1a]
    a.T2 = T2_rearr
    Q2 = a.Q2.copy()
    
    replay_exp2 = np.empty((0, 4))
    for sr in range(8):
        for ar in range(4):
            this_action_probas = T2_rearr[sr, ar, :]
            s1r  = np.argmax(this_action_probas)
            rr = np.sum(worlds[c%2].ravel()*this_action_probas)
            
            this_replay = np.array([sr, ar, rr, s1r])
            replay_exp2 = np.vstack((replay_exp2, this_replay))
            
    Q2, replay_gain, replay_backups, _ = replay_1move(Q2, replay_exp2, 0.00001, p['alpha2'], p['beta'])
    a.Q2 = Q2
    
    if os.path.isdir(os.path.join(save_path0, 'Offtask')):
        shutil.rmtree(os.path.join(save_path0, 'Offtask'))
    os.mkdir(os.path.join(save_path0, 'Offtask'))
    
    save_name = os.path.join(save_path0, 'Offtask', 'data.npz')
    np.savez(save_name, Q2=Q2, T2=T2_rearr, replay_gain=replay_gain, replay_backups=replay_backups)