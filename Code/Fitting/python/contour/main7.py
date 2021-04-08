import numpy as np
import os, csv
from agent import Agent
from misc import state2idcs, get_new_state

# World 1
world1 = np.array([[0, 9, 3, 5],
                   [8, 2, 1, 10]], dtype=int)

idcs1  = np.array([[0, 1, 2, 3], 
                   [4, 5, 6, 7]])

save_path = os.path.join('/u/gantonov/data/contour/rho_tau_evm_smooth/')

a = -1.105
b = 1.088
d = -(a+b)

p = {'beta': 1.5,
     'beta2': 1.433,
     'beta1': 0.530,
     'alpha2': 0.5,
     'alpha1': 0.369,
     'Q_init': 0.088,  
     'T_learn_rate': 0.98,
     'opp_T_learn_rate': 0.70,
     'rho': 0.983,
     'tau': 0.958,
     'biases': np.array([a, b, 0, d]),
     'tau_forget_block': 0.111,
     'T_forget_block': 0.902,
     'evm_threshold': 0.0005,
     'num_trials': None}

Q_true = np.zeros((8,4))
for s in range(8):
    si = state2idcs(s, idcs1)
    for a in range(4):
        s1i, r = get_new_state(si, a, world1, idcs1)
        Q_true[s, a] = r
            
states  = np.random.choice(np.arange(8), 300, replace=True)
np.save(os.path.join(save_path, 'states.npy'), states)

max_rwd = np.empty(0)
for s in states:
    max_rwd = np.append(max_rwd, np.amax(Q_true[s, :]))
    
tau  = np.linspace(0.5, 1, 100)
rho  = np.linspace(0.5, 0.6, 20)

perf_matrix = np.zeros((len(rho), len(tau)))
N = 200

for rh in range(len(rho)):
    p['rho'] = rho[rh]
    
    for ta in range(len(tau)):
        p['tau'] = tau[ta]

        tmp = []
        for _ in range(20):
            a = Agent(p)
            r_hist = a.explore_one_move(world1, idcs1, states=states, num_trials=len(states), save_folder=None)
            tmp.append(np.mean(r_hist[N:]/max_rwd[N:]))
        perf_matrix[rh, ta] = np.mean(np.mean(tmp))
        
        print('Done for tau: %u, rho: %u'%(ta, rh), flush=True)
                
np.save(os.path.join(save_path, 'perf_matrix_10.npy'), perf_matrix)
