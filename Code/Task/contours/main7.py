import numpy as np
import os, csv
from agent import Agent
from misc import state2idcs, get_new_state, idcs2state, get_a_opp
from pathos.multiprocessing import ProcessingPool as Pool

# World 1
world1 = np.array([[0, 9, 3, 5],
                   [8, 2, 1, 10]], dtype=int)

idcs1  = np.array([[0, 1, 2, 3], 
                   [4, 5, 6, 7]])

root_path = '/Users/GA/Documents/Dayan_lab/Optimism_And_Pessimism_In_Optimised_Replay/Code/Task/contours/'
perf_save_path = os.path.join(root_path, 'perf_matrix')
rat_save_path  = os.path.join(root_path, 'rat_matrix')

a = -1.105
b = 1.088
d = -(a+b)
p = {'beta': 1.552,
     'beta2': 1.433,
     'beta1': 0.530,
     'alpha2': 0.615,
     'alpha1': 0.369,
     'Q_init': 0.088,  
     'T_learn_rate': 0.98,
     'opp_T_learn_rate': 0.70,
     'rho': 0.983,
     'tau': 0.958,
     'biases':np.array([a, b, 0, d]),
     'tau_forget_block': 0.111,
     'T_forget_block': 0.902,
     'evm_threshold': 0.0001,
     'num_trials': None}

Q_true = np.zeros((8,4))
for s in range(8):
    si = state2idcs(s, idcs1)
    for a in range(4):
        s1i, r = get_new_state(si, a, world1, idcs1)
        Q_true[s, a] = r
        
states  = np.random.choice(np.arange(8), 300, replace=True)
np.save(os.path.join(root_path, 'states.npy'), states)

# states = np.load(os.path.join(root_path, 'states.npy'))

max_rwd = np.empty(0)
for s in states:
    max_rwd = np.append(max_rwd, np.amax(Q_true[s, :]))
    
rho  = np.linspace(0.5, 1, 80)
tau  = np.linspace(0.5, 1, 80)

N = 200
# N = 0

po = Pool(2)
for rh in range(3, len(rho)):
    p['rho'] = rho[rh]
    # this_folder = os.path.join(save_path, str(rh))
    # os.mkdir(this_folder)
    perf_matrix = np.zeros(len(tau))
    rat_matrix   = np.zeros(len(tau))
    for ta in range(len(tau)):
        p['tau'] = tau[ta]

        tmp_rew = []
        tmp_rat = []
        
        for k in range(5):
            
            res = po.map(Agent.explore_one_move, [Agent(p) for i in range(4)], [world1]*4, [idcs1]*4, [states]*4, [len(states)]*4)

            tmp_rew += [np.mean(res[i][0][N:]/max_rwd[N:]) for i in range(4)]
            tmp_rat += [np.mean(res[i][1][N:]) for i in range(4)]

        perf_matrix[ta] = np.mean(tmp_rew)
        rat_matrix[ta]  = np.mean(tmp_rat)
        print('Done for rho: ', rh, 'tau: ', ta)
        # config_path    = os.path.join(save_path0, 'config.json')
        # with open(config_path, 'w', newline="") as csv_file:
        #     writer = csv.writer(csv_file)
        #     for key, value in p.items():
        #         writer.writerow([key, value])

    np.save(os.path.join(perf_save_path, 'perf_matrix_%d.npy'%rh), perf_matrix)
    np.save(os.path.join(rat_save_path, 'rat_matrix_%d.npy'%rh), rat_matrix)

po.close()
po.join()  
po.clear()