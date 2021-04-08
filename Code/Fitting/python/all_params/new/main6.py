import numpy as np
import pandas as pd
from scipy.io import loadmat
import sys, os, glob, csv
from misc import idcs2state, sample_params, get_a_opp, replay_2moves, state2idcs, get_new_state
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

def main(p):
        
    rwd_all = []

    # Initialise the agent
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
            m = a.explore_two_moves(world1, idcs1, states=these_states[i*18+6:i*18+6+12], num_trials=12)
            for mi in m:
                rwd.append(mi)
    rwd_all.append(rwd)

    Q1 = a.Q1
    Q2 = a.Q2
    T2 = a.T2
    
    idcs = []
    for ai in [0, 1]:
        for aj in [0, 1]:
            idcs.append(ai*4+aj)
    for ai in [2, 3]:
        for aj in [2, 3]:
            ai_opp = get_a_opp(ai)
            if aj == ai_opp:
                idcs.append(ai*4+aj)
                    
    replay_exp2 = np.empty((0, 4))
    for sr in range(8):
        for ar in range(4):
            this_action_probas = T2[sr, ar, :]
            s1r  = np.argmax(this_action_probas)
            rr = np.sum(world2.ravel()*this_action_probas)
            
            this_replay = np.array([sr, ar, rr, s1r])
            replay_exp2 = np.vstack((replay_exp2, this_replay))
                    
    Q1, Q2, _, _, _, _, _, _ = replay_2moves(Q1, Q2, replay_exp2, T2, 
                                            world2, p['evm_threshold'], 
                                            idcs, p['beta2'], p['beta1'], 
                                            p['alpha2'], p['alpha1'])
    
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

     # World change
    # Rearrange the model
    T2       = a.T2.copy()
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
    
    replay_exp2 = np.empty((0, 4))
    for sr in range(8):
        for ar in range(4):
            this_action_probas = T2[sr, ar, :]
            s1r  = np.argmax(this_action_probas)
            rr = np.sum(world2.ravel()*this_action_probas)
            
            this_replay = np.array([sr, ar, rr, s1r])
            replay_exp2 = np.vstack((replay_exp2, this_replay))
                    
    Q1, Q2, _, _, _, _, _, _ = replay_2moves(Q1, Q2, replay_exp2, new_T2, 
                                            world2, p['evm_threshold'], 
                                            idcs, p['beta2'], p['beta1'], 
                                            p['alpha2'], p['alpha1'])
    
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
    
    perf_agent = []
    perf_true  = []
    rwd_all    = np.array(rwd_all)
    for i in range(5):
        for j in range(0, 54, 18):
            perf_agent.append(np.sum(rwd_all[i][j:j+18])/np.sum(blocks_max_rwd[i][j:j+18]))
            perf_true.append(np.sum(blocks_rwd[i][j:j+18])/np.sum(blocks_max_rwd[i][j:j+18]))
            
    err = np.sqrt(np.sum(np.power((np.array(perf_true) - np.array(perf_agent)), 2)))
    
    return (p, err)

# Run
#data_path = os.path.join('/u/gantonov/data', 'Parameter_fits.xlsx')
#df = pd.read_excel(data_path)
#s  = this_sub + 1
save_path = '/u/gantonov/out/save_params_%u'%this_sub

num_cores = os.cpu_count()
N   = 100000
eps = 0.4

print('Using %u cores'%num_cores)

# Create workers pool
pool = Pool(num_cores)

for it in range(N):
    p_list = [sample_params() for i in range(num_cores)]                         
    res = pool.map(main, p_list)
    for i in range(len(res)):
        err = res[i][1]
        if err < eps:

            p = res[i][0]
            p['err'] = err
            
            save_folder = os.path.join(save_path, str(it))
            if not os.path.isdir(save_folder):
                os.mkdir(save_folder)
             
            this_num       = np.random.randint(100000000)
            config_path    = os.path.join(save_folder, 'config%u.json'%this_num)

            with open(config_path, 'w', newline="") as csv_file:  
                writer = csv.writer(csv_file)
                for key, value in p.items():
                    writer.writerow([key, value])
pool.close()
