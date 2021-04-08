import numpy as np
import pandas as pd
from scipy.io import loadmat
import sys, os, glob, csv
from misc import idcs2state, sample_params
from agent import Agent
from analysis_new import analyse_1move, analyse_2moves

# World 1
world1 = np.array([[0, 9, 3, 5],
                   [8, 2, 1, 10]], dtype=int)

# World 2
world2 = np.array([[3, 1, 0, 9],
                   [5, 10, 8, 2]], dtype=int)

# World 3
world3 = np.array([[10, 1, 0, 8], 
                   [5, 3, 9, 2]], dtype=int)
# Load data
subs = [0,1,4,5,6,7,9,10,11,12,13,14,15,16,17,18,20,22,23,24,25,26,29,30,32,34,35,36,37,38,39]

for l in subs:
    this_sub  = l
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
            this_state = idcs2state(these_idcs, world1)
            tmp.append(this_state)
        blocks_sts.append(tmp)

    # Coolio almost done
    # Some moves are nan what the heck. Will change to optimal moves instead because why not.
    # Actually no I will just ignore those moves. Because this why not is better. 

    # Initialise the agent
    params_path = '/Users/GA/Documents/Dayan_lab/Data/Eran/My_fits/all_params/%u/best/config.json'%this_sub
    with open(params_path, 'r') as csv_file:  
        reader = csv.reader(csv_file)
        p = {rows[0]:rows[1] for rows in reader}
        for k, v in p.items():
            if k == 'biases':
                p[k] = [float(i.rstrip(',')) for i in v[1:-1].split()]
            elif k == 'num_trials':
                p[k] = None
            else:
                p[k] = float(v)
                
    a = Agent(p)
    # ----------------
    #  Start the task
    # ----------------
    # We have 2 blocks. Each block has 3 epochs with 6 1-move trials followed by 12 2-move trials
    # Every 6 consecutive trials have distinct starting locations. Except for first 24 2-move trials – 
    # in these each starting location is repeated once
    save_path = os.path.join('/Users/GA/Documents/Dayan_lab/Data/Eran/Task', str(this_sub))

    sts = blocks_sts.copy()

    bl  = 0
    these_states = sts[bl]
    save_path0 = os.path.join(save_path, str(bl))
    c = 0
    for i in range(3):
        a.explore_one_move(world1, states=these_states[i*18:i*18+6], num_trials=6, save_folder=os.path.join(save_path0, str(c)))
        analyse_1move(os.path.join(save_path0, str(c)), world1)
        c+=1
        a.explore_two_moves(world1, states=these_states[i*18+6:i*18+6+12], num_trials=12, save_folder=os.path.join(save_path0, str(c)))
        analyse_2moves(os.path.join(save_path0, str(c)), world1)
        c+=1

    bl += 1
    these_states = sts[bl]
    save_path0 = os.path.join(save_path, str(bl))
    c = 0
    for i in range(3):
        if i == 0:
            a.explore_one_move(world1, states=these_states[i*18:i*18+6], online_learning=False, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            analyse_1move(os.path.join(save_path0, str(c)), world1)
            c+=1
            a.explore_two_moves(world1, states=these_states[i*18+6:i*18+6+6], online_learning=False, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            analyse_2moves(os.path.join(save_path0, str(c)), world1)
            c+=1
            a.explore_two_moves(world1, states=these_states[i*18+6+6:i*18+6+6+6], online_learning=True, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            analyse_2moves(os.path.join(save_path0, str(c)), world1)
            c+=1
        else:
            a.explore_one_move(world1, states=these_states[i*18:i*18+6], num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            analyse_1move(os.path.join(save_path0, str(c)), world1)
            c+=1
            a.explore_two_moves(world1, states=these_states[i*18+6:i*18+6+12], num_trials=12, save_folder=os.path.join(save_path0, str(c)))
            analyse_2moves(os.path.join(save_path0, str(c)), world1)
            c+=1

    bl += 1
    these_states = sts[bl]
    save_path0 = os.path.join(save_path, str(bl))
    c = 0
    for i in range(3):
        if i == 0:
            a.explore_one_move(world2, states=these_states[i*18:i*18+6], online_learning=False, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            analyse_1move(os.path.join(save_path0, str(c)), world2)
            c+=1
            a.explore_two_moves(world2, states=these_states[i*18+6:i*18+6+6], online_learning=False, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            analyse_2moves(os.path.join(save_path0, str(c)), world2)
            c+=1
            a.explore_two_moves(world2, states=these_states[i*18+6+6:i*18+6+6+6], online_learning=True, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            c+=1
        else:
            a.explore_one_move(world2, states=these_states[i*18:i*18+6], num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            analyse_1move(os.path.join(save_path0, str(c)), world2)
            c+=1
            a.explore_two_moves(world2, states=these_states[i*18+6:i*18+6+12], num_trials=12, save_folder=os.path.join(save_path0, str(c)))
            analyse_2moves(os.path.join(save_path0, str(c)), world2)
            c+=1

    bl += 1
    these_states = sts[bl]
    save_path0 = os.path.join(save_path, str(bl))
    c = 0
    for i in range(3):
        if i == 0:
            a.explore_one_move(world2, states=these_states[i*18:i*18+6], online_learning=False, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            analyse_1move(os.path.join(save_path0, str(c)), world2)
            c+=1
            a.explore_two_moves(world2, states=these_states[i*18+6:i*18+6+6], online_learning=False, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            analyse_2moves(os.path.join(save_path0, str(c)), world2)
            c+=1
            a.explore_two_moves(world2, states=these_states[i*18+6+6:i*18+6+6+6], online_learning=True, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            analyse_2moves(os.path.join(save_path0, str(c)), world2)
            c+=1
        else:
            a.explore_one_move(world2, states=these_states[i*18:i*18+6], num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            analyse_1move(os.path.join(save_path0, str(c)), world2)
            c+=1
            a.explore_two_moves(world2, states=these_states[i*18+6:i*18+6+12], num_trials=12, save_folder=os.path.join(save_path0, str(c)))
            analyse_2moves(os.path.join(save_path0, str(c)), world2)
            c+=1

    # World change
    # Rearrange the model
    T2 = a.T2.copy()
    new_T2 = p['T_forget_block'] * T2 + (1-p['T_forget_block']) * 1/7
    a.T2 = new_T2

    # Forget MF Q values
    Q2  = a.Q2.copy()
    Q1 = a.Q1.copy()
    av_rew2  = np.mean(a.rew_history2)
    av_rew1 = np.mean(a.rew_history1)

    # for i in range(8):
    #      for j in range(4):
    dist = (Q2 - av_rew2)
    Q2 = Q2 - (1-p['tau_forget_block'])*dist

    dist = (Q1 - av_rew1)
    Q1 = Q1 - (1-p['tau_forget_block'])*dist
            
    a.Q2 = Q2
    a.Q1 = Q1

    bl += 1
    these_states = sts[bl]
    save_path0 = os.path.join(save_path, str(bl))
    c = 0
    for i in range(3):
        if i == 0:
            a.explore_one_move(world3, states=these_states[i*18:i*18+6], online_learning=False, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            analyse_1move(os.path.join(save_path0, str(c)), world3)
            c+=1
            a.explore_two_moves(world3, states=these_states[i*18+6:i*18+6+6], online_learning=False, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            analyse_2moves(os.path.join(save_path0, str(c)), world3)
            c+=1
            a.explore_two_moves(world3, states=these_states[i*18+6+6:i*18+6+6+6], online_learning=True, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            analyse_2moves(os.path.join(save_path0, str(c)), world3)
            c+=1
        else:
            a.explore_one_move(world3, states=these_states[i*18:i*18+6], num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            analyse_1move(os.path.join(save_path0, str(c)), world3)
            c+=1
            a.explore_two_moves(world3, states=these_states[i*18+6:i*18+6+12], num_trials=12, save_folder=os.path.join(save_path0, str(c)))
            analyse_2moves(os.path.join(save_path0, str(c)), world3)
            c+=1
    print(l)