import numpy as np
import sys, os, glob, csv, shutil
from misc import idcs2state, get_a_opp, replay_2moves, state2idcs, get_new_state, convert_params
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

data_path = '/Users/GA/Documents/Dayan_lab/Data/Eran/subject_data'

# Load data
subs = [37]

for sub in subs:
    sub_data_path  = os.path.join(data_path, str(sub))
    blocks_sts     = np.load(os.path.join(sub_data_path, 'blocks_sts.npy'), allow_pickle=True)
    save_path      = os.path.join('/Users/GA/Documents/Dayan_lab/Data/Eran/TTASKK', str(sub))
    
    # if not os.path.isdir(save_path):
    #     os.makedirs(save_path)
    # else:
    #     shutil.rmtree(save_path)
    #     os.makedirs(save_path)
    
    params_path = '/Users/GA/Documents/Dayan_lab/Data/Eran/fits/save_params_%u/params.npy'%sub
    p_arr = np.load(params_path)
    p = convert_params(p_arr)
        
    for rep in range(34, 50):

        save_path_rep = os.path.join(save_path, str(rep))
        if not os.path.isdir(save_path_rep):
            os.mkdir(save_path_rep)
        # else:
        #     shutil.rmtree(save_path_rep)
        #     os.makedirs(save_path_rep)
        # Initialise the agent

        a = Agent(p)
        # ----------------
        #  Start the task
        # ----------------
        # We have 2 blocks. Each block has 3 epochs with 6 1-move trials followed by 12 2-move trials
        # Every 6 consecutive trials have distinct starting locations. Except for first 24 2-move trials – 
        # in these each starting location is repeated once

        sts = blocks_sts.copy()

        # Training
        bl = 0
        these_states = sts[bl]
        a.explore_one_move(world1, idcs1, states=these_states, num_trials=6)

        for bl in range(1, 6):
            these_states = sts[bl]
            a.explore_one_move(world1, idcs1, states=these_states, num_trials=12)

        bl = 6
        these_states = sts[bl]
        a.explore_one_move(world1, idcs1, states=these_states, num_trials=48)
        print('Done training')

        bl  = 0
        sts = sts[7:]
        these_states = sts[bl]

        save_path0   = os.path.join(save_path_rep, str(bl))
        if not os.path.isdir(save_path0):
            os.mkdir(save_path0)

        c = 0
        for i in range(3):
            m = a.explore_one_move(world1, idcs1, states=these_states[i*18:i*18+6], num_trials=6)
            np.save(os.path.join(save_path0, 'moves%u.npy'%c), m)
            c+=1
            m = a.explore_two_moves(world1, idcs1, states=these_states[i*18+6:i*18+6+12], num_trials=12)
            np.save(os.path.join(save_path0, 'moves%u.npy'%c), m)
            c+=1

        bl += 1
        these_states = sts[bl]
        save_path0   = os.path.join(save_path_rep, str(bl))
        if not os.path.isdir(save_path0):
            os.mkdir(save_path0)
        c = 0
        forget = False
        for i in range(3):
            if i == 0:
                m = a.explore_one_move(world1, idcs1, states=these_states[i*18:i*18+6], online_learning=False, num_trials=6)
                np.save(os.path.join(save_path0, 'moves%u.npy'%c), m)
                c+=1
                m = a.explore_two_moves(world1, idcs1, states=these_states[i*18+6:i*18+6+6], online_learning=False, num_trials=6)
                np.save(os.path.join(save_path0, 'moves%u.npy'%c), m)
                c+=1
                m = a.explore_two_moves(world1, idcs1, states=these_states[i*18+6+6:i*18+6+6+6], online_learning=True, num_trials=6)
                np.save(os.path.join(save_path0, 'moves%u.npy'%c), m)
                c+=1
            else:
                m = a.explore_one_move(world1, idcs1, states=these_states[i*18:i*18+6], num_trials=6)
                np.save(os.path.join(save_path0, 'moves%u.npy'%c), m)
                c+=1
                if i == 2:
                    forget=True
                m = a.explore_two_moves(world1, idcs1, states=these_states[i*18+6:i*18+6+12], num_trials=12, T_forget=forget)
                np.save(os.path.join(save_path0, 'moves%u.npy'%c), m)
                c+=1

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
        save_path0   = os.path.join(save_path_rep, str(bl))
        if not os.path.isdir(save_path0):
            os.mkdir(save_path0)
        c = 0
        for i in range(3):
            if i == 0:
                m = a.explore_one_move(world2, idcs1, states=these_states[i*18:i*18+6], online_learning=False, num_trials=6)
                np.save(os.path.join(save_path0, 'moves%u.npy'%c), m)
                c+=1
                m = a.explore_two_moves(world2, idcs1, states=these_states[i*18+6:i*18+6+6], online_learning=False, num_trials=6)
                np.save(os.path.join(save_path0, 'moves%u.npy'%c), m)
                c+=1
                m = a.explore_two_moves(world2, idcs1, states=these_states[i*18+6+6:i*18+6+6+6], online_learning=True, num_trials=6)
                np.save(os.path.join(save_path0, 'moves%u.npy'%c), m)
                c+=1
            else:
                m = a.explore_one_move(world2, idcs1, states=these_states[i*18:i*18+6], num_trials=6)
                np.save(os.path.join(save_path0, 'moves%u.npy'%c), m)
                c+=1
                m = a.explore_two_moves(world2, idcs1, states=these_states[i*18+6:i*18+6+12], num_trials=12)
                np.save(os.path.join(save_path0, 'moves%u.npy'%c), m)
                c+=1

        bl += 1
        these_states = sts[bl]
        save_path0   = os.path.join(save_path_rep, str(bl))
        if not os.path.isdir(save_path0):
            os.mkdir(save_path0)
        c = 0
        forget=False
        for i in range(3):
            if i == 0:
                m = a.explore_one_move(world2, idcs1, states=these_states[i*18:i*18+6], online_learning=False, num_trials=6)
                np.save(os.path.join(save_path0, 'moves%u.npy'%c), m)
                c+=1
                m = a.explore_two_moves(world2, idcs1, states=these_states[i*18+6:i*18+6+6], online_learning=False, num_trials=6)
                np.save(os.path.join(save_path0, 'moves%u.npy'%c), m)
                c+=1
                m = a.explore_two_moves(world2, idcs1, states=these_states[i*18+6+6:i*18+6+6+6], online_learning=True, num_trials=6)
                np.save(os.path.join(save_path0, 'moves%u.npy'%c), m)
                c+=1
            else:
                m = a.explore_one_move(world2, idcs1, states=these_states[i*18:i*18+6], num_trials=6)
                np.save(os.path.join(save_path0, 'moves%u.npy'%c), m)
                c+=1
                if i == 2:
                    forget = True
                m = a.explore_two_moves(world2, idcs1, states=these_states[i*18+6:i*18+6+12], num_trials=12, T_forget=forget)
                np.save(os.path.join(save_path0, 'moves%u.npy'%c), m)
                c+=1

        # World change
        # Rearrange the model
        T2 = a.T2.copy()
        T2 = p['Block_forget']*T2 + (1-p['Block_forget'])*(1./7)
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

        for i in range(T2_rearr.shape[0]):
            for j in range(T2_rearr.shape[1]):
                T2_rearr[i, j, i]  = 0
                
                row = T2_rearr[i, j, :]
                tmp = np.sum(row)
                if tmp > 0:
                    T2_rearr[i, j, :] = row / tmp
            
        new_T2 = (1-p['T_forget_block']) * T2 + p['T_forget_block'] * T2_rearr

        for i in range(new_T2.shape[0]):
            for j in range(new_T2.shape[1]):
                new_T2[i, j, i]  = 0
                
                row = new_T2[i, j, :]
                tmp = np.sum(row)
                if tmp > 0:
                    new_T2[i, j, :] = row / tmp
        a.T2 = new_T2

        # Forget MF Q values
        Q2 = a.Q2.copy()
        Q1 = a.Q1.copy()

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
        save_path0   = os.path.join(save_path_rep, str(bl))
        if not os.path.isdir(save_path0):
            os.mkdir(save_path0)
        c = 0
        for i in range(3):
            if i == 0:
                m = a.explore_one_move(world2, idcs2, states=these_states[i*18:i*18+6], online_learning=False, num_trials=6)
                np.save(os.path.join(save_path0, 'moves%u.npy'%c), m)
                c+=1
                m = a.explore_two_moves(world2, idcs2, states=these_states[i*18+6:i*18+6+6], online_learning=False, num_trials=6)
                np.save(os.path.join(save_path0, 'moves%u.npy'%c), m)
                c+=1
                m = a.explore_two_moves(world2, idcs2, states=these_states[i*18+6+6:i*18+6+6+6], online_learning=True, num_trials=6)
                np.save(os.path.join(save_path0, 'moves%u.npy'%c), m)
                c+=1
            else:
                m = a.explore_one_move(world2, idcs2, states=these_states[i*18:i*18+6], num_trials=6)
                np.save(os.path.join(save_path0, 'moves%u.npy'%c), m)
                c+=1
                m = a.explore_two_moves(world2, idcs2, states=these_states[i*18+6:i*18+6+12], num_trials=12)
                np.save(os.path.join(save_path0, 'moves%u.npy'%c), m)
                c+=1
        
        print('Done sub %u rep %u '%(sub, rep))