import numpy as np
import os
from misc import replay_2moves, convert_params, normalise, model_rearrange
from analysis import analyse_1move, analyse_2moves
from agent import Agent

# set paths 
root_path    = '/Users/GA/Documents/Dayan_lab/Optimism_And_Pessimism_In_Optimised_Replay/'
data_path    = os.path.join(root_path, 'Data/subject_data')
path_to_save = os.path.join(root_path, 'Data/task')

N_subs       = 40
ZERO_MF      = False
# subs_who_replay = np.load(os.path.join(root_path, 'Data/task/Analysis/subs_who_replay.npy'))

# Worlds & idcs
world1 = np.load(os.path.join(root_path, 'Data/world1.npy'))
world2 = np.load(os.path.join(root_path, 'Data/world2.npy'))
idcs1  = np.load(os.path.join(root_path, 'Data/idcs1.npy'))
idcs2  = np.load(os.path.join(root_path, 'Data/idcs2.npy'))

def main():
    
    # Load data
    for sub in range(N_subs):

        sub_data_path = os.path.join(data_path, str(sub))
        blocks_sts    = np.load(os.path.join(sub_data_path, 'blocks_sts.npy'), allow_pickle=True)
        save_path     = os.path.join(path_to_save, str(sub))

        # Initialise the agent
        p_arr = np.load(os.path.join(root_path, 'Data/fits/save_params_%u/params.npy'%sub))
        p     = convert_params(p_arr)
        a     = Agent(**p)
        
        sts = blocks_sts.copy()
        
        # -------------------
        #  Start the training
        # -------------------

        save_path0 = os.path.join(save_path, 'training')
        for bl in range(0, 7):
            these_states = sts[bl]
            if bl == 0:
                a.explore_one_move(world1, idcs1, states=these_states, num_trials=6, save_folder=os.path.join(save_path0, str(bl)))
            elif bl == 6:
                a.explore_one_move(world1, idcs1, states=these_states, num_trials=48, save_folder=os.path.join(save_path0, str(bl)))
            else:
                a.explore_one_move(world1, idcs1, states=these_states, num_trials=12, save_folder=os.path.join(save_path0, str(bl)))
            
        print('Done training')
        
        # -------------------
        #  Start the task
        # -------------------
        bl  = 0
        sts = sts[7:]
        
        these_states = sts[bl]
        save_path0   = os.path.join(save_path, str(bl))
        c = 0
        for i in range(3):
            a.explore_one_move(world1, idcs1, states=these_states[i*18:i*18+6], num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            analyse_1move(os.path.join(save_path0, str(c)), world1, idcs1)
            c+=1
            a.explore_two_moves(world1, idcs1, states=these_states[i*18+6:i*18+6+12], num_trials=12, save_folder=os.path.join(save_path0, str(c)))
            analyse_2moves(os.path.join(save_path0, str(c)), world1, idcs1)
            c+=1

        bl += 1
        these_states = sts[bl]
        save_path0   = os.path.join(save_path, str(bl))
        c = 0
        forget = False
        for i in range(3):
            if i == 0:
                a.explore_one_move(world1, idcs1, states=these_states[i*18:i*18+6], online_learning=False, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
                analyse_1move(os.path.join(save_path0, str(c)), world1, idcs1)
                c+=1
                a.explore_two_moves(world1, idcs1, states=these_states[i*18+6:i*18+6+6], online_learning=False, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
                analyse_2moves(os.path.join(save_path0, str(c)), world1, idcs1)
                c+=1
                a.explore_two_moves(world1, idcs1, states=these_states[i*18+6+6:i*18+6+6+6], online_learning=True, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
                analyse_2moves(os.path.join(save_path0, str(c)), world1, idcs1)
                c+=1
            else:
                a.explore_one_move(world1, idcs1, states=these_states[i*18:i*18+6], num_trials=6, save_folder=os.path.join(save_path0, str(c)))
                analyse_1move(os.path.join(save_path0, str(c)), world1, idcs1)
                c+=1
                if i == 2:
                    forget=True
                a.explore_two_moves(world1, idcs1, states=these_states[i*18+6:i*18+6+12], num_trials=12, save_folder=os.path.join(save_path0, str(c)), T_forget=forget)
                analyse_2moves(os.path.join(save_path0, str(c)), world1, idcs1)
                c+=1

        # Off-task replay
        Q1 = a.Q1.copy()
        Q2 = a.Q2.copy()

        # Obliviate MF knowledge
        if ZERO_MF:
            Q2 = np.zeros(Q2.shape)
            Q1 = np.zeros(Q1.shape)
        else:
            av_rew2 = np.mean(a.rew_history2)
            av_rew1 = np.mean(a.rew_history1)

            dist = (Q2 - av_rew2)
            Q2   = Q2 - (1-p['tau_forget_block'])*dist

            dist = (Q1 - av_rew1)
            Q1   = Q1 - (1-p['tau_forget_block'])*dist

        T2 = a.T2
        T2 = p['rho'] * T2 + (1-p['rho']) * 1/7 
        T2 = normalise(T2)
        a.T2 = T2
                
        Q1, Q2, Q1_history, Q2_history, replay_backups, replay_gain1, replay_gain2 = replay_2moves(Q1.copy(), Q2.copy(), T2, world2, p['xi'], p['beta2'], p['beta1'], p['alpha2r'], p['alpha1r'])

        save_name = os.path.join(save_path0, 'offline')
        np.savez(save_name, T=T2, Q1_history=Q1_history, Q2_history=Q2_history, replay_backups=replay_backups, replay_gain1=replay_gain1, replay_gain2=replay_gain2, rew_history=[a.rew_history1, a.rew_history2])
        a.Q1 = Q1
        a.Q2 = Q2

        bl += 1
        these_states = sts[bl]
        save_path0   = os.path.join(save_path, str(bl))
        c = 0
        for i in range(3):
            if i == 0:
                a.explore_one_move(world2, idcs1, states=these_states[i*18:i*18+6], online_learning=False, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
                analyse_1move(os.path.join(save_path0, str(c)), world2, idcs1)
                c+=1
                a.explore_two_moves(world2, idcs1, states=these_states[i*18+6:i*18+6+6], online_learning=False, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
                analyse_2moves(os.path.join(save_path0, str(c)), world2, idcs1)
                c+=1
                a.explore_two_moves(world2, idcs1, states=these_states[i*18+6+6:i*18+6+6+6], online_learning=True, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
                analyse_2moves(os.path.join(save_path0, str(c)), world2, idcs1)
                c+=1
            else:
                a.explore_one_move(world2, idcs1, states=these_states[i*18:i*18+6], num_trials=6, save_folder=os.path.join(save_path0, str(c)))
                analyse_1move(os.path.join(save_path0, str(c)), world2, idcs1)
                c+=1
                a.explore_two_moves(world2, idcs1, states=these_states[i*18+6:i*18+6+12], num_trials=12, save_folder=os.path.join(save_path0, str(c)))
                analyse_2moves(os.path.join(save_path0, str(c)), world2, idcs1)
                c+=1

        bl += 1
        these_states = sts[bl]
        save_path0   = os.path.join(save_path, str(bl))
        c = 0
        forget=False
        for i in range(3):
            if i == 0:
                a.explore_one_move(world2, idcs1, states=these_states[i*18:i*18+6], online_learning=False, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
                analyse_1move(os.path.join(save_path0, str(c)), world2, idcs1)
                c+=1
                a.explore_two_moves(world2, idcs1, states=these_states[i*18+6:i*18+6+6], online_learning=False, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
                analyse_2moves(os.path.join(save_path0, str(c)), world2, idcs1)
                c+=1
                a.explore_two_moves(world2, idcs1, states=these_states[i*18+6+6:i*18+6+6+6], online_learning=True, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
                analyse_2moves(os.path.join(save_path0, str(c)), world2, idcs1)
                c+=1
            else:
                a.explore_one_move(world2, idcs1, states=these_states[i*18:i*18+6], num_trials=6, save_folder=os.path.join(save_path0, str(c)))
                analyse_1move(os.path.join(save_path0, str(c)), world2, idcs1)
                c+=1
                if i == 2:
                    forget = True
                a.explore_two_moves(world2, idcs1, states=these_states[i*18+6:i*18+6+12], num_trials=12, save_folder=os.path.join(save_path0, str(c)), T_forget=forget)
                analyse_2moves(os.path.join(save_path0, str(c)), world2, idcs1)
                c+=1

        # World change
        # Rearrange the model
        T2 = a.T2.copy()
        T2 = p['Block_forget']*T2 + (1-p['Block_forget'])*(1./7)
        T2 = normalise(T2)
                    
        T2_rearr = model_rearrange(T2, idcs1, idcs2, world2)
        T2_rearr = normalise(T2_rearr)
            
        new_T2 = (1-p['T_forget_block']) * T2 + p['T_forget_block'] * T2_rearr
        new_T2 = normalise(new_T2)

        # Forget MF Q values
        Q2 = a.Q2.copy()
        Q1 = a.Q1.copy()

        if ZERO_MF:
            Q2 = np.zeros(Q2.shape)
            Q1 = np.zeros(Q1.shape)
        else:
            av_rew2 = np.mean(a.rew_history2)
            av_rew1 = np.mean(a.rew_history1)

            dist = (Q2 - av_rew2)
            Q2   = Q2 - (1-p['tau_forget_block'])*dist

            dist = (Q1 - av_rew1)
            Q1   = Q1 - (1-p['tau_forget_block'])*dist

        Q1, Q2, Q1_history, Q2_history, replay_backups, replay_gain1, replay_gain2 = replay_2moves(Q1.copy(), Q2.copy(), new_T2, world2, p['xi'], p['beta2'], p['beta1'], p['alpha2r'], p['alpha1r'])

        save_name = os.path.join(save_path0, 'offline')
        np.savez(save_name, T=new_T2, Q1_history=Q1_history, Q2_history=Q2_history, replay_backups=replay_backups, replay_gain1=replay_gain1, replay_gain2=replay_gain2, rew_history=[a.rew_history1, a.rew_history2])

        a.Q2 = Q2
        a.Q1 = Q1

        bl += 1
        these_states = sts[bl]
        save_path0   = os.path.join(save_path, str(bl))
        c = 0
        for i in range(3):
            if i == 0:
                a.explore_one_move(world2, idcs2, states=these_states[i*18:i*18+6], online_learning=False, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
                analyse_1move(os.path.join(save_path0, str(c)), world2, idcs2)
                c+=1
                a.explore_two_moves(world2, idcs2, states=these_states[i*18+6:i*18+6+6], online_learning=False, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
                analyse_2moves(os.path.join(save_path0, str(c)), world2, idcs2)
                c+=1
                a.explore_two_moves(world2, idcs2, states=these_states[i*18+6+6:i*18+6+6+6], online_learning=True, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
                analyse_2moves(os.path.join(save_path0, str(c)), world2, idcs2)
                c+=1
            else:
                a.explore_one_move(world2, idcs2, states=these_states[i*18:i*18+6], num_trials=6, save_folder=os.path.join(save_path0, str(c)))
                analyse_1move(os.path.join(save_path0, str(c)), world2, idcs2)
                c+=1
                a.explore_two_moves(world2, idcs2, states=these_states[i*18+6:i*18+6+12], num_trials=12, save_folder=os.path.join(save_path0, str(c)))
                analyse_2moves(os.path.join(save_path0, str(c)), world2, idcs2)
                c+=1
                
        print('Done: ', sub)
        
if __name__ == '__main__':
    main()