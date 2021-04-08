import numpy as np
from agent_eran_gen import Agent
from misc_eran_gen import get_new_state, state2idcs, idcs2state
import pandas as pd
from scipy.io import loadmat
import os

## -------------------------------------------- ##
#                Initialise worlds               #
## -------------------------------------------- ##
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

num_states  = 8
num_actions = 4 # 0 - up, 1 - down, 2 - left, 3 - right 
                                                          
#############################################
#                                           #
#        World 1              World 2       #
#      __________           __________      #
#     | 0 9 3  5 | ------> | 3  1 0 9 |     #
#     | 8 2 1 10 | ------> | 5 10 8 2 |     #
#      ----------           ----------      #
#                                           #
#############################################

## ------------------------------------------------------------ ## 
#                       Set Agent parameters                     #
## ------------------------------------------------------------ ##
p = {'beta': 0.40356094, # ..................................... # Inverse temperature for 1-move trials
     'beta1': 0.16, # .......................................... # Inverse temperature for 1st move in 2-move trials
     'beta2': 0.27, # .......................................... # Inverse temperature for 2nd move in 2-move trials
     'beta_mb':0.3, # .......................................... #
     'k': 0.8,
     'alpha1': 0.16167969, # ................................... # Learning rate for 1st move in 2-move trials
     'alpha2': 0.5485062, # .................................... # Learning rate for 2nd move in 2-move trials
     'Q_init': -2.7504501, # ................................... # Mean value for initialising Q values
     'T_learn_rate': 0.94392949, # ............................. # State transition learning rate
     'opp_T_learn_rate': 0.78202522, # ......................... # Opposite state transition learning rate
     'tau': (1-0.026778573), # ................................. # Q-value remembrance parameter
     'num_trials': 54, # ....................................... # Maximum number of episodes in the simulation
     'biases': [-0.72859412, 0.050869249, 0, 0.48746705] # ..... # Biases in favour of each action
     }

## ------------------------------------ ##
#                Main part               #
## ------------------------------------ ##
data_path = '/Users/GA/Documents/Dayan_lab/Data/Eran/Parameter_fits.xlsx'
df = pd.read_excel(data_path)

for s in range(1, len(df.columns)):
    
    this_sub  = s
    data_path = os.path.join('/Users/GA/Documents/Dayan_lab/Data/Eran', 'Behavioral Data.mat')
    data      = loadmat(data_path)

    task_data      = data['TSK'][0, (this_sub-1)]
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
    #             blocks_rwd[i][j] = blocks_max_rwd[i][j]

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
        
    save_path = '/Users/GA/Documents/Dayan_lab/Data/Sim_data/Eran/Generate_data/%u'%s

    # Get parameters
    p['beta']             = df[s].iloc[0]
    p['beta2']            = df[s].iloc[1]
    p['beta1']            = df[s].iloc[2]
    p['beta_mb']          = df[s].iloc[3]
    p['alpha2']           = df[s].iloc[4]
    p['alpha1']           = df[s].iloc[5]
    p['Q_init']           = df[s].iloc[6]
    p['tau']              = df[s].iloc[7]
    p['T_learn_rate']     = df[s].iloc[9]
    p['rho']              = df[s].iloc[10]
    p['k']                = df[s].iloc[12]
    p['opp_T_learn_rate'] = df[s].iloc[14]

    p['biases']           = [df[s].iloc[16], df[s].iloc[17], 0, df[s].iloc[16]]

    tau_forget_block      = (1-df[s].iloc[8])
    T_forget_block        = (1-df[s].iloc[11])

    # Initialise the agent
    a = Agent(p)

    # ----------------
    #  Start training
    # ----------------
    sts = blocks_sts.copy()

    bl  = 0
    these_states = sts[bl]
    save_path0 = os.path.join(save_path, str(bl))
    c = 0
    for i in range(3):
        a.explore_one_move(world1, idcs1, states=these_states[i*18:i*18+6], num_trials=6, save_folder=os.path.join(save_path0, str(c)))
        c+=1
        a.explore_two_moves(world1, idcs1, states=these_states[i*18+6:i*18+6+12], num_trials=12, save_folder=os.path.join(save_path0, str(c)))
        c+=1

    bl += 1
    these_states = sts[bl]
    save_path0 = os.path.join(save_path, str(bl))
    c = 0
    for i in range(3):
        if i == 0:
            a.explore_one_move(world1, idcs1, states=these_states[i*18:i*18+6], online_learning=False, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            c+=1
            a.explore_two_moves(world1, idcs1, states=these_states[i*18+6:i*18+6+6], online_learning=False, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            c+=1
            a.explore_two_moves(world1, idcs1, states=these_states[i*18+6+6:i*18+6+6+6], online_learning=True, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            c+=1
        else:
            a.explore_one_move(world1, idcs1, states=these_states[i*18:i*18+6], num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            c+=1
            a.explore_two_moves(world1, idcs1, states=these_states[i*18+6:i*18+6+12], num_trials=12, save_folder=os.path.join(save_path0, str(c)))
            c+=1

    bl += 1
    these_states = sts[bl]
    save_path0 = os.path.join(save_path, str(bl))
    c = 0
    for i in range(3):
        if i == 0:
            a.explore_one_move(world2, idcs1, states=these_states[i*18:i*18+6], online_learning=False, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            c+=1
            a.explore_two_moves(world2, idcs1, states=these_states[i*18+6:i*18+6+6], online_learning=False, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            c+=1
            a.explore_two_moves(world2, idcs1, states=these_states[i*18+6+6:i*18+6+6+6], online_learning=True, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            c+=1
        else:
            a.explore_one_move(world2, idcs1, states=these_states[i*18:i*18+6], num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            c+=1
            a.explore_two_moves(world2, idcs1, states=these_states[i*18+6:i*18+6+12], num_trials=12, save_folder=os.path.join(save_path0, str(c)))
            c+=1

    bl += 1
    these_states = sts[bl]
    save_path0 = os.path.join(save_path, str(bl))
    c = 0
    for i in range(3):
        if i == 0:
            a.explore_one_move(world2, idcs1, states=these_states[i*18:i*18+6], online_learning=False, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            c+=1
            a.explore_two_moves(world2, idcs1, states=these_states[i*18+6:i*18+6+6], online_learning=False, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            c+=1
            a.explore_two_moves(world2, idcs1, states=these_states[i*18+6+6:i*18+6+6+6], online_learning=True, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            c+=1
        else:
            a.explore_one_move(world2, idcs1, states=these_states[i*18:i*18+6], num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            c+=1
            a.explore_two_moves(world2, idcs1, states=these_states[i*18+6:i*18+6+12], num_trials=12, save_folder=os.path.join(save_path0, str(c)))
            c+=1
        
    # World change
    # Rearrange the model
    idx      = [5, 1, 2, 6, 4, 0, 3, 7]
    T2       = a.T2.copy()
    T2_rearr = T2[:, :, idx]
    new_T2   = (1-T_forget_block) * T2 + T_forget_block * T2_rearr
    a.T2     = new_T2

    # Forget MF Q values
    Q2 = a.Q2.copy()
    Q1 = a.Q1.copy()


    # for i in range(8):
    #      for j in range(4):
    Q2 = tau_forget_block*Q2 - (1-tau_forget_block)*p['Q_init']
    Q1 = tau_forget_block*Q1 - (1-tau_forget_block)*p['Q_init']
            
    a.Q2 = Q2
    a.Q1 = Q1

    # Again same 2 blocks with 3 epochs. First epoch has no feedback = no online learning in those
    bl += 1
    these_states = sts[bl]
    save_path0 = os.path.join(save_path, str(bl))
    c = 0
    for i in range(3):
        if i == 0:
            a.explore_one_move(world2, idcs2, states=these_states[i*18:i*18+6], online_learning=False, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            c+=1
            a.explore_two_moves(world2, idcs2, states=these_states[i*18+6:i*18+6+6], online_learning=False, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            c+=1
            a.explore_two_moves(world2, idcs2, states=these_states[i*18+6+6:i*18+6+6+6], online_learning=True, num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            c+=1
        else:
            a.explore_one_move(world2, idcs2, states=these_states[i*18:i*18+6], num_trials=6, save_folder=os.path.join(save_path0, str(c)))
            c+=1
            a.explore_two_moves(world2, idcs2, states=these_states[i*18+6:i*18+6+12], num_trials=12, save_folder=os.path.join(save_path0, str(c)))
            c+=1
    
    print('Done for sess [%u/%u]'%(s, len(df.columns)-1))
          