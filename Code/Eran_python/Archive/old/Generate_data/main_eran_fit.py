import numpy as np
from agent_eran_fit import Agent
from misc_eran_fit import get_new_state, state2idcs
import pandas as pd
import os, json

np.random.seed(3)
## -------------------------------------------- ##
#                Initialise worlds               #
## -------------------------------------------- ##
# World 1
world1 = np.array([[0, 9, 3, 5],
                   [8, 2, 1, 10]], dtype=int)

# World 2
world2 = np.array([[3, 1, 0, 9], 
                   [5, 10, 8, 2]], dtype=int)

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

# True Q-values for the two worlds (1-move trials)
w1_Q_true = np.array([[ 8,  8,  5,  9],
                      [ 2,  2,  0,  3],
                      [ 1,  1,  9,  5],
                      [10, 10,  3,  0],
                      [ 0,  0, 10,  2],
                      [ 9,  9,  8,  1],
                      [ 3,  3,  2, 10],
                      [ 5,  5,  1,  8]])

w2_Q_true = np.array([[ 5,  5,  9,  1],
                      [10, 10,  3,  0],
                      [ 8,  8,  1,  9],
                      [ 2,  2,  0,  3],
                      [ 3,  3,  2, 10],
                      [ 1,  1,  5,  8],
                      [ 0,  0, 10,  2],
                      [ 9,  9,  8,  5]])

# True Q-values for first moves (2-move trials)
w1_Q_true_move1 = np.array([[18, 18, 15,  12],
                            [11, 11,  9,  12],
                            [11, 11, 12,  15],
                            [18, 18, 12,   9],
                            [ 9,  9, 18,  11],
                            [12, 12, 18,  11],
                            [12, 12, 11,  18],
                            [15, 15, 11,  18]])

w2_Q_true_move1 = np.array([[15, 15, 12, 11],
                            [18, 18, 12,  9],
                            [18, 18, 11, 12],
                            [11, 11,  9, 12],
                            [12, 12, 11, 18],
                            [11, 11, 15, 18],
                            [ 9,  9, 18, 11],
                            [12, 12, 18, 15]])

# Optimal moves for 1&2 moves in both worlds 
w1_move1 = np.array([[0, 1], [0, 1], [3], [0, 1], [2], [2], [3], [3]])
w1_move  = np.array([[3], [3], [2], [0, 1], [2], [0, 1], [3], [3]]) # Second move in 2-move trials

w2_move1 = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [3], [3], [2], [2]])
w2_move  = np.array([[2], [0, 1], [3], [3], [3], [3], [2], [0, 1]]) # Second move in 2-move trials

## ------------------------------------------------------------ ## 
#                       Set Agent parameters                     #
## ------------------------------------------------------------ ##
p = {'beta': 0.40356094, # ..................................... # Inverse temperature for 1-move trials
     'beta1': 0.16, # .......................................... # Inverse temperature for 1st move in 2-move trials
     'beta2': 0.27, # .......................................... # Inverse temperature for 2nd move in 2-move trials
     'alpha1': 0.16167969, # ................................... # Learning rate for 1st move in 2-move trials
     'alpha2': 0.5485062, # .................................... # Learning rate for 2nd move in 2-move trials
     'gamma': 0.9, # ........................................... # 2nd move discounting
     'Q_init': -2.7504501, # ................................... # Mean value for initialising Q values
     'T_learn_rate': 0.94392949, # ............................. # State transition learning rate
     'opp_T_learn_rate': 0.78202522, # ......................... # Opposite state transition learning rate
     'rho': (1-0.0021629888), # ................................ # Model memory
     'tau': (1-0.026778573), # ................................. # Q-value remembrance parameter
     'evm_threshold': 0.00007, # ................................ # EVM threshold
     'num_trials': 54, # ....................................... # Maximum number of episodes in the simulation
     'max_num_replays': 2000, # ................................ # Maximum number of planning steps
     'biases': [-0.72859412, 0.050869249, 0, 0.48746705] # ..... # Biases in favour of each action
     }

## ------------------------------------ ##
#                Main part               #
## ------------------------------------ ##
data_path = '/Users/GA/Documents/Dayan_lab/Data/Parameter_fits.xlsx'
df = pd.read_excel(data_path)
s = 5
     
# Get parameters
beta                  = df[s].iloc[0]
beta2                 = df[s].iloc[1]
beta1                 = df[s].iloc[2]
alpha2                = df[s].iloc[4]
alpha1                = df[s].iloc[5]
p['gamma']            = df[s].iloc[12]
p['Q_init']           = df[s].iloc[6]
p['T_learn_rate']     = df[s].iloc[9]
p['opp_T_learn_rate'] = df[s].iloc[14]
rho                   = df[s].iloc[10]
tau                   = df[s].iloc[7]
p['biases']           = [df[s].iloc[16], df[s].iloc[17], 0, df[s].iloc[16]]

tau_forget_block = df[s].iloc[8]
T_forget_block   = df[s].iloc[11]

y = 0
for b in np.linspace(beta/2, beta+(beta/2), 6):
    p['beta'] = b
    for b2 in np.linspace(beta2/2, beta2+(beta2/2), 6):
        p['beta2'] = b2
        for b1 in np.linspace(beta1/2, beta1+(beta1/2), 6):
            p['beta1'] = b1
            for a2 in np.linspace(alpha2/2, alpha2+(alpha2/2), 6):
                p['alpha2'] = a2
                for a1 in np.linspace(alpha1/2, alpha1+(alpha1/2), 6):
                    p['alpha1'] = a1
                    for r in np.linspace(rho/2, rho+(rho/2), 8):
                        p['rho'] = rho
                        for t in np.linspace(tau/2, tau+(tau/2), 8):
                            p['tau'] = tau
                            for e in np.linspace(0.00005, 0.001, 8):
                                p['evm_threshold'] = e
                                print('Done for %u'%y)
                                y += 1
                                save_path = '/Users/GA/Documents/Dayan_lab/Data/Sim_data/Eran/Fit_data/%u'%y
                                
                                # Initialise the agent
                                a = Agent(p)

                                # ----------------
                                #  Start training
                                # ----------------
                                # 6 blocks of 12 1-move trials starting in only 2 states
                                states = np.arange(8)
                                these_states = np.random.choice(states, 2, replace=False)
                                for i in range(6):
                                    path1 = os.path.join(save_path, 'Training/%u'%i)
                                    start_states = np.random.choice(these_states, 12, replace=True)
                                    a.explore_one_move(world1, states=start_states, num_trials=12, save_folder=path1)

                                # 1 block of 48 1-move trials where in the first 24 trials 
                                # each starting location is repeated in the next trial
                                start_states = np.random.choice(states, 12, replace=True)
                                start_states = np.repeat(start_states, 2)
                                for _ in range(24):
                                    start_states = np.append(start_states, np.random.choice(states, 1)[0])
                                    
                                path2 = os.path.join(save_path, 'Training/%u'%(i+1))
                                a.explore_one_move(world1, states=start_states, num_trials=48, save_folder=path2)

                                # ----------------
                                #  Start the task
                                # ----------------
                                # We have 2 blocks. Each block has 3 epochs with 6 1-move trials followed by 12 2-move trials
                                # Every 6 consecutive trials have distinct starting locations. Except for first 24 2-move trials – 
                                # in these each starting location is repeated once
                                av_rew2 = np.mean(a.rew_history2)
                                a.Q1 = np.random.normal(av_rew2, 1, (8, 4))
                                c = 0
                                for i in range(3):
                                    path3 = os.path.join(save_path, 'Task/World1/Block1/%u'%c)
                                    
                                    start_states = np.random.choice(states, 6, replace=False)
                                    a.explore_one_move(world1, states=start_states, num_trials=6, save_folder=path3)
                                    # plot_simulation_1move(path3, world1, w1_move, w1_Q_true, d=None)
                                    c += 1
                                    
                                    path4 = os.path.join(save_path, 'Task/World1/Block1/%u'%c)
                                    
                                    if i != 2:
                                        start_states = np.repeat(np.random.choice(states, 6, replace=False), 2)
                                    else:
                                        start_states = np.append(np.random.choice(states, 6, replace=False), np.random.choice(states, 6, replace=False))
                                    a.explore_two_moves(world1, states=start_states, num_trials=12, save_folder=path4)
                                    # plot_simulation_2moves(path4, world1, [w1_Q_true_move1, w1_Q_true])
                                    c += 1
                                    
                                c = 0
                                for i in range(3):
                                    path3 = os.path.join(save_path, 'Task/World1/Block2/%u'%c)
                                    
                                    start_states = np.random.choice(states, 6, replace=False)
                                    a.explore_one_move(world1, states=start_states, num_trials=6, save_folder=path3)
                                    c += 1
                                    
                                    path4 = os.path.join(save_path, 'Task/World1/Block2/%u'%c)
                                    
                                    start_states = np.append(np.random.choice(states, 6, replace=False), np.random.choice(states, 6, replace=False))
                                    a.explore_two_moves(world1, states=start_states, num_trials=12, save_folder=path4)
                                    c += 1
                                    
                                # World change
                                # Rearrange the model
                                T2 = a.T2.copy()
                                new_T2 = T_forget_block * T2 + (1-T_forget_block) * 1/7
                                a.T2 = new_T2

                                # Forget MF Q values
                                Q2  = a.Q2.copy()
                                Q1 = a.Q1.copy()
                                av_rew2  = np.mean(a.rew_history2)
                                av_rew1 = np.mean(a.rew_history1)

                                # for i in range(8):
                                #      for j in range(4):
                                dist = (Q2 - av_rew2)
                                Q2 = Q2 - (1-tau_forget_block)*dist

                                dist = (Q1 - av_rew1)
                                Q1 = Q1 - (1-tau_forget_block)*dist
                                        
                                a.Q2 = Q2
                                a.Q1 = Q1

                                # Again same 2 blocks with 3 epochs. First epoch has no feedback = no online learning in those
                                c = 0
                                for i in range(3):
                                    if i == 0:
                                        online = False
                                    else:
                                        online = True
                                        
                                    path3 = os.path.join(save_path, 'Task/World2/Block1/%u'%c)
                                    
                                    start_states = np.random.choice(states, 6, replace=False)
                                    a.explore_one_move(world2, states=start_states, num_trials=6, save_folder=path3)
                                    c += 1
                                    
                                    path4 = os.path.join(save_path, 'Task/World2/Block1/%u'%c)
                                    
                                    start_states = np.append(np.random.choice(states, 6, replace=False), np.random.choice(states, 6, replace=False))
                                    a.explore_two_moves(world2, states=start_states, num_trials=12, save_folder=path4)
                                    c += 1
                                    
                                c = 0
                                for i in range(3):
                                    if i == 0:
                                        online = False
                                    else:
                                        online = True
                                        
                                    path3 = os.path.join(save_path, 'Task/World2/Block2/%u'%c)
                                    
                                    start_states = np.random.choice(states, 6, replace=False)
                                    a.explore_one_move(world2, states=start_states, num_trials=6, save_folder=path3)
                                    c += 1
                                    
                                    path4 = os.path.join(save_path, 'Task/World2/Block2/%u'%c)
                                    
                                    start_states = np.append(np.random.choice(states, 6, replace=False), np.random.choice(states, 6, replace=False))
                                    a.explore_two_moves(world2, states=start_states, num_trials=12, save_folder=path4)
                                    c += 1
                                    
                                
                                config_path = os.path.join(save_path, 'config.json')
                                with open(config_path, 'w', encoding='utf-8') as f:
                                    json.dump(p, f, ensure_ascii=False, indent=4)
                                
        
            