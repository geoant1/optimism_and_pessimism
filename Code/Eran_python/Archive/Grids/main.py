#!/usr/bin/env python
#%%
import numpy as np
from agent_h import Agent

# np.random.seed(1234)

# Set world parameters
maze = np.zeros((7, 9))
maze[1:, :4] = 1
maze[1:, 5:] = 1

start_idcs = np.array([6, 4])

r = {'pos_val':  5, # ........ # reward value
     'pos_loc': np.array([0, 0]), # .. # reward locationd
     'neg_val': -5,
     'neg_loc': np.array([0, 8])
     }

R = np.zeros((7*9*4))
R[(1*4+2)] = 5
# R[(10*4+0)] = 5
R[(7*4+3)] = -5
# R[(19*4+0)] = -10

# Set agent parameters
p = {'temperature': 0.2, # .............. # Temperature policy parameter
     'gamma': 0.99, #..................... # Discount factor
     'alpha': 0.2, # .................... # Learning rate
     'lambda': 0, # ..................... # Eligibility trace parameter
     't_learn_rate': 0.9, # ............. # State transition learning rate
     'tau': 0.98,
     'rho': 0.9,
     'opp_t_learn_rate': 1,
     'EVM_threshold': 0.00001, # ......... # EVM threshold
     'num_sim_steps': int(1e8), # ....... # Number of steps in the simulation
     'max_num_episodes': 150, # .......... # Maximum number of episodes in the simulation
     'num_plan_steps': 500, # .......... # Maximum number of planning steps
     'plan_at_start_or_goal': True, # ... # Plan only at start or goal locations
     'expand_further': True, # .......... # Expand previous backup
     'allow_loops': False, # ............ # Allow loops in backups
     'start_to_goal': True, # ........... # Add start-to-goal transitions
     'start_location': start_idcs, # .... # Start location indices
     'r_params': r, # ................... # Reward parametersc
     'R': R
     }


#Main
#%%
a = Agent(p, maze)

a.pre_explore()

save_folder = '/Users/GA/Documents/Dayan_lab/Data/Sim_data/Grids/Data/All_average'
a.run_simulation(plot_location=False, save_folder=save_folder)



# %%
