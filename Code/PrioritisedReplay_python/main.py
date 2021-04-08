#!/usr/bin/env python
#%%
import numpy as np
from misc import init_maze
from agent import Agent

np.random.seed(1234)

# Set world parameters
side_i = 6
side_j = 9
obst   = [[1, 2], [2, 2], [3, 2], 
          [4, 5], 
          [0, 7], [1, 7], [2, 7]]

maze = init_maze(side_i, side_j, obst)
##############################################
# __________________________________________ #
#           1s correspond to walls           # 
#            ____________________            # 
#           | 0 0 0 0 0 0 0 1 r | <-- goal   #
#           | 0 0 1 0 0 0 0 1 0 |            #
# start --> | s 0 1 0 0 0 0 1 0 |            #
#           | 0 0 1 0 0 0 0 0 0 |            #
#           | 0 0 0 0 0 1 0 0 0 |            #
#           | 0 0 0 0 0 0 0 0 0 |            #
#            --------------------            #
# __________________________________________ #
##############################################

start_idcs = np.array([2, 0])
goal_idcs  = np.array([0, 8])

r = {'val':  1, # ........ # reward value
     'std':  0.1, # ...... # reward std
     'prob': 1, # ........ # reward probability
     'loc': goal_idcs # .. # reward location
     }

# Set agent parameters
p = {'act_policy': 'softmax', # ......... # Policy type
     'temperature': 5, # ................ # Temperature policy parameter
     'on_vs_off_policy': 'off_policy', #  # Policy learning 
     'gamma': 0.9, #..................... # Discount factor
     'alpha': 1, # ...................... # Learning rate
     'lambda': 0, # ..................... # Eligibility trace parameter
     't_learn_rate': 0.9, # ............. # State transition learning rate
     'baseline_gain': 1e-10, # .......... # Baseline gain 
     'EVM_threshold': 0.0, # ............ # EVM threshold
     'num_sim_steps': int(1e5), # ....... # Number of steps in the simulation
     'max_num_episodes': 50, # .......... # Maximum number of episodes in the simulation
     'num_plan_steps': 20, # ............ # Maximum number of planning steps
     'plan_at_start_or_goal': True, # ... # Plan only at start or goal locations
     'expand_further': True, # .......... # Expand previous backup
     'allow_loops': False, # ............ # Allow loops in backups
     'remove_same_state': True, # ....... # Remove experiences reaching the same state
     'start_to_goal': True, # ........... # Add start-to-goal transitions
     'start_location': start_idcs, # .... # Start location indices
     'goal_location': goal_idcs, # ...... # Goal location indices
     'r_params': r # .................... # Reward parameters
     }


#Main
a = Agent(p, maze)

a.pre_explore()

a.run_simulation(plot_location=True)



# %%
