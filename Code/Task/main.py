import numpy as np
import os
from misc import convert_params
from agent import Agent

root_folder    = '/Users/GA/Documents/Dayan_lab/Optimism_And_Pessimism_In_Optimised_Replay/'
data_path      = os.path.join(root_folder, 'Data/subject_data')
params_path    = os.path.join(root_folder, 'Data/fits')

# load data
this_sub       = 8 # subject number
sub_data_path  = os.path.join(data_path, str(this_sub)) # path to subject data

blocks_sts     = np.load(os.path.join(sub_data_path, 'blocks_sts.npy'), allow_pickle=True)
save_folder    = os.path.join(root_folder, 'Data/test', str(this_sub))

p_arr          = np.load(os.path.join(params_path, 'save_params_%u/params.npy'%this_sub))
p              = convert_params(p_arr)

# function that simulates data
def main(params):
    
    # Initialise the agent
    a = Agent(**params)

    training_states = blocks_sts[:7]
    task_states     = blocks_sts[7:]
    
    # Training
    a.train(training_states, save_folder)

    # Task
    a.task(task_states, save_folder)
    
    return None

if __name__ == '__main__':
    main(p)