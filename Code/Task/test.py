import numpy as np
import pandas as pd
import os
from misc import convert_params
from agent import Agent
from misc_analysis import get_replay_benefit
import scipy.stats

root_folder   = '/Users/GA/Documents/Dayan_lab/Optimism_And_Pessimism_In_Optimised_Replay/'
data_path     = os.path.join(root_folder, 'Data/subject_data')
params_path   = os.path.join(root_folder, 'Data/new_new_fits')

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
    
    this_sub = 38
    
    for t in range(10):
        
        np.random.seed(this_sub)
        
        # load data
        sub_data_path = os.path.join(data_path, str(this_sub)) # path to subject data

        blocks_sts    = np.load(os.path.join(sub_data_path, 'blocks_sts.npy'), allow_pickle=True)
        # save_folder   = os.path.join(root_folder, 'Data/test', str(this_sub))
        save_folder   = os.path.join(root_folder, 'Data/tmp_test/', str(this_sub))

        # p_arr         = np.load(os.path.join(params_path, 'save_params_%u/params.npy'%this_sub))
        p_arr         = pd.read_csv(os.path.join(params_path, 'save_params_%u'%this_sub, str(t), 'backup.txt'), sep='\t').iloc[-1].values[:-2]
        p             = convert_params(p_arr)
        
        # run
        main(p)
        
        opt_sub = get_replay_benefit(save_folder, p_arr, 'objective', 'value')
        
        _, p_val = scipy.stats.ttest_1samp(opt_sub, 0)
        
        print('t: %u, av: %.3E, p_val: %3E'%(t, np.mean(opt_sub), p_val))