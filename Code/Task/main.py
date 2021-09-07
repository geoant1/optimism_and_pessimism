import numpy as np
import pandas as pd
import os
from misc import convert_params
from agent import Agent

root_folder = '/Users/GA/Documents/Dayan_lab/Optimism_And_Pessimism_In_Optimised_Replay/'
data_path   = os.path.join(root_folder, 'Data/subject_data')
params_path = os.path.join(root_folder, 'Data/new_new_fits')
num_subs    = 40

# Zero-out MF Q values before off-task replay
ZERO_MF     = True

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
    # 8, 18, 39
    for this_sub in [27]:
        
        np.random.seed(this_sub)
        
        # load data
        sub_data_path = os.path.join(data_path, str(this_sub)) # path to subject data
        blocks_sts    = np.load(os.path.join(sub_data_path, 'blocks_sts.npy'), allow_pickle=True)
        
        if ZERO_MF:
            save_folder = os.path.join(root_folder, 'Data/tmp_zeromf/', str(this_sub))
        else:
            save_folder   = os.path.join(root_folder, 'Data/tmp/', str(this_sub))

        print(save_folder)
        
        # p_arr         = np.load(os.path.join(params_path, 'save_params_%u/params.npy'%this_sub))
        p_arr         = pd.read_csv(os.path.join(params_path, 'save_params_%u'%this_sub, 'backup.txt'), sep='\t').iloc[-1].values[:-2]
        p             = convert_params(p_arr)
        p['ZERO_MF']  = ZERO_MF
        
        # run
        main(p)
        
        print('Done with sub %u'%this_sub)