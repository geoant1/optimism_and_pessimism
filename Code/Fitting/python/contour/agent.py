import numpy as np
import os
import shutil
from misc import *

class Agent:
    
    def __init__(self, p: dict):    
        
        # ---- Agent parameters ----
        # Inverse temperatures
        self.beta             = p['beta']
        self.beta1            = p['beta1']
        self.beta2            = p['beta2']        
        # Learning rates
        self.alpha1           = p['alpha1']
        self.alpha2           = p['alpha2']
        
        # MF and MB parameters
        self.Q_init           = p['Q_init']
        self.T_learn_rate     = p['T_learn_rate']
        self.opp_T_learn_rate = p['opp_T_learn_rate']
        self.rho              = p['rho']
        self.tau              = p['tau']
        
        self.evm_thresh       = p['evm_threshold']
        self.biases           = p['biases']
        
        # Simulation parameters
        self.num_trials       = p['num_trials']
        
        # World parameters – get rid of this later
        self.num_states  = 8
        self.num_actions = 4 # as always
        
        # Pre-initialise all learning neccessities
        # 1-move trials
        self.Q2 = np.random.normal(self.Q_init, 1, (8, 4)) # MF Q values for 1-move trials and second moves in 2-move trials
        self.T2 = np.full((self.num_states, self.num_actions, self.num_states), 1/7, dtype=float) # state transition probability model 
        for i in range(self.num_states):
            for j in range(self.num_actions):
                self.T2[i, j, i] = 0 # Self-transitions are not allowed
        
        # Buffers for rewards experienced by the agent
        self.rew_history2 = []
                
        return
    
    def explore_one_move(self, world, state_arr, states=None, num_trials=None, online_learning=True, save_folder=None):
        '''Run 1-move trials'''
        
        if save_folder is not None:
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)
            else:
                shutil.rmtree(save_folder)
                os.makedirs(save_folder)
                
        if num_trials:
            self.num_trials = num_trials
            
        # Initialise in a random state (unless specific states provided)
        if states is not None:
            s_counter = 0
            s = states[s_counter]
        else:
            s = np.random.randint(self.num_states)

        si = state2idcs(s, state_arr)
        
        r_list = []
        # Start exploring the maze
        for trial in range(self.num_trials):
            
            # Choose an action
            q_vals = self.Q2[s, :]
            probs  = policy_choose(q_vals, self.beta, self.biases)
            a      = np.random.choice(range(self.num_actions), p=probs)
            
            # Execute the action, move to the next state and receive reward
            s1i, r = get_new_state(si, a, world, state_arr)
            s1     = idcs2state(s1i, state_arr)
            
            # online TD-learning
            if online_learning:
                Q_target = r        
                delta    = Q_target - self.Q2[s, a]
                self.Q2[s, a] = self.Q2[s, a] + self.alpha2 * delta
                self.rew_history2.append(r)
                
                # Update the state transition model
                delta_t = 1 - self.T2[s, a, s1]
                self.T2[s, a, s1] += self.T_learn_rate * delta_t
                
                # NB we also learn about opposite transitions
                a_opp = get_a_opp(a)
                delta_t_opp = 1 - self.T2[s1, a_opp, s]
                self.T2[s1, a_opp, s] += self.T_learn_rate * self.opp_T_learn_rate * delta_t_opp
                
                # Normalise the transition matrix
                for i in range(self.num_states):
                    for j in range(self.num_actions):
                        self.T2[i, j, i] = 0
                        row = self.T2[i, j, :]
                        tmp = np.sum(row)
                        if tmp > 0:
                            self.T2[i, j, :] = row / tmp
            
            # Prepare replay
            replay = True # initialise replay 
            # Generate replays from the model
            replay_exp = np.empty((0, 4))
            for sr in range(8):
                for ar in range(4):
                    this_action_probas = self.T2[sr, ar, :]
                    s1r  = np.argmax(this_action_probas)
                    rr = np.sum(world.ravel()*this_action_probas)
                    
                    this_replay = np.array([sr, ar, rr, s1r])
                    replay_exp = np.vstack((replay_exp, this_replay))
            
            replay_gain     = np.empty((0, 8*4))
            replay_backups  = np.empty((0, 4))
            
            while replay:

                # Gain & Need
                gain = compute_gain(self.Q2, replay_exp, self.alpha2, self.beta)
                        
                # Expected value of each memory
                evm = 1/8 * gain
                
                max_evm = evm.max()
                # if replay_counter == 0:
                #     evm_thresh = np.percentile(evm, sigmoid(H))
                
                # if the value is above threshold
                if max_evm > self.evm_thresh:
                    max_evm_idx = np.where(evm == max_evm)[0]
                    max_evm_idx = max_evm_idx[-1]
                    
                    # Retrieve information from this experience
                    curr_path = replay_exp[max_evm_idx, :]
                    sr        = int(curr_path[0])
                    ar        = int(curr_path[1])
                    rr        = curr_path[2]
                    
                    Q_target = rr
                    
                    delta = Q_target - self.Q2[sr, ar]
                    self.Q2[sr, ar] = self.Q2[sr, ar] + self.alpha2 * delta
                    
                    replay_gain    = np.vstack((replay_gain, gain))
                    replay_backups = np.vstack((replay_backups, [sr, ar, rr, s1r]))
                    
                else:
                    replay = False
                    break
            
            if save_folder is not None:
                save_name = os.path.join(save_folder, 'move%u'%trial)
                np.savez(save_name, move=[s, a, r, s1], T=self.T2, replay_gain=replay_gain, 
                         replay_backups=replay_backups)
                
            # Gradually forget the model towards average reward
            av_rew = np.mean(self.rew_history2)
            dist2 = (self.Q2 - av_rew)
            self.Q2 = self.Q2 - (1-self.tau)*dist2
            
            # Forget the state transition model
            self.T2 = self.rho * self.T2 + (1-self.rho) * 1/7
            
            # Normalise the transition matrix
            for i in range(self.num_states):
                for j in range(self.num_actions):
                    self.T2[i, j, i] = 0
                    row = self.T2[i, j, :]
                    tmp = np.sum(row)
                    if tmp > 0:
                        self.T2[i, j, :] = row / tmp
                                 
            if trial == (self.num_trials-1):
                return np.array(self.rew_history2)
                        
            # Complete this step – prepare next trial
            if states is not None:
                s_counter += 1
                s = states[s_counter]
            else:
                s = np.random.randint(self.num_states)
            si = state2idcs(s, state_arr)
            
