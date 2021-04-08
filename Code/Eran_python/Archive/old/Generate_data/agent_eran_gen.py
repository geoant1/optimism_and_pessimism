import numpy as np
import json
import os
import shutil
from misc_eran_gen import *

class Agent:
    
    def __init__(self, p: dict):    
        
        # ---- Agent parameters ----
        # Inverse temperatures
        self.beta             = p['beta']
        self.beta1            = p['beta1']
        self.beta2            = p['beta2']    
        self.beta_mb          = p['beta_mb']    
        # Learning rates
        self.alpha1           = p['alpha1']
        self.alpha2           = p['alpha2']
        
        # Discounnt factor
        self.k                = p['k']
        self.rho              = p['rho']
        
        # MF and MB parameters
        self.Q_init           = p['Q_init']
        self.T_learn_rate     = p['T_learn_rate']
        self.opp_T_learn_rate = p['opp_T_learn_rate']
        self.tau              = p['tau']
        
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
        
        # 2-move trials
        self.Q1   = np.random.normal(self.Q_init, 1, (8, 4*4)) #  MF Q values for first moves in 2-move trials
        return
    
    def explore_one_move(self, world, state_arr, states=None, num_trials=None, online_learning=True, save_folder=None):
        '''Run 1-move trials'''
        
        if save_folder:
            config_folder = os.path.join(save_folder, 'Config')
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)
                os.mkdir(config_folder)
            else:
                shutil.rmtree(save_folder)
                os.makedirs(save_folder)
                os.mkdir(config_folder)
        
        if num_trials:
            self.num_trials = num_trials
            
        # Initialise in a random state (unless specific states provided)
        if states is not None:
            s_counter = 0
            s = states[s_counter]
        else:
            s = np.random.randint(self.num_states)
        
        si = state2idcs(s, state_arr)
        
        # Start exploring the maze
        for trial in range(self.num_trials):
            
            # Choose an action
            q_vals = self.Q2[s, :]
            q_vals_mb = np.zeros(4)
            for i in range(4):
                q_vals_mb[i] = np.sum(self.T2[s, i, :]*world.ravel())
            probs  = policy_choose_moves([q_vals, q_vals_mb], [self.beta, self.beta_mb], self.biases)
            a      = np.random.choice(range(self.num_actions), p=probs)
            
            # Execute the action, move to the next state and receive reward
            s1i, r = get_new_state(si, a, world, state_arr)
            s1 = idcs2state(s1i, state_arr)
            
            # online TD-learning
            if online_learning:
                # Gradually forget the model towards average reward
                self.Q2 = self.tau*self.Q2 + (1-self.tau)*self.Q_init
                
                Q_target = r        
                delta = Q_target - self.Q2[s, a]
                self.Q2[s, a] += self.alpha2 * delta
                
                # Forget the state transition model
                self.T2 = self.rho * self.T2 + (1-self.rho) * 1/7
                
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
            
            else:
                self.Q2 = self.tau*self.Q2 + (1-self.tau)*self.Q_init
                
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
                            
            if save_folder is not None:
                save_name = os.path.join(save_folder, 'move%u'%trial)
                np.savez(save_name, move=[s, a, r, s1], T=self.T2, Q=self.Q2)
               
            if trial >= (self.num_trials-1):
                break
                        
            # Complete this step – prepare next trial
            if states is not None:
                s_counter += 1
                s = states[s_counter]
            else:
                s = np.random.randint(self.num_states)
                
            si = state2idcs(s, state_arr)
            
            
    def explore_two_moves(self, world, state_arr, states=None, num_trials=None, online_learning=True, save_folder=None):
        
        if save_folder:
            config_folder = os.path.join(save_folder, 'Config')
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)
                os.mkdir(config_folder)
            else:
                shutil.rmtree(save_folder)
                os.makedirs(save_folder)
                os.mkdir(config_folder)
        
        idcs = []
        for ai in [0, 1]:
            for aj in [0, 1]:
                idcs.append(ai*4+aj)
        for ai in [2, 3]:
            for aj in [2, 3]:
                ai_opp = get_a_opp(ai)
                if aj == ai_opp:
                    idcs.append(ai*4+aj)
                    
        if num_trials:
            self.num_trials = num_trials
            
        num_steps    = 0
        num_episodes = 0
        sas          = np.empty((0, 4), dtype=int)
        
        # Initialise in a random state (unless specific states provided)
        if states is not None:
            s_counter = 0
            s = states[s_counter]
        else:
            s = np.random.randint(self.num_states)
        
        si = state2idcs(s, state_arr)
        
        # Start exploring maze
        for trial in range(self.num_trials*2):
                    
            # First move
            if num_steps == 0:
                # Choose an action
                q_vals1 = np.empty(0)
                for i in range(self.num_actions):
                    tmp     = self.Q1[s, (i*4):(i*4+4)].copy()
                    probs   = policy_choose(tmp, self.beta1, self.biases)
                    q_vals1 = np.append(q_vals1, np.sum(tmp*probs))
                
                q_vals_mb = np.empty(0)
                for i in range(self.num_actions):
                    tmp = 0
                    for jj in range(self.num_states):
                        max_val = 0
                        for ii in range(self.num_actions):
                            this_val = np.sum(self.T2[jj, ii, :]*world.ravel())
                            if this_val > max_val:
                                max_val = this_val
                        tmp += self.T2[s, i, jj]*(world.ravel()[jj] + self.k*max_val)
                    q_vals_mb = np.append(q_vals_mb, tmp)

                q_vals2 = self.Q2[s, :]
                probs   = policy_choose_moves([q_vals1, q_vals2, q_vals_mb], [self.beta1, self.beta2, self.beta_mb], self.biases)
                a       = np.random.choice(range(self.num_actions), p=probs)
                
                # Execute the action, move to the next state and receive reward
                s1i, r = get_new_state(si, a, world, state_arr)
                s1 = idcs2state(s1i, state_arr)
                
                # Remember the first move for later
                sas = np.vstack((sas, [s, a, r, s1]))
                
            # Second move    
            else:
                # Choose an action
                prev_s   = sas[-1, 0]
                prev_a   = sas[-1, 1]
                
                if online_learning:
                    q_vals1   = self.Q1[prev_s, (prev_a*4):(prev_a*4+4)]
                    q_vals2   = self.Q2[s, :].copy()
                    q_vals_mb = np.empty(0)
                    for i in range(4):
                        q_vals_mb = np.append(q_vals_mb, np.sum(self.T2[s, i, :]*world.ravel()))
                    
                    if prev_a == 0 or prev_a == 1:
                        a_opp = [0, 1]
                        probs  = policy_choose_moves([np.delete(q_vals1, a_opp), np.delete(q_vals2, a_opp), np.delete(q_vals_mb, a_opp)], [self.beta1, self.beta2, self.beta_mb], np.delete(self.biases, a_opp))
                        a      = np.random.choice(range(self.num_actions-2), p=probs)
                        a += 2
                    else:
                        prev_a_opp = get_a_opp(prev_a)
                        probs  = policy_choose_moves([np.delete(q_vals1, prev_a_opp), np.delete(q_vals2, prev_a_opp), np.delete(q_vals_mb, prev_a_opp)], [self.beta1, self.beta2, self.beta_mb], np.delete(self.biases, prev_a_opp))
                        a      = np.random.choice(range(self.num_actions-1), p=probs)
                        if a_opp <= a:
                            a += 1
                            
                else:
                    q_vals1 = self.Q1[prev_s, (prev_a*4):(prev_a*4+4)]
                    
                    # q_vals2 = np.zeros(4)
                    # for i in range(8):
                    #     tmp      = self.Q2[i, :]
                    #     probs    = policy_choose(tmp, self.beta2, self.biases)
                    #     q_vals2 += tmp*probs
                    for i in range(8):
                        tmp      = self.Q2[i, :]
                        q_vals2 += tmp
                    q_vals2 /= 8
                    
                    q_vals_mb = np.zeros(4)
                    for j in range(8):
                        for i in range(4):
                            q_vals_mb[i] += np.sum(self.T2[prev_s, prev_a, j]*(np.sum(self.T2[j, i, :]*world.ravel())))
                    
                    if prev_a == 0 or prev_a == 1:
                        prev_a_opp = [0, 1]
                        probs  = policy_choose_moves([np.delete(q_vals1, prev_a_opp), np.delete(q_vals2, prev_a_opp), np.delete(q_vals_mb, prev_a_opp)], [self.beta1, self.beta2, self.beta_mb], np.delete(self.biases, prev_a_opp))
                        a      = np.random.choice([2, 3], p=probs)
                    else:
                        prev_a_opp = get_a_opp(prev_a)
                        probs  = policy_choose_moves([np.delete(q_vals1, prev_a_opp), np.delete(q_vals2, prev_a_opp), np.delete(q_vals_mb, prev_a_opp)], [self.beta1, self.beta2, self.beta_mb], np.delete(self.biases, prev_a_opp))
                        a      = np.random.choice(range(self.num_actions-1), p=probs)
                        if prev_a_opp <= a:
                            a += 1
                    
                    # if prev_a == 0 or prev_a == 1:
                    #     prev_a_opp = [0, 1]
                    #     probs  = policy_choose(np.delete(q_vals2, prev_a_opp), self.beta2, np.delete(self.biases, prev_a_opp))
                    #     a      = np.random.choice([2,3], p=probs)
                    # else:
                    #     prev_a_opp = get_a_opp(prev_a)
                    #     probs  = policy_choose(np.delete(q_vals2, prev_a_opp), self.beta2, np.delete(self.biases, prev_a_opp))
                    #     a      = np.random.choice(range(self.num_actions-1), p=probs)
                    #     if prev_a_opp <= a:
                    #         a += 1
                            
                # Execute the action, move to the next state and receive reward
                s1i, r = get_new_state(si, a, world, state_arr)
                s1 = idcs2state(s1i, state_arr)
                
                # Online TD-learning
                self.Q2 = self.tau*self.Q2 + (1-self.tau)*self.Q_init
                self.Q1 = self.tau*self.Q1 + (1-self.tau)*self.Q_init
                self.Q1[:, idcs] = -200
                if online_learning:
            
                    Q_target = r        
                    delta    = Q_target - self.Q2[s, a]
                    self.Q2[s, a] += self.alpha2 * delta
                    
                    # Update first move Q-values
                    prev_s = sas[-1, 0]
                    prev_a = sas[-1, 1]
                    prev_r = sas[-1, 2]
                    
                    Q_target = r + prev_r
                    delta    = Q_target = self.Q1[prev_s, prev_a*4+a]
                    self.Q1[prev_s, prev_a*4+a] += self.alpha1 * delta
                    
            # Update the state transition model
            self.T2 = self.rho * self.T2 + (1-self.rho) * 1/7
            if online_learning:
                
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
                        
            # Complete step
            num_steps += 1
            
            # Move agent to the next state    
            s = s1
            si = s1i
                            
            if num_steps == 2:
                
                if save_folder is not None:
                    save_name = os.path.join(save_folder, 'move%u'%num_episodes)
                    np.savez(save_name, move=[[sas[-1, 0], sas[-1, 1], sas[-1, 2], sas[-1, 3]], [s, a, r, s1]])
                
                if num_episodes >= (self.num_trials-1):
                    break
                
                num_steps = 0
                num_episodes += 1
                
                if states is not None:
                    s_counter += 1
                    s = states[s_counter]
                else:
                    s = np.random.randint(self.num_states)
                    
                si = state2idcs(s, state_arr)
                            