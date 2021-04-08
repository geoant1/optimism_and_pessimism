import numpy as np
import json
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
        self.Q2 = np.random.normal(self.Q_init, 1, (8, 4)) # MF Q values for 1-move trials and second moves in 2-move trials
        self.T2 = np.full((self.num_states, self.num_actions, self.num_states), 1/7, dtype=float) # state transition probability model 
        for i in range(self.num_states):
            for j in range(self.num_actions):
                self.T2[i, j, i] = 0 # Self-transitions are not allowed
        
        # 2-move trials
        self.Q1 = np.random.normal(self.Q_init, 1, (8, 16)) #  MF Q values for first moves in 2-move trials
        
        # Buffers for rewards experienced by the agent
        self.rew_history2 = []
        self.rew_history1 = []
                
        return
    
    def explore_one_move(self, world, state_arr, states=None, num_trials=None, online_learning=True):
        '''Run 1-move trials'''
        
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
            
            r_list.append(r)
            
            # online TD-learning
            if online_learning:
                self.rew_history2.append(r)

                # Update Q values
                Q_target = r        
                delta    = Q_target - self.Q2[s, a]
                self.Q2[s, a] = self.Q2[s, a] + self.alpha2 * delta
            
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
            R_vec  = world.flatten()
            # Generate replays from the model
            replay_exp = np.empty((0, 4))
            for sr in range(8):
                for ar in range(4):
                    this_action_probas = self.T2[sr, ar, :]
                    s1r  = np.argmax(this_action_probas)
                    rr = np.sum(R_vec*this_action_probas)
                    
                    this_replay = np.array([sr, ar, rr, s1r])
                    replay_exp  = np.vstack((replay_exp, this_replay))
                        
            while replay:

                # Gain & Need
                gain = compute_gain(self.Q2, replay_exp, self.alpha2, self.beta)
                        
                # Expected value of each memory
                evm = 1/8 * gain                
                max_evm = evm.max()

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
                    
                else:
                    replay = False
                    break
            
            # Forget Q values
            av_rew  = np.mean(self.rew_history2)
            dist2   = (self.Q2 - av_rew)
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
                return r_list
                        
            # Complete this step – prepare next trial
            if states is not None:
                s_counter += 1
                s = states[s_counter]
            else:
                s = np.random.randint(self.num_states)
            si = state2idcs(s, state_arr)
            
            
    def explore_two_moves(self, world, state_arr, states=None, num_trials=None, online_learning=True):
                
        if num_trials:
            self.num_trials = num_trials
            
        num_steps    = 0
        num_episodes = 0
        sars         = np.empty((0, 4), dtype=int)
        
        idcs = []
        for ai in [0, 1]:
            for aj in [0, 1]:
                idcs.append(ai*4+aj)
        for ai in [2, 3]:
            for aj in [2, 3]:
                ai_opp = get_a_opp(ai)
                if aj == ai_opp:
                    idcs.append(ai*4+aj)
        self.Q1[:, idcs] = np.nan
        
        # Initialise in a random state (unless specific states provided)
        if states is not None:
            s_counter = 0
            s = states[s_counter]
        else:
            s = np.random.randint(self.num_states)
        si = state2idcs(s, state_arr)
        
        r_list = []
        # Start exploring maze
        for trial in range(self.num_trials*2):
                    
            # First move
            if num_steps == 0:
                # Choose an action
                q_vals1 = np.zeros(4)
                for i in range(self.num_actions):
                    tmp     = self.Q1[s, (i*4):(i*4+4)].copy()
                    probs   = policy_choose(tmp, self.beta1, self.biases)
                    q_vals1[i] = np.nansum(tmp*probs)
                q_vals2 = self.Q2[s, :].copy()
                
                probs  = policy_choose_moves([q_vals1, q_vals2], [self.beta1, self.beta2], self.biases)
                a      = np.random.choice(range(self.num_actions), p=probs)
                
                # Execute the action, move to the next state and receive reward
                s1i, r = get_new_state(si, a, world, state_arr)
                s1     = idcs2state(s1i, state_arr)
                
                sars = np.vstack((sars, [s, a, r, s1]))
                    
            # Second move    
            else:
                # Choose an action
                prev_s  = sars[-1, 0]
                prev_a  = sars[-1, 1]
                prev_r  = sars[-1, 2]
                
                q_vals1 = self.Q1[prev_s, (prev_a*4):(prev_a*4+4)]
                
                if online_learning:
                    q_vals2   = self.Q2[s, :].copy()
                    
                    probs      = policy_choose_moves([q_vals1, q_vals2], [self.beta1, self.beta2], self.biases)
                    a          = np.random.choice(range(self.num_actions), p=probs)
                    
                    # Execute the action, move to the next state and receive reward
                    s1i, r = get_new_state(si, a, world, state_arr)
                    s1     = idcs2state(s1i, state_arr)
                    
                    Q_target = r        
                    delta = Q_target - self.Q2[s, a]
                    self.Q2[s, a] = self.Q2[s, a] + self.alpha2 * delta
                    
                    Q_target = r + prev_r
                    delta = Q_target - self.Q1[prev_s, prev_a*4 + a]
                    self.Q1[prev_s, prev_a*4 + a] = self.Q1[prev_s, prev_a*4 + a] + self.alpha1 * delta
                    
                    self.rew_history2.append(r)
                    self.rew_history1.append(prev_r + r)
                
                # No online learning            
                else:
                    q_vals2 = np.zeros(4)
                    for i in range(8):
                        q_vals2  += self.Q2[i, :]
                    q_vals2 /= 8
                    
                    probs      = policy_choose_moves([q_vals1, q_vals2], [self.beta1, self.beta2], self.biases)
                    a          = np.random.choice(range(self.num_actions), p=probs)

                    # Execute the action, move to the next state and receive reward
                    s1i, r = get_new_state(si, a, world, state_arr)
                    s1     = idcs2state(s1i, state_arr)
                
                r_list.append(prev_r+r)
                     
            # Update the state transition model
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
                                             
            # Prepare replay
            replay = False # initialise counter
            if num_steps != 0:
                replay = True
                replay_exp2 = np.empty((0, 4))
                
                R_vec = world.flatten()
                # Generate 2nd move experiences
                for sr in range(8):
                    for ar in range(4):
                        
                        this_action_probas = self.T2[sr, ar, :]
                        s1r  = np.argmax(this_action_probas)
                        rr = np.sum(this_action_probas * R_vec)
                                
                        this_replay = np.array([sr, ar, rr, s1r])
                        replay_exp2 = np.vstack((replay_exp2, this_replay)) 
                
            while replay:
                # Keep separate experiences for moves 1 & 2
                replay_exp1 = np.empty((0, 4))   
                  
                # Generate 1st move experiences
                for sr in range(8):
                    for ar in range(4):
                        s1r = np.argmax(self.T2[sr, ar, :])
                        for ar2 in range(4):
                            rr = 0
                            for s2r in range(8):
                                
                                q_vals2 = self.Q2[s2r, :]
                                probs  = policy_choose(q_vals2, self.beta2)
                                
                                rr += self.T2[sr, ar, s2r]*(R_vec[s2r] + probs[ar2]*np.sum(self.T2[s2r, ar2, :] * R_vec))
                                
                            this_replay = np.array([sr, ar*4+ar2, rr, s1r*8+s2r])
                            replay_exp1 = np.vstack((replay_exp1, this_replay)) 
                            
                # Gain & Need 
                gain_move1 = compute_gain(self.Q1, replay_exp1, self.alpha1, self.beta1)
                gain_move2 = compute_gain(self.Q2, replay_exp2, self.alpha2, self.beta2)
                
                # Expected value of memories
                evm_move1 = 1/8 * gain_move1
                evm_move2 = 1/8 * gain_move2
                
                # Compare memories from both moves and choose in which plane to relay
                max_evm_move1 = evm_move1.max()
                max_evm_move2 = evm_move2.max()
                
                if max_evm_move1 >= max_evm_move2:
                    replay_exp = replay_exp1
                    max_evm  = max_evm_move1
                    evm      = evm_move1
                    plane    = 0
                else:
                    replay_exp = replay_exp2
                    max_evm  = max_evm_move2
                    evm      = evm_move2
                    plane    = 1
                
                # if the value is above threshold
                if max_evm > self.evm_thresh:
                    max_evm_idx = np.where(evm == max_evm)[0]
                    max_evm_idx = max_evm_idx[-1]
                                        
                    # Retrieve information from this experience
                    curr_path = replay_exp[max_evm_idx, :]
                    sr        = int(curr_path[0])
                    ar        = int(curr_path[1])
                    rr        = curr_path[2]
                    s1r       = int(curr_path[3])                    
                    
                    if plane == 0:
                        Q_target = rr
                        delta = Q_target - self.Q1[sr, ar]
                        self.Q1[sr, ar] = self.Q1[sr, ar] + self.alpha1 * delta
                        
                    else:
                        Q_target = rr 
                        delta = Q_target - self.Q2[sr, ar]
                        self.Q2[sr, ar] = self.Q2[sr, ar] + self.alpha2 * delta
                                        
                else:
                    replay = False
                    break    
                                
            # Complete step
            num_steps += 1
            
            # Move agent to the next state    
            s = s1
            si = s1i
            
            if num_steps == 2:
                
                # Forget Q values
                av_rew2 = np.mean(self.rew_history2)
                dist2   = np.subtract(self.Q2, av_rew2)
                self.Q2 = self.Q2 - (1-self.tau)*dist2
                
                av_rew1 = np.mean(self.rew_history1)
                dist1   = np.subtract(self.Q1, av_rew1)
                self.Q1 = self.Q1 - (1-self.tau)*dist1
                # self.Q1[:, idcs] = -100
                
                # Forget state-transition model 
                self.T2 = self.rho * self.T2 + (1-self.rho) * 1/7
                
                # Normalise the transition matrix
                for i in range(self.num_states):
                    for j in range(self.num_actions):
                        self.T2[i, j, i] = 0
                        row = self.T2[i, j, :]
                        tmp = np.sum(row)
                        if tmp > 0:
                            self.T2[i, j, :] = row / tmp  
                            
                if num_episodes >= (self.num_trials-1):
                    return r_list
                
                num_steps = 0
                num_episodes += 1
                
                if states is not None:
                    s_counter += 1
                    s = states[s_counter]
                else:
                    s = np.random.randint(self.num_states)
                si = state2idcs(s, state_arr)
