import numpy as np
import json
import os
import shutil
from misc_eran_fit import *

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
        
        # Discounnt factor
        self.gamma            = p['gamma']
        
        # MF and MB parameters
        self.Q_init           = p['Q_init']
        self.T_learn_rate     = p['T_learn_rate']
        self.opp_T_learn_rate = p['opp_T_learn_rate']
        self.rho              = p['rho']
        self.tau              = p['tau']
        
        self.evm_thresh       = p['evm_threshold']
        self.biases           = p['biases']
        
        self.max_num_replays  = p['max_num_replays']
        
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
        self.Q1 = np.random.normal(self.Q_init, 1, (8, 4)) #  MF Q values for first moves in 2-move trials
        
        # Buffers for rewards experienced by the agent
        self.rew_history2 = []
        self.rew_history1 = []
                
        return
    
    def explore_one_move(self, world, states=None, num_trials=None, online_learning=True, save_folder=None):
        '''Run 1-move trials'''
        
        if save_folder:
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
        si = state2idcs(s, world)
        
        # Start exploring the maze
        for trial in range(self.num_trials):
            
            # Choose an action
            q_vals = self.Q2[s, :]
            probs  = policy_choose(q_vals, self.beta, self.biases)
            a      = np.random.choice(range(self.num_actions), p=probs)
            
            # Execute the action, move to the next state and receive reward
            s1i, r = get_new_state(si, a, world)
            s1     = idcs2state(s1i, world)
            
            # online TD-learning
            if online_learning:
                Q_target = r        
                delta = Q_target - self.Q2[s, a]
                self.Q2[s, a] = self.Q2[s, a] + self.alpha2 * delta
                self.rew_history2.append(r)
                
                # Update the state transition model
                delta_t = 1 - self.T2[s, a, s1]
                self.T2[s, a, s1] += self.T_learn_rate * delta_t
                
                # NB we also learn about opposite transitions
                a_opp = None
                if a == 0:
                    a_opp = 1
                elif a == 1:
                    a_opp = 0
                elif a == 2:
                    a_opp = 3
                else:
                    a_opp = 2
                    
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
            replay_counter = 0 # initialise replay counter

            # Generate replays from the model
            replay_exp = np.empty((0, 4))
            for sr in range(8):
                for ar in range(4):
                    if sr == s & ar == a:
                        rr  = r
                        s1r = s1
                    else:
                        this_action_probas = self.T2[sr, ar, :]
                        s1r  = np.argmax(this_action_probas)
                        rr = np.sum(world.ravel()*this_action_probas)
                    
                    this_replay = np.array([sr, ar, rr, s1r])
                    replay_exp = np.vstack((replay_exp, this_replay))
            
            while replay_counter < self.max_num_replays:

                # Gain & Need
                gain = compute_gain(self.Q2, replay_exp, self.alpha2, self.beta, self.biases)
                need = [1/8]*len(gain) # here we assume no knowledge of next (random) transition
                        
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
                    
                    replay_counter += 1
                else:
                    break
            
            if save_folder:
                path_to_save = os.path.join(save_folder, 'Move%u'%trial)
                np.savez(path_to_save, move=[s, a, r, s1])
                        
            if trial >= (self.num_trials-1):
                break
            
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
                        
            # Complete this step – prepare next trial
            if states is not None:
                s_counter += 1
                s = states[s_counter]
            else:
                s = np.random.randint(self.num_states)
            si = state2idcs(s, world)
            
            
    def explore_two_moves(self, world, states=None, num_trials=None, online_learning=True, save_folder=None):
        
        if save_folder:
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)
            else:
                shutil.rmtree(save_folder)
                os.makedirs(save_folder)
                
        if num_trials:
            self.num_trials = num_trials
            
        num_steps    = 0
        num_episodes = 0
        
        # Initialise in a random state (unless specific states provided)
        if states is not None:
            s_counter = 0
            s = states[s_counter]
        else:
            s = np.random.randint(self.num_states)
        si = state2idcs(s, world)
        
        # Start exploring maze
        for trial in range(self.num_trials*2):
                    
            # First move
            if num_steps == 0:
                # Choose an action
                q_vals = self.Q1[s, :].copy()
                probs  = policy_choose(q_vals, self.beta1, self.biases)
                a      = np.random.choice(range(self.num_actions), p=probs)
                
                # Execute the action, move to the next state and receive reward
                s1i, r = get_new_state(si, a, world)
                s1     = idcs2state(s1i, world)
                
                # Remember the first move for later
                prev_s  = s
                prev_a  = a
                prev_r  = r
                prev_s1 = s1
                
                # Online TD-learning
                if online_learning:
                    s1_val = np.amax(self.Q2[s1, :])
                    Q_target = r + self.gamma*s1_val
                    delta = Q_target - self.Q1[s, a]
                    self.Q1[s, a] = self.Q1[s, a] + self.alpha1 * delta
                      
            # Second move    
            else:
                # Choose an action
                q_vals = self.Q2[s, :].copy()
                probs  = policy_choose(q_vals, self.beta2, self.biases)
                a      = np.random.choice(range(self.num_actions), p=probs)
                
                # Execute the action, move to the next state and receive reward
                s1i, r = get_new_state(si, a, world)
                s1     = idcs2state(s1i, world)
                
                # Online TD-learning
                if online_learning:
                    Q_target = r        
                    delta = Q_target - self.Q2[s, a]
                    self.Q2[s, a] = self.Q2[s, a] + self.alpha2 * delta
                    
                    self.rew_history2.append(r)
                    self.rew_history1.append(prev_r + r)
                    
            # Update the state transition model
            if online_learning:
                delta_t = 1 - self.T2[s, a, s1]
                self.T2[s, a, s1] += self.T_learn_rate * delta_t
                
                # NB we also learn about opposite transitions
                a_opp = None
                if a == 0:
                    a_opp = 1
                elif a == 1:
                    a_opp = 0
                elif a == 2:
                    a_opp = 3
                else:
                    a_opp = 2
                    
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
            if num_steps == 0:
                replay_counter = np.inf
            else:
                replay_counter = 0 # initialise counter
                # Generate 2nd move experiences
                replay_exp2 = np.empty((0, 4))
                for sr in range(8):
                    for ar in range(4):
                        # For the most recent experience we use 'true' observations
                        if sr == s & ar == a:
                            rr = r
                            s1r = s1
                        else:
                            this_action_probas = self.T2[sr, ar, :]
                            s1r  = np.argmax(this_action_probas)
                            rr = np.sum(this_action_probas * world.ravel())
                                
                        this_replay = np.array([sr, ar, rr, s1r])
                        replay_exp2 = np.vstack((replay_exp2, this_replay))

            while replay_counter < self.max_num_replays:
                # Keep separate experiences for moves 1 & 2
                replay_exp1 = np.empty((0, 4))
                # Generate 1st move experiences
                for sr in range(8):
                    for ar in range(4):
                        
                        # For the most recent experience we use 'true' observations
                        if sr == prev_s & ar == prev_a:
                            rr  = prev_r + self.gamma * np.amax(self.Q2[prev_s1, :])
                            s1r = prev_s1
                            
                        # All other experiences are generated from the agent's model
                        else:
                            this_action_probas1 = self.T2[sr, ar, :].copy()
                            s1r  = np.argmax(this_action_probas1)
                            
                            av_val = 0
                            for ar2 in range(4):
                                tmp = 0
                                for s2r in range(8):
                                
                                    q_vals = self.Q2[s2r, :].copy()
                                    probs  = policy_choose(q_vals, self.beta2, self.biases)
                                    this_action_probas2 = self.T2[s2r, ar2, :].copy()

                                    av_val += probs[ar2] * this_action_probas2[s2r] * world.ravel()[s2r]
                                
                            rr = np.sum(this_action_probas1 * world.ravel()) + np.sum(this_action_probas1 * av_val)
                                
                        this_replay = np.array([sr, ar, rr, s1r])
                        replay_exp1 = np.vstack((replay_exp1, this_replay)) 
                
                # Gain & Need 
                gain_move1 = compute_gain(self.Q1, replay_exp1, self.alpha1, self.beta1, self.biases)
                gain_move2 = compute_gain(self.Q2, replay_exp2, self.alpha2, self.beta2, self.biases)
                
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
                        # s1_val   = np.amax(self.Q2[s1r, :])
                        # Q_target = rr + self.gamma * s1_val
                        Q_target = rr
                        delta = Q_target - self.Q1[sr, ar]
                        self.Q1[sr, ar] = self.Q1[sr, ar] + self.alpha1 * delta
                        
                    else:
                        Q_target = rr 
                        delta = Q_target - self.Q2[sr, ar]
                        self.Q2[sr, ar] = self.Q2[sr, ar] + self.alpha2 * delta
                                        
                    replay_counter += 1
                else:
                    break
            
            if save_folder and num_steps == 1:
                
                path_to_save = os.path.join(save_folder, 'Episode%u_move%u'%(num_episodes, num_steps))
                move1 = [prev_s, prev_a, prev_r, prev_s1]
                move2 = [s, a, r, s1]
                np.savez(path_to_save, move1=move1, move2=move2)
            
            # Complete step
            num_steps += 1
            
            # Move agent to the next state    
            s = s1
            si = s1i
            
            if num_steps == 2:
                num_steps = 0
                num_episodes += 1
                
                if num_episodes >= (self.num_trials-1):
                    break
                
                if states is not None:
                    s_counter += 1
                    s = states[s_counter]
                else:
                    s = np.random.randint(self.num_states)
                si = state2idcs(s, world)

                # Gradually forget towards the average reward
                av_rew1 = np.mean(self.rew_history1)
                av_rew2 = np.mean(self.rew_history2)

                dist2 = (self.Q2 - av_rew2)
                self.Q2 = self.Q2 - (1-self.tau)*dist2
                
                dist1 = (self.Q1 - av_rew1)
                self.Q1 = self.Q1 - (1-self.tau)*dist1
                
                self.T2 = self.rho * self.T2 + (1-self.rho) * 1/7
                
                # Normalise the transition matrix
                for i in range(self.num_states):
                    for j in range(self.num_actions):
                        self.T2[i, j, i]  = 0
                        
                        row = self.T2[i, j, :]
                        tmp = np.sum(row)
                        if tmp > 0:
                            self.T2[i, j, :] = row / tmp
                            