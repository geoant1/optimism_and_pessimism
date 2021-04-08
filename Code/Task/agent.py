import numpy as np
import os
import shutil
from misc import policy_choose, get_new_state, idcs2state, get_a_opp, normalise, replay_1move, replay_2moves, state2idcs, policy_choose_moves

class Agent:
    
    def __init__(self, **kwargs):    
        
        # ---- Agent parameters ----
        self.__dict__.update(**kwargs)
        
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
        self.Q1 = np.random.normal(self.Q_init, 1, (8, 16)) #  MF Q values for first moves in 2-move trials
        
        # Buffers for rewards experienced by the agent
        self.rew_history2 = []
        self.rew_history1 = []
                
        return
    
    def explore_one_move(self, world, state_arr, states, num_trials, online_learning=True, save_folder=None):
        '''Run 1-move trials'''
        
        if save_folder is not None:
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)
            else:
                shutil.rmtree(save_folder)
                os.makedirs(save_folder)
            
        s_counter = 0
        s  = states[s_counter]
        si = state2idcs(s, state_arr)
        
        # Start exploring the maze
        for trial in range(num_trials):
            
            # Choose an action
            q_vals = self.Q2[s, :]
            probs  = policy_choose(q_vals, self.beta, self.biases)
            a      = np.random.choice(range(self.num_actions), p=probs)
            
            # Execute the action, move to the next state and receive reward
            s1i, r = get_new_state(si, a, world, state_arr)
            s1     = idcs2state(s1i, state_arr)
            
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
                self.T2 = normalise(self.T2)
            
            # Prepare replay
            replay = True # initialise replay 
            
            if replay:
                Q2, replay_backups, Q_history, replay_gain = replay_1move(self.Q2, self.T2, world, self.beta, self.alpha2r, self.evm_thresh)
                self.Q2 = Q2
            
            if save_folder is not None:
                save_name = os.path.join(save_folder, 'move%u'%trial)
                np.savez(save_name, move=[s, a, r, s1], T=self.T2, 
                         Q_history=Q_history, replay_backups=replay_backups, rew_history=self.rew_history2, replay_gain=replay_gain)
            
            if online_learning:
                # Gradually forget the model towards average reward
                av_rew  = np.mean(self.rew_history2)
                dist2   = (self.Q2 - av_rew)
                self.Q2 = self.Q2 - (1-self.tau)*dist2
            
                # Forget the state transition model
                self.T2 = self.rho * self.T2 + (1-self.rho) * 1/7
            
                # Normalise the transition matrix
                self.T2 = normalise(self.T2)
                                 
            if trial == (num_trials-1):
                break
                        
            # Complete this step – prepare next trial
            s_counter += 1
            s = states[s_counter]
            si = state2idcs(s, state_arr)
            
            
    def explore_two_moves(self, world, state_arr, states, num_trials, online_learning=True, save_folder=None, T_forget=False):
        
        if save_folder is not None:
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)
            else:
                shutil.rmtree(save_folder)
                os.makedirs(save_folder)
            
        num_steps    = 0
        num_episodes = 0
        
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

        s_counter = 0
        s  = states[s_counter]
        si = state2idcs(s, state_arr)
        
        # Start exploring maze
        for _ in range(num_trials*2):
                    
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
                
                prev_s  = s
                prev_a  = a
                prev_r  = r   
                prev_s1 = s1      
                
                if online_learning:
                    Q_target = r
                    delta = Q_target - self.Q2[s, a]
                    self.Q2[s, a] = self.Q2[s, a] + self.alpha2 * delta

                    # Update state-transition model
                    delta_t = 1 - self.T2[s, a, s1]
                    self.T2[s, a, s1] += self.T_learn_rate * delta_t
                    
                    # NB we also learn about opposite transitions
                    a_opp = get_a_opp(a)
                    delta_t_opp = 1 - self.T2[s1, a_opp, s]
                    self.T2[s1, a_opp, s] += self.T_learn_rate * self.opp_T_learn_rate * delta_t_opp
                    
                    # Normalise the transition matrix
                    self.T2 = normalise(self.T2)
                            
            # Second move    
            else:
                
                q_vals1 = self.Q1[prev_s, (prev_a*4):(prev_a*4+4)]
                
                if online_learning:
                    q_vals2   = self.Q2[s, :].copy()
                    
                    probs      = policy_choose_moves([q_vals1, q_vals2], [self.beta1, self.beta2], self.biases)
                    a          = np.random.choice(range(self.num_actions), p=probs)
                    
                    # Execute the action, move to the next state and receive reward
                    s1i, r = get_new_state(si, a, world, state_arr)
                    s1     = idcs2state(s1i, state_arr)
                    
                    # Update both Q values
                    Q_target = r        
                    delta = Q_target - self.Q2[s, a]
                    self.Q2[s, a] = self.Q2[s, a] + self.alpha2 * delta
                    
                    Q_target = r + prev_r
                    delta = Q_target - self.Q1[prev_s, prev_a*4 + a]
                    self.Q1[prev_s, prev_a*4 + a] = self.Q1[prev_s, prev_a*4 + a] + self.alpha1 * delta
                    
                    # Update state0transition model
                    delta_t = 1 - self.T2[s, a, s1]
                    self.T2[s, a, s1] += self.T_learn_rate * delta_t
                    
                    # NB we also learn about opposite transitions
                    a_opp = get_a_opp(a)
                    delta_t_opp = 1 - self.T2[s1, a_opp, s]
                    self.T2[s1, a_opp, s] += self.T_learn_rate * self.opp_T_learn_rate * delta_t_opp
                    
                    # Normalise the transition matrix
                    self.T2 = normalise(self.T2)
                            
                    self.rew_history2.append(r)
                    self.rew_history1.append(prev_r + r)
                    
                else:
                    q_vals2 = np.zeros(4)
                    for i in np.delete(range(8), prev_s):
                        q_vals2  += self.Q2[i, :]
                    q_vals2 /= 7
                    
                    probs      = policy_choose_moves([q_vals1, q_vals2], [self.beta1, self.beta2], self.biases)
                    a          = np.random.choice(range(self.num_actions), p=probs)

                    # Execute the action, move to the next state and receive reward
                    s1i, r = get_new_state(si, a, world, state_arr)
                    s1     = idcs2state(s1i, state_arr)
                                    
            # Prepare replay
            replay = False # initialise counter
            if num_steps != 0:
                replay = True
                
            if replay:
                Q1, Q2, Q1_history, Q2_history, replay_backups, replay_gain1, replay_gain2 = replay_2moves(self.Q1, self.Q2, self.T2, world, self.evm_thresh, self.beta2, self.beta1, self.alpha2r, self.alpha1r)
                self.Q1 = Q1
                self.Q2 = Q2
            
            if num_steps == 0 and online_learning:
                av_rew2 = np.mean(self.rew_history2)
                dist2   = np.subtract(self.Q2, av_rew2)
                self.Q2 = self.Q2 - (1-self.tau)*dist2
                
                self.T2 = self.rho * self.T2 + (1-self.rho) * 1/7
                # Normalise the transition matrix
                self.T2 = normalise(self.T2)
                                
            # Complete step
            num_steps += 1
            
            if num_steps == 2:
                
                if save_folder is not None:
                    save_name = os.path.join(save_folder, 'move%u'%num_episodes)
                    np.savez(save_name, move=[[prev_s, prev_a, prev_r, prev_s1], [s, a, r, s1]], T=self.T2, 
                             Q1_history=Q1_history, Q2_history=Q2_history, replay_backups=replay_backups, replay_gain1=replay_gain1, replay_gain2=replay_gain2,
                             rew_history=[self.rew_history1, self.rew_history2])
                    
                # Gradually forget towards the average reward
                if T_forget and num_episodes >= (num_trials-1):
                    pass
                else:
                    if online_learning:
                        # Forget Q values
                        av_rew2 = np.mean(self.rew_history2)
                        dist2   = np.subtract(self.Q2, av_rew2)
                        self.Q2 = self.Q2 - (1-self.tau)*dist2
                
                        av_rew1 = np.mean(self.rew_history1)
                        dist1   = np.subtract(self.Q1, av_rew1)
                        self.Q1 = self.Q1 - (1-self.tau)*dist1
                        # self.Q1[:, idcs] = -100

                        self.T2 = self.rho * self.T2 + (1-self.rho) * 1/7
                       
                        # Normalise the transition matrix
                        self.T2 = normalise(self.T2)
                            
                if num_episodes >= (num_trials-1):
                    break
                
                num_steps = 0
                num_episodes += 1
                
                s_counter += 1
                s = states[s_counter]
                si = state2idcs(s, state_arr)
                
            else:
                # Move agent to the next state    
                s = s1
                si = s1i
                            