import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from misc import *
import json
import os, shutil

class Agent:
    
    def __init__(self, p: dict, maze):  
        
        self.max_num_episodes      = p['max_num_episodes']
        self.num_plan_steps        = p['num_plan_steps'] 
        self.num_sim_steps         = p['num_sim_steps']     
        self.plan_at_start_or_goal = p['plan_at_start_or_goal']
        self.expand_further        = p['expand_further']
        self.allow_loops           = p['allow_loops']
        self.start_to_goal         = p['start_to_goal']
        self.r_params              = p['r_params']
        self.R = p['R']

        self.t                = p['temperature']
        self.t_learn_rate     = p['t_learn_rate']
        self.tau              = p['tau']
        self.rho              = p['rho']
        self.opp_t_learn_rate = p['opp_t_learn_rate']
        self.gamma            = p['gamma'] 
        self.alpha            = p['alpha']    
        self.lam              = p['lambda']    
        self.baseline_gain    = p['baseline_gain']
        self.evm_thresh       = p['EVM_threshold']

        self.maze   = maze
        self.side_i = maze.shape[0]
        self.side_j = maze.shape[1]
        self.num_states  = self.side_i * self.side_j
        self.num_actions = 4 # 0 = up, 1 = down, 2 = left, 3 = right from each state
        self.start_idcs  = p['start_location']
        self.goal_idcs   = self.r_params['pos_loc']
        
        self.world = np.zeros_like(self.maze)
        self.world[0, 0] = 10
        self.world[0, 4] = -10
        
        self.Q_table  = np.zeros((self.num_states, self.num_actions)) # table of Q values 
        self.T_matrix = np.zeros((self.num_states, self.num_actions, self.num_states), dtype=np.float64)  # transition probability matrix 
        self.H = np.eye(self.num_states*self.num_actions)
        
        self.rew_history = []
        self.exp_list = np.empty((0, 4))
        
        return
    
    def pre_explore(self):
        '''
        Define allowed transitions that respect the world's boundaries
        '''
        
        # From each state we can only transition into a valid state
        idcs = np.where(self.maze.ravel()==0)[0]
        for i in idcs:
            for j in idcs:
                if j != i:
                    self.T_matrix[i, :, j] = 1
        
        # Correct for actions from valid states that lead into walls
        for s in idcs:
            s_idcs = state2idcs(s, self.maze)
            for a in range(self.num_actions):
                s1_idcs, _ = get_new_state(s_idcs, a, self.r_params, self.maze)
                s1 = idcs2state(s1_idcs, self.maze)
                if s == s1:
                    self.T_matrix[s, a, :] = 0
        
        # Add start-to-goal transitions
        g = idcs2state(self.goal_idcs, self.maze)  # get the goal state
        s = idcs2state(self.start_idcs, self.maze) # get the start state
        self.T_matrix[g, :, :] = 0
        self.T_matrix[g, :, s] = 1
        
        # save the allowed transitions schema
        T_allowed = self.T_matrix.copy()
        
        # normalise the transition matrix
        for i in range(self.num_states):
            for j in range(self.num_actions):
                row = self.T_matrix[i, j, :]
                tmp = np.sum(row)
                if tmp > 0:
                    self.T_matrix[i, j, :] = row / tmp
                    
        self.exp_list = np.append(self.exp_list, np.array([[1, 1, 0, np.nan]]), axis=0)
        self.exp_list = np.append(self.exp_list, np.array([[1, 1, 0, np.nan]]), axis=0)
            
        return T_allowed, self.T_matrix
    
    def run_simulation(self, T_allowed, T_init, plot_location=True, save_folder=None):
        
        if save_folder:
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)
            else:
                shutil.rmtree(save_folder)
                os.makedirs(save_folder)
        os.chdir(save_folder)
        
        num_episodes = 0
        num_steps    = 0
        
        s      = idcs2state(self.start_idcs, self.maze)
        s_idcs = self.start_idcs
                
        if plot_location:
            plt.ion()
            fig1 = plt.figure(figsize=(10, 8))
            ax1  = plt.axes([0.1, 0.1, 0.8, 0.8])

        # Start exploring maze
        for t in range(self.num_sim_steps):
            print('%d steps; %d episodes'%(t, num_episodes))
            
            if plot_location and num_episodes>20:
                plot_each_step(ax1, s_idcs, self.maze, self.Q_table, 'Exploring', 0.2)
            
            # Choose an action
            
            q_vals = self.Q_table[s, :]
            probs = policy_choose(q_vals, 'softmax', temperature=self.t)
            a = np.random.choice(range(self.num_actions), p=probs)
            
            # Execute the action, move to the next state and receive reward
            s1_idcs, r = get_new_state(s_idcs, a, self.r_params, self.maze)
            s1 = idcs2state(s1_idcs, self.maze)
            if s1 == s:
                while True:
                    q_vals = self.Q_table[s, :]
                    probs = policy_choose(q_vals, 'softmax', temperature=self.t)
                    a = np.random.choice(range(self.num_actions), p=probs)
                    
                    # Execute the action, move to the next state and receive reward
                    s1_idcs, r = get_new_state(s_idcs, a, self.r_params, self.maze)
                    s1 = idcs2state(s1_idcs, self.maze)
                    if s1 != s:
                        break
                    
            if r > 0:
                self.rew_history.append(r)
            
            # Update Q-values
            # off-policy learning
            q     = self.Q_table[s, a]
            q1    = np.max(self.Q_table[s1, :])
            
            Q_target = r + self.gamma * q1
            delta    =  Q_target - q # temporal difference
            self.Q_table[s, a] += self.alpha*delta
            
            self.exp_list = np.append(self.exp_list, np.array([[s, a, r, s1]]), axis=0)
            
            # Update the state transition model
            delta_t = 1 - self.T_matrix[s, a, s1]
            self.T_matrix[s, a, s1] += self.t_learn_rate * delta_t
            
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
                
            delta_t_opp = 1 - self.T_matrix[s1, a_opp, s]
            self.T_matrix[s1, a_opp, s] += self.t_learn_rate * self.opp_t_learn_rate * delta_t_opp
            
            self.T_matrix *= T_allowed
            # Normalise the transition matrix
            for i in range(self.num_states):
                for j in range(self.num_actions):
                    row = self.T_matrix[i, j, :]
                    tmp = np.sum(row)
                    if tmp > 0:
                        self.T_matrix[i, j, :] = row / tmp
            
            # Prepare plannnings
            p_count = 0 # initialise planinng counter
            if self.plan_at_start_or_goal: # only replay if current or last state was a goal or a start state
                curr_step_is_goal  = np.array_equal(state2idcs(self.exp_list[-1, 3], self.maze), self.goal_idcs)
                last_step_was_goal = np.array_equal(state2idcs(self.exp_list[-2, 3], self.maze), self.goal_idcs) or np.array_equal(state2idcs(self.exp_list[-2, 0], self.maze), self.goal_idcs)
                if not (curr_step_is_goal or last_step_was_goal):
                    p_count = np.inf # do not plan if other states
            if r == 0 and num_episodes == 0:
                p_count = np.inf # do not plan before the first reward is encountered

            if p_count == 0:
                # Compute T_pi and H(sa, s'a')
                T_pi = np.zeros((self.num_states*self.num_actions, self.num_states*self.num_actions))
                for i in range(self.num_states):
                    for j in range(self.num_actions):
                        for ii in range(self.num_states):
                            for jj in range(self.num_actions):
                                q_vals = self.Q_table[ii, :]
                                probas = policy_choose(q_vals, 'softmax', temperature=self.t)
                                T_pi[i*self.num_actions+j, ii*self.num_actions+jj] = self.T_matrix[i, j, ii]*probas[jj]
                    # Normalise in case something goes wrong
                    row = T_pi[i*self.num_actions+j, :]
                    tmp = np.sum(row)
                    if tmp > 0:
                        T_pi[i*self.num_actions+j, :] = row/tmp
                
                # Generate replays from the model
                plan_exp = np.empty((0, 4))
                H = np.linalg.inv(np.identity(T_pi.shape[0]) - self.gamma*T_pi)
                for sn in range(self.num_states):
                    for an in range(self.num_actions):
                        these_probas = self.T_matrix[sn, an, :]
                        if np.all(these_probas==0) is not True:
                            s1n = np.argmax(these_probas)
                            rn  = np.dot(H[(sn*self.num_actions+an), :], self.R)[0]
                            this_exp = [sn, an, rn, s1n]
                            plan_exp = np.vstack((plan_exp, this_exp))
                            
                replay_backups = np.empty((0, 4))

            while p_count < self.num_plan_steps:
                print('Replay %u/%u'%(p_count+1, self.num_plan_steps))

                # Gain & Need
                gain = compute_gain(self.Q_table, plan_exp, self.gamma, self.alpha, self.t)
                need = compute_need([s, a], H, plan_exp, self.gamma)
                         
                # Expected value of memories
                EVM = gain*need
                
                if plot_location and num_episodes>20:
                    plot_each_step(ax1, s_idcs, self.maze, self.Q_table, 'Replay', 0.4)
                        
                if np.amax(EVM) > self.evm_thresh:
                    max_EVM_idx = np.where(EVM == np.amax(EVM))[0]
                    max_EVM_idx = max_EVM_idx[0]
                    
                    curr_path = plan_exp[max_EVM_idx, :]
                    sp        = int(curr_path[0])
                    spi       = state2idcs(sp, self.maze)
                    ap        = int(curr_path[1])
                    s1p       = int(curr_path[3]) # final state reached in the trajectory
                    rp        = curr_path[2]

                    # print([sp, ap, rp, s1p])
                    if plot_location and num_episodes>20:
                        plot_each_step(ax1, s_idcs, self.maze, self.Q_table, 'Replay', 1, [spi, ap])
                    
                    s1_value = np.amax(self.Q_table[s1p, :])
                    Q_target = rp + s1_value
                    delta = Q_target - self.Q_table[sp, ap]
                    self.Q_table[sp, ap] += self.alpha*delta
                    
                    replay_backups = np.vstack((replay_backups, [sp, ap, rp, s1p]))

                    p_count += 1
                else:
                    break
                            
            # Move agent to the next state    
            s = s1
            s_idcs = s1_idcs
                        
            # Complete step
            num_steps += 1
            if np.array_equal(s_idcs, self.goal_idcs): # We are in the terminal state
                
                s1 = idcs2state(self.start_idcs, self.maze)
                s1_idcs = self.start_idcs
                if self.start_to_goal:
                    delta_t = 1 - self.T_matrix[s, :, s1]
                    self.T_matrix[s, :, s1] += self.t_learn_rate * delta_t

                av_rew = np.mean(self.rew_history)
                dist = (self.Q_table - av_rew)
                self.Q_table = self.Q_table - (1-self.tau)*dist
                
                # Forget the state transition model
                self.T_matrix = self.rho * self.T_matrix + (1-self.rho) * T_init
                
                self.T_matrix *= T_allowed
                # Normalise the transition matrix
                for i in range(self.num_states):
                    for j in range(self.num_actions):
                        row = self.T_matrix[i, j, :]
                        tmp = np.sum(row)
                        if tmp > 0:
                            self.T_matrix[i, j, :] = row / tmp
                        
                s = s1
                s_idcs = s1_idcs
                            
                if save_folder:
                    path_to_save = os.path.join(save_folder, 'Episode%u'%num_episodes)
                    np.savez(path_to_save, T_pi = T_pi, T_matrix =self.T_matrix, H=H, replay_backups=replay_backups)
                         
                num_episodes += 1
                num_steps = 0
                
            if num_episodes > self.max_num_episodes:
                break