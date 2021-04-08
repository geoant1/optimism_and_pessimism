import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from misc import *

class Agent:

    def __init__(self, p: dict, maze):  
        
        self.max_num_episodes      = p['max_num_episodes']
        self.num_plan_steps        = p['num_plan_steps'] 
        self.num_sim_steps         = p['num_sim_steps']     
        self.plan_at_start_or_goal = p['plan_at_start_or_goal']
        self.expand_further        = p['expand_further']
        self.allow_loops           = p['allow_loops']
        self.remove_same_state     = p['remove_same_state']
        self.start_to_goal         = p['start_to_goal']
        self.r_params              = p['r_params']

        self.act_policy       = p['act_policy']
        self.t                = p['temperature']
        self.on_vs_off_policy = p['on_vs_off_policy']
        self.t_learn_rate     = p['t_learn_rate']
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
        self.goal_idcs   = p['goal_location']
        
        self.Q_table  = np.zeros((self.num_states, self.num_actions), dtype=np.float64) # table of Q values 
        self.T_matrix = np.zeros((self.num_states, self.num_states), dtype=np.float64)  # transition probability matrix 
        self.E_matrix = np.zeros((self.num_states, self.num_actions), dtype=np.float64) # eligibility matrix
        
        self.exp_list = np.empty((0, 4), float) # array for storing experiences
        self.exp_last_step = np.empty((self.num_states, self.num_actions)) # next state in each entry
        self.exp_last_step[:] = np.nan                           
        
        self.exp_last_rew  = np.empty((self.num_states, self.num_actions)) # next reward 
        self.exp_last_rew[:] = np.nan

        return
    
    def pre_explore(self):
        
        for s in range(self.num_states):
            s_idcs = state2idcs(s, self.maze)
            si = s_idcs[0]
            sj = s_idcs[1]
            for a in range(self.num_actions):
                if self.maze[si, sj] == 0 and not np.array_equal(s_idcs, self.goal_idcs): # do not explore walls and goal location
                    [s1_idcs, _] = get_new_state(s_idcs, a, self.r_params, self.maze)
                    s1 = idcs2state(s1_idcs, self.maze)
                                
                    self.exp_list = np.append(self.exp_list, np.array([[s, a, 0, s1]]), axis=0)
                    self.exp_last_step[s, a] = s1
                    self.exp_last_rew[s, a] = 0
                    self.T_matrix[s, s1] += 1
        
        # normalise the transition matrix
        for i in range(self.num_states):
            row = self.T_matrix[i, :]
            tmp = np.sum(row)
            if tmp > 0:
                self.T_matrix[i, :] = row / tmp

        # Add start-to-goal transitions
        if self.start_to_goal:
            g = idcs2state(self.goal_idcs, self.maze)  # get the goal state
            s = idcs2state(self.start_idcs, self.maze) # get the start state
            self.T_matrix[g, :] = 0
            self.T_matrix[g, s] = 1
            
        return
    
    def run_simulation(self, plot_location=True):
        
        num_episodes = 0
        num_steps    = 0
        
        s      = idcs2state(self.start_idcs, self.maze)
        s_idcs = self.start_idcs
                
        if plot_location:
            plt.ion()
            fig1 = plt.figure(figsize=(10, 8))
            ax1  = plt.axes([0, 0, 1, 1])

        # Start exploring maze
        for t in range(self.num_sim_steps):
            if t%100 == 0:
                print('%d steps; %d episodes'%(t, num_episodes))
            
            if plot_location and num_episodes>0:
                plot_each_step(ax1, s_idcs, self.maze, self.Q_table, 'Exploring', 0.2)
            
            # Choose an action
            q_vals = self.Q_table[s, :]
            probs = policy_choose(q_vals, self.act_policy, temperature=self.t)
            a = np.random.choice(range(self.num_actions), p=probs)
            
            # Execute the action, move to the next state and receive reward
            s1_idcs, r = get_new_state(s_idcs, a, self.r_params, self.maze)
            s1 = idcs2state(s1_idcs, self.maze)
            
            # Update transition matrix and experience list
            targ_vec = np.zeros(self.num_states)
            targ_vec[s1] = 1
            self.T_matrix[s, :] = self.T_matrix[s, :] + self.t_learn_rate*(targ_vec - self.T_matrix[s, :])
            
            self.exp_list = np.append(self.exp_list, np.array([[s, a, r, s1]]), axis=0)
            self.exp_last_step[s, a] = s1
            self.exp_last_rew[s, a]  = r
            
            # Update Q-values
            # off-policy learning
            q     = self.Q_table[s, a]
            q1    = np.max(self.Q_table[s1, :])
            
            Q_target = r + self.gamma * q1
            delta    =  Q_target - q # temporal difference

            self.E_matrix[s, :] = 0
            self.E_matrix[s, a] = 1
            
            self.Q_table += self.alpha * self.E_matrix * delta # TD-learning
            self.E_matrix *= self.lam * self.gamma             # decay eligibility trace
                                    
            # Prepare plannning
            p = 0 # initialise planinng counter
            if self.plan_at_start_or_goal: # only replay if current or last state was a goal or a start state
                curr_step_is_goal  = np.array_equal(state2idcs(self.exp_list[-1, 3], self.maze), self.goal_idcs)
                last_step_was_goal = np.array_equal(state2idcs(self.exp_list[-2, 3], self.maze), self.goal_idcs) or np.array_equal(state2idcs(self.exp_list[-2, 0], self.maze), self.goal_idcs)
                if not (curr_step_is_goal or last_step_was_goal):
                    p = np.inf # do not plan if other states
            if r == 0 and num_episodes == 0:
                p = np.inf # do not plan before the first reward is encountered

            # Pre-allocate variable for planning
            planning_backups = np.empty((0, 5))
            backups_gain     = []
            backups_need     = []
            backupsEVM       = []

            while p < self.num_plan_steps:
                print('Replay')
                col0     = np.tile(np.arange(self.num_states).reshape(-1, 1), (self.num_actions, 1))
                col1     = np.repeat(np.arange(self.num_actions).reshape(-1, 1), self.num_states, axis=0)
                col2     = np.transpose(self.exp_last_rew).ravel()
                col3     = np.transpose(self.exp_last_step).ravel()
                plan_exp = np.column_stack((col0, col1, col2, col3))
                plan_exp = plan_exp[~np.isnan(plan_exp).any(axis=1)]
                if self.remove_same_state:
                    plan_exp = plan_exp[~(plan_exp[:, 0] == plan_exp[:, 3])]
                
                # reformat
                plan_exp = [np.expand_dims(plan_exp[i], axis=0) for i in range(plan_exp.shape[0])]
                # expand previous backup with one extra action
                if self.expand_further and (planning_backups.shape[0] > 0):
                    seq_start = np.where(planning_backups[:, 4] == 1)[0][-1]
                    seq_so_far = planning_backups[seq_start:, :4]
                    sn = int(seq_so_far[-1, 3])
                    
                    if self.on_vs_off_policy == 'off_policy': # appended experience is sampled greedily
                        probs = np.zeros(self.num_actions)
                        qn = self.Q_table[sn, :]
                        max_qn = np.amax(qn)
                        if max_qn == 0:
                            probs = [0.25]*4
                        else:
                            idcs =  np.argwhere(qn == max_qn).ravel()
                            probs[idcs] = max_qn/np.sum(qn[idcs])
                    an  = np.random.choice(range(self.num_actions), p=probs) # select action to append using the same policy as in real experience
                    sn1 = self.exp_last_step[sn, an] # new state sn1 resulting from taking action an in state sn 
                    rn  = self.exp_last_rew[sn, an] # reward received for transitioning to sn1 from sn 
                    
                    next_step_is_nan = np.isnan(sn1) or np.isnan(rn)
                    next_step_is_repeated = (sn1 in seq_so_far[:, 0]) or (sn1 in seq_so_far[:, 3])
                            
                    if not (next_step_is_nan) and (self.allow_loops or (not next_step_is_repeated)):
                        seq_so_far = [seq_so_far[i] for i in range(seq_so_far.shape[0])]
                        seq_updated = np.vstack((seq_so_far, [sn, an, rn, sn1]))
                        plan_exp.append(seq_updated)
                

                # Gain & Need
                gain, _ = compute_gain(self.Q_table, plan_exp, self.gamma, self.alpha)
                need, _ = compute_need(s, self.T_matrix, plan_exp, self.gamma)
                                
                # Expected value of memories
                EVM = np.zeros(len(plan_exp))
                for i in range(len(plan_exp)):
                    EVM[i] = np.sum(np.array(need[i][-1]) * np.array(compare(gain[i], self.baseline_gain)))
                
                if plot_location:
                    plot_each_step(ax1, s_idcs, self.maze, self.Q_table, 'Replay', 0.4)
                
                opport_cost = np.nanmean(self.exp_list[:, 2]) # expected reward from a random act
                EVM_thresh  = np.array([opport_cost, self.evm_thresh]).min()
                
                if np.amax(EVM) > EVM_thresh:
                    max_EVM_idx = np.where(EVM == np.amax(EVM))[0]
                    if len(max_EVM_idx) > 1:
                        n_steps = [plan_exp[i].shape[0] for i in max_EVM_idx] # number of steps in each trajectory
                        max_EVM_idx = max_EVM_idx[np.argmin(n_steps)] # select the one corresponsding to a shorter trajectory
                    else:
                        max_EVM_idx = max_EVM_idx[0]
                    
                    # Retrieve information from this experience
                    for n in range(plan_exp[max_EVM_idx].shape[0]):
                        curr_path = plan_exp[max_EVM_idx]
                        sp        = int(curr_path[n, 0])
                        spi       = state2idcs(sp, self.maze)
                        ap        = int(curr_path[n, 1])
                        s1p       = int(curr_path[-1, 3]) # final state reached in the trajectory
                        r_to_end  = curr_path[n:, 2]
                        num_p     = len(r_to_end)
                        rp        = np.dot(self.gamma**np.arange(num_p), r_to_end)
                        
                        if plot_location:
                            plot_each_step(ax1, s_idcs, self.maze, self.Q_table, 'Replay', 0.4, [spi, ap])
                        # plt.waitforbuttonpress()
                        s1_value = np.amax(self.Q_table[s1p, :])
                        Q_target = rp + (self.gamma**num_p)*s1_value
                        self.Q_table[sp, ap] = self.Q_table[sp, ap] + self.alpha*(Q_target - self.Q_table[sp, ap])
                        
                    backups_gain.append(gain[max_EVM_idx])
                    backups_need.append(need[max_EVM_idx])
                    backupsEVM.append(EVM[max_EVM_idx])
                    planning_backups = np.vstack((planning_backups, np.append(plan_exp[max_EVM_idx][-1], plan_exp[max_EVM_idx].shape[0])))

                    p += 1
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
                    targ_vec = np.zeros(self.num_states)
                    targ_vec[s1] = 1
                    self.T_matrix[s, :] = self.T_matrix[s, :] + self.t_learn_rate*(targ_vec - self.T_matrix[s, :])
                    self.exp_list = np.vstack((self.exp_list, [s, np.nan, np.nan, s1]))
                s = s1
                s_idcs = s1_idcs
                self.E_matrix = np.zeros_like(self.E_matrix)
                num_episodes += 1
                num_steps = 0
                
            if num_episodes > self.max_num_episodes:
                break