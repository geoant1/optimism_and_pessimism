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
        self.evm_thresh       = p['EVM_threshold']

        self.maze   = maze
        self.side_i = maze.shape[0]
        self.side_j = maze.shape[1]
        self.num_states  = self.side_i * self.side_j
        self.num_actions = 4 # 0 = up, 1 = down, 2 = left, 3 = right from each state
        self.start_idcs  = p['start_location']
        self.goal_idcs   = self.r_params['pos_loc']
        
        self.Q = np.ones((self.num_states, self.num_actions)) # table of Q values 
        self.H = np.eye(self.num_states*self.num_actions)
        
        self.rew_history = []
        self.sars = np.empty((0, 4), dtype=int)
        
        return
    
    def pre_explore(self):
        '''
        Define allowed transitions that respect the world's boundaries
        '''
        
        # From each state we can only transition into a valid state
        idcs = np.where(self.maze.ravel()==1)[0]
        for i in idcs:
            self.Q[i, :] = 0
            for j in range(self.num_actions):
                self.H[i*4+j, :] = 0
        
        # Correct for actions from valid states that lead into walls
        idcs = np.where(self.maze.ravel()==0)[0]
        for s in idcs:
            s_idcs = state2idcs(s, self.maze)
            for a in range(self.num_actions):
                s1_idcs, _ = get_new_state(s_idcs, a, self.r_params, self.maze)
                s1 = idcs2state(s1_idcs, self.maze)
                if s == s1:
                    self.Q[s, a] = 0
                    self.H[s*4+a, :] = 0
                    
        # gs  = idcs2state(self.goal_idcs, self.maze)
        # ss = idcs2state(self.start_idcs, self.maze)
        # self.H[ss*4:(ss*4+4), gs*4:(gs*4+4)] = 0
    
    def run_simulation(self, plot_location=True, save_folder=None):
        
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
            
        goal_state  = idcs2state(self.goal_idcs, self.maze)
        start_state = idcs2state(self.start_idcs, self.maze)

        Q_mask = np.ma.masked_equal(self.Q, 0)
        H_mask = np.where(~self.H.any(axis=1))[0]
        self.Q[Q_mask.mask] = -200
        # idcs = np.where(self.maze.ravel()==1)[0]
        moves = np.empty((0, 4))
        preplay = False
        replay  = False
        # Start exploring maze
        for t in range(self.num_sim_steps):
            print('%d steps; %d episodes'%(t, num_episodes))
            
            if plot_location and num_episodes>7:
                plot_each_step(ax1, s_idcs, self.maze, self.Q, 'Exploring', 0.2)
            
            # Choose an action
            
            q_vals = self.Q[s, :]
            probs = policy_choose(q_vals, 'softmax', temperature=self.t)
            a = np.random.choice(range(self.num_actions), p=probs)
            
            # Execute the action, move to the next state and receive reward
            s1_idcs, r = get_new_state(s_idcs, a, self.r_params, self.maze)
            s1 = idcs2state(s1_idcs, self.maze)
            if s1 == s:
                while True:
                    q_vals = self.Q[s, :]
                    probs = policy_choose(q_vals, 'softmax', temperature=self.t)
                    a = np.random.choice(range(self.num_actions), p=probs)
                    
                    # Execute the action, move to the next state and receive reward
                    s1_idcs, r = get_new_state(s_idcs, a, self.r_params, self.maze)
                    s1 = idcs2state(s1_idcs, self.maze)
                    if s1 != s:
                        break
            
            if r != 0:
                self.rew_history.append(r)
            
            # Update Q-values
            # off-policy learning
            q     = self.Q[s, a]
            q1    = np.max(self.Q[s1, :])
            
            Q_target = r + self.gamma * q1
            delta    =  Q_target - q # temporal difference
            self.Q[s, a] += self.alpha*delta
            
            # Prepare replay & update the SR 
            
            p_count = np.inf
            if t > 0:
                
                prev_s  = self.sars[-1, 0]
                prev_s1 = self.sars[-1, 3]
                prev_a  = self.sars[-1, 1]
                
                tmp = np.zeros(self.num_states*self.num_actions)
                tmp[prev_s*4+prev_a] = 1
                delta = tmp + self.gamma*self.H[(s*4+a), :] - self.H[(prev_s*4+prev_a), :]
                self.H[(prev_s*4+prev_a), :] += self.t_learn_rate*delta
                
                # Also learn about opposite transition
                # prev_a_opp = get_a_opp(prev_a)
                # a_opp      = get_a_opp(a)
                # tmp = np.zeros(self.num_states*self.num_actions)
                # tmp[s*4+a_opp] = 1
                # delta = tmp + self.gamma*self.H[(prev_s*4+prev_a_opp), :] - self.H[(s*4+a_opp), :]
                # self.H[(s*4+a_opp), :] += self.alpha*delta
                
                self.H[H_mask, :] = 0
                
                cond1  = (s1 == goal_state)
                if cond1:
                    replay = True
                else: replay = False
                    
                cond2  = (s  == start_state) and (prev_s1 == goal_state)
                if cond2:
                    preplay = True
                else: preplay = False
                
                if not cond2 and not cond1:
                    preplay = False
                    replay  = False
                    
                if replay or preplay:
                    p_count = 0
                    plan_exp = np.empty((0, 4))
                    for sn in range(self.num_states):
                        for an in range(self.num_actions):
                            these_visits = self.H[sn*4+an, :]
                            # print(these_visits)
                            # if np.all(these_visits==0) is not True:
                            s1n = np.argmax(these_visits)//4
                            rn  = np.dot(these_visits, self.R)
                            this_exp = [sn, an, rn, s1n]
                            plan_exp = np.vstack((plan_exp, this_exp))
                                
                    replay_backups = np.empty((0, 5))
                    replay_gain    = np.empty((self.num_states*self.num_actions))
                    Q_history      = np.empty((self.num_states*self.num_actions))
                    
            self.sars = np.vstack((self.sars, [s, a, r, s1]))
            moves = np.vstack((moves, [s, a, r, s1]))
            
            while p_count < self.num_plan_steps:
                print('Replay %u/%u'%(p_count+1, self.num_plan_steps))
                # Gain & Need
                # Gain & Need
                gain = compute_gain(self.Q, plan_exp, self.gamma, self.alpha, self.t)
                
                # need = []
                # for i in range(self.num_states):
                #     for j in range(self.num_actions):
                #         need.append(compute_need([i, j], self.H, plan_exp, self.gamma))
                
                need = compute_need([s, a], self.H, plan_exp, self.gamma)
                # Expected value of memories
                # EVM = []
                # for i in range(self.num_actions):
                #     for j in range(self.num_states):
                #         EVM.append(gain*need[i*j + j])
                
                EVM = gain*need
                        
                # max_EVM = 0
                # k = 0
                # for i in range(self.num_actions):
                #     for j in range(self.num_states):
                #         this_EVM = EVM[i*j + j]
                #         this_EVM_max = np.amax(this_EVM)
                #         if this_EVM_max > max_EVM:
                #             max_EVM = this_EVM_max
                #             max_EVM_idx = np.where(this_EVM == np.amax(max_EVM))[0]
                #             max_EVM_idx = max_EVM_idx[0]
                #             k = i*j + j
                
                if plot_location and num_episodes>7:
                    plot_each_step(ax1, s_idcs, self.maze, self.Q, 'Replay', 0.4)
                        
                if np.amax(EVM) > self.evm_thresh:
                    max_EVM_idx = np.where(EVM == np.amax(EVM))[0]
                    max_EVM_idx = max_EVM_idx[0]
                    
                    curr_path = plan_exp[max_EVM_idx, :]
                    sp        = int(curr_path[0])
                    spi       = state2idcs(sp, self.maze)
                    ap        = int(curr_path[1])
                    s1p       = int(curr_path[3]) # final state reached in the trajectory
                    rp        = curr_path[2]

                    if plot_location and num_episodes>7:
                        plot_each_step(ax1, s_idcs, self.maze, self.Q, 'Replay', 1, [spi, ap])
                    
                    Q_target = rp 
                    delta = Q_target - self.Q[sp, ap]
                    self.Q[sp, ap] += self.alpha*delta
                    
                    replay_backups = np.vstack((replay_backups, [sp, ap, rp, s1p, delta]))
                    replay_gain    = np.vstack((replay_gain, gain))
                    # replay_need    = np.append(replay_need, need[k][max_EVM_idx])
                    Q_history      = np.vstack((Q_history, self.Q.reshape(1, self.num_states*self.num_actions)))
                                        
                    p_count += 1
                else:
                    break
                                                    
            # Complete step
            num_steps += 1
                    
            if preplay or replay: # We are in the terminal state
                
                if replay:
                    av_rew = np.mean(self.rew_history)
                    dist   = (self.Q - av_rew)
                    self.Q = self.Q - (1-self.tau)*dist
                    self.Q[Q_mask.mask] = -200
                    
                    tmp = np.copy(self.H)
                    tmp[H_mask, :] = np.nan
                    tmp[:, H_mask] = np.nan
                    av = np.nanmean(tmp[:])
                    for i in range(self.H.shape[0]):
                        this_row = self.H[i, :].copy()
                        dist     = this_row - av
                        tmp      = self.rho*this_row - (1-self.rho)*dist
                        tmp[H_mask] = 0
                        if np.sum(tmp) > 0:
                            tmp = tmp/np.max(tmp)
                        self.H[i, :] = tmp
                        
                    self.H[H_mask, :] = 0
                    
                    if save_folder:
                        path_to_save = os.path.join(save_folder, 'Episode%u'%num_episodes)
                        np.savez(path_to_save, move=moves, H=self.H.flatten(), replay_backups=replay_backups, replay_gain=replay_gain, replay_need=need, Q_history=Q_history, rew_history=self.rew_history)
                    
                    s = start_state
                    s_idcs = self.start_idcs
                    
                    moves = np.empty((0, 4))
                
                if preplay:
                    if save_folder:
                        path_to_save = os.path.join(save_folder, 'Episode%u'%num_episodes)
                        np.savez(path_to_save, H=self.H.flatten(), replay_backups=replay_backups, replay_gain=replay_gain, replay_need=need, Q_history=Q_history, rew_history=self.rew_history)
                    
                    s = s1
                    s_idcs = s1_idcs
                
                num_episodes += 1
                num_steps = 0
                
            else:
                s = s1
                s_idcs = s1_idcs
                
            if num_episodes > self.max_num_episodes:
                break