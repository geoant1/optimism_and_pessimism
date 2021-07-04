import numpy as np
from misc import normalise, model_rearrange, state2idcs, idcs2state, get_new_state, get_a_opp
from misc_fit import policy_choose, replay_2moves, replay_1move, policy_choose_moves

class Agent:
    
    def __init__(self, **kwargs):    
        
        # ---- Agent parameters ----
        self.__dict__.update(**kwargs)
    
        self.num_states  = 8
        self.num_actions = 4 # as always
        
        # Worlds
        self.world1 = np.array([[0, 9, 3, 5], [8, 2, 1, 10]], dtype=int)
        self.idcs1  = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=int)
        self.world2 = np.array([[3, 1, 0, 9], [5, 10, 8, 2]], dtype=int)
        self.idcs2  = np.array([[5, 1, 2, 6], [4, 0, 3, 7]], dtype=int)
        
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
        
        self.r_list = []
                
        return None
    
    def train(self, states):
        
        # Training
        world     = self.world1
        state_arr = self.idcs1
        for bl in range(7):
            these_states = states[bl]
            self.one_move(world, state_arr, states=these_states)

        return None
    
    def task(self, states):
        
        self.r_blocks = []
        # Task
        for bl in range(5):
            self.r_list  = []
            
            these_states = states[bl]
            
            if bl in [0, 1]:
                world     = self.world1
                state_arr = self.idcs1
            elif bl in [2, 3]: # reward association change
                world     = self.world2
                state_arr = self.idcs1
            else: # spatial rearrangement
                world     = self.world2
                state_arr = self.idcs2
            
            for i in range(3):
                forget = False
                learn  = True
                if (i == 0):
                    if (bl > 0):
                        learn = False # no feedback in some first trials after block 0
                    self.one_move(world, state_arr, states=these_states[i*18:i*18+6], online_learning=learn)
                    self.two_moves(world, state_arr, states=these_states[i*18+6:i*18+6+6], online_learning=learn)
                    self.two_moves(world, state_arr, states=these_states[i*18+6+6:i*18+6+6+6])
                else:
                    if ((bl == 1) or (bl == 3)) and (i == 2):
                        forget = True # do not forget at the end – special parameter in self.offtask_replay()
                    self.one_move(world, state_arr, states=these_states[i*18:i*18+6])
                    self.two_moves(world, state_arr, states=these_states[i*18+6:i*18+6+12], T_forget=forget)
            
            if (bl == 1):
                self.offtask_replay()
            elif (bl == 3):
                self.offtask_replay(rearrange=True)
            
            self.r_blocks += [self.r_list]
            
        return None
                
    def offtask_replay(self, rearrange=False):
        
        av_rew2 = np.mean(self.rew_history2)
        av_rew1 = np.mean(self.rew_history1)

        dist    = (self.Q2 - av_rew2)
        self.Q2 = self.Q2 - (1-self.tau_forget_block)*dist

        dist    = (self.Q1 - av_rew1)
        self.Q1 = self.Q1 - (1-self.tau_forget_block)*dist
        
        if rearrange: # rearrange the model 
            self.T2  = self.Block_forget * self.T2 + (1-self.Block_forget) * 1/7
            self.T2  = normalise(self.T2)
            T2_rearr = model_rearrange(self.T2, self.idcs1, self.idcs2, self.world2)
            T2_rearr = normalise(T2_rearr)
            new_T2   = (1-self.T_forget_block) * self.T2 + self.T_forget_block * T2_rearr
            self.T2  = new_T2
        else:
            self.T2  = self.rho * self.T2 + (1-self.rho) * 1/7 
            self.T2  = normalise(self.T2)
            
        self.Q1, self.Q2 = replay_2moves(self.Q1, self.Q2, self.T2, self.world2, self.xi, self.beta2, self.beta1, self.alpha2r, self.alpha1r)
        
        return None
    
    def one_move(self, world, state_arr, states, online_learning=True):
        '''Run 1-move trials'''
        
        num_trials = len(states)
            
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
            
            self.r_list.append(r)
            
            # online TD-learning
            if online_learning:
                self.rew_history2.append(r)

                # Update Q values
                Q_target      = r        
                delta         = Q_target - self.Q2[s, a]
                self.Q2[s, a] = self.Q2[s, a] + self.alpha2 * delta
            
                # Update the state transition model
                delta_t = 1 - self.T2[s, a, s1]
                self.T2[s, a, s1] += self.T_learn_rate * delta_t
                
                # NB we also learn about opposite transitions
                a_opp = get_a_opp(a)
                delta_t_opp = 1 - self.T2[s1, a_opp, s]
                self.T2[s1, a_opp, s] += self.T_learn_rate * self.opp_T_learn_rate * delta_t_opp
                
                # Normalise the transition matrix
                self.T2  = normalise(self.T2)
            
            # Prepare replay
            replay = True # initialise replay 
            if replay:
                self.Q2 = replay_1move(self.Q2.copy(), self.T2, world, self.beta, self.alpha2r, self.xi)
            
            # Forget Q values
            if online_learning:
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
            
            
    def two_moves(self, world, state_arr, states, online_learning=True, T_forget=False):
                
        num_trials = len(states)
            
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
        
        s_counter = 0
        s = states[s_counter]
        si = state2idcs(s, state_arr)
        
        # Start exploring maze
        for trial in range(num_trials*2):
                    
            # First move
            if num_steps == 0:
                # Choose an action
                q_vals1 = np.zeros(4)
                for i in range(self.num_actions):
                    tmp   = self.Q1[s, (i*4):(i*4+4)].copy()
                    probs = policy_choose(tmp, self.beta1, self.biases)
                    q_vals1[i] = np.nansum(tmp*probs)
                q_vals2 = self.Q2[s, :].copy()
                
                probs  = policy_choose_moves(q_vals1, q_vals2, self.beta1, self.beta2, self.biases)
                a      = np.random.choice(range(self.num_actions), p=probs)
                
                # Execute the action, move to the next state and receive reward
                s1i, r = get_new_state(si, a, world, state_arr)
                s1     = idcs2state(s1i, state_arr)
                                
                sars = np.vstack((sars, [s, a, r, s1]))
                
                if online_learning:
                    # Update Q values
                    Q_target      = r
                    delta         = Q_target - self.Q2[s, a]
                    self.Q2[s, a] = self.Q2[s, a] + self.alpha2 * delta

                    # Update state-transition model
                    delta_t = 1 - self.T2[s, a, s1]
                    self.T2[s, a, s1] += self.T_learn_rate * delta_t
                    
                    # NB we also learn about opposite transitions
                    a_opp = get_a_opp(a)
                    delta_t_opp = 1 - self.T2[s1, a_opp, s]
                    self.T2[s1, a_opp, s] += self.T_learn_rate * self.opp_T_learn_rate * delta_t_opp
                    
                    # Normalise the transition matrix
                    self.T2  = normalise(self.T2)
                            
            # Second move    
            else:
                # Choose an action
                prev_s  = sars[-1, 0]
                prev_a  = sars[-1, 1]
                prev_r  = sars[-1, 2]
                
                q_vals1 = self.Q1[prev_s, (prev_a*4):(prev_a*4+4)]
                
                if online_learning:
                    q_vals2 = self.Q2[s, :].copy()
                    
                    probs   = policy_choose_moves(q_vals1, q_vals2, self.beta1, self.beta2, self.biases)
                    a       = np.random.choice(range(self.num_actions), p=probs)
                    
                    # Execute the action, move to the next state and receive reward
                    s1i, r  = get_new_state(si, a, world, state_arr)
                    s1      = idcs2state(s1i, state_arr)
                    
                    # Update both Q values
                    Q_target      = r        
                    delta         = Q_target - self.Q2[s, a]
                    self.Q2[s, a] = self.Q2[s, a] + self.alpha2 * delta
                    
                    Q_target      = r + prev_r
                    delta         = Q_target - self.Q1[prev_s, prev_a*4 + a]
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
                
                # No online learning            
                else:
                    q_vals2 = np.zeros(4)
                    for i in np.delete(range(8), prev_s):
                        q_vals2  += self.Q2[i, :]
                    q_vals2 /= 7
                    
                    probs    = policy_choose_moves(q_vals1, q_vals2, self.beta1, self.beta2, self.biases)
                    a        = np.random.choice(range(self.num_actions), p=probs)

                    # Execute the action, move to the next state and receive reward
                    s1i, r   = get_new_state(si, a, world, state_arr)
                    s1       = idcs2state(s1i, state_arr)
                
                self.r_list.append(prev_r+r)
                
            # Prepare replay
            replay = False # initialise counter
            if num_steps != 0:
                replay = True

            if replay:
                self.Q1, self.Q2 = replay_2moves(self.Q1.copy(), self.Q2.copy(), self.T2, world, self.xi, self.beta2, self.beta1, self.alpha2r, self.alpha1r)
            
            # Forget after first move
            if num_steps == 0 and online_learning:
                # Forget Q values
                av_rew2  = np.mean(self.rew_history2)
                dist2    = np.subtract(self.Q2, av_rew2)
                self.Q2  = self.Q2 - (1-self.tau)*dist2
                
                self.T2  = self.rho * self.T2 + (1-self.rho) * 1/7
                # Normalise the transition matrix
                self.T2  = normalise(self.T2)
                            
            # Complete step
            num_steps += 1
            
            # Move agent to the next state    
            s = s1
            si = s1i
            
            if num_steps == 2:
                
                # Do not forget after certain blocks
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
                
                num_steps     = 0
                num_episodes += 1
                
                s_counter += 1
                s  = states[s_counter]
                si = state2idcs(s, state_arr)