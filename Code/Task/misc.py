import numpy as np

# Supplementary funcitons 
def normalise(T):
    '''
    Normalise the transition matrix
    
    Parameters:
    T – transition matrix
    '''
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            T[i, j, i]  = 0
            
            row = T[i, j, :]
            tmp = np.sum(row)
            if tmp > 0:
                T[i, j, :] = row / tmp
    return T
                                
def get_new_state(si, a, world, state_idcs):
    '''
    Get new state from current state index and action
    
    Parameters:
    si     – index of the current state
    a      – chosen action
    world  - reward function
    state_idcs – spatial arrangement of state_idcs
    '''
    
    i_coord = si[0]
    j_coord = si[1]
    s1i = []
    
    side_i = world.shape[0]
    side_j = world.shape[1]
        
    cond1 = (i_coord == 0 and a == 0)          # top row, move up
    cond2 = (i_coord == (side_i-1) and a == 1) # bottom row, move down
    cond3 = (j_coord == 0 and a == 2)          # left column, move left
    cond4 = (j_coord == (side_j-1) and a == 3) # right column, move right
    
    if cond1:
        s1i = [side_i-1, j_coord]
    elif cond2:
        s1i = [0, j_coord]
    elif cond3:
        s1i = [i_coord, side_j-1]
    elif cond4:
        s1i = [i_coord, 0]
    else:
        if a == 0:   # up
            s1i = [i_coord-1, j_coord]
        elif a == 1: # down
            s1i = [i_coord+1, j_coord]
        elif a == 2: # left
            s1i = [i_coord, j_coord-1]
        else:        # right
            s1i = [i_coord, j_coord+1]

    s = idcs2state(s1i, state_idcs)
    r = world.ravel()[s]
    
    return s1i, r

def replay_1move(Q2, T, world, beta, alpha2, xi):
    '''
    Generate replay experiences for 1-move trials
    
    Parameters:
    Q2         – Q values for 1-move trials
    T          – state-transition matrix
    world      – image-reward associations
    beta       – inverse temperature parameter
    alpha2     – replay learning rate
    xi         – evm threshold
    '''
    
    replay_gain    = np.full(32, -200)
    replay_backups = np.full(4, -200)
    Q_history      = Q2.copy().reshape(1, 32)
    R_vec          = world.flatten()
    
    # Generate replays from the model
    replays = np.empty((32, 4))
    for sr in range(8):
        for ar in range(4):
            this_action_probas = T[sr, ar, :]
            s1r  = np.argmax(this_action_probas)
            rr   = np.sum(R_vec*this_action_probas)
            
            replays[sr*4 +ar, :] = np.array([sr, ar, rr, s1r])
                    
    while True:
        # Gain
        gain = compute_gain(Q2, replays, alpha2, beta)
                
        # Expected value of each memory
        evm         = 1/8 * gain                
        max_evm_idx = evm.argmax()

        # if the value is above threshold
        if evm[max_evm_idx] > xi:
            
            # Retrieve information from this experience
            this_replay = replays[max_evm_idx, :]
            sr          = int(this_replay[0])
            ar          = int(this_replay[1])
            rr          = this_replay[2]
            s1r         = int(this_replay[3])
            
            Q_target   = rr
            delta      = Q_target - Q2[sr, ar]
            Q2[sr, ar] = Q2[sr, ar] + alpha2 * delta
            
            replay_gain    = np.vstack((replay_gain, gain.reshape(1, 32)))
            replay_backups = np.vstack((replay_backups, [sr, ar, rr, s1r]))
            Q_history      = np.vstack((Q_history, Q2.reshape(1, 32)))
        else:
            return Q2, replay_backups, Q_history, replay_gain
                    

def replay_2moves(Q1, Q2, T, world, xi, beta2, beta1, alpha2, alpha1):
    '''
    Generate replay experiences for first moves in 2-move trials
    
    Parameters:
    Q1         – Q values for forst moves 
    Q2         – Q values for 1-move trials
    T          – state-transition matrix
    world      – image-reward associations
    xi         – evm threshold
    beta2      – inverse temperature parameter for Q2
    beta1      – inverse temperature parameter for Q1
    alpha2     – replay learning rate for Q2
    alpha1     – replay learning rate for Q1
    '''
    replays2 = np.empty((32, 4), dtype=np.float64)
    for sr in range(8):
        for ar in range(4):
            this_action_probas = T[sr, ar, :].copy()
            s1r  = np.argmax(this_action_probas)
            rr   = np.sum(world.ravel()*this_action_probas)
            
            replays2[sr*4 + ar, :] = np.array([sr, ar, rr, s1r])
    
    replay_gain1   = np.full((8*16), -200)
    replay_gain2   = np.full(32, -200)
    replay_backups = np.full(5, -200)
    Q1_history     = Q1.copy().reshape(1, 8*16)
    Q2_history     = Q2.copy().reshape(1, 32)
    # Generate 1st move experiences
    while True:
        replays1 = np.empty((8*16, 4), dtype=np.float64)  
        for sr in range(8):
            for ar in range(4):
                s1r = np.argmax(T[sr, ar, :])
                for ar2 in range(4):
                    rr = 0
                    for s2r in range(8):
                        
                        q_vals2 = Q2[s2r, :]
                        probs   = policy_choose(q_vals2, beta2)
                        
                        these_action_probas     = T[s2r, ar2, :].copy()
                        these_action_probas[sr] = 0
                        these_action_probas     = these_action_probas / np.sum(these_action_probas)
                        rr += T[sr, ar, s2r]*(world.ravel()[s2r] + probs[ar2]*np.sum(these_action_probas * world.ravel()))
                        
                    replays1[sr*16 + (ar*4 +ar2), :] = np.array([sr, ar*4+ar2, rr, s1r*8+s2r], dtype=np.float64)
                    
        # Gain & Need 
        gain_move1 = compute_gain(Q1, replays1, alpha1, beta1)
        gain_move2 = compute_gain(Q2, replays2, alpha2, beta2)
        
        # Expected value of memories
        evm_move1 = 1/8 * gain_move1
        evm_move2 = 1/8 * gain_move2
        
        # Compare memories from both moves and choose in which plane to relay
        max_evm_move1 = evm_move1.max()
        max_evm_move2 = evm_move2.max()
        
        if max_evm_move1 >= max_evm_move2:
            replay_exp = replays1
            max_evm    = max_evm_move1
            evm        = evm_move1
            plane      = 0
        else:
            replay_exp = replays2
            max_evm    = max_evm_move2
            evm        = evm_move2
            plane      = 1
        
        # if the value is above threshold
        if max_evm > xi:
            max_evm_idx = np.where(evm == max_evm)[0]
            max_evm_idx = max_evm_idx[-1]
                                
            # Retrieve information from this experience
            curr_path = replay_exp[max_evm_idx, :]
            sr        = int(curr_path[0])
            ar        = int(curr_path[1])
            rr        = curr_path[2]
            s1r       = int(curr_path[3])                    
            
            if plane == 0:
                Q_target     = rr
                delta        = Q_target - Q1[sr, ar]
                Q1[sr, ar]   = Q1[sr, ar] + alpha1 * delta
                
                Q1_history   = np.vstack((Q1_history, Q1.reshape(1, 8*16)))
                replay_gain1 = np.vstack((replay_gain1, gain_move1.reshape(1, 8*16)))
                
            else:
                Q_target     = rr 
                delta        = Q_target - Q2[sr, ar]
                Q2[sr, ar]   = Q2[sr, ar] + alpha2 * delta
                
                Q2_history   = np.vstack((Q2_history, Q2.reshape(1, 32)))
                replay_gain2 = np.vstack((replay_gain2, gain_move2.reshape(1, 32)))
            
            replay_backups = np.vstack((replay_backups, [sr, ar, rr, s1r, plane]))
        else:
            return Q1, Q2, Q1_history, Q2_history, replay_backups, replay_gain1, replay_gain2

def convert_params(param):
    
    p = {}
    p['beta']             = param[0]
    p['beta2']            = param[1]
    p['beta1']            = param[2]
    p['alpha2']           = param[3]
    p['alpha1']           = param[4]
    p['Q_init']           = param[5]
    p['T_learn_rate']     = param[6]
    p['opp_T_learn_rate'] = param[7]
    p['rho']              = param[8]
    p['tau']              = param[9]
    
    a = param[10]
    b = param[11]
    d = -(a+b)

    p['biases']           = np.array([a, b, 0, d])
    p['Block_forget']     = param[12]
    p['T_forget_block']   = param[13]
    p['tau_forget_block'] = param[14]
    p['xi']               = param[15]
    p['alpha2r']          = param[16]
    p['alpha1r']          = param[17]
    return p

def idcs2state(idcs: list, state_idcs):
    '''Convert state idcs to state id'''
    si = idcs[0]
    sj = idcs[1]
    
    return state_idcs[si, sj]

def state2idcs(s: int, state_idcs):
    '''Convert state id to state idcs'''
    return np.argwhere(s == state_idcs)[0]

def get_a_opp(a):
    '''Get opposite action'''
    if a == 0:
        a_opp = 1
    elif a == 1:
        a_opp = 0
    elif a == 2:
        a_opp = 3
    else:
        a_opp = 2
    
    return a_opp

def policy_choose(q_values, temp, biases=None):
    '''Choose an action'''
    q_vals = q_values.copy()
    
    if biases is not None:
        num = q_vals*temp + biases
    else:
        num = q_vals*temp
    
    num      = np.exp(num - np.nanmax(num))
    den      = np.nansum(num)
    pol_vals = num/den
    
    pol_vals[np.isnan(pol_vals)] = 0
    
    if pol_vals.sum() == 0:
        return np.repeat(1/len(q_vals), len(q_vals))
    
    return pol_vals/pol_vals.sum()

def policy_choose_moves(q_vals1, q_vals2, temp1, temp2, biases):
    '''Choose an action'''
    
    num = q_vals1*temp1 + q_vals2*temp2 + biases
    
    num = np.exp(num - np.nanmax(num))
    den = np.nansum(num)
    pol_vals = num/den
    
    pol_vals[np.isnan(pol_vals)] = 0
    
    if pol_vals.sum() == 0:
        return np.repeat(1/len(q_vals2), len(q_vals2))
    
    return pol_vals/pol_vals.sum()

def compute_gain(Q_table, plan_exp, alpha, temp):
    '''Compute gain term for each experience'''
    Q  = Q_table.copy()
    
    num_exp = plan_exp.shape[0]
    gain    = np.zeros(num_exp)

    for i in range(num_exp):
        curr_exp = plan_exp[i, :]
        s = int(curr_exp[0])
        q_vals = Q[s, :]

        # Policy before backup
        probs_pre = policy_choose(q_vals, temp)
        a_taken   = int(curr_exp[1])
        
        if np.isnan(q_vals[a_taken]):
            gain[i] = -200
        else:
            # Next state value
            r = curr_exp[2]
            # Q_target = r

            q_vals_after = q_vals.copy()
            delta = r - q_vals_after[a_taken]
            q_vals_after[a_taken] = q_vals_after[a_taken] + alpha*delta
            probs_post = policy_choose(q_vals_after, temp)
            
            # Calculate gain
            EV_pre = np.nansum(probs_pre*q_vals_after)
            EV_post = np.nansum(probs_post*q_vals_after)
            gain[i] = (EV_post - EV_pre)
    return gain

def compute_gain_first_move(Q_list: list, plan_exp, alpha:list, temp:list, biases):
    '''Compute gain term for each experience'''
    Q1 = Q_list[0].copy()
    Q2 = Q_list[1].copy() 
    
    beta1 = temp[0]
    beta2 = temp[1]
    
    num_exp = plan_exp.shape[0]
    gain    = np.zeros(num_exp)

    for i in range(num_exp):
        curr_exp = plan_exp[i, :]
        s  = int(curr_exp[0])
        a  = int(curr_exp[1])
        
        q_vals1_raw = Q1[s, :].copy()
        
        q_vals1 = np.empty(0)
        for ai in range(4):
            tmp     = Q1[s, (ai*4):(ai*4+4)].copy()
            probs   = policy_choose(tmp, beta1, biases)
            q_vals1 = np.append(q_vals1, np.sum(tmp*probs))
            
        q_vals2 = Q2[s, :].copy()
        
        # Policy before backup
        probs_pre = policy_choose_moves(q_vals1, q_vals2, beta1, beta2, biases)
        a_taken   = a
        
        # Next state value
        r = curr_exp[2]
        # Q_target = r

        q_vals1_raw_after = q_vals1_raw.copy()
        delta = r - q_vals1_raw_after[a_taken]
        q_vals1_raw_after[a_taken] = q_vals1_raw_after[a_taken] + alpha*delta
        
        q_vals1_after = np.empty(0)
        for ai in range(4):
            tmp     = q_vals1_raw_after[(ai*4):(ai*4+4)].copy()
            probs   = policy_choose(tmp, beta1, biases)
            q_vals1_after = np.append(q_vals1_after, np.sum(tmp*probs))
            
        probs_post = policy_choose_moves(q_vals1_after, q_vals2, beta1, beta2, biases)
        
        # Calculate gain
        EV_pre = np.sum(probs_pre*q_vals1_after)
        EV_post = np.sum(probs_post*q_vals1_after)
        gain[i] = (EV_post - EV_pre)
        
    return gain

def model_rearrange(T, idcs1, idcs2, world2):
    
    T_rearr = T.copy()
    for s in range(8):
        for act in range(4):
            si = state2idcs(s, idcs1)
            s1i_b, _ = get_new_state(si, act, world2, idcs1)
            s1b = idcs2state(s1i_b, idcs1)
            si = state2idcs(s, idcs2)
            s1i_a, _ = get_new_state(si, act, world2, idcs2)
            s1a = idcs2state(s1i_a, idcs2)
            
            T_rearr[s, act, s1a] = T[s, act, s1b]
            T_rearr[s, act, s1b] = T[s, act, s1a]
            
    return T_rearr