import numpy as np
import time

# Supplementary funcitons 
def tic():
    return time.time()

def toc(a):
    return time.time() - a

def get_new_state(si, a, world, states):
    '''Get new state from current state and action'''
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

    s = idcs2state(s1i, states)
    r = world.ravel()[s]
    
    return s1i, r

def replay_2moves(Q1, Q2, replay_exp2, T2, world, evm_thresh, beta2, beta1, alpha2, alpha1):
    replay_gain2     = np.empty((0, 8*4))
    replay_backups   = np.empty((0, 5))
    Q2_history       = Q2.flatten()
    
    replay_gain1     = np.empty((0, 8*16))
    Q1_history       = Q1.flatten()
    replay_planes    = []
    idcs_rmv         = []
    
    # Generate 1st move experiences
    while True:
        replay_exp1 = np.empty((0, 4))  
        for sr in range(8):
            for ar in range(4):
                s1r = np.argmax(T2[sr, ar, :])
                for ar2 in range(4):
                    rr = 0
                    for s2r in range(8):
                        
                        q_vals2 = Q2[s2r, :]
                        probs  = policy_choose(q_vals2, beta2)
                        
                        rr += T2[sr, ar, s2r]*(world.ravel()[s2r] + probs[ar2]*np.sum(T2[s2r, ar2, :] * world.ravel()))
                        
                    this_replay = np.array([sr, ar*4+ar2, rr, s1r*8+s2r])
                    replay_exp1 = np.vstack((replay_exp1, this_replay)) 
                    
        # Gain & Need 
        gain_move1 = compute_gain(Q1, replay_exp1, alpha1, beta1)
        gain_move2 = compute_gain(Q2, replay_exp2, alpha2, beta2)
        
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
        if max_evm > evm_thresh:
            max_evm_idx = np.where(evm == max_evm)[0]
            max_evm_idx = max_evm_idx[-1]
                                
            # Retrieve information from this experience
            curr_path = replay_exp[max_evm_idx, :]
            sr        = int(curr_path[0])
            ar        = int(curr_path[1])
            rr        = curr_path[2]
            s1r       = int(curr_path[3])                    
            
            if plane == 0:
                # s1_val   = np.amax(Q2[s1r, :])
                # Q_target = rr + gamma * s1_val
                Q_target = rr
                delta = Q_target - Q1[sr, ar]
                Q1[sr, ar] = Q1[sr, ar] + alpha1 * delta
                
            else:
                Q_target = rr 
                delta = Q_target - Q2[sr, ar]
                Q2[sr, ar] = Q2[sr, ar] + alpha2 * delta
            
            replay_gain1    = np.vstack((replay_gain1, gain_move1))
            Q1_history      = np.vstack((Q1_history, Q1.flatten()))
            
            replay_gain2    = np.vstack((replay_gain2, gain_move2))
            Q2_history      = np.vstack((Q2_history, Q2.flatten()))
            replay_planes.append(plane)
            replay_backups = np.vstack((replay_backups, [sr, ar, rr, s1r, plane]))
                    
        else:
            return Q1, Q2, replay_gain1, replay_gain2, Q1_history, Q2_history, replay_planes, replay_backups

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

    p['biases']           = [a, b, 0, d] 
    p['tau_forget_block'] = param[12]
    p['T_forget_block']   = param[13]
    p['evm_threshold']    = param[14]
    p['num_trials'] = None
    return p

def idcs2state(idcs: list, states):
    '''Convert state idcs to state id'''
    si = idcs[0]
    sj = idcs[1]
    
    return states[si, sj]

def state2idcs(s: int, states):
    '''Convert state id to state idcs'''
    return np.argwhere(s == states)[0]

def get_a_opp(a):
    
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
        
    num = np.exp(num - np.nanmax(num))
    den = np.nansum(num)
    pol_vals = num/den
    
    pol_vals[np.isnan(pol_vals)] = 0
    
    if pol_vals.sum() == 0:
        return [1/len(q_vals)]*len(q_vals)
    
    return pol_vals/pol_vals.sum()

def policy_choose_moves(Q:list, temp:list, biases):
    '''Choose an action'''
    q_vals1 = Q[0].copy()
    temp1   = temp[0]
    
    q_vals2 = Q[1].copy()
    temp2   = temp[1]
    
    num = q_vals1*temp1 + q_vals2*temp2 + biases
    
    num = np.exp(num - np.nanmax(num))
    den = np.nansum(num)
    pol_vals = num/den
    
    pol_vals[np.isnan(pol_vals)] = 0
    
    if pol_vals.sum() == 0:
        return [1/len(q_vals1)]*len(q_vals1)
    
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
        probs_pre = policy_choose_moves([q_vals1, q_vals2], [beta1, beta2], biases)
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
            
        probs_post = policy_choose_moves([q_vals1_after, q_vals2], [beta1, beta2], biases)
        
        # Calculate gain
        EV_pre = np.sum(probs_pre*q_vals1_after)
        EV_post = np.sum(probs_post*q_vals1_after)
        gain[i] = (EV_post - EV_pre)
        
    return gain

def compute_need(s, T_matrix, plan_exp, gamma):
    '''Compute need term for each experience'''
    T = T_matrix.copy()
    num_exp = plan_exp.shape[0]
    need    = np.zeros(num_exp)
    
    # Calculate the Successor Representation
    SR = np.linalg.inv(np.identity(T.shape[0]) - gamma*T)
    SRi = SR[s, :]
    SR_or_SD = SRi
    for i in range(num_exp):
        curr_exp = plan_exp[i]
        s = int(curr_exp[0])
        need[i] = (SR_or_SD[s])
    return need
                        
