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

def sample_params():
    
    p = {}
    p['beta']             = np.random.gamma(1, 1)
    p['beta2']            = np.random.gamma(1, 1)
    p['beta1']            = np.random.gamma(1, 1)
    p['alpha2']           = np.random.beta(1, 1)
    p['alpha1']           = np.random.beta(1, 1)
    p['Q_init']           = np.random.normal(0, 1)
    p['T_learn_rate']     = np.random.beta(1, 1)
    p['opp_T_learn_rate'] = np.random.beta(1, 1)
    p['rho']              = np.random.beta(1, 1)
    p['tau']              = np.random.beta(1, 1)
    
    a = np.random.normal(0, 1)
    b = np.random.normal(0, 1)
    c = 0
    d = -(a + b)    

    p['biases']           = [a, b, c, d] 
    p['tau_forget_block'] = np.random.beta(1, 1)
    p['T_forget_block']   = np.random.beta(1, 1)
    p['evm_threshold']    = np.random.gamma(1, 0.01)
    p['num_trials'] = None
    return p

def replay_1move(Q2, replay_exp, evm_thresh, alpha, beta):
    replay_gain     = np.empty((0, 8*4))
    replay_backups  = np.empty((0, 4))
    Q_history       = Q2.flatten()
    
    while True:
        # Gain & Need
        gain = compute_gain(Q2, replay_exp, alpha, beta)
                
        # Expected value of each memory
        evm = 1/8 * gain
        
        max_evm = evm.max()
        # if replay_counter == 0:
        #     evm_thresh = np.percentile(evm, sigmoid(H))
        
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
            
            Q_target = rr
            
            delta = Q_target - Q2[sr, ar]
            Q2[sr, ar] = Q2[sr, ar] + alpha * delta
            
            replay_gain    = np.vstack((replay_gain, gain))
            replay_backups = np.vstack((replay_backups, [sr, ar, rr, s1r]))
            Q_history      = np.vstack((Q_history, Q2.flatten()))
            
        else:
            return Q2, replay_gain, replay_backups, Q_history

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
        num = np.exp(q_vals*temp + biases)
    else:
        num = np.exp(q_vals*temp)
        
    den = np.sum(num)
    pol_vals = num/den
    
    return pol_vals

def policy_choose_moves(Q:list, temp:list, biases):
    '''Choose an action'''
    q_vals1 = Q[0].copy()
    temp1   = temp[0]

    q_vals2 = Q[1].copy()
    temp2   = temp[1]

    num = q_vals1*temp1 + q_vals2*temp2 + biases

    num -= np.max(num)
    if np.isnan(num).any():
        return [1/len(num)]*len(num)
    check = np.any(num < 0)
    if check:
        num = np.exp(num)
        den = np.sum(num)
        pol_vals = num/den
    else:
        den = np.log(np.sum(np.exp(num)))
        pol_vals = np.exp(num-den)

    return pol_vals

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
        
        # Next state value
        r = curr_exp[2]
        # Q_target = r

        q_vals_after = q_vals.copy()
        delta = r - q_vals_after[a_taken]
        q_vals_after[a_taken] = q_vals_after[a_taken] + alpha*delta
        probs_post = policy_choose(q_vals_after, temp)
        
        # Calculate gain
        EV_pre = np.sum(probs_pre*q_vals_after)
        EV_post = np.sum(probs_post*q_vals_after)
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
                        
