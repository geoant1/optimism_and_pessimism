import numpy as np
from numba import jit
import time

# Supplementary funcitons 
def tic():
    return time.time()

def toc(a):
    return time.time() - a

def get_new_state(si, a, world):
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

    r = world[s1i[0], s1i[1]]
    
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

    p['biases']           = np.array([a, b, c, d])
    p['tau_forget_block'] = np.random.beta(1, 1)
    p['T_forget_block']   = np.random.beta(1, 1)
    p['evm_threshold']    = np.random.gamma(1, 0.01)
    p['num_trials']       = None
    p['Block_forget']     = np.random.beta(1, 1)
    return p

def idcs2state(idcs: list, maze):
    '''Convert state idcs to state id'''
    si = idcs[0]
    sj = idcs[1]
    side_j = maze.shape[1]
    
    return si*side_j + sj

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

def state2idcs(s: int, maze):
    '''Convert state id to state idcs'''
    side_j = maze.shape[1]
    si = s // side_j
    sj = s % side_j
    return [si, sj]
    
def policy_choose(q_values, temp, biases):
    '''Choose an action'''
    q_vals = q_values.copy()
    num = np.exp(q_vals*temp + biases)
    den = np.sum(num)
    pol_vals = num/den
    
    return pol_vals

def policy_choose_moves(q_values:list, temp:list, biases):
    '''Choose an action'''
    q_vals1 = q_values[0].copy()
    temp1   = temp[0]
    
    q_vals2 = q_values[1].copy()
    temp2   = temp[1]
    
    if len(q_values) == 3:
        q_vals3 = q_values[2].copy()
        temp3   = q_values[2].copy()
        num = q_vals1*temp1 + q_vals2*temp2 + q_vals3*temp3 + biases
    else:
        num = q_vals1*temp1 + q_vals2*temp2 + biases
    
    num -= np.max(num)
    check = np.any(num < 0)
    if check:
        num = np.exp(num)
        den = np.sum(num)
        pol_vals = num/den
    else:
        den = np.log(np.sum(np.exp(num)))
        pol_vals = np.exp(num-den)
        
    return pol_vals

def compute_gain(Q_table, plan_exp, alpha, temp, biases):
    '''Compute gain term for each experience'''
    Q  = Q_table.copy()
    
    num_exp = plan_exp.shape[0]
    gain    = np.zeros(num_exp)

    for i in range(num_exp):
        curr_exp = plan_exp[i, :]
        s = int(curr_exp[0])
        q_vals = Q[s, :]

        # Policy before backup
        probs_pre = policy_choose(q_vals, temp, biases)
        a_taken   = int(curr_exp[1])
        
        # Next state value
        r = curr_exp[2]
        # Q_target = r

        q_vals_after = q_vals.copy()
        delta = r - q_vals_after[a_taken]
        q_vals_after[a_taken] = q_vals_after[a_taken] + alpha*delta
        probs_post = policy_choose(q_vals_after, temp, biases)
        
        # Calculate gain
        EV_pre = np.sum(probs_pre*q_vals_after)
        EV_post = np.sum(probs_post*q_vals_after)
        gain[i] = (EV_post - EV_pre)
        
    return gain

def compute_gain_first_move(Q_list: list, plan_exp, gamma, alpha, temp, biases):
    '''Compute gain term for each experience'''
    Q_move1  = Q_list[0].copy()
    Q_move2  = Q_list[1].copy()
    
    num_exp = plan_exp.shape[0]
    gain    = np.zeros(num_exp)

    for i in range(num_exp):
        curr_exp = plan_exp[i, :]
        s = int(curr_exp[0])
        q_vals = Q_move1[s, :]
        
        # Policy before backup
        probs_pre = policy_choose(q_vals, temp, biases)
        a_taken   = int(curr_exp[1])
        
        # Next state value
        r = curr_exp[2]
        s1 = int(curr_exp[3])
        Q_target = r# + gamma * np.amax(Q_move2[s1, :])

        delta = Q_target-q_vals[a_taken]
        q_vals[a_taken] += alpha*delta
        
        probs_post = policy_choose(q_vals, temp, biases)
        
        # Calculate gain
        EV_pre = np.sum(probs_pre*q_vals)
        EV_post = np.sum(probs_post*q_vals)
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
                        
