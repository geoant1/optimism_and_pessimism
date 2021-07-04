import numpy as np
import scipy
from numba import jit

def sample_prior():
    
    p = [np.random.gamma(1, 1),
         np.random.gamma(1, 1),
         np.random.gamma(1, 1),
         np.random.uniform(0, 1),
         np.random.uniform(0, 1),
         np.random.normal(0, 1),
         np.random.beta(6, 2),
         np.random.beta(6, 2),
         np.random.beta(6, 2),
         np.random.beta(6, 2),
         np.random.normal(0, 1),
         np.random.normal(0, 1),
         np.random.uniform(0, 1),
         np.random.uniform(0, 1),
         np.random.uniform(0, 1),
         scipy.stats.loggamma.rvs(0.02, loc=-1, scale=1/100, size=1)[0],
         np.random.uniform(0, 1),
         np.random.uniform(0, 1)]
    
    return p

def perturb(theta, sig):
    
    c0 = [0, 1, 2]
    c1 = [3, 4, 6, 7, 8, 9, 12, 13, 14, 16, 17]
    c2 = 15
    p = np.zeros(18)
    for i in range(18):
        while True:
            p[i] = theta[i] + np.random.normal(0, sig)
            if i in c0:
                if p[i] < 0:
                    continue
                else: break
            elif i in c1:
                if p[i] < 0 or p[i] > 1:
                    continue
                else: break
            elif i == c2:
                if p[i] >= 0:
                    continue
                else: break
            else: break
    return p

def backup(out_file, it, thresh, params):
    if it == 0:
        f = open(out_file, 'w')
        for np in range(len(params)):
            f.write("param#%s \t "%np)
        f.write("dist \t")
        f.write('It \n')
        for p in range(len(params)):
            if p == 15:
                f.write("%.8f \t" % 10**params[p])
            else:
                f.write("%.8f \t" % params[p])
        f.write('%.4f \t'%thresh)
        f.write('%u \n'%it)
    else:
        f = open(out_file, 'a')
        for p in range(len(params)):
            if p == 15:
                f.write("%.8f \t" % 10**params[p])
            else:
                f.write("%.8f \t" % params[p])
        f.write('%.4f \t'%thresh)
        f.write('%u \n'%it)
    f.flush()
    f.close()
    
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
    p['xi']               = 10**param[15]
    p['alpha2r']          = param[16]
    p['alpha1r']          = param[17]
    return p

@jit(nopython=True)
def replay_1move(Q2, T2, world, beta, alpha2, xi):
    R_vec  = world.flatten()
    # Generate replays from the model
    replay_exp = np.empty((32, 4))
    for sr in range(8):
        for ar in range(4):
            this_action_probas = T2[sr, ar, :]
            s1r  = np.argmax(this_action_probas)
            rr = np.sum(R_vec*this_action_probas)
            
            replay_exp[sr*4 +ar, :] = np.array([sr, ar, rr, s1r])
                    
    while True:
        # Gain & Need
        gain = compute_gain(Q2, replay_exp, alpha2, beta)
                
        # Expected value of each memory
        evm = 1/8 * gain                
        max_evm = evm.max()

        # if the value is above threshold
        if max_evm > xi:
            max_evm_idx = np.where(evm == max_evm)[0]
            max_evm_idx = max_evm_idx[-1]
            
            # Retrieve information from this experience
            curr_path = replay_exp[max_evm_idx, :]
            sr        = int(curr_path[0])
            ar        = int(curr_path[1])
            rr        = curr_path[2]
            
            Q_target = rr
            
            delta = Q_target - Q2[sr, ar]
            Q2[sr, ar] = Q2[sr, ar] + alpha2 * delta
            
        else:
            return Q2
                    

@jit(nopython=True)
def replay_2moves(Q1, Q2, T2, world, xi, beta2, beta1, alpha2, alpha1):
    
    replay_exp2 = np.empty((32, 4), dtype=np.float64)
    for sr in range(8):
        for ar in range(4):
            this_action_probas = T2[sr, ar, :]
            s1r  = np.argmax(this_action_probas)
            rr = np.sum(world.ravel()*this_action_probas)
            
            replay_exp2[sr*4 +ar, :] = np.array([sr, ar, rr, s1r])
            
    # Generate 1st move experiences
    while True:
        replay_exp1 = np.empty((8*16, 4), dtype=np.float64)  
        for sr in range(8):
            for ar in range(4):
                s1r = np.argmax(T2[sr, ar, :])
                for ar2 in range(4):
                    rr = 0
                    for s2r in range(8):
                        
                        q_vals2 = Q2[s2r, :]
                        probs  = policy_choose(q_vals2, beta2)
                        
                        these_action_probas     = T2[s2r, ar2, :].copy()
                        these_action_probas[sr] = 0
                        these_action_probas = these_action_probas / np.sum(these_action_probas)                      
                        rr += T2[sr, ar, s2r]*(world.ravel()[s2r] + probs[ar2]*np.sum(these_action_probas * world.ravel()))
                        
                    replay_exp1[sr*16 + (ar*4 +ar2), :] = np.array([sr, ar*4+ar2, rr, s1r*8+s2r], dtype=np.float64)
                    
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
                # s1_val   = np.amax(Q2[s1r, :])
                # Q_target = rr + gamma * s1_val
                Q_target = rr
                delta = Q_target - Q1[sr, ar]
                Q1[sr, ar] = Q1[sr, ar] + alpha1 * delta
                
            else:
                Q_target = rr 
                delta = Q_target - Q2[sr, ar]
                Q2[sr, ar] = Q2[sr, ar] + alpha2 * delta
        else:
            return Q1, Q2
        
@jit(nopython=True)
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
        return np.repeat(1/len(q_vals), len(q_vals))
    
    return pol_vals/pol_vals.sum()

@jit(nopython=True)
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

@jit(nopython=True)
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

@jit(nopython=True)
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
            
        probs_post = policy_choose_moves([q_vals1_after, q_vals2], [beta1, beta2], biases)
        
        # Calculate gain
        EV_pre = np.sum(probs_pre*q_vals1_after)
        EV_post = np.sum(probs_post*q_vals1_after)
        gain[i] = (EV_post - EV_pre)
        
    return gain