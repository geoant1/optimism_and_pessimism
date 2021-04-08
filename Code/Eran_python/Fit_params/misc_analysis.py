import numpy as np
import os, glob
from misc import policy_choose, policy_choose_moves, get_new_state, state2idcs, idcs2state, get_a_opp

root_path = '/Users/GA/Documents/Dayan_lab/Optimism_And_Pessimism_In_Optimised_Replay/'

# Worlds & idcs
world1 = np.load(os.path.join(root_path, 'Data/Eran/world1.npy'))
world2 = np.load(os.path.join(root_path, 'Data/Eran/world2.npy'))
idcs1  = np.load(os.path.join(root_path, 'Data/Eran/idcs1.npy'))
idcs2  = np.load(os.path.join(root_path, 'Data/Eran/idcs2.npy'))

def get_sorted_files(path):
    
    os.chdir(path)
    all_files = glob.glob('*.npz')
    # Sort files by episode number
    episodes = [int(''.join(c for c in f if c.isdigit())) for f in all_files]
    idcs_ep  = [i[0] for i in sorted(enumerate(episodes), key=lambda x:x[1])]
    all_files_sorted = [all_files[i] for i in idcs_ep] 
    
    return all_files_sorted

def get_entropy(T, s, a):
    
    out = 0
    for i in np.delete(range(8), s):
        out -= T[s, a, i]*np.log2(T[s, a, i])
    return out

def get_joint_entropy(T, s1, a1, a2):
    
    out = 0
    for i in np.delete(range(8), s1):
        
        probas = T[i, a2, :].copy()
        probas[s1] = 0
        probas = probas / np.sum(probas)
        for j in np.delete(range(8), [s1, i]):
            out -= T[s1, a1, i]*probas[j]*np.log2(T[s1, a1, i]*probas[j])
            
    return out
            
def policy_improve(q_before, q_after, Q_true, s, beta, biases, world, idcs, mode='value', benefit='objective'):
    
    probs_before = policy_choose(q_before, beta)
    probs_after  = policy_choose(q_after, beta)
    
    if mode != 'value':
        a_opt = get_optimal_move(s, Q_true)

        opt = 0
        for a in a_opt:
            diff = probs_after[a] - probs_before[a]
            opt += diff
            
        out = opt*100
    
    else:
        if benefit == 'objective':
            r = np.empty(0)
            si = state2idcs(s, idcs)
            for a in range(4):
                _, ri = get_new_state(si, a, world, idcs)
                r = np.append(r, ri)
            out = np.sum(r*probs_after) - np.sum(r*probs_before)
        else:
            out = np.sum(q_after*probs_after) - np.sum(q_after*probs_before)
    return out

def policy_improve_2moves(q1_before, q1_after, q2_before, q2_after, Q1_true, Q2_true, s, beta1, beta2, biases, world, idcs, mode='value', benefit='objective'):
    
    # q_vals1b = np.zeros(4)
    # for a in range(4):
    #     tmp     = q1_before[(a*4):(a*4+4)].copy()
    #     probs   = policy_choose(tmp, beta1, biases)
    #     q_vals1b[a] = np.nansum(tmp*probs)
    # q_vals2b = q2_before[s, :].copy()

    # probs_before1 = policy_choose_moves([q_vals1b, q_vals2b], [beta1, beta2], biases)
    
    # q_vals1a = np.zeros(4)
    # for a in range(4):
    #     tmp     = q1_after[(a*4):(a*4+4)].copy()
    #     probs   = policy_choose(tmp, beta1, biases)
    #     q_vals1a[a] = np.nansum(tmp*probs)
    # q_vals2a = q2_after[s, :].copy()

    # probs_after1 = policy_choose_moves([q_vals1a, q_vals2a], [beta1, beta2], biases)
    
    # si = state2idcs(s, idcs)
                
    # if mode != 'value':
    #     a1_opt = get_optimal_move(s, Q1_true)//4
    #     a2_opt = get_optimal_move(s, Q1_true)%4

    #     opt = 0
    #     for a1 in a1_opt:
    #         s1i, _ = get_new_state(si, a1, world, idcs)
    #         s1     = idcs2state(s1i, idcs)
    #         for a2 in a2_opt:
    #             probs_before2 = policy_choose_moves([q1_before[(a1*4):(a1*4+4)], q2_before[s1, :]], [beta1, beta2], biases)
    #             probs_after2  = policy_choose_moves([q1_after[(a1*4):(a1*4+4)], q2_after[s1, :]], [beta1, beta2], biases)
                
    #             diff = probs_after1[a1]*probs_after2[a2] - probs_before1[a1]*probs_before2[a2]
    #             opt += diff
        
    #     out = opt*100
    
    # else:
    #     if benefit == 'objective':
    #         si = state2idcs(s, idcs)
    #         tmpb = []
    #         tmpa = []
    #         for a in range(4):
    #             s1i, _        = get_new_state(si, a, world, idcs)
    #             s1            = idcs2state(s1i, idcs)
    #             probs_before2 = policy_choose_moves([q1_before[(a*4):(a*4+4)], q2_before[s1, :]], [beta1, beta2], biases)
    #             tmpb.append(np.nansum(probs_before2*Q1_true[s, (a*4):(a*4+4)]))
            
    #             probs_after2 = policy_choose_moves([q1_after[(a*4):(a*4+4)], q2_after[s1, :]], [beta1, beta2], biases)
    #             tmpa.append(np.nansum(probs_after2*Q1_true[s, (a*4):(a*4+4)]))
                
    #         out = np.sum(probs_after1*tmpa) - np.sum(probs_before1*tmpb)
            
    #     else:
    #         si = state2idcs(s, idcs)
    #         tmpb = []
    #         tmpa = []
    #         for a in range(4):
    #             s1i, _        = get_new_state(si, a, world, idcs)
    #             s1            = idcs2state(s1i, idcs)
    #             probs_before2 = policy_choose_moves([q1_before[(a*4):(a*4+4)], q2_before[s1, :]], [beta1, beta2], biases)
    #             tmpb.append(np.nansum(probs_before2*q1_after[(a*4):(a*4+4)]))
            
    #             probs_after2 = policy_choose_moves([q1_after[(a*4):(a*4+4)], q2_after[s1, :]], [beta1, beta2], biases)
    #             tmpa.append(np.nansum(probs_after2*q1_after[(a*4):(a*4+4)]))
                
    #         out = np.sum(probs_after1*tmpa) - np.sum(probs_before1*tmpb)
    probs_before = policy_choose(q1_before, beta1)
    probs_after  = policy_choose(q1_after, beta1)
    
    if mode != 'value':
        
        a_opt = get_optimal_move(s, Q1_true)
        opt = 0
        
        for a in a_opt:
                
                diff = probs_after[a] - probs_before[a]
                opt += diff
        
        out = opt*100
    else:
        if benefit == 'objective':
             
            out = np.nansum(probs_after*Q1_true[s, :]) - np.nansum(probs_before*Q1_true[s, :])   
            
        else:
            
            out = np.nansum(probs_after*q1_after) - np.nansum(probs_before*q1_after)
    
    return out

def histogram(arr, bins=None):

    arr  = np.array(arr)
    arr  = arr[~np.isnan(arr)]
    if bins is not None:
        h, b = np.histogram(arr, bins=bins)
    else:
        try:
            h, b = np.histogram(arr, bins=range(0,int(np.max(arr)+2)))
        except:
            h, b = np.histogram(arr)
    h    = h/np.sum(h)
    w    = b[1] - b[0]
    
    return h, b[:-1], w

def run_permutation_test(pooled, nx, ny, rep):

    shuffled = np.random.permutation(pooled)
    shuffX   = np.random.choice(shuffled, size=nx, replace=rep)
    shuffY   = np.random.choice(shuffled, size=ny, replace=rep)
    
    return np.mean(shuffX) - np.mean(shuffY)
    
def permutation_test(X, Y, num_perm, replace=False, return_tests=False):
    
    X = np.array(X)
    X = X[~np.isnan(X)]

    Y = np.array(Y)
    Y = Y[~np.isnan(Y)]

    pooled = np.hstack([X,Y])
    
    delta  =  np.mean(X) - np.mean(Y)
    
    tests  = []
    for i in range(num_perm):
        tests += [run_permutation_test(pooled,len(X),len(Y), rep=replace)]
        
    diff_count = len(np.where(tests >= delta)[0])
    p_value    = 1.0 - (float(diff_count)/float(num_perm))
    
    if return_tests:
        return tests, delta, p_value
    else:
        return delta, p_value

def get_optimal_move(s, Q_true, a=None):
    
    if a is None:
        a_opt  = np.argwhere(Q_true[s, :] == np.nanmax(Q_true[s, :])).flatten()
    else:
        a_opt  = np.argwhere(Q_true[s, a*4:a*4+4] == np.nanmax(Q_true[s, a*4:a*4+4])).flatten()
                        
    return a_opt

def get_Q_true(world, state_arr):
    
    idcs = []
    for ai in [0, 1]:
        for aj in [0, 1]:
            idcs.append(ai*4+aj)
    for ai in [2, 3]:
        for aj in [2, 3]:
            ai_opp = get_a_opp(ai)
            if aj == ai_opp:
                idcs.append(ai*4+aj)
                    
    Q1_true = np.zeros((8,16))
    Q2_true = np.zeros((8,4))
    for s in range(8):
        si = state2idcs(s, state_arr)
        for a in range(4):
            s1i, r = get_new_state(si, a, world, state_arr)
            s1 = idcs2state(s1i, state_arr)
            Q2_true[s, a] = r

            for a2 in range(4):
                s2i, r2 = get_new_state(s1i, a2, world, state_arr)
                Q1_true[s, a*4+a2] = r + r2
        Q1_true[:, idcs] = np.nan
        
    return Q1_true, Q2_true


def get_obtained_reward(sub_task_folder):
    
    rwd_obt = []
    for i in range(5):
        if i == 0:
            this_range = 6
        else:
            this_range = 7
        tmp = []
        for j in range(this_range):
            # Prepare to load files
            epoch_folder = os.path.join(sub_task_folder, str(i), str(j))
            all_files_sorted = get_sorted_files(epoch_folder)
            
            for f in range(len(all_files_sorted)):
                this_file      = all_files_sorted[f]
                data      = np.load(os.path.join(epoch_folder, this_file), allow_pickle=True)
                move = data['move']
                if len(move) != 2:
                    r = move[2]
                    tmp.append(r)
                else:
                    move1 = move[0]
                    move2 = move[1]
                    tmp.append(move1[2]+move2[2])
        rwd_obt.append(tmp)
    
    return rwd_obt

def analyse_recent_replays(sub_task_folder):
    
    H_opt_single    = []
    H_opt_paired    = []
    H_subopt_single = []
    H_subopt_paired = []
    opt             = []
    subopt          = []
    
    for i in range(5):
        if i == 0:
            this_range = 6
        else:
            this_range = 7
            
        if i in [0, 1]:
            Q1_true, Q2_true = get_Q_true(world1, idcs1)
        elif i in [2, 3]:
            Q1_true, Q2_true = get_Q_true(world2, idcs1)
        else:
            Q1_true, Q2_true = get_Q_true(world2, idcs2)
        
        for j in range(this_range):
            if i > 0 and j < 2:
                pass
            else:
                # Prepare to load files
                epoch_folder = os.path.join(sub_task_folder, str(i), str(j))
                all_files_sorted = get_sorted_files(epoch_folder)

                for f in range(len(all_files_sorted)):

                    this_file = all_files_sorted[f]
                    data      = np.load(os.path.join(epoch_folder, this_file), allow_pickle=True)
                    move      = data['move']
                    T         = data['T']
                    replay_backups = np.atleast_2d(data['replay_backups'])

                    if len(move) == 2:

                        s1 = move[0][0]
                        a1 = move[0][1]

                        s2 = move[1][0]
                        a2 = move[1][1]

                        a  = a1*4 + a2

                        paired_move_optimal        = a  in get_optimal_move(s1, Q1_true)
                        second_single_move_optimal = a2 in get_optimal_move(s2, Q2_true)

                        # up/down replays correspond to the same 'experience' – pool these together
                        ja  = [a1*4 + a2]
                        ja1 = [a1]
                        ja2 = [a2]
                        if a1 in [0, 1]:
                            ja  = [a1*4 + a2, (1-a1)*4 + a2] # up (0) /down (1) + second move
                            ja1 = [a1, (1-a1)]
                        if a2 in [0, 1]:
                            ja  = [a1*4 + a2, a1*4+ (1-a2)] # first move + up (0) /down (1)
                            ja2 = [a2, (1-a2)] # up (0) /down (1) in the second move
                        
                        if replay_backups.shape[0] > 1:
                            
                            idcs_paired_similar = []
                            idcs_single_similar = []
                            
                            # Find replays of the paired move
                            s_cond      = replay_backups[:, 0] == s1
                            a_cond      = (replay_backups[:, 1] == ja[0])
                            p_cond      = (replay_backups[:, 4] == 0)
                            idcs_paired = np.argwhere(s_cond & a_cond & p_cond).flatten()
                            if len(ja) > 1:
                                a_cond_similar      = (replay_backups[:, 1] == ja[1])
                                idcs_paired_similar = np.argwhere(s_cond & a_cond_similar & p_cond).flatten()
                            
                            # Find replays of the second move
                            # NB replays of the first move are not included since optimal first moves in 
                            # 2-move trials are usually not the same as optimal moves in 1-move trials
                            s_cond = replay_backups[:, 0] == s2
                            a_cond = (replay_backups[:, 1] == ja2[0])
                            p_cond = (replay_backups[:, 4] == 1)
                            idcs_single = np.argwhere(s_cond & a_cond & p_cond).flatten()
                            if len(ja2) > 1:
                                a_cond_similar      = (replay_backups[:, 1] == ja2[1])
                                idcs_single_similar = np.argwhere(s_cond & a_cond_similar & p_cond).flatten()

                            num_paired         = len(idcs_paired)
                            num_paired_similar = len(idcs_paired_similar)
                            num_single         = len(idcs_single)
                            num_single_similar = len(idcs_single_similar)
                            
                            # If the sequence was optimal, then both the sequence and the 
                            # second move were optimal
                            if paired_move_optimal:

                                opt += [num_paired + num_single + num_paired_similar + num_single_similar]

                                if num_paired > 0:
                                    H_opt_paired += [get_joint_entropy(T, s1, ja[0]//4, ja[0]%4)]*num_paired
                                if num_paired_similar > 0:
                                    H_opt_paired += [get_joint_entropy(T, s1, ja[1]//4, ja[1]%4)]*num_paired_similar

                                if num_single > 0:
                                    H_opt_single += [get_entropy(T, s2, ja2[0])]*num_single
                                if num_single_similar > 0:
                                    H_opt_single += [get_entropy(T, s2, ja2[1])]*num_single_similar

                            # If the sequence was sub-optimal that could be due to the 
                            # first move. The second move can still be optimal
                            if second_single_move_optimal and not paired_move_optimal:

                                # Second move thus optimal
                                opt    += [num_single + num_single_similar]
                                if num_single > 0:
                                    H_opt_single += [get_entropy(T, s2, ja2[0])]*num_single
                                if num_single_similar > 0:
                                    H_opt_single += [get_entropy(T, s2, ja2[1])]*num_single_similar

                                # Sequence sub-optimal
                                subopt += [num_paired + num_paired_similar]
                                if num_paired > 0:
                                    H_subopt_paired += [get_joint_entropy(T, s1, ja[0]//4, ja[0]%4)]*num_paired
                                if num_paired_similar > 0:
                                    H_subopt_paired += [get_joint_entropy(T, s1, ja[1]//4, ja[1]%4)]*num_paired_similar

                            # If the subject screwed up everything
                            if not second_single_move_optimal and not paired_move_optimal:

                                subopt += [num_paired + num_single + num_paired_similar + num_single_similar]

                                if num_paired > 0:
                                    H_subopt_paired += [get_joint_entropy(T, s1, ja[0]//4, ja[0]%4)]*num_paired
                                if num_paired_similar > 0:
                                    H_subopt_paired += [get_joint_entropy(T, s1, ja[1]//4, ja[1]%4)]*num_paired_similar

                                if num_single > 0:
                                    H_subopt_single += [get_entropy(T, s2, ja2[0])]*num_single
                                if num_single_similar > 0:
                                    H_subopt_single += [get_entropy(T, s2, ja2[1])]*num_single_similar

                        else:
                            if paired_move_optimal:
                                opt    += [0]
                            else:
                                subopt += [0]

                    else:
                        s2 = move[0]
                        a2 = move[1]
                        single_move_optimal = a2 in get_optimal_move(s2, Q2_true)

                        ja2 = [a2]
                        if a2 in [0, 1]:
                            ja2 = [a2, (1-a2)]
                            
                        if replay_backups.shape[0] > 1:
                            
                            idcs_single_similar = []
                            
                            s_cond      = replay_backups[:, 0] == s2
                            a_cond      = replay_backups[:, 1] == ja2[0]
                            idcs_single = np.argwhere(s_cond & a_cond).flatten()
                            if len(ja2) > 1:
                                a_cond_similar      = (replay_backups[:, 1] == ja2[1])
                                idcs_single_similar = np.argwhere(s_cond & a_cond_similar).flatten()

                            num_replays = len(idcs_single)
                            num_replays_similar = len(idcs_single_similar)

                            if single_move_optimal:

                                opt   += [num_replays + num_replays_similar]
                                if num_replays > 0:
                                    H_opt_single += [get_entropy(T, s2, ja2[0])]*num_replays
                                if num_replays_similar > 0:
                                    H_opt_single += [get_entropy(T, s2, ja2[1])]*num_replays_similar
                                    
                            else:
                                subopt   += [num_replays + num_replays_similar]
                                if num_replays > 0:
                                    H_subopt_single += [get_entropy(T, s2, ja2[0])]*num_replays
                                if num_replays_similar > 0:
                                    H_subopt_single += [get_entropy(T, s2, ja2[1])]*num_replays_similar
                                    
    return opt, subopt, H_opt_single, H_subopt_single, H_opt_paired, H_subopt_paired

def analyse_other_replays(sub_task_folder):
    
    H_opt_single    = []
    H_opt_paired    = []
    H_subopt_single = []
    H_subopt_paired = []
    opt             = []
    subopt          = []
    
    for i in range(5):
        if i == 0:
            this_range = 6
        else:
            this_range = 7
            
        if i in [0, 1]:
            Q1_true, Q2_true = get_Q_true(world1, idcs1)
        elif i in [2, 3]:
            Q1_true, Q2_true = get_Q_true(world2, idcs1)
        else:
            Q1_true, Q2_true = get_Q_true(world2, idcs2)
        
        for j in range(this_range):
            if i > 0 and j < 2:
                pass
            else:
                # Prepare to load files
                epoch_folder = os.path.join(sub_task_folder, str(i), str(j))
                all_files_sorted = get_sorted_files(epoch_folder)

                for f in range(len(all_files_sorted)):

                    tmp_opt = 0
                    tmp_subopt = 0

                    this_file = all_files_sorted[f]
                    data      = np.load(os.path.join(epoch_folder, this_file), allow_pickle=True)
                    move      = data['move']
                    T         = data['T']

                    replay_backups = np.atleast_2d(data['replay_backups'])
                    if replay_backups.shape[0] > 1:

                        if len(move) == 2:

                            s1 = move[0][0]
                            a1 = move[0][1]

                            s2 = move[1][0]
                            a2 = move[1][1]

                            a  = a1*4 + a2

                            # Ditto
                            ja  = [a1*4 + a2]
                            ja1 = [a1]
                            ja2 = [a2]
                            if a1 in [0, 1]:
                                ja  = [a1*4 + a2, (1-a1)*4 + a2] # up (0) /down (1) + second move
                                ja1 = [a1, (1-a1)]
                            if a2 in [0, 1]:
                                ja  = [a1*4 + a2, a1*4+ (1-a2)] # first move + up (0) /down (1)
                                ja2 = [a2, (1-a2)] # up (0) /down (1) in the second move
                            
                            for rid in range(replay_backups.shape[0]):
                                
                                this_replay = replay_backups[rid, :]
                                sr = int(this_replay[0])
                                ar = int(this_replay[1])
                                pl = int(this_replay[4])

                                if sr != -200 and ar != -200:

                                    if pl == 0:
                                        if sr != s1 and ar not in ja:
                                            paired_move_optimal = ar in get_optimal_move(sr, Q1_true)
                                            if paired_move_optimal:
                                                H_opt_paired += [get_joint_entropy(T, sr, ar//4, ar%4)]
                                                tmp_opt += 1
                                            else:
                                                H_subopt_paired += [get_joint_entropy(T, sr, ar//4, ar%4)]
                                                tmp_subopt += 1
                                    else:
                                        cond1 = sr != s1 and ar not in ja1
                                        cond2 = sr != s2 and ar not in ja2
                                        if cond1 and cond2:
                                            second_single_move_optimal = ar in get_optimal_move(sr, Q2_true)
                                            if second_single_move_optimal:
                                                H_opt_single += [get_entropy(T, sr, ar)]
                                                tmp_opt += 1
                                            else:
                                                H_subopt_single += [get_entropy(T, sr, ar)]
                                                tmp_subopt += 1
                        else:
                            s2 = move[0]
                            a2 = move[1]

                            ja2 = [a2]
                            if a2 in [0, 1]:
                                ja2 = [a2, (1-a2)]
                                
                            replay_backups = np.atleast_2d(data['replay_backups'])
                            if replay_backups.shape[0] > 1:

                                for rid in range(replay_backups.shape[0]):
                                    this_replay = replay_backups[rid, :]
                                    sr = int(this_replay[0])
                                    ar = int(this_replay[1])

                                    if sr != -200 and ar != -200:

                                        if sr != s2 and ar not in ja2:
                                            single_move_optimal = ar in get_optimal_move(sr, Q2_true)

                                            if single_move_optimal:
                                                H_opt_single += [get_entropy(T, sr, ar)]
                                                tmp_opt += 1
                                            else:
                                                H_subopt_single += [get_entropy(T, sr, ar)]
                                                tmp_subopt += 1
                    else:
                        tmp_opt     = 0
                        tmp_subopt  = 0

                    opt    += [tmp_opt]
                    subopt += [tmp_subopt]
    
    return opt, subopt, H_opt_single, H_subopt_single, H_opt_paired, H_subopt_paired

def get_replay_benefit(sub_task_folder, p, mode, ben):
        
    beta  = p[0]
    beta2 = p[1]
    beta1 = p[2]
    a     = p[10]
    b     = p[11]
    d     = -(a+b)
    biases = np.array([a, b, 0, d])
    
    for i in range(0, 5):
        if i == 0:
            this_range = 6
        else:
            this_range = 7
        
        if i in [0, 1]:
            world = world1
            idcs  = idcs1
        elif i in [2, 3]:
            world = world2
            idcs  = idcs1
        else:
            world = world2
            idcs  = idcs2
            
        Q1_true, Q2_true = get_Q_true(world, idcs)
        
        for j in range(this_range):
            if i > 0 and j < 2:
                pass
            else:
                # Prepare to load files
                epoch_folder = os.path.join(sub_task_folder, str(i), str(j))
                all_files_sorted = get_sorted_files(epoch_folder)

                for f in range(len(all_files_sorted)):

                    tmp_opt1, tmp_opt2 = [], []

                    this_file = all_files_sorted[f]
                    data      = np.load(os.path.join(epoch_folder, this_file), allow_pickle=True)
                    move      = data['move']

                    if len(move) == 2:

                        replay_backups = np.atleast_2d(data['replay_backups'])
                        if replay_backups.shape[0] > 1:

                            Q1 = data['Q1_history']
                            Q2 = data['Q2_history']

                            Q1_before = Q1[0, :].reshape(8, -1)
                            Q1_after  = Q1[-1, :].reshape(8, -1)

                            Q2_before = Q2[0, :].reshape(8, -1)
                            Q2_after  = Q2[-1, :].reshape(8, -1)

                            states1 = np.unique(replay_backups[np.argwhere(replay_backups[:, 4] == 0), 0])
                            states1 = [int(i) for i in states1]
                            
                            states2 = np.unique(replay_backups[np.argwhere(replay_backups[:, 4] == 1), 0])
                            states2 = [int(i) for i in states2]

                            if len(states1) > 0:
                                for s in states1:
                                    opt1 = policy_improve_2moves(Q1_before[s, :], Q1_after[s, :], Q2_before, Q2_after, Q1_true, Q2_true, s, beta1, beta2, biases, world, idcs, mode=mode, benefit=ben)
                                    tmp_opt1 += [opt1]
                            
                            if len(states2) > 0:
                                for s in states2:
                                    opt2 = policy_improve(Q2_before[s, :], Q2_after[s, :], Q2_true, s, beta, biases, world, idcs, mode=mode, benefit=ben)
                                    tmp_opt2  += [opt2]
                            
                            if len(tmp_opt1)>0 or len(tmp_opt1)>0:
                                opt_sub += [np.mean(tmp_opt2+tmp_opt1)]
                                    
                    else:
                        replay_backups = np.atleast_2d(data['replay_backups'])
                        if replay_backups.shape[0] > 1:

                            Q = data['Q_history']

                            Q_before = Q[0, :].reshape(8, -1)
                            Q_after  = Q[-1, :].reshape(8, -1)

                            states = np.unique(replay_backups[1:, 0])
                            states = [int(i) for i in states]

                            if len(states)>0:
                                for s in states:
                                    opt = policy_improve(Q_before[s, :], Q_after[s, :], Q2_true, s, beta, biases, world, idcs, mode=mode, benefit=ben)
                                    tmp_opt2    += [opt]
                                    
                                opt_sub += [np.mean(tmp_opt2)]
                                
    return opt_sub