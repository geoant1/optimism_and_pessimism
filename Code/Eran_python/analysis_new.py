import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style("white")
import os, shutil, glob
from tools import plot_history, plot_q_values_maintenance, plot_replay_frequency, plot_performance, plot_replay_vs_evm
from misc import state2idcs, idcs2state, get_a_opp, get_new_state


def analyse_1move(data_path, world):
    
    save_folder = os.path.join(data_path, 'Analysis')
    if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
    
    os.chdir(data_path)
    all_files = glob.glob('*.npz')
    
    # Sort files by episode number
    episodes = [int(''.join(c for c in f if c.isdigit())) for f in all_files]
    idcs     = [i[0] for i in sorted(enumerate(episodes), key=lambda x:x[1])]
    all_files_sorted = [all_files[i] for i in idcs] 
    
    Q_true = np.zeros((8,4))
    for s in range(8):
        si = state2idcs(s, world)
        for a in range(4):
            s1i, r = get_new_state(si, a, world)
            Q_true[s, a] = r
    
    # Pre-allocate variables
    # ----------------------------
    # --- Q-values maintenance ---
    Q_all  = np.empty((0, 32))
    
    # --- Replay frequency ---
    pos_TD = np.zeros(8*4)
    neg_TD = np.zeros(8*4)
    
    # --- Performance actual moves ---
    ylab_actual   = 'Prop of available reward collected'
    y_exp         = []
    act_r_history = []
    max_r_history = []
    
    # --- Replay stats ---
    # ylab_replay = 'Prop optimal moves replayed per trial'
    # y_replay    = [] 
    
    # --- Replay vs EVM ---
    these_evms = [[] for i in range(32)]
    
    # --- Replay vs Entropy ---
    H = [[] for i in range(32)]
    
    for f in range(len(all_files_sorted)):
        
        this_file = os.path.join(data_path, all_files_sorted[f])
        data      = np.load(this_file)
        
        this_move = data['move']
        s   = int(this_move[0])
        a   = int(this_move[1])
        r   = this_move[2]
        
        Q    = data['Q_history']
        evm  = data['replay_gain'] * 1/8
        T    = data['T']
        
        # --- Q-values maintenance ---
        # Append Q-values & average reward across all trials
        Q_all  = np.vstack((Q_all, Q))
        
        # --- Performance actual moves ---
        max_reward = np.amax(Q_true[s, :])

        act_r_history.append(r)
        max_r_history.append(max_reward)
        y_exp.append(np.sum(act_r_history)/np.sum(max_r_history))
        
        replay_backups = data['replay_backups']
        replay_history = []
        if replay_backups.shape[0] > 0:
            for i in range(replay_backups.shape[0]):
                
                sp  = int(replay_backups[i, 0])
                ap  = int(replay_backups[i, 1])
    
                # --- Replay stats ---
                # is_replay_move_optimal = int(ap in correct_moves[sp])  
                # replay_history.append([is_replay_move_optimal])
                
                this_Q  = Q[i].reshape(8, 4)
                
                if this_Q[sp, ap] <= Q_true[sp, ap]:
                    pos_TD[(sp*4 + ap)] += 1
                else:
                    neg_TD[(sp*4 + ap)] += 1
                
                # --- evm vs Replay ---
                tmp_evm = evm[i, :]
                this_evm = tmp_evm[(sp*4 + ap)]
                these_evms[(sp*4 + ap)].append(this_evm)
                
                # --- Replay vs Entropy
                tmp_entropy = 0
                for j in range(8):
                    if j == sp:
                        pass
                    else:
                        tmp_entropy -= T[sp, ap, j]*np.log2(T[sp, ap, j])
                H[(sp*4 + ap)].append(tmp_entropy)
                
            # y_replay.append(np.sum(replay_history)/len(replay_history))
            
    os.chdir(save_folder)
    # ---------------------------------
    # --- Plot Q-values maintenance ---
    this_save_folder = 'Q_values_maintenance'
    this_save_name   = 'Q_values_maintenance.png'
    this_title       = 'Q value steady-state maintenance'
    this_save_path   = os.path.join(this_save_folder, this_save_name)
    if not os.path.isdir(this_save_folder):
        os.mkdir(this_save_folder)
        
    plot_q_values_maintenance(np.mean(act_r_history), Q_all, Q_true, this_title, this_save_path)
    
    # ---------------------------------
    # --- Plot replay frequency ---
    this_save_folder = 'Replay_frequency'
    this_save_name   = 'Replay_frequency.png'
    this_title       = 'Replay frequency'
    this_save_path   = os.path.join(this_save_folder, this_save_name)
    if not os.path.isdir(this_save_folder):
        os.mkdir(this_save_folder)
    plot_replay_frequency(pos_TD, neg_TD, Q_true, this_title, this_save_path)
    
    # ---------------------------------
    # --- Plot performance ---
    this_save_name   = 'Learning_progress.png'
    this_title       = 'Learning progress'
    this_save_path   = this_save_name
    if not os.path.isdir(this_save_folder):
        os.mkdir(this_save_folder)
    plot_performance(y_exp, ylab_actual, this_title, this_save_path)
    
    # ---------------------------------
    # --- Plot replay stats ---
    # this_save_name   = 'Replay_stats.png'
    # this_title       = 'Replay stats'
    # this_save_path   = this_save_name
    # if not os.path.isdir(this_save_folder):
    #     os.mkdir(this_save_folder)
    # plot_performance(y_replay, ylab_replay, this_title, this_save_path)
    
    # ---------------------------------
    # --- Plot replay vs evm ---
    this_save_folder = 'evm_viol'
    this_save_name   = 'evm_viol.png'
    this_title       = 'evm vs replay density'
    this_save_path   = os.path.join(this_save_folder, this_save_name)
    if not os.path.isdir(this_save_folder):
        os.mkdir(this_save_folder)
    plot_replay_vs_evm(these_evms, 'evm', this_title, this_save_path)
    
    # ---------------------------------
    # --- Plot replay vs Entropy ---
    this_save_folder = 'Entropy_viol'
    this_save_name   = 'Entropy_viol.png'
    this_title       = 'Entropy vs replay density'
    this_save_path   = os.path.join(this_save_folder, this_save_name)
    if not os.path.isdir(this_save_folder):
        os.mkdir(this_save_folder)
    plot_replay_vs_evm(H, 'Entropy', this_title, this_save_path)
    
    return
    
def analyse_2moves(data_path, world):
    
    save_folder = os.path.join(data_path, 'Analysis')
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    
    os.chdir(data_path)
    all_files = glob.glob('*.npz')
    
    # Sort files by episode number
    episodes = [int(''.join(c for c in f if c.isdigit())) for f in all_files]
    idcs     = [i[0] for i in sorted(enumerate(episodes), key=lambda x:x[1])]
    all_files_sorted = [all_files[i] for i in idcs] 
    
    Q1_true = np.zeros((8,4))
    Q2_true = np.zeros_like(Q1_true)

    for s in range(8):
        si = state2idcs(s, world)
        for a in range(4):
            s1i, r = get_new_state(si, a, world)
            s1 = idcs2state(s1i, world)
            Q2_true[s, a] = r
            
            tmp = 0
            for a2 in range(4):
                s2i, r2 = get_new_state(s1i, a2, world)
                if (r + r2) > tmp:
                    tmp = (r + r2)
            Q1_true[s, a] = tmp
    
    Q_true_list = [Q1_true, Q2_true]
    
    for d in range(0, 2):
        
        Q_true = Q_true_list[d]
        # correct_moves = correct_moves_list[d-1]
            
        # Pre-allocate variables
        # ----------------------------
        # --- Q-values maintenance ---
        Q_all  = np.empty((0, 32))
        
        # --- Replay frequency ---
        pos_TD = np.zeros(8*4)
        neg_TD = np.zeros(8*4)
        
        # # --- Replay stats ---
        # ylab_replay = 'Prop optimal moves replayed per trial'
        # y_replay    = [] 
        
        # --- Replay vs evm ---
        these_evms = [[] for i in range(32)]
        
        # --- Replay vs Entropy ---
        H = [[] for i in range(32)]
        
        for f in range(len(all_files_sorted)):
            
            this_file = os.path.join(data_path, all_files_sorted[f])
            data      = np.load(this_file, allow_pickle=True)
            
            this_move = data['move'][d]
            s   = int(this_move[0])
            a   = int(this_move[1])
            
            Q   = data['Q_history'][d]
            evm = data['replay_gain'][d] * 1/8
            T   = data['T']
            planes = data['replay_planes']
            
            # --- Q-values maintenance ---
            # Append Q-values & average reward across all trials
            Q_all = np.vstack((Q_all, Q))
            if f == (len(all_files_sorted)-1):
                rew_his = data['rew_history']
                av_rew = np.mean(rew_his[d])
            
            replay_backups = data['replay_backups']
            replay_history = []
            if replay_backups.shape[0] > 0:
                idcs = np.where(planes == d)[0]
                if idcs.shape[0] > 0:
                    for i in idcs:
                        
                        sp  = int(replay_backups[i, 0])
                        ap  = int(replay_backups[i, 1])
            
                        # --- Replay stats ---
                        # is_replay_move_optimal = int(ap in correct_moves[sp])  
                        # replay_history.append([is_replay_move_optimal])
                        
                        this_Q  = Q[i].reshape(8, 4)
                        
                        if this_Q[sp, ap] <= Q_true[sp, ap]:
                            pos_TD[(sp*4 + ap)] += 1
                        else:
                            neg_TD[(sp*4 + ap)] += 1
                        
                        # --- evm vs Replay ---
                        tmp_evm = evm[i, :]
                        this_evm = tmp_evm[(sp*4 + ap)]
                        these_evms[(sp*4 + ap)].append(this_evm)
                        
                        # --- Replay vs Entropy
                        tmp_entropy = 0
                        for j in range(8):
                            if j == sp:
                                pass
                            else:
                                tmp_entropy -= T[sp, ap, j]*np.log2(T[sp, ap, j])
                        H[(sp*4 + ap)].append(tmp_entropy)
                    
                # y_replay.append(np.sum(replay_history)/len(replay_history))
                
        os.chdir(save_folder)
        # ---------------------------------
        # --- Plot Q-values maintenance ---
        this_save_folder = 'Q_values_maintenance_move%u'%d
        this_save_name   = 'Q_values_maintenance.png'
        this_title       = 'Q value steady-state maintenance'
        this_save_path   = os.path.join(this_save_folder, this_save_name)
        if not os.path.isdir(this_save_folder):
            os.mkdir(this_save_folder)
            
        plot_q_values_maintenance(av_rew, Q_all, Q_true, this_title, this_save_path)
        
        # ---------------------------------
        # --- Plot replay frequency ---
        this_save_folder = 'Replay_frequency_move%u'%d
        this_save_name   = 'Replay_frequency.png'
        this_title       = 'Replay frequency'
        this_save_path   = os.path.join(this_save_folder, this_save_name)
        if not os.path.isdir(this_save_folder):
            os.mkdir(this_save_folder)
        plot_replay_frequency(pos_TD, neg_TD, Q_true, this_title, this_save_path)
        
        # ---------------------------------
        # --- Plot replay stats ---
        # this_save_name   = 'Replay_stats_move%u.png'%d
        # this_title       = 'Replay stats'
        # this_save_path   = this_save_name
        # if not os.path.isdir(this_save_folder):
        #     os.mkdir(this_save_folder)
        # plot_performance(y_replay, ylab_replay, this_title, this_save_path)
        
        # ---------------------------------
        # --- Plot replay vs evm ---
        this_save_folder = 'evm_viol_move%u'%d
        this_save_name   = 'evm_viol.png'
        this_title       = 'evm vs replay density'
        this_save_path   = os.path.join(this_save_folder, this_save_name)
        if not os.path.isdir(this_save_folder):
            os.mkdir(this_save_folder)
        plot_replay_vs_evm(these_evms, 'evm', this_title, this_save_path)
        
        # ---------------------------------
        # --- Plot replay vs Entropy ---
        this_save_folder = 'Entropy_viol_move%u'%d
        this_save_name   = 'Entropy_viol.png'
        this_title       = 'Entropy vs replay density'
        this_save_path   = os.path.join(this_save_folder, this_save_name)
        if not os.path.isdir(this_save_folder):
            os.mkdir(this_save_folder)
        plot_replay_vs_evm(H, 'Entropy', this_title, this_save_path)
    
    # --- Performance actual moves ---
    ylab_actual   = 'Prop of available reward collected'
    y_exp         = []
    act_r_history = []
    max_r_history = []
    
    for f in range(len(all_files_sorted)):
            
            this_file = os.path.join(data_path, all_files_sorted[f])
            data      = np.load(this_file)
            
            this_move1 = data['move'][0]
            s1 = int(this_move1[0])
            r1 = int(this_move1[2])
            
            this_move2 = data['move'][1]
            r2 = int(this_move2[2])
            
            Q_true = Q_true_list[0]
            max_reward = np.amax(Q_true[s1, :])

            act_r_history.append([r1 + r2])
            max_r_history.append([max_reward])
            y_exp.append(np.sum(act_r_history)/np.sum(max_r_history))
            
    this_save_name  = 'Learning_progress_overall.png'
    this_title      = 'Learning progress'
    this_save_path  = this_save_name
    if not os.path.isdir(this_save_folder):
        os.mkdir(this_save_folder)
    plot_performance(y_exp, ylab_actual, this_title, this_save_path)
    return