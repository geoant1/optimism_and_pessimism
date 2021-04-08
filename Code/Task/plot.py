import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from tools import plot_each_step, plot_history, plot_evm, plot_Q, plot_Q_2moves, plot_evm_2moves
from misc import state2idcs
import os, glob, shutil
import numpy as np
from misc_analysis import histogram

def plot_performance(perf_agent, perf_human, mode='single'):
    
    fig = plt.figure(figsize=(14, 7))
    
    if mode == 'all':
        for i in range(0, 15, 3):
            y_h = perf_human[:, i:i+3]
            y_h_plot = np.mean(y_h, axis=0)
            
            y_a = perf_agent[:, i:i+3]
            y_a_plot = np.mean(y_a, axis=0)
            
            x = np.arange(3)+i
            
            std_h = np.std(y_h, axis=0)
            sem_h = std_h/np.sqrt(40)
            
            plt.plot(x, y_h_plot, c='k', linewidth=5)
            plt.fill_between(x, (y_h_plot-sem_h), (y_h_plot+sem_h), color='k', alpha=.2)
            
            plt.plot(x, y_a_plot, c='#00D667', linewidth=2)
        
        plt.scatter(range(0,15), np.mean(perf_human, axis=0), label='Humans', c='k', marker='h', s=300)
        plt.scatter(range(0,15), np.mean(perf_agent, axis=0), label='Agent', c='#00D667', marker='h', s=100)
        plt.ylim([.58,.82])
    
    else:
        for i in range(0, 15, 3):
            x = np.arange(3)+i
            if i == 12:
                plt.scatter(x, perf_human[i:i+3], c='k', marker='h', label='Human', s=150)
                plt.scatter(x, perf_agent[i:i+3], c='#00D667', marker='h', label='Agent', s=100)
            else:
                plt.scatter(x, perf_human[i:i+3], c='k', marker='h', s=150)
                plt.scatter(x, perf_agent[i:i+3], c='#00D667', marker='h', s=100)

        plt.ylim([0, 1])
        err = np.sqrt(np.sum(np.power((np.array(perf_human) - np.array(perf_agent)), 2)))
        plt.title('Performance, err = %.3f'%err)
    
    plt.axvline(2.5, linestyle='--')
    plt.axvline(5.5, linestyle='--', color='g', label='Rearrangement')
    plt.axvline(8.5, linestyle='--')
    plt.axvline(11.5, linestyle='--', color='r', label='World change')
        
    plt.legend(prop={'size': 9}, loc='best')
    # plt.grid(alpha=0.5)
    plt.ylabel('Prop of available reward collected')
    plt.xlabel('Epochs')

    return fig

def plot_replays_entropy(opt, subopt, H_opt_single, H_subopt_single, H_opt_paired, H_subopt_paired):
    
    fig = plt.figure(figsize=(14, 4))

    # Replays
    plt.subplot(1, 2, 1)
    
    subopt_h, subopt_b, subopt_w = histogram(subopt)
    plt.bar(subopt_b, subopt_h, width=subopt_w, facecolor='b')
    opt_h, opt_b, _ = histogram(opt)
    plt.bar(opt_b, opt_h, width=subopt_w, facecolor='orange', alpha=0.8)
    plt.xlabel('Number of replays per trial', fontsize=14)
    plt.xlim(0-subopt_w/2, np.max(subopt_b)+subopt_w/2)
    
    # Single entropy
    plt.subplot(1, 2, 2)
    
    bins=np.linspace(0, np.log2(7)+np.log2(6), 40)
    H_subopt_hs, H_subopt_b, H_subopt_w = histogram(H_subopt_single, bins)
    plt.bar(H_subopt_b, H_subopt_hs, width=H_subopt_w, facecolor='b')
    H_opt_hs, H_opt_b, _ = histogram(H_opt_single, bins)
    plt.bar(H_opt_b, H_opt_hs, width=H_subopt_w, facecolor='orange', alpha=0.8)
    plt.xlim(0-H_subopt_w/2, np.log2(7)+np.log2(6)-H_subopt_w/2)
    
    # Paired entropy
    H_subopt_hp, H_subopt_b, _ = histogram(H_subopt_paired, bins)
    plt.bar(H_subopt_b, -H_subopt_hp, width=H_subopt_w, facecolor='b')
    H_opt_hp, H_opt_b, _ = histogram(H_opt_paired, bins)
    plt.bar(H_opt_b, -H_opt_hp, width=H_subopt_w, facecolor='orange', alpha=0.8)

    plt.axhline(0, c='k')
    plt.ylabel('Proportion of replays', fontsize=14)
    plt.xlabel('Action entropy', fontsize=14)
    
    max_val = max(np.max(H_opt_hp), np.max(H_subopt_hs), np.max(H_opt_hs), np.max(H_subopt_hp))
    plt.ylim(-max_val-0.1, max_val+0.1)
    
    plt.tight_layout()
    
    return fig

def plot_replays(opt, subopt):
    
    fig = plt.figure(figsize=(4, 4))

    plt.bar(1, np.mean(opt), facecolor='orange', alpha=0.6, align='center')
    plt.scatter([1]*len(opt), opt, c='orange')
    plt.bar(2, np.mean(subopt), facecolor='b', alpha=0.6, align='center')
    plt.scatter([2]*len(subopt), subopt, c='b')
    plt.axhline(0, c='k')
    
    return fig

def plot_policy(x):
    
    fig = plt.figure(figsize=(4, 5))
    ax  = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.bar(1, np.nanmean(x), facecolor='#22CF00', alpha=0.4, width=0.1)
    ax.scatter([1]*len(x), x, c='#22CF00', alpha=1)
    ax.axhline(0, c='k')
    ax.set_ylabel(r'$Average \; \; \Delta \pi(a \mid s) \; \; over \; \; states$', fontsize=14)
    
    return fig

def plot_simulation_1move(data_path, world, corr_moves, Q_true, d=None):

    save_folder = os.path.join(data_path, 'Figures')
    info_folder = os.path.join(save_folder, 'Replay_info')
    
    if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
            os.mkdir(info_folder)
    else:
        shutil.rmtree(save_folder)
        os.makedirs(save_folder)
        os.mkdir(info_folder)

    fig = plt.figure(1, dpi=150, figsize=(16, 10))
    ax1 = fig.add_axes([0.02, 0.33, 0.35, 0.35])  # left, bottom, width, height
    ax2 = fig.add_axes([0.40, 0.52, 0.25, 0.44])  # EVM
    ax3 = fig.add_axes([0.71, 0.52, 0.25, 0.44])  # Need 
    ax4 = fig.add_axes([0.71, 0.04, 0.25, 0.44])  # Gain
    ax5 = fig.add_axes([0.40, 0.04, 0.25, 0.44])  # Q-values

    os.chdir(data_path)
    all_files = glob.glob('*.npz')
    
    # Sort files by episode number
    episodes = [int(''.join(c for c in f if c.isdigit())) for f in all_files]
    idcs     = [i[0] for i in sorted(enumerate(episodes), key=lambda x:x[1])]
    all_files_sorted = [all_files[i] for i in idcs] 
    
    if d:
        p1 = d[0]
        p2 = d[1]
    else:
        p1 = 0
        p2 = len(episodes)
    
    for f in range(p1, p2):
        this_file = os.path.join(data_path, all_files_sorted[f])
        data = np.load(this_file)

        this_move = data['move']
        s   = int(this_move[0])
        si  = state2idcs(s, world)
        a   = int(this_move[1])
        s1  = int(this_move[3])
        s1i = state2idcs(s1, world)
        # print('\nMove: ', this_move)
        
        replay_backups = data['replay_backups']
        replay_exp     = data['replay_exp']
        gain           = data['gain']
        need           = data['need']
        evm            = data['evm']
        rew_history2   = data['rew_history2']
        
        Q = data['Q_values']
        # plt.waitforbuttonpress()
        
        title = 'Move %u'%f
        av_rew = np.mean(rew_history2)
        plot_each_step(ax1, s1i, world, title, [a, si])
        plot_Q(ax5, Q[0, :], Q_true, av_rew)
        # Save this move
        plt.savefig(os.path.join(save_folder, 'Move%d.png'%f))
        # plt.waitforbuttonpress()
        
        # Write this move to the info txt file
        info = open(os.path.join(info_folder, 'Move%d.txt'%f), 'w')
        info.write('Move: [')
        info.writelines([" %s " % item  for item in this_move])
        info.write(']\n')
        
        if replay_backups.shape[0] > 0:
            for i in range(replay_backups.shape[0]):
                this_gain = gain[i].reshape(8, 4)
                this_need = need[i].reshape(8, 4)
                this_evm  = evm[i].reshape(8, 4)
                
                info.write('\n-- Replay %d\n-- Considered experiences:\n'%i)
                for z in range(replay_exp.shape[0]):
                    az = int(replay_exp[z, 1])
                    sz = int(replay_exp[z, 0])
                    
                    info.write('\nExp: [')
                    info.writelines([" %.1f " % item  for item in replay_exp[z, :]])
                    info.write('], Gain: [ %.5f ], Need: [ %.5f ], evm: [ %.5f ]\n'%(this_gain[sz, az], this_need[sz, az], this_evm[sz, az]))
                    
                sp  = int(replay_backups[i, 0])
                spi = state2idcs(sp, world)
                ap  = int(replay_backups[i, 1])
                rp  = replay_backups[i, 2]
                s1p = int(replay_backups[i, 3])
                s1pi = state2idcs(s1p, world)
                
                info.write('\nReplaying exp [ %u %u %.2f %u ]\nevm: [ %.5f ]\n'%(sp, ap, rp, s1p, this_evm[sp, ap]))
                
                # print('\nReplaying ', [sp, ap, rp, s1p], '\nevm: ', evm[i].reshape(8, 4)[sp, ap], '\n')
                
                max_need = abs(max(np.nanmin(this_need[:]), np.nanmax(this_need[:]), key=abs))
                max_gain = abs(max(np.nanmin(this_gain[:]), np.nanmax(this_gain[:]), key=abs))
                max_evm  = abs(max(np.nanmin(this_evm[:]), np.nanmax(this_evm[:]), key=abs))
                
                title = 'Replay %u, episode %u'%(i,f)
                plot_each_step(ax1, s1i, world, title, [a, si], [ap, spi, s1pi]) # ok
                
                plot_evm(ax3, this_need, this_need/max_need, 'Need')
                plot_evm(ax4, this_gain, this_gain/max_gain, 'Gain')
                plot_evm(ax2, this_evm, this_evm/max_evm, 'Memory values')                
                plt.savefig(os.path.join(save_folder, 'Move%u_replay_before%u.png'%(f, i))) # Save evm with old q-values
                
                plot_Q(ax5, Q[i+1, :], Q_true, av_rew)
                plt.savefig(os.path.join(save_folder, 'Move%u_replay_after%u.png'%(f, i)))

    plt.close(fig)
    return

def plot_simulation_2moves(data_path, world, Q_true: list, d=None):

    save_folder = os.path.join(data_path, 'Figures')
    info_folder = os.path.join(save_folder, 'Replay_info')
    
    if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
            os.mkdir(info_folder)
    else:
        shutil.rmtree(save_folder)
        os.makedirs(save_folder)
        os.mkdir(info_folder)
    
    # plt.ion()
    fig = plt.figure(1, dpi=170, figsize=(16, 10))
    
    ax11 = fig.add_axes([0.02, 0.12, 0.30, 0.30])  # left, bottom, width, height
    ax12 = fig.add_axes([0.02, 0.54, 0.30, 0.30])

    ax21 = fig.add_axes([0.37, 0.505, 0.27, 0.20])  # EVM move 1
    ax22 = fig.add_axes([0.37, 0.76, 0.27, 0.20])  # EVM move 2

    ax31 = fig.add_axes([0.69, 0.505, 0.27, 0.20])  # Need move 1
    ax32 = fig.add_axes([0.69, 0.76, 0.27, 0.20])  # Need move 2

    ax41 = fig.add_axes([0.69, 0.03, 0.27, 0.18])  # Gain move 1
    ax42 = fig.add_axes([0.69, 0.27, 0.27, 0.18])  # Gain move 2

    ax51 = fig.add_axes([0.37, 0.03, 0.27, 0.18])  # Q-values move 1
    ax52 = fig.add_axes([0.37, 0.27, 0.27, 0.18])  # Q-values move 2

    os.chdir(data_path)
    all_files = glob.glob('*.npz')
    
    # Sort files by episode number
    episodes = [int(''.join(c for c in f if c.isdigit())) for f in all_files]
    idcs     = [i[0] for i in sorted(enumerate(episodes), key=lambda x:x[1])]
    all_files_sorted = [all_files[i] for i in idcs]  
    
    if d:
        p1 = d[0]
        p2 = d[1]
    else:
        p1 = 0
        p2 = len(episodes)
    
    es = [i for i in range(p1, int(p2-((p2-p1)/2)))]
    c  = 0
    
    for f in range(p1, p2):
        this_file = os.path.join(data_path, all_files_sorted[f])
        data = np.load(this_file)

        this_move1 = data['move1']
        s   = int(this_move1[0])
        si  = state2idcs(s, world)
        a   = int(this_move1[1])
        s1  = int(this_move1[3])
        s1i = state2idcs(s1, world)
        
        this_move2 = data['move2']
        a2  = int(this_move2[1])
        s2  = int(this_move2[3])
        s2i = state2idcs(s2, world)

        replay_backups = data['replay_backups']
        replay_planes  = data['replay_planes']
        
        replay_exp1 = data['replay_exp1']
        gain1       = data['gain1']
        need1       = data['need1']
        evm1        = data['evm1']
        
        replay_exp2   = data['replay_exp2']
        gain2       = data['gain2']
        need2       = data['need2']
        evm2        = data['evm2']
        
        rew_history1     = data['rew_history1']
        rew_history2     = data['rew_history2']
        
        av_rew1 = np.mean(rew_history1)
        av_rew2 = np.mean(rew_history2)
        
        Q1 = data['Q1']
        Q2 = data['Q2']
        
        Q1_true = Q_true[0]
        Q2_true = Q_true[1]
        
        title = 'Episode %u'%f
        plot_each_step(ax11, s1i, world, title, [a, si])
        plot_each_step(ax12, s2i, world, title, [a2, s1i])
        plot_Q_2moves(ax51, Q1[0, :], Q1_true, av_rew1)
        plot_Q_2moves(ax52, Q2[0, :], Q2_true, av_rew2)
            
        # Save this move
        plt.savefig(os.path.join(save_folder, 'Episode%u.png'%f))
        
        # Write this move to the info txt file
        info = open(os.path.join(info_folder, 'Episode%u.txt'%f), 'w')
        info.write('Move 1: [')
        info.writelines([" %s " % item  for item in this_move1])
        info.write(']\n')
        info.write('Move 2: [')
        info.writelines([" %s " % item  for item in this_move2])
        info.write(']\n')
        
        if replay_backups.shape[0] > 0:
            
            for i in range(replay_backups.shape[0]):
                
                this_gain = gain1[i].reshape(8, 4)
                this_need = need1[i].reshape(8, 4)
                this_evm  = evm1[i].reshape(8, 4)
                info.write('\n-- Replay %d\n-- Considered experiences from move 1:\n'%i)
                for z in range(replay_exp1.shape[0]):
                    
                    az = int(replay_exp1[z, 1])
                    sz = int(replay_exp1[z, 0])
                    
                    info.write('\nExp: [')
                    info.writelines([" %u " % item  for item in replay_exp1[z, :]])
                    info.write('], Gain: [ %.5f ], Need: [ %.5f ], evm: [ %.5f ]\n'%(this_gain[sz, az], this_need[sz, az], this_evm[sz, az]))
                this_gain = gain2[i].reshape(8, 4)
                this_need = need2[i].reshape(8, 4)
                this_evm  = evm2[i].reshape(8, 4)
                info.write('\n-- Considered experiences from move 2:\n')
                for z in range(replay_exp2.shape[0]):
                    
                    az = int(replay_exp2[z, 1])
                    sz = int(replay_exp2[z, 0])
                    
                    info.write('\nExp: [')
                    info.writelines([" %u " % item  for item in replay_exp2[z, :]])
                    info.write('], Gain: [ %.5f ], Need: [ %.5f ], evm: [ %.5f ]\n'%(this_gain[sz, az], this_need[sz, az], this_evm[sz, az]))
                    
                sp  = int(replay_backups[i, 0])
                spi = state2idcs(sp, world)
                ap  = int(replay_backups[i, 1])
                rp  = replay_backups[i, 2]
                s1p = int(replay_backups[i, 3])
                s1pi = state2idcs(s1p, world)
                plane = int(replay_planes[i])
                
                this_gain1 = gain1[i].reshape(8, 4)
                this_need1 = need1[i].reshape(8, 4)
                this_evm1  = evm1[i].reshape(8, 4)
                
                this_gain2 = gain2[i].reshape(8, 4)
                this_need2 = need2[i].reshape(8, 4)
                this_evm2  = evm2[i].reshape(8, 4)
                
                info.write('\nReplaying exp [ %u %u %.2f %u ] from move %u\nevm: [ %.5f ]\n'%(sp, ap, rp, s1p, plane, this_evm[sp, ap]))
                                
                max_need1 = abs(max(np.nanmin(this_need1[:]), np.nanmax(this_need1[:]), key=abs))
                max_gain1 = abs(max(np.nanmin(this_gain1[:]), np.nanmax(this_gain1[:]), key=abs))
                max_evm1  = abs(max(np.nanmin(this_evm1[:]), np.nanmax(this_evm1[:]), key=abs))
                
                max_need2 = abs(max(np.nanmin(this_need2[:]), np.nanmax(this_need2[:]), key=abs))
                max_gain2 = abs(max(np.nanmin(this_gain2[:]), np.nanmax(this_gain2[:]), key=abs))
                max_evm2  = abs(max(np.nanmin(this_evm2[:]), np.nanmax(this_evm2[:]), key=abs))
                
                title = 'Replay %u, episode %u'%(i,es[c])
                
                if plane == 0:
                    plot_each_step(ax11, s1i, world, title, [a, si], [ap, spi, s1pi]) # ok
                    plot_each_step(ax12, s2i, world, title, [a2, s1i])
                else:
                    plot_each_step(ax11, s1i, world, title, [a, si]) # ok
                    plot_each_step(ax12, s2i, world, title, [a2, s1i], [ap, spi, s1pi]) # ok
                
                plot_evm_2moves(ax31, this_need1, this_need1/max_need1, 'Need')
                plot_evm_2moves(ax41, this_gain1, this_gain1/max_gain1, 'Gain')
                plot_evm_2moves(ax21, this_evm1, this_evm1/max_evm1, 'Memory values')     
                
                plot_evm_2moves(ax32, this_need2, this_need2/max_need2, 'Need')
                plot_evm_2moves(ax42, this_gain2, this_gain2/max_gain2, 'Gain')
                plot_evm_2moves(ax22, this_evm2, this_evm2/max_evm2, 'Memory values')   
                           
                plt.savefig(os.path.join(save_folder, 'Episode%u_replay_before%u.png'%(f, i))) # Save evm with old q-values
                
                plot_Q_2moves(ax51, Q1[i+1, :], Q1_true, av_rew1)
                plot_Q_2moves(ax52, Q2[i+1, :], Q2_true, av_rew2)
                plt.savefig(os.path.join(save_folder, 'Episode%u_replay_after%u.png'%(f, i)))
           
    plt.close(fig)
    return