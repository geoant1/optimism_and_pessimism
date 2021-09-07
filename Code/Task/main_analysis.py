import numpy as np
import pandas as pd
import os, glob
import scipy.stats
import matplotlib.pyplot as plt
from misc_analysis import *
from plot import plot_performance, plot_replays_entropy, plot_replays, plot_policy

root_path = '/Users/GA/Documents/Dayan_lab/Optimism_And_Pessimism_In_Optimised_Replay/'

# Worlds & idcs
world1 = np.load(os.path.join(root_path, 'Data/world1.npy'))
world2 = np.load(os.path.join(root_path, 'Data/world2.npy'))
idcs1  = np.load(os.path.join(root_path, 'Data/idcs1.npy'))
idcs2  = np.load(os.path.join(root_path, 'Data/idcs2.npy'))

def analyse_task_performance(task_folder):
    
    os.chdir(task_folder)
    s = glob.glob('*')
    if 'Analysis' in s:
        s.remove('Analysis')
    
    folders = [int(i) for i in s]
    folders.sort()
    nsubs   = len(folders)

    ### ------------------------ ###
    ### Analyse task performance ###
    ### ------------------------ ###

    path        = os.path.join(root_path, 'Data/subject_data')

    perf_agent  = np.zeros((nsubs, 15))
    perf_human  = np.zeros((nsubs, 15))

    for sub in folders:
        
        sub_path       = os.path.join(path, str(sub))
    
        blocks_max_rwd = np.load(os.path.join(sub_path, 'blocks_max_rwd.npy'), allow_pickle=True)[7:]
        blocks_obt_rwd = np.load(os.path.join(sub_path, 'blocks_obt_rwd.npy'), allow_pickle=True)[7:]

        sub_task_folder = os.path.join(task_folder, str(sub))
        rwd_obt = get_obtained_reward(sub_task_folder)

        count = 0
        for i in range(5):
            for j in range(0, 54, 18):
                ag = np.sum(rwd_obt[i][j:j+18])/np.sum(blocks_max_rwd[i][j:j+18])
                hm = np.sum(blocks_obt_rwd[i][j:j+18])/np.sum(blocks_max_rwd[i][j:j+18])
                
                perf_agent[sub, count] = ag
                perf_human[sub, count] = hm
                
                count += 1
        
        fig = plot_performance(perf_agent[sub, :], perf_human[sub, :])
        plt.savefig(os.path.join(sub_task_folder, 'performance.png'))
        np.save(os.path.join(sub_task_folder, 'performance_agent.npy'), perf_agent[sub, :])
        np.save(os.path.join(sub_task_folder, 'performance_sub.npy'), perf_human[sub, :])
        plt.close()
        print('Done with sub %u'%sub)
    
    analysis_folder = os.path.join(task_folder, 'Analysis')
    if not os.path.isdir(analysis_folder):
        os.makedirs(analysis_folder)
        
    fig = plot_performance(perf_agent, perf_human, mode='all')
    plt.savefig(os.path.join(analysis_folder, 'overall_fit.svg'), format='svg', transparent=True)
    np.save(os.path.join(analysis_folder, 'performance_agent_all.npy'), perf_agent)
    np.save(os.path.join(analysis_folder, 'performance_sub_all.npy'), perf_human)
    plt.close()
    
def analyse_replay_statistics(task_folder, thresh):

    H_opt_single_all    = []
    H_opt_paired_all    = []
    H_subopt_single_all = []
    H_subopt_paired_all = []

    opt_all    = []
    subopt_all = []

    subs_who_replay = []
    # subs_who_replay = np.load('/Users/GA/Documents/Dayan_lab/Optimism_And_Pessimism_In_Optimised_Replay/Data/task/Analysis/subs_who_replay.npy')

    ### ---------------------------------------- ###
    ### Analyse the replay of recent transitions ###
    ### ---------------------------------------- ###
    print('\nAnalysing replay of recent transitions...\n')
    
    os.chdir(task_folder)
    s = glob.glob('*')
    if 'Analysis' in s:
        s.remove('Analysis')
    
    folders = [int(i) for i in s]
    folders.sort()
    nsubs   = len(folders)
    
    for sub in folders:
        
        sub_task_folder = os.path.join(task_folder, str(sub))
        opt, subopt, H_opt_single, H_subopt_single, H_opt_paired, H_subopt_paired = analyse_recent_replays(sub_task_folder)
        
        if np.nanmean(opt) > thresh or np.nanmean(subopt) > thresh:

            H_opt_single_all    += H_opt_single
            H_opt_paired_all    += H_opt_paired
            H_subopt_single_all += H_subopt_single
            H_subopt_paired_all += H_subopt_paired
            
            opt_all             += [np.nanmean(opt)]
            subopt_all          += [np.nanmean(subopt)]
            
            subs_who_replay     += [sub]
            
            if not os.path.isdir(sub_task_folder):
                os.makedirs(sub_task_folder)
                
            stats_file = os.path.join(sub_task_folder, 'stats_recent.txt')
            with open(stats_file, 'w') as f:
                f.write('Action entropy tests \n')
                tp, pp = scipy.stats.ranksums(H_opt_paired, H_subopt_paired)
                f.write('H_opt_paired recent vs H_subopt_paired recent: W = %.3E, p = %.3E\n'%(tp, pp))
                ts, ps = scipy.stats.ranksums(H_opt_single, H_subopt_single)
                f.write('H_opt_single recent vs H_subopt_single recent: W = %.3E, p = %.3E\n'%(ts, ps))
                
                f.write('\nReplay tests \n')
                t, p = scipy.stats.ranksums(opt, subopt)
                f.write('opt vs subopt: W = %.3E, p = %.3E\n'%(t, p))
            
            fig = plot_replays_entropy(opt, subopt, H_opt_single, H_subopt_single, H_opt_paired, H_subopt_paired)
            plt.savefig(os.path.join(sub_task_folder, 'recent_replays_entropy.svg'), format='svg', transparent=True)
            np.savez(os.path.join(sub_task_folder, 'recent_replays_entropy_data.npz'), opt=opt, subopt=subopt, H_opt_single=H_opt_single, H_subopt_single=H_subopt_single, H_opt_paired=H_opt_paired, H_subopt_paired=H_subopt_paired)
            plt.close()
            
        print('Done with sub %u'%sub)
    
    # keep the recent replays for later comparison 
    H_opt_paired_exp    = H_opt_paired_all.copy()
    H_subopt_paired_exp = H_subopt_paired_all.copy()
    H_opt_single_exp    = H_opt_single_all.copy()
    H_subopt_single_exp = H_subopt_single_all.copy()
    opt_all_exp         = opt_all.copy()
    subopt_all_exp      = subopt_all.copy()
    
    # create the analysis folder
    analysis_folder = os.path.join(task_folder, 'Analysis')
    if not os.path.isdir(analysis_folder):
        os.makedirs(analysis_folder)
    
    np.save(os.path.join(analysis_folder, 'subs_who_replay.npy'), subs_who_replay)
    
    # run and save the stats
    stats_file = os.path.join(analysis_folder, 'stats_recent.txt')
    with open(stats_file, 'w') as f:
        f.write('Action entropy tests \n')
        tp, pp = scipy.stats.ranksums(H_opt_paired_all, H_subopt_paired_all)
        f.write('H_opt_paired recent vs H_subopt_paired recent: W = %.3E, p = %.3E\n'%(tp, pp))
        ts, ps = scipy.stats.ranksums(H_opt_single_all, H_subopt_single_all)
        f.write('H_opt_single recent vs H_subopt_single recent: W = %.3E, p = %.3E\n'%(ts, ps))
        
        f.write('\nReplay tests \n')
        t, p = scipy.stats.ranksums(opt_all, subopt_all)
        f.write('opt vs subopt: t = %.3E, p = %.3E\n'%(t, p))
    # plot
    fig = plot_replays_entropy(opt_all, subopt_all, H_opt_single_all, H_subopt_single_all, H_opt_paired_all, H_subopt_paired_all)
    plt.savefig(os.path.join(analysis_folder, 'recent_replays_entropy.svg'), format='svg', transparent=True)
    np.savez(os.path.join(analysis_folder, 'recent_replays_entropy_data.npz'), opt_all=opt_all, subopt_all=subopt_all, H_opt_single_all=H_opt_single_all, H_subopt_single_all=H_subopt_single_all, H_opt_paired_all=H_opt_paired_all, H_subopt_paired_all=H_subopt_paired_all)
    plt.close()
    
    fig = plot_replays(opt_all, subopt_all)
    plt.savefig(os.path.join(analysis_folder, 'recent_replays.svg'), format='svg', transparent=True)
    np.savez(os.path.join(analysis_folder, 'recent_replays_data.npz'), opt_all=opt_all, subopt_all=subopt_all)
    plt.close()

    print('\nDone with replay of recent transitions')
    print('Found %u subjects who replay > %.2f transitions per trial'%(len(subs_who_replay), thresh))
    
    ### --------------------------------------- ###
    ### Analyse the replay of other transitions ###
    ### --------------------------------------- ###
    print('\nAnalysing replay of other transitions in these subjects...\n')
    
    H_opt_single_all    = []
    H_opt_paired_all    = []
    H_subopt_single_all = []
    H_subopt_paired_all = []

    opt_all    = []
    subopt_all = []

    subs_who_replay = np.load(os.path.join('/Users/GA/Documents/Dayan_lab/Optimism_And_Pessimism_In_Optimised_Replay/Data/tmp/Analysis', 'subs_who_replay.npy'))
    
    for sub in subs_who_replay:
    # for sub in folders:
        
        H_opt_single    = []
        H_opt_paired    = []
        H_subopt_single = []
        H_subopt_paired = []
        opt             = []
        subopt          = []
        
        sub_task_folder = os.path.join(task_folder, str(sub))
        opt, subopt, H_opt_single, H_subopt_single, H_opt_paired, H_subopt_paired = analyse_other_replays(sub_task_folder)

        opt_all             += [np.nanmean(opt)]
        subopt_all          += [np.nanmean(subopt)]
        H_opt_single_all    += H_opt_single
        H_opt_paired_all    += H_opt_paired
        H_subopt_single_all += H_subopt_single
        H_subopt_paired_all += H_subopt_paired

        if not os.path.isdir(sub_task_folder):
            os.makedirs(sub_task_folder)
            
        stats_file = os.path.join(sub_task_folder, 'stats_other.txt')
        with open(stats_file, 'w') as f:
            f.write('Action entropy tests \n')
            tp, pp = scipy.stats.ranksums(H_opt_paired, H_subopt_paired)
            f.write('H_opt_paired other vs H_subopt_paired other: W = %.3E, p = %.3E\n'%(tp, pp))
            ts, ps = scipy.stats.ranksums(H_opt_single, H_subopt_single)
            f.write('H_opt_single other vs H_subopt_single other: W = %.3E, p = %.3E\n'%(ts, ps))
            
            f.write('\nReplay tests \n')
            t, p = scipy.stats.ranksums(opt, subopt)
            f.write('opt vs subopt: W = %.3E, p = %.3E\n'%(t, p))
        
        fig = plot_replays_entropy(opt, subopt, H_opt_single, H_subopt_single, H_opt_paired, H_subopt_paired)
        plt.savefig(os.path.join(sub_task_folder, 'other_replays_entropy.svg'), format='svg', transparent=True)
        np.savez(os.path.join(sub_task_folder, 'other_replays_entropy_data.npz'), opt=opt, subopt=subopt, H_opt_single=H_opt_single, H_subopt_single=H_subopt_single, H_opt_paired=H_opt_paired, H_subopt_paired=H_subopt_paired)
        plt.close()
        print('Done with sub %u'%sub)
    
    stats_file = os.path.join(analysis_folder, 'stats_other.txt')
    with open(stats_file, 'w') as f:
        f.write('Action entropy tests \n')
        tp, pp = scipy.stats.ranksums(H_opt_paired_all, H_subopt_paired_all)
        f.write('H_opt_paired other vs H_subopt_paired other: W = %.3E, p = %.3E\n'%(tp, pp))
        ts, ps = scipy.stats.ranksums(H_opt_single_all, H_subopt_single_all)
        f.write('H_opt_single other vs H_subopt_single other: W = %.3E, p = %.3E\n'%(ts, ps))
        
        f.write('\nReplay tests \n')
        t, p   = scipy.stats.ranksums(opt_all, subopt_all)
        f.write('opt vs subopt: W = %.3E, p = %.3E\n'%(t, p))
    
    fig = plot_replays_entropy(opt_all, subopt_all, H_opt_single_all, H_subopt_single_all, H_opt_paired_all, H_subopt_paired_all)
    plt.ylim(-0.5, 0.5)
    plt.savefig(os.path.join(analysis_folder, 'other_replays_entropy.svg'), format='svg', transparent=True)
    np.savez(os.path.join(analysis_folder, 'other_replays_entropy_data.npz'), opt_all=opt_all, subopt_all=subopt_all, H_opt_single_all=H_opt_single_all, H_subopt_single_all=H_subopt_single_all, H_opt_paired_all=H_opt_paired_all, H_subopt_paired_all=H_subopt_paired_all)
    
    plt.close()
    
    fig = plot_replays(opt_all, subopt_all)
    plt.savefig(os.path.join(analysis_folder, 'other_replays.svg'), format='svg', transparent=True)
    np.savez(os.path.join(analysis_folder, 'other_replays_data.npz'), opt_all=opt_all, subopt_all=subopt_all)
    plt.close()

    stats_file = os.path.join(analysis_folder, 'stats_recent_vs_other.txt')
    with open(stats_file, 'w') as f:
        f.write('Action entropy tests \n')
        t, p_ent_opt = scipy.stats.ranksums(H_opt_paired_exp, H_opt_paired)
        f.write('H_opt_paired recent vs H_opt_paired other:       W = %.3E, p = %.3E\n'%(t, p_ent_opt))
        t, p_ent_opt = scipy.stats.ranksums(H_opt_single_exp, H_opt_single)
        f.write('H_opt_single recent vs H_opt_single other:       W = %.3E, p = %.3E\n'%(t, p_ent_opt))
        t, p_ent_subopt = scipy.stats.ranksums(H_subopt_paired_exp, H_subopt_paired)
        f.write('H_subopt_paired recent vs H_subopt_paired other: W = %.3E, p = %.3E\n'%(t, p_ent_subopt))
        t, p_ent_subopt = scipy.stats.ranksums(H_subopt_single_exp, H_subopt_single)
        f.write('H_subopt_single recent vs H_subopt_single other: W = %.3E, p = %.3E\n'%(t, p_ent_subopt))
        
        f.write('\nReplay tests \n')
        t, p_subopt = scipy.stats.ranksums(subopt_all_exp, subopt_all)
        f.write('subopt recent vs subopt other: W = %.3E, p = %.3E\n'%(t, p_subopt))
        t, p_opt = scipy.stats.ranksums(opt_all_exp, opt_all)
        f.write('opt recent vs opt other:       W = %.3E, p = %.3E'%(t, p_opt))
    print('\nDone with replay of other transitions\n')

def analyse_replay_benefit(task_folder):
    
    ### ------------------------- ###
    ### Analyse benefit of replay ###
    ### ------------------------- ###
    
    analysis_folder = os.path.join(task_folder, 'Analysis')
    subs_who_replay = np.load(os.path.join('/Users/GA/Documents/Dayan_lab/Optimism_And_Pessimism_In_Optimised_Replay/Data/tmp/Analysis', 'subs_who_replay.npy'))
    # subs_who_replay = np.load(os.path.join(analysis_folder, 'subs_who_replay.npy'))
    print('Analysing benefit of replay...\n')

    modes   = ['value', 'probs', 'value']
    
    for mode in range(len(modes)):
        opt_all = []
        
        if mode == 2:
            ben = 'subjective'
        else:
            ben = 'objective'
        
        for sub in subs_who_replay:
            
            sub_task_folder = os.path.join(task_folder, str(sub))
            # params = os.path.join(root_path, 'Data/new_fits/save_params_%u/params.npy'%sub)
            # p      = np.load(params)
            p       = pd.read_csv(os.path.join(root_path, 'Data/new_new_fits/save_params_%u'%sub, 'backup.txt'), sep='\t').iloc[-1].values[:-2]
            
            opt_sub = get_replay_benefit(sub_task_folder, p, modes[mode], ben)
            
            opt_all.append(np.mean(opt_sub))
            
            fig = plot_policy(opt_sub)
            plt.savefig(os.path.join(sub_task_folder, 'policy_improve_%s_%s.svg'%(modes[mode], ben)), format='svg', transparent=True)
            np.save(os.path.join(sub_task_folder, 'policy_improve_%s_%s.npy'%(modes[mode], ben)), opt_sub)
            plt.close()
            
            stats_file = os.path.join(sub_task_folder, 'stats_policy_%s_%s.txt'%(modes[mode], ben))
            with open(stats_file, 'w') as f:
                f.write('Policy improvement \n')
                f.write('\n Average: %.3f'%np.nanmean(opt_sub))
                f.write('Opt policy change: t: %.3f,  p-value: %.3E' % scipy.stats.ttest_1samp(opt_sub, 0))
                # f.write('Opt policy change: t: %.3f,  p-value: %.3E' % scipy.stats.wilcoxon(opt_sub))
        
        fig = plot_policy(opt_all)
        plt.savefig(os.path.join(analysis_folder, 'policy_improve_%s_%s.svg'%(modes[mode], ben)), format='svg', transparent=True)
        np.save(os.path.join(analysis_folder, 'policy_improve_%s_%s.npy'%(modes[mode], ben)), opt_all)
        plt.close()
        
        stats_file = os.path.join(analysis_folder, 'stats_policy_%s_%s.txt'%(modes[mode], ben))
        with open(stats_file, 'w') as f:
            f.write('Policy improvement \n')
            f.write('\n Average: %.3f'%np.nanmean(opt_all))
            f.write('Opt policy change: t: %.3f,  p-value: %.3E' % scipy.stats.ttest_1samp(opt_all, 0))
            # f.write('Opt policy change: t: %.3f,  p-value: %.3E' % scipy.stats.wilcoxon(opt_all))
            
this_folder = os.path.join(root_path, 'Data/tmp_zeromf')
# analyse_task_performance(this_folder)
analyse_replay_statistics(this_folder, 0.3)
analyse_replay_benefit(this_folder)