import numpy as np
import seaborn as sns
sns.set_palette("husl")
cm = sns.light_palette("green")
import matplotlib
from matplotlib.patches import RegularPolygon, Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from misc import state2idcs, get_new_state
import os
import shutil
import random

def plot_each_step(ax, idcs, maze, title: str, a=None, a_plan=None):
    '''Plot trajectory and state values'''
    ax.clear()

    patches = []                         #  x    y   width  height
    for i in range(4):
        for j in range(2):
            patches.append(Rectangle((1+1*i, 1-1*j),  1,   1, edgecolor='k', fill=False)) # centre
            # ax.annotate('%d'%(maze[j,i]), [1.04+1*i, 0.15+1*j])
            ax.annotate('%.2f'%(maze[j,i]), [1.04+1*i, 0.15+1*j], fontsize=8)
            for act in range(4):
                if act == 0:
                    patches.append(RegularPolygon((1.5+1*i, 1.2-1*j), 3, radius=0.1, lw=0.5, orientation=np.pi, edgecolor='k', fill=False))
                elif act == 1:
                    patches.append(RegularPolygon((1.5+1*i, 1.8-1*j), 3, radius=0.1, lw=0.5, orientation=0, edgecolor='k', fill=False))
                elif act == 2:
                    patches.append(RegularPolygon((1.25+1*i, 1.5-1*j), 3, radius=0.1, lw=0.5, orientation=np.pi/2, edgecolor='k', fill=False))
                else: 
                    patches.append(RegularPolygon((1.75+1*i, 1.5-1*j), 3, radius=0.1, lw=0.5, orientation=-np.pi/2, edgecolor='k', fill=False))
    
    # Plot agent location
    if idcs:
        i = idcs[1]
        j = idcs[0]
        ax.scatter(i+1.5, j+0.5, s=70)
    
    # Plot agent's action
    if a:
        i = a[1][1]
        j = a[1][0]
        a = a[0]
        
        if a == 0:
            patches.append(RegularPolygon((1.5+1*i, 0.2+1*j), 3, radius=0.1, lw=0.5, orientation=np.pi, edgecolor='k', facecolor='r', fill=True))
        elif a == 1:
            patches.append(RegularPolygon((1.5+1*i, 0.8+1*j), 3, radius=0.1, lw=0.5, orientation=0, edgecolor='k', facecolor='r', fill=True))
        elif a == 2:
            patches.append(RegularPolygon((1.25+1*i, 0.5+1*j), 3, radius=0.1, lw=0.5, orientation=np.pi/2, edgecolor='k', facecolor='r', fill=True))
        else: 
            patches.append(RegularPolygon((1.75+1*i, 0.5+1*j), 3, radius=0.1, lw=0.5, orientation=-np.pi/2, edgecolor='k', facecolor='r', fill=True))
    
    if a_plan:
        ip = a_plan[1][1]
        jp = a_plan[1][0]
        
        i_loc = a_plan[2][1]
        j_loc = a_plan[2][0]
        a_plan = a_plan[0]
        
        ax.scatter(i_loc+1.5, j_loc+0.5, s=70, color='orange')
        
        if a_plan == 0:
            patches.append(RegularPolygon((1.5+1*ip, 0.2+1*jp), 3, radius=0.1, lw=0.5, orientation=np.pi, edgecolor='k', facecolor='g', fill=True))
        elif a_plan == 1:
            patches.append(RegularPolygon((1.5+1*ip, 0.8+1*jp), 3, radius=0.1, lw=0.5, orientation=0, edgecolor='k', facecolor='g', fill=True))
        elif a_plan == 2:
            patches.append(RegularPolygon((1.25+1*ip, 0.5+1*jp), 3, radius=0.1, lw=0.5, orientation=np.pi/2, edgecolor='k', facecolor='g', fill=True))
        else: 
            patches.append(RegularPolygon((1.75+1*ip, 0.5+1*jp), 3, radius=0.1, lw=0.5, orientation=-np.pi/2, edgecolor='k', facecolor='g', fill=True))
    
    collection = PatchCollection(patches, match_original=True)
    ax.add_collection(collection)
    
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.invert_yaxis()
    ax.set_title(title)
    # plt.show()
    
    return 


def plot_evm(ax, evm, evm_to_plot, title: str):
    '''Plot trajectory and state values'''
    ax.clear()
    ax2 = ax.twinx()
    
    # evm = evm.reshape(8, 4)
    # evm_to_plot = evm_to_plot.reshape(8, 4)
    
    for_colors = evm.ravel()
    colors = np.array([(0,0,1)]*len(for_colors))
    colors[for_colors >= 0] = (1,0,0)
    colors[np.isnan(for_colors)] = (0,0,0)
    mult = np.reshape(np.repeat(np.abs(for_colors)/np.max(np.abs(for_colors[~np.isnan(for_colors)])),3),(len(for_colors),3))
    colors =  mult*colors

    y_labels = ['N', 'S', 'W', 'E']
    y_pos = np.arange(0.2, 2.6, 0.3)
    add   = [-0.09, -0.03, 0.03, 0.09][::-1]
    y_ticks = []
    y_ticks2 = []

    for i in range(evm.shape[0]):
        this_evm = evm[i, :]
        for j in range(evm.shape[1]):
            y = y_pos[i]+add[j]
            y_ticks.append(y)
            c = colors[i*4+j]
            if this_evm[j] > 0:
                ax.hlines(y, 0, evm_to_plot[i, j], color=c, lw=3)
                # ax.text(evm_to_plot[i, j]+0.12, y_pos[i]+add[j]-0.015, 
                #         '%.3f'%(this_evm[j]), ha='center', fontsize=6)
                ax.text(evm_to_plot[i, j]+0.14, y_pos[i]+add[j]-0.015, 
                        '%.2fe2'%(this_evm[j]*100), ha='center', fontsize=6)
            else:
                ax.hlines(y, evm_to_plot[i, j], 0, color=c, lw=3)
                # ax.text(evm_to_plot[i, j]-0.14, y_pos[i]+add[j]-0.015, 
                #         '%.3f'%(this_evm[j]), ha='center', fontsize=6)
                ax.text(evm_to_plot[i, j]-0.14, y_pos[i]+add[j]-0.015, 
                        '%.2fe2'%(this_evm[j]*100), ha='center', fontsize=6)

    ax.axvline(0, 0, 1, color='k', alpha=0.4)

    y_tick_labels = y_labels*evm.shape[0]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontsize=7)
    ax.set_ylabel('Action')

    y2_tick_labels = ['']+[str(i) for i in range(evm.shape[0])]+['']
    ax2.set_yticks(np.arange(0, evm.shape[0]+2))
    ax2.set_yticklabels(y2_tick_labels, fontsize=7)
    ax2.set_ylabel('State')

    ax.set_title(title)
    max_val = abs(max(np.nanmin(evm_to_plot[:]), np.nanmax(evm_to_plot[:]), key=abs))
    # ax.set_xlim((-max_val-0.3, max_val+0.3))
    ax.set_xlim((-max_val-0.3, max_val+0.3))
    ax.get_xaxis().set_visible(False)
    # plt.show()
    
    return 
    
def plot_Q(ax, Q, Q_true, av_rew):
    ax.clear()
    ax2 = ax.twinx()

    Q = Q.reshape(8, 4)
    
    y_labels = ['N', 'S', 'W', 'E']
    y_pos = np.arange(0.2, 2.6, 0.3)
    add   = [-0.09, -0.03, 0.03, 0.09][::-1]
    y_ticks = []

    Q_true = Q_true.reshape(8, 4)
    max_val = 0
    for i in range(Q.shape[0]):
        this_Q = Q[i, :]
        this_Q_true = Q_true[i, :]
        for j in range(Q.shape[1]):
            y = y_pos[i]+add[j]
            y_ticks.append(y)
            
            if this_Q[j] > this_Q_true[j]:
                ax.hlines(y, 0, this_Q[j], color='k', lw=3)
                this_x = this_Q[j]+0.59
                ax.text(this_x, y_pos[i]+add[j]-0.01, '%.2f/%.2f'%(this_Q[j], this_Q_true[j]), ha='center', fontsize=6)
            else:
                ax.hlines(y, this_Q[j], this_Q_true[j], color='r', lw=3)
                ax.hlines(y, 0, this_Q[j], color='k', lw=3)
                this_x = this_Q_true[j]+0.59
                ax.text(this_x, y_pos[i]+add[j]-0.012, '%.2f/%.2f'%(this_Q[j], this_Q_true[j]), ha='center', fontsize=6)
                
            if this_x > max_val:
                max_val = this_x
    
    ax.axvline(av_rew, 0, 1, color='k', alpha=0.4)
    
    y_tick_labels = y_labels*Q.shape[0]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontsize=7)
    ax.set_ylabel('Action')

    y2_tick_labels = ['']+[str(i) for i in range(Q.shape[0])]+['']
    ax2.set_yticks(np.arange(0, Q.shape[0]+2))
    ax2.set_yticklabels(y2_tick_labels, fontsize=7)
    ax2.set_ylabel('State')

    ax.set_title('Q-values')
    # ax.set_xlim((0, np.max(Q_true[:])+1.9))
    ax.set_xlim((0, max_val+0.73))
    ax.get_xaxis().set_visible(False)
    # plt.show()
    
def plot_history(ax, x, y, ylab, title):
    
    ax.clear()
    
    ax.set_title(title)
    ax.set_xlabel('Steps')
    ax.set_ylabel(ylab)
    ax.scatter(x, y, s=20)
    ax.set_ylim([-0.05, 1.05])
    # plt.show()
    
    return 
    
def plot_Q_2moves(ax, Q, Q_true, av_rew):
    
    ax.clear()
    Q = Q.reshape(8, 4)
    ax2 = ax.twiny()
    
    x_labels = ['N', 'S', 'W', 'E']
    x_pos    = np.arange(0.2, 2.6, 0.3)
    add      = [-0.06, -0.02, 0.02, 0.06]
    x_ticks  = []
    x_ticks2 = []

    max_val = 0
    for i in range(Q.shape[0]):
        this_Q = Q[i, :]
        this_Q_true = Q_true[i, :]
        for j in range(Q.shape[1]):
            x = x_pos[i]+add[j]
            x_ticks.append(x)
            ax.vlines(x, this_Q[j], this_Q_true[j], color='r', lw=3)
            ax.vlines(x, 0, this_Q[j], color='k', lw=3)
            if this_Q[j] > this_Q_true[j]:
                this_y = this_Q[j]+0.2
                ax.text(x_pos[i]+add[j]+0.005, this_y, '%.2f/%.2f'%(this_Q[j], this_Q_true[j]), 
                        ha='center', fontsize=4, rotation=90)
            else:
                this_y = this_Q_true[j]+0.2
                ax.text(x_pos[i]+add[j]+0.005, this_y, '%.2f/%.2f'%(this_Q[j], this_Q_true[j]), 
                        ha='center', fontsize=4, rotation=90)
            if this_y > max_val:
                max_val = this_y
    
    ax.axhline(av_rew, c='k', linestyle=':')
    x_tick_labels = x_labels*Q.shape[0]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, fontsize=5)
    # ax.set_xlabel('Action', fontsize=7)

    x2_tick_labels = ['']+[str(i) for i in range(Q.shape[0])]+['']
    ax2.set_xticks(np.arange(0, Q.shape[0]+2))
    ax2.set_xticklabels(x2_tick_labels, fontsize=6)
    # ax2.set_xlabel('State', fontsize=7)

    # ax.set_title('Q-values', fontsize=7)
    ax.set_ylim((0, max_val+1.5))
    ax.get_yaxis().set_visible(False)
    
def plot_evm_2moves(ax, evm, evm_to_plot, title: str):
    
    ax.clear()
    ax2 = ax.twiny()

    for_colors = evm.ravel()
    colors = np.array([(0,0,1)]*len(for_colors))
    colors[for_colors >= 0] = (1,0,0)
    colors[np.isnan(for_colors)] = (0,0,0)
    mult = np.reshape(np.repeat(np.abs(for_colors)/np.max(np.abs(for_colors[~np.isnan(for_colors)])),3),(len(for_colors),3))
    colors =  mult*colors

    x_labels = ['N', 'S', 'W', 'E']
    x_pos    = np.arange(0.2, 2.6, 0.3)
    add      = [-0.06, -0.02, 0.02, 0.06]
    x_ticks  = []
    x_ticks2 = []

    for i in range(evm.shape[0]):
        this_evm = evm_to_plot[i, :]
        for j in range(evm.shape[1]):
            x = x_pos[i]+add[j]
            x_ticks.append(x)
            c = colors[i*4+j]
            if this_evm[j] > 0:
                ax.vlines(x, 0, this_evm[j], color=c, lw=3)
                ax.text(x_pos[i]+add[j]+0.005, this_evm[j]+0.1, 
                        '%.2f'%(evm[i, j]), ha='center', fontsize=4, rotation=90)
            else:
                ax.vlines(x, this_evm[j], 0, color=c, lw=3)
                ax.text(x_pos[i]+add[j]+0.005, this_evm[j]-0.3, 
                        '%.2f'%(evm[i, j]), ha='center', fontsize=4, rotation=90)

    ax.axhline(0, 0, 1, color='k', alpha=0.4)

    x_tick_labels = x_labels*evm.shape[0]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, fontsize=5)
    # ax.set_xlabel('Action')

    x2_tick_labels = ['']+[str(i) for i in range(evm.shape[0])]+['']
    ax2.set_xticks(np.arange(0, evm.shape[0]+2))
    ax2.set_xticklabels(x2_tick_labels, fontsize=6)
    # ax2.set_xlabel('State')

    # ax.set_title('Gain')
    max_val = abs(max(np.min(evm_to_plot[:]), np.max(evm_to_plot[:]), key=abs))
    ax.set_ylim((-max_val-.5, max_val+.5))
    ax.get_yaxis().set_visible(False)
    
def plot_q_values_maintenance(av_rew, Q_all, Q_true, title: str, save_path: str):
        
    fig = plt.figure(dpi=160, figsize=(16, 10))
    ax  = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    x_pos    = np.arange(0, 8)
    num_acts = int(Q_all.shape[1]/8)
    add      = np.linspace(-0.35, 0.35, num_acts)
    
    if num_acts == 4:
        x_ticks  = ['N','S','W','E']*8
        c = ['r', 'g', 'b', 'orange']
        
    else:
        acts = ['N','S','W','E']
        x_ticks = []
        for a1 in range(4):
            for a2 in range(4):
                x_ticks.append(acts[a1]+acts[a2])
        c = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(num_acts)]

    Q_av = np.mean(Q_all, axis=0)
    err  = []
    for i in range(Q_all.shape[1]):
        err.append(np.nanmean((Q_all[:, i]-Q_av[i])**2))

    Q_av = Q_av.reshape(8, num_acts)
        
    w = 0.04
    x_xticks = []
    for s in range(8):
        for a in range(num_acts):
            this_x = x_pos[s]+add[a]
            ax.bar(this_x, Q_av[s, a], width=w, yerr=np.sqrt(err[s*4 + a]), capsize=1, color=c[a], alpha=0.5)
            ax.hlines(Q_true[s, a], this_x-w/2, this_x+w/2)
            # ax.annotate('%s'%Q_true[s, a], (this_x-w/2 +0.02, 0.03), fontsize=5, color='k')
            x_xticks.append(this_x)

    these_lines = Line2D([0], [0], color='k', linewidth=2, linestyle='-', label='True Q-values')
    true_rew   = ax.axhline(av_rew, c='k', linestyle=':',label='Average reward')
    ax.set_xticks(x_xticks)
    ax.set_xticklabels(x_ticks, fontsize=4)

    ax.set_ylim(bottom=0)

    ax2 = ax.twiny()
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(i) for i in range(8)])
    ax2.xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
    ax2.xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom
    ax2.spines['bottom'].set_position(('outward', 18))
    ax2.set_xlim(ax.get_xlim())

    ax.set_title(title)
    ax.legend(handles=[true_rew, these_lines])
    
    plt.savefig(save_path)
    plt.close(fig)
    
def plot_replay_frequency(pos_TD, neg_TD, Q_true, title: str, save_path: str):
    
    num_acts = int(Q_true.shape[1])
    
    Q_true = Q_true.ravel()
    tmp = pos_TD + neg_TD
    max_val = np.nanmax(tmp[:])
    
    fig = plt.figure(dpi=150, figsize=(16, 10))
    ax  = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    x_pos    = np.arange(0, 8)
    x_add    = np.linspace(-0.35, 0.35, num_acts)
    
    if num_acts == 4:
        x_ticks  = ['N','S','W','E']*8
        c = ['r', 'g', 'b', 'orange']
        
    else:
        acts = ['N','S','W','E']
        x_ticks = []
        for a1 in range(4):
            for a2 in range(4):
                x_ticks.append(acts[a1]+acts[a2])
        c = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(num_acts)]

    width = 0.03
    idcs = [i+x_add[j] for i in range(8) for j in range(num_acts)]
    
    # if max_val > 0:
    #     pos_TD /= max_val
    #     neg_TD /= max_val

    p1 = ax.bar(idcs, pos_TD, width, color='r', alpha=0.5)
    p2 = ax.bar(idcs, neg_TD, width, bottom=pos_TD, color='b', alpha=0.5)
    
    for i in range(len(idcs)):
        ax.annotate('%.1f'%Q_true[i], (idcs[i]-width/2 +0.02, 0.03), fontsize=6, color='k')

    ax.set_xticks(idcs)
    ax.set_xticklabels(x_ticks, fontsize=6)
    ax.set_ylim(bottom=0)
    
    ax2 = ax.twiny()
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(i) for i in range(8)])
    ax2.xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
    ax2.xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom
    ax2.spines['bottom'].set_position(('outward', 18))
    ax2.set_xlim(ax.get_xlim())

    ax.legend((p1[0], p2[0]), ('Positive TD replay', 'Negative TD replay'))
    ax.set_title(title)
    
    plt.savefig(save_path)
    plt.close(fig)
    
def plot_performance(y, ylab: str, title: str, save_path: str):
    
    fig = plt.figure(1, dpi=160, figsize=(9, 7))
    ax  = fig.add_axes([.1, .1, .8, .8])
    
    plot_history(ax, range(len((y))), y, ylab, title)
    plt.savefig(save_path)
    
    plt.close(fig)
    
def plot_replay_vs_evm(these_evms, ylab: str, title: str, save_path: str):
    
    fig = plt.figure(1, dpi=160, figsize=(9, 7))
    ax  = fig.add_axes([.1, .1, .8, .8])
    
    x_ticks = ['N','S','W','E']

    vl = sns.violinplot(data=these_evms, scale='count', inner='points', linewidth=0, ax=ax)
    
    pos = [[i]*len(these_evms[i]) for i in range(32)]
    
    for i in range(len(pos)):
        if len(pos[i]) == 0:
            pass
        else:    
            ax.scatter(pos[i], these_evms[i], color='k', alpha=0.1, s=8)

    ax.set_xticks(np.arange(32))
    ax.set_xticklabels(x_ticks*8, fontsize=7)
    ax.set_ylabel(ylab)
    ax.set_title(title)

    ax2 = ax.twiny()
    ax2.set_xticks(np.arange(1.5, 32, 4))
    ax2.set_xticklabels([str(i) for i in range(8)])
    ax2.xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
    ax2.xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom
    ax2.spines['bottom'].set_position(('outward', 18))
    ax2.set_xlim(ax.get_xlim())
    
    plt.savefig(save_path)
    ax.clear()
    
    c = 0
    for s in range(0, 32, 4):
        these_data = these_evms[s:(s+4)]
        vl = sns.violinplot(data=these_data, scale='count', inner='points', linewidth=0, ax=ax)
        
        pos = [[i]*len(these_data[i]) for i in range(4)]
        for i in range(len(pos)):
            if len(pos[i]) == 0:
                pass
            else:
                ax.scatter(pos[i], these_data[i], color='k', alpha=0.3, s=8)
        
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(x_ticks, fontsize=7)
        ax.set_ylabel(ylab)
        ax.set_title((title + ' for state %u'%c))
        plt.savefig(save_path[:-4] + '_state%u.png'%c)
        
        ax.clear()
        c+=1
    plt.close(fig)
    

        