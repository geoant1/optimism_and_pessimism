import numpy as np
import seaborn as sns
sns.set_palette("husl")
cm = sns.light_palette("green")
import matplotlib
from matplotlib.patches import RegularPolygon, Rectangle, Patch
from matplotlib.collections import PatchCollection
matplotlib.use('Qt5Agg')
# from matplotlib import animation
import matplotlib.pyplot as plt
import time

# Supplementary funcitons 
def tic():
    return time.time()

def toc(a):
    return time.time() - a
    
def get_new_state(s, a, r_params, maze):
    '''Get new state from current state and action'''
    si = s[0]
    sj = s[1]
    
    side_i = maze.shape[0]
    side_j = maze.shape[1]
    s1_idcs = []

    # do not transition outside the maze
    cond1 = (si == 0 and a == 0)
    cond2 = (si == (side_i-1) and a == 1)
    cond3 = (sj == 0 and a == 2)
    cond4 = (sj == (side_j-1) and a == 3)
    if cond1 or cond2 or cond3 or cond4:
        return s, 0
    else:
        if a == 0: # up
            s1_idcs = [si-1, sj]
        elif a == 1: # down
            s1_idcs = [si+1, sj]
        elif a == 2: # left
            s1_idcs = [si, sj-1]
        else:      # right
            s1_idcs = [si, sj+1]

        # do not transition into walls
        if maze[s1_idcs[0], s1_idcs[1]] == 1:
            return s, 0
        
        if np.array_equal(r_params['pos_loc'], s1_idcs):
            r = r_params['pos_val']
        elif np.array_equal(r_params['neg_loc'], s1_idcs):
            r = r_params['neg_val']
        else:
            r = 0
        
        return s1_idcs, r

def idcs2state(idcs: list, maze):
    '''Convert state idcs to state id'''
    side_j = maze.shape[1]
    side_i = maze.shape[0]
    
    si = idcs[0]
    sj = idcs[1]
    
    return si*side_j + sj

def state2idcs(s: int, maze):
    '''Convert state id to state idcs'''
    side_j = maze.shape[1]
    side_i = maze.shape[0]
    
    si = s // side_j
    sj = s % side_j
    
    return [si, sj]
    
def policy_choose(q_values, act_policy, temperature=None, eps=None):
    '''Choose an action'''
    q_vals = q_values.copy()
    if act_policy == 'softmax' and temperature:
        num   = np.exp(q_vals*temperature)
        denom = np.sum(np.exp(q_vals*temperature))
        pol_vals = num / denom
    
    return pol_vals

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

def compute_gain(Q_table, plan_exp, gamma, alpha, temp):
    '''Compute gain term for each experience'''
    Q = Q_table.copy()
    num_exp = len(plan_exp)
    gain    = []

    for i in range(num_exp):
        curr_exp = plan_exp[i, :]
        sn  = int(curr_exp[0])
        an  = int(curr_exp[1])
        rn  = curr_exp[2]
        
        q_vals = Q[sn, :].copy()
        
        # Policy before backup
        probs_pre = policy_choose(q_vals, 'softmax', temperature=temp)
        
        # Next state value
        Q_target = rn
        delta    = Q_target - q_vals[an]
        
        q_vals[an] += alpha*delta
        
        probs_post = policy_choose(q_vals, 'softmax', temperature=temp)
        
        # Calculate gain
        EV_pre = np.sum(probs_pre*q_vals)
        EV_post = np.sum(probs_post*q_vals)
        
        gain.append(EV_post - EV_pre)

    return np.array(gain)

def compute_need(sa: list, H, plan_exp, gamma):
    '''
    Compute need term for each experience
    '''
    s = sa[0]
    a = sa[1]
    num_exp = len(plan_exp)
    need    = []
    
    # Calculate the Successor Representation
    SRi = H[(s*4 + a), :]
    
    for i in range(num_exp):
        curr_exp = plan_exp[i, :]
        sn = int(curr_exp[0])
        an = int(curr_exp[1])
        need.append(SRi[sn*4+an])
    return np.array(need)

def compare(gain: list, bl_gain: float):
    '''Compare experience gain to baseline gain'''
    for i in gain:
        if i > bl_gain:
            return gain
    return bl_gain

def normalise(a):
    m = np.max(a[:])
    assert m > 0, 'Can\'t divide by 0 in normalise()'
    return a / m
    
def plot_each_step(ax, idcs, maze, Q_table, title: str, halt: float, idcs_plan=None):
    '''Plot trajectory and state values'''

    ax.clear()
    x_side = maze.shape[1]
    y_side = maze.shape[0]
    ax.set_xlim(0, x_side)
    ax.set_ylim(0, y_side)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.invert_yaxis()
    # q = np.mean(Q_table, axis=1)
    
    # for_colors = q.copy()
    # colors = np.array([(0,0,1)]*len(for_colors))
    # colors[for_colors >= 0] = (1,0,0)
    # mult = np.reshape(np.repeat(np.abs(for_colors)/np.max(np.abs(for_colors)),3),(len(for_colors),3))
    # colors =  mult*colors
    
    
    patches = []
    q = Q_table.copy()
    max_q = np.max(np.abs(q))
    state = 0
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):            #  x     y   width  height
            this_state    = q[state, :]
            these_colours = np.array([(0,0,1)]*len(this_state))
            these_colours[this_state > 0] = (1, 0, 0)
            mult = np.reshape(np.repeat(np.abs(this_state)/max_q,3),(len(this_state),3))
            colours =  mult*these_colours
            
            for act in range(4):
                if act == 0:
                    patches.append(RegularPolygon((0.5+1*j, 0.2+1*i), 3, radius=0.1, lw=0.5, orientation=np.pi, edgecolor='k', facecolor=colours[act], fill=True))
                elif act == 1:
                    patches.append(RegularPolygon((0.5+1*j, 0.8+1*i), 3, radius=0.1, lw=0.5, orientation=0, edgecolor='k', facecolor=colours[act], fill=True))
                elif act == 2:
                    patches.append(RegularPolygon((0.25+1*j, 0.5+1*i), 3, radius=0.1, lw=0.5, orientation=np.pi/2, edgecolor='k', facecolor=colours[act], fill=True))
                else: 
                    patches.append(RegularPolygon((0.75+1*j, 0.5+1*i), 3, radius=0.1, lw=0.5, orientation=-np.pi/2, edgecolor='k', facecolor=colours[act], fill=True))
                    
                if maze[i, j] == 0:
                    patches.append(Rectangle((j, i),  1,   1, edgecolor='k', alpha=0.5, fill=False)) # centre
                else:
                    patches.append(Rectangle((j, i),  1,   1, edgecolor='k', alpha=1, fill=True, facecolor='k')) # centre
            state += 1
        # patches.append(Rectangle((4, 2),  1,   1, edgecolor='k', alpha=1, fill=True, facecolor='k')) # centre

    collection = PatchCollection(patches, match_original=True)
    ax.add_collection(collection)
    
    y = idcs[0]+0.5
    x = idcs[1]+0.5
    sc   = ax.scatter(x, y, s=100, c='blue')
    ax.scatter(20+0.40, 20+0.5, s=110, c='crimson', marker=r'$\clubsuit$', alpha=0.7)
    if idcs_plan:
        yp = idcs_plan[0][0]+0.5
        xp = idcs_plan[0][1]+0.5
        ap = idcs_plan[1]
        if ap == 0: # up
            dx = 0
            dy = -0.2
        elif ap == 1: # down
            dx = 0
            dy = +0.2
        elif ap == 2: # left
            dx = -0.2
            dy = 0
        else:
            dx = 0.2
            dy = 0
        arrow_params = {'width': 0.01, 'alpha': 0.6, 'ec': 'k', 'fc': 'r', 'color': 'r'}
        pl = ax.scatter(xp, yp, s=100, color='r', alpha=0.6)
        plt.pause(0.2)
        ar = ax.arrow(xp, yp, dx, dy, **arrow_params)
    if title == 'Replay':
        plt.title(title, fontsize=20, color='red', alpha=0.6)
    elif title == 'Exploring':
        plt.title(title, fontsize=20, color='blue')
        
    plt.show()
    plt.pause(0.2)
    
    