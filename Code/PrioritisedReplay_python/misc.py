import numpy as np
import seaborn as sns
sns.set_palette("husl")
cm = sns.light_palette("green")
import matplotlib
import matplotlib.patches as patches
matplotlib.use('Qt5Agg')
# from matplotlib import animation
import matplotlib.pyplot as plt
import time

# Supplementary funcitons 
def tic():
    return time.time()

def toc(a):
    return time.time() - a

def check_coordinates(f):
    '''Do not change co-ordinates if we decide to move into a wall'''
    def check_current_state(s, a, rew_params, maze):
        side_i = maze.shape[0]
        side_j = maze.shape[1]
        i_coord = s[0]
        j_coord = s[1]
        # do not transition outside the maze
        cond1 = (i_coord == 0 and a == 0)
        cond2 = (i_coord == (side_i-1) and a == 1)
        cond3 = (j_coord == 0 and a == 2)
        cond4 = (j_coord == (side_j-1) and a == 3)
        if cond1 or cond2 or cond3 or cond4:
            return s, 0
        else:
            si, rew = f(s, a, rew_params, maze)
            # do not transition into walls
            if maze[si[0], si[1]] == 1:
                return s, 0
            else:
                return si, rew
    return check_current_state

@check_coordinates
def get_new_state(s, a, r_params, maze):
    '''Get new state from current state and action'''
    i_coord = s[0]
    j_coord = s[1]
    si = []
    
    if a == 0: # up
        si = [i_coord-1, j_coord]
    elif a == 1: # down
        si = [i_coord+1, j_coord]
    elif a == 2: # left
        si = [i_coord, j_coord-1]
    else:      # right
        si = [i_coord, j_coord+1]

    if np.array_equal(r_params['loc'], si):
        r = r_params['val'] + np.random.randn()*r_params['std']
        r = r * (r > 0)
    else:
        r = 0
    
    return si, r

def init_maze(side_i, side_j, obst: list):
    '''Initialise the maze environment'''
    maze   = np.zeros((side_i, side_j))
    for i in range(len(obst)):
        maze[obst[i][0], obst[i][1]] = 1
    return maze

def idcs2state(idcs: list, maze):
    '''Convert state idcs to state id'''
    si = idcs[0]
    sj = idcs[1]
    side_j = maze.shape[1]
    
    return si*side_j + sj

def state2idcs(s: int, maze):
    '''Convert state id to state idcs'''
    side_j = maze.shape[1]
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

def compute_gain(Q_table, plan_exp, gamma, alpha):
    '''Compute gain term for each experience'''
    Q = Q_table.copy()
    num_exp = len(plan_exp)
    gain    = [[] for i in range(num_exp)]
    gain_sa = np.zeros_like(Q)

    for i in range(num_exp):
        curr_exp = plan_exp[i]
        
        for j in range(curr_exp.shape[0]):
            q_vals = Q[int(curr_exp[j, 0]), :].copy()
            
            # Policy before backup
            probs_pre = policy_choose(q_vals, 'softmax', temperature=5)
            
            # Next state value
            s1_val = np.amax(Q[int(curr_exp[-1, 3]), :])
            a_taken  = int(curr_exp[j, 1])
            steps_to_end = curr_exp.shape[0]-j
            
            r = np.dot(gamma ** np.arange(steps_to_end), curr_exp[j:, 2])
            Q_target = r + (gamma**steps_to_end)*s1_val
            
            q_vals[a_taken] += alpha*(Q_target-q_vals[a_taken])
            
            probs_post = policy_choose(q_vals, 'softmax', temperature=5)
            
            # Calculate gain
            EV_pre = np.sum(probs_pre*q_vals)
            EV_post = np.sum(probs_post*q_vals)
            
            gain[i].append(EV_post - EV_pre)
            gain_sa[int(curr_exp[j, 0]), int(curr_exp[j, 1])] = np.amax([gain_sa[int(curr_exp[j, 0]), int(curr_exp[j, 1])], gain[i][j]])

    return gain, gain_sa

def compute_need(s, T_matrix, plan_exp, gamma):
    '''Compute need term for each experience'''
    T = T_matrix.copy()
    num_exp = len(plan_exp)
    need    = [[] for i in range(num_exp)]
    
    # Calculate the Successor Representation
    SR = np.linalg.inv(np.identity(T.shape[0]) - gamma*T)
    SRi = SR[s, :]
    SR_or_SD = SRi
    
    for i in range(num_exp):
        curr_exp = plan_exp[i]
        for j in range(curr_exp.shape[0]):
            need[i].append(SR_or_SD[int(curr_exp[j, 0])])
    return need, SR_or_SD

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

# def plot_trajectory(traj, maze, q):

    # cm = sns.light_palette("green")
    
    # fig = plt.figure(figsize=(8, 6))
    # ax = plt.axes((0, 0, 1, 1))
    
    # q = np.reshape(np.amax(q, axis=1), (maze.shape[0], maze.shape[1]))
    
    # def animate(i):

    #     ax.clear()
    #     im = sns.heatmap(q, linewidths=1, linecolor='k', cmap=cm, 
    #                      cbar=False, square=True, yticklabels=False, 
    #                      xticklabels=False)

    #     rect1 = patches.Rectangle((2,1),1,3,linewidth=1,edgecolor='k',facecolor='k')
    #     rect2 = patches.Rectangle((5,4),1,1,linewidth=1,edgecolor='k',facecolor='k')
    #     rect3 = patches.Rectangle((7,0),1,3,linewidth=1,edgecolor='k',facecolor='k')

    #     ax.add_patch(rect1)
    #     ax.add_patch(rect2)
    #     ax.add_patch(rect3)
        
    #     y = traj[i][0]+0.5
    #     x = traj[i][1]+0.5
    #     sc = ax.scatter(x, y, c='blue')

    #     return im, sc

    # anim = animation.FuncAnimation(fig, animate, frames=len(traj), interval=100, blit=True)
    # plt.show(block=True)
    
def plot_each_step(ax, idcs, maze, Q_table, title: str, halt: float, idcs_plan=None):
    '''Plot trajectory and state values'''
    q = Q_table.copy()
    ax.clear()
    q = np.reshape(np.amax(q, axis=1), (maze.shape[0], maze.shape[1]))
    annot = q
    annot = np.round(annot, 3)
    annot = annot.astype('str')
    annot[0, 8] = 'Goal'
    im = sns.heatmap(q, linewidths=1, linecolor='k', cmap=cm, 
                         cbar=False, vmin=0, vmax=1.3, square=True, yticklabels=False, 
                         xticklabels=False, annot=annot, fmt='', annot_kws={'ha': 'center', 'va': 'bottom'})

    for t in ax.texts:
        trans = t.get_transform()
        offs = matplotlib.transforms.ScaledTranslation(0, 0.46,
                        matplotlib.transforms.IdentityTransform())
        t.set_transform( offs + trans )
    
    rect1 = patches.Rectangle((2,1),1,3,linewidth=1,edgecolor='k',facecolor='k')
    rect2 = patches.Rectangle((5,4),1,1,linewidth=1,edgecolor='k',facecolor='k')
    rect3 = patches.Rectangle((7,0),1,3,linewidth=1,edgecolor='k',facecolor='k')
    
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    
    y = idcs[0]+0.5
    x = idcs[1]+0.5
    sc   = ax.scatter(x, y, s=100, c='blue')
    goal = ax.scatter(8.49, 0.49, s=1000, c='crimson', marker=r'$\clubsuit$', alpha=0.7)
    if idcs_plan:
        yp = idcs_plan[0][0]+0.5
        xp = idcs_plan[0][1]+0.5
        ap = idcs_plan[1]
        if ap == 0: # up
            dx = 0
            dy = -0.2
        elif ap == 1: # down
            dx = 0
            dy = 0.2
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
    if title == 'Planning':
        plt.title(title, fontsize=20, color='red', alpha=0.6)
    elif title == 'Exploring':
        plt.title(title, fontsize=20, color='blue')
    plt.show()
    plt.pause(halt)
    
    