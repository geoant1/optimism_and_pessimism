import numpy as np
import os, shutil
from misc import state2idcs, get_new_state, get_files

# World 1
world1 = np.array([[0, 9, 3, 5],
                   [8, 2, 1, 10]], dtype=int)

idcs1  = np.array([[0, 1, 2, 3], 
                   [4, 5, 6, 7]])

Q_true = np.zeros((8,4))
for s in range(8):
    si = state2idcs(s, idcs1)
    for a in range(4):
        s1i, r = get_new_state(si, a, world1, idcs1)
        Q_true[s, a] = r

root_path = os.path.join('/u/gantonov/data/contour/rho_tau')
os.chdir(root_path)

prop_actual_replayed_matrix  = np.zeros((50, 50))
prop_optimal_replayed_matrix = np.zeros((50, 50))

N = 200

dirs1 = next(os.walk('.'))[1]
c1 = 0
for dir1 in dirs1:
    dir1_path = os.path.join(root_path, dir1)
    os.chdir(dir1_path)
    dirs2 = next(os.walk('.'))[1]
    c2 = 0
    for dir2 in dirs2:
        dir2_path = os.path.join(dir1_path, dir2)
        os.chdir(dir2_path)
        all_files_sorted = get_files()
        all_files_sorted = all_files_sorted[N:]
        
        is_move_optimal       = []
        prop_actual_replayed  = []
        prop_optimal_replayed = []
        
        for f in all_files_sorted:
            data = np.load(f)
            move = data['move']
            
            s = move[0]
            a = move[1]
            opt_move_a = np.argmax(Q_true[s, :])
            
            if a == opt_move_a:
                is_move_optimal.append(1)
            else:
                is_move_optimal.append(0)
            
            replay_backups = data['replay_backups']
            num_backups = replay_backups.shape[0]
            
            pa = 0
            po = 0
            
            if num_backups > 0:
                for i in range(num_backups):
                    this_backup = replay_backups[i, :]
                    sr = int(this_backup[0])
                    ar = int(this_backup[1])
                    opt_move_r = np.argmax(Q_true[sr, :])
                    
                    if (s == sr) and (a == ar):
                        pa += 1
                    if (ar == opt_move_r):
                        po += 1
                        
                prop_actual_replayed.append(pa/num_backups)
                prop_optimal_replayed.append(po/num_backups)
            else:
                prop_actual_replayed.append(0)
                prop_optimal_replayed.append(0)    

        if not os.path.isdir('Analysis'):
            os.mkdir('Analysis')
        else:
            shutil.rmtree('Analysis')
            os.mkdir('Analysis')
            
        np.save('./Analysis/prop_actual_replayed.npy',  np.array(prop_actual_replayed))
        np.save('./Analysis/prop_optimal_replayed.npy', np.array(prop_optimal_replayed))
        np.save('./Analysis/is_move_optimal', np.array(is_move_optimal))
        
        prop_actual_replayed_matrix[c1, c2]  = np.mean(prop_actual_replayed)
        prop_optimal_replayed_matrix[c1, c2] = np.mean(prop_optimal_replayed)
        c2 += 1
    c1 += 1
    
np.save(os.path.join(root_path, 'prop_actual_replayed_matrix.npy'), prop_actual_replayed_matrix)
np.save(os.path.join(root_path, 'prop_optimal_replayed_matrix.npy'), prop_optimal_replayed_matrix)