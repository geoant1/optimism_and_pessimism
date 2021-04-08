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

root_path = os.path.join('/u/gantonov/data/contour/rho_tau_evm')
os.chdir(root_path)

dirs1 = next(os.walk('.'))[1]
for dir1 in dirs1:
    dir1_path = os.path.join(root_path, dir1)
    os.chdir(dir1_path)
    dirs2 = next(os.walk('.'))[1]

    for dir2 in dirs2:
        dir2_path = os.path.join(dir1_path, dir2)
        os.chdir(dir2_path)
        all_files_sorted = get_files()

        transition_matrix = np.empty(8*4*8)
        replayed_moves = np.empty(8*4)
        moves = []
        
        for f in all_files_sorted:
            data = np.load(f)
            move = data['move']
            T = data['T']
            
            s = move[0]
            a = move[1]
            
            moves.append(s*4+a)
            transition_matrix = np.vstack((transition_matrix, T.flatten()))            

            replay_backups = data['replay_backups']
            num_backups = replay_backups.shape[0]
            
            tmp = np.zeros(8*4)
            
            if num_backups > 0:
                for i in range(num_backups):
                    this_backup = replay_backups[i, :]
                    sr = int(this_backup[0])
                    ar = int(this_backup[1])
                    
                    tmp[sr*4+ar] += 1

            replayed_moves = np.vstack((replayed_moves, tmp))
        if os.path.isdir(os.path.join(dir2_path, 'Analysis')):
            shutil.rmtree(os.path.join(dir2_path, 'Analysis'))
        os.mkdir(os.path.join(dir2_path, 'Analysis'))
        os.chdir(os.path.join(dir2_path, 'Analysis'))

        np.save('moves.npy', np.array(moves))
        np.save('replayed_moves.npy', replayed_moves)
        np.save('transition_matrix.npy', transition_matrix)

