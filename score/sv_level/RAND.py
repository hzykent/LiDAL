import os
import numpy as np
import argparse
import glob
import pickle

from nuscenes.utils.splits import create_splits_scenes


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = 'Random selection in the super voxel level')
    parser.add_argument('--dataset_name', type = str, required = True,
                        help = 'name of the used dataset')  
    parser.add_argument('--r_id', type = int, required = True,
                        help = 'current training r_id')

    args = parser.parse_args()

    if args.dataset_name == 'SK':
        # Sequences
        train_split = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
        train_point_num = 2349559532
    if args.dataset_name == 'NU':
        scene_splits = create_splits_scenes()
        train_split = scene_splits['train']
        train_point_num = 976677792

    if not os.path.exists('Processing_files/{}/sv_flag/KMeans/RAND'.format(args.dataset_name)):
        os.makedirs('Processing_files/{}/sv_flag/KMeans/RAND'.format(args.dataset_name))
    if not os.path.exists('Processing_files/{}/sv_flag/KMeans/RAND/{}r'.format(args.dataset_name, args.r_id)):
        os.makedirs('Processing_files/{}/sv_flag/KMeans/RAND/{}r'.format(args.dataset_name, args.r_id))

    # Load id2sv
    with open('Processing_files/{}/super_voxel/KMeans/id2sv.pickle'.format(args.dataset_name), 'rb') as f:
        id2sv = pickle.load(f)

    # Load previous labeled super voxels
    all_sv_flags = np.array([])
    sv_offsets = [0]
    saving_names = []
    for i_folder in train_split:
        if not os.path.exists('Processing_files/{}/sv_flag/KMeans/RAND/{}r/{}'.format(args.dataset_name, args.r_id, i_folder)):
            os.makedirs('Processing_files/{}/sv_flag/KMeans/RAND/{}r/{}'.format(args.dataset_name, args.r_id, i_folder))
        if args.r_id == 1:
            sv_flags = sorted(glob.glob('Processing_files/{}/sv_flag/KMeans/0r/{}/*.npy'.format(args.dataset_name, i_folder)))
        else:
            sv_flags = sorted(glob.glob('Processing_files/{}/sv_flag/KMeans/RAND/{}r/{}/*.npy'.format(args.dataset_name, args.r_id - 1, i_folder)))
        # Load flags
        for sv_flag in sv_flags:
            flag = np.load(sv_flag)
            all_sv_flags = np.append(all_sv_flags, flag)
            sv_offsets += [sv_offsets[-1] + len(flag)]
            saving_names += ['Processing_files/{}/sv_flag/KMeans/RAND/{}r/{}/'.format(args.dataset_name, args.r_id, i_folder) + sv_flag.split('/')[-1]]
            
    # Add 1% more
    point_limit = int(np.round(0.01 * train_point_num))
    for idx in np.random.choice(len(id2sv), len(id2sv)):
        if all_sv_flags[idx] == False:
            # Load sv_info
            seq_id, frame_id, sv_id_frame = id2sv[idx]
            with open('Processing_files/{}/super_voxel/KMeans/{}/{}.pickle'.format(args.dataset_name, seq_id, frame_id), 'rb') as f:
                sv_id, sv2point = pickle.load(f)
            assert sv_id[sv_id_frame] == idx
            point_limit -= len(sv2point[sv_id_frame])
            if point_limit < 0:
                break
            all_sv_flags[idx] = True
        
    # Save sv_flag
    for idx, name in enumerate(saving_names):
        sv_flag = all_sv_flags[sv_offsets[idx] : sv_offsets[idx + 1]]
        np.save(name, sv_flag)

    