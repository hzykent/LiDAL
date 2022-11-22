import os
import argparse
import numpy as np
import glob
from nuscenes.utils.splits import create_splits_scenes

from sklearn.metrics import pairwise_distances

####################################### Score ###############################################

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = 'Active selection in the frame level based on core set selection')
    parser.add_argument('--dataset_name', type = str, required = True,
                        help = 'name of the used dataset')  
    parser.add_argument('--model_name', type = str, required = True,
                        help = 'name of the trained model providing prob inference')
    parser.add_argument('--r_id', type = int, required = True,
                        help = 'current training r_id')
    args = parser.parse_args()

    if args.dataset_name == 'SK':
        # Sequences
        train_split = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
    if args.dataset_name == 'NU':
        scene_splits = create_splits_scenes()
        train_split = scene_splits['train']

    # Since there are no connections between sequences, we process each sequence separately

    # Current labeled dataset
    all_frame_flag = np.array([], dtype=bool)
    seq_offsets = [0]
    
    save_folder = 'Processing_files/{}/frame_flag/{}'.format(args.dataset_name, args.model_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_folder = 'Processing_files/{}/frame_flag/{}/CSET'.format(args.dataset_name, args.model_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder) 
    save_folder = 'Processing_files/{}/frame_flag/{}/CSET/{}r'.format(args.dataset_name, args.model_name, args.r_id)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for seq_id in train_split:
        assert args.r_id >= 1
        if args.r_id == 1:
            frame_flag = np.load('Processing_files/{}/frame_flag/0r/{}.npy'.format(args.dataset_name, seq_id))
        else:
            frame_flag = np.load('Processing_files/{}/frame_flag/{}/CSET/{}r/{}.npy'.format(args.dataset_name, args.model_name, args.r_id - 1, seq_id))
        all_frame_flag = np.append(all_frame_flag, frame_flag)
        seq_offsets.append(seq_offsets[-1] + frame_flag.shape[0])
    
    # Representative feature for each frame in the dataset
    all_feats = []

    for seq_id in train_split:
        
        # Load outfeat files
        if args.r_id == 1:
            outfeat_files = sorted(glob.glob('Processing_files/{}/outfeat/{}/fr/0r/{}/*.npy'.format(args.dataset_name, args.model_name, seq_id)))
        else:
            outfeat_files = sorted(glob.glob('Processing_files/{}/outfeat/{}/fr/CSET/{}r/{}/*.npy'.format(args.dataset_name, args.model_name, args.r_id - 1, seq_id)))
                
        for outfeat_file in outfeat_files:
            outfeat = np.load(outfeat_file)
            frame_feat = outfeat.mean(0)
        
            # Save
            all_feats += [frame_feat.reshape(1, -1)]
        
    all_feats = np.concatenate(all_feats, 0)
    
    labeled_ids = np.where(all_frame_flag == True)[0]
    # Iteratively select the furthest feat from current labeled set
    # Init
    cluster_centers = all_feats[labeled_ids]
    dist = pairwise_distances(all_feats, cluster_centers, metric='euclidean')
    min_dist = np.min(dist, axis=1).reshape(-1, 1)
    
    selected_ids = set(labeled_ids)
    num_add = round(0.01 * all_frame_flag.shape[0])
    for _ in range(num_add):
        ind = np.argmax(min_dist)
        # New examples should not be in already selected since those points
        # should have min_distance of zero to a cluster center.
        assert ind not in selected_ids
        # Update min_dist
        dist = pairwise_distances(all_feats, all_feats[ind].reshape(1, -1), metric='euclidean')
        min_dist = np.minimum(min_dist, dist)
        all_frame_flag[ind] = True
        selected_ids.add(ind)
    
    # Save
    for idx, seq_id in enumerate(train_split):
        frame_flag = all_frame_flag[seq_offsets[idx] : seq_offsets[idx + 1]]
        np.save('Processing_files/{}/frame_flag/{}/CSET/{}r/{}.npy'.format(args.dataset_name, args.model_name, args.r_id, seq_id), frame_flag)

    

