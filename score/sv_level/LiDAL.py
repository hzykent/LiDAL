import os
import argparse
import numpy as np
import glob
import pickle
from multiprocessing import Pool
from scipy.special import kl_div
from scipy.stats import entropy

from nuscenes.utils.splits import create_splits_scenes

####################################### Score ###############################################

# A global dictionary storing the variables passed from the initializer.
var_dict = {}

def init_worker(sv_pre, nei_num, dis_thresh, seq_id, prob_files, kdtree_files, sv_info_files):
    var_dict['sv_pre'] = sv_pre
    var_dict['nei_num'] = nei_num
    var_dict['dis_thresh'] = dis_thresh
    var_dict['seq_id'] = seq_id
    var_dict['prob_files'] = prob_files
    var_dict['kdtree_files'] = kdtree_files
    var_dict['sv_info_files'] = sv_info_files

# Calculate inter-scan divergence and entropy for idth frame
def worker_func(id):
    
    # Load from buffer
    sv_pre = var_dict['sv_pre']
    nei_num = var_dict['nei_num']
    dis_thresh = var_dict['dis_thresh']
    seq_id = var_dict['seq_id']
    prob_files = var_dict['prob_files']
    kdtree_files = var_dict['kdtree_files']
    sv_info_files = var_dict['sv_info_files']
    
    print("Processing frame {}_{}".format(seq_id, id))
    
    # Neighboring ids
    nei_ids = [(id - offset - 1) if (id - offset - 1) >= 0 else (int(nei_num / 2) + offset + 1) for offset in np.arange(int(nei_num / 2))]
    nei_ids += [(id + offset + 1) if (id + offset + 1) <= (len(prob_files) - 1) else (len(prob_files) - 2 - int(nei_num / 2) - offset) for offset in np.arange(int(nei_num / 2))]

    # Load prob_maps
    query_prob = np.load(prob_files[id])
    nei_probs = []

    for n_id in nei_ids:
        nei_probs += [np.load(prob_files[n_id])]

    # Load kdtrees
    with open(kdtree_files[id], 'rb') as f:
        query_tree = pickle.load(f)
    nei_trees = []
    for n_id in nei_ids:
        with open(kdtree_files[n_id], 'rb') as f:
            nei_trees += [pickle.load(f)]
    
    # For each point in frame_id, find its nearest point in neighboring frames
    query_points = np.asarray(query_tree.data)
    map_count = np.ones(query_prob.shape[0])
    interd_points = np.zeros(query_points.shape[0])
    sum_prob = query_prob.copy()
    epsilon = 0.00001
    for n_prob, n_tree in zip(nei_probs, nei_trees):
        dists, nearest_ids = n_tree.query(query_points, k=1, return_distance=True, dualtree=False, breadth_first=False)
        dists = dists.squeeze()
        nearest_ids = nearest_ids.squeeze()
        match_mask = dists <= dis_thresh
        sum_prob[match_mask] += n_prob[nearest_ids][match_mask]
        interd_points[match_mask] += np.sum(kl_div(query_prob[match_mask] + epsilon, n_prob[nearest_ids][match_mask] + epsilon), axis=1)
        map_count[match_mask] += 1
    
    # Calculate inter-frame entropy
    sum_prob /= np.expand_dims(map_count, 1)
    intere_points = entropy(sum_prob, axis=1)
    
    # Calculate inter-frame divergence
    map_count = map_count - 1
    map_mask = map_count > 0
    interd_points[map_mask] /= map_count[map_mask]
    
    # Load sv info
    with open(sv_info_files[id], 'rb') as f:
        sv_id, sv2point = pickle.load(f)
    
    # Calculate inter-frame divergence and entropy for each super voxel in the frame
    sv_interds = np.zeros_like(sv_id, dtype=np.float32)
    sv_interes = np.zeros_like(sv_id, dtype=np.float32)
    if not sv_pre:
        sv_pnums = np.zeros_like(sv_id, dtype=int)
        sv_centers = np.zeros((len(sv_id), 3), dtype=np.float32)
    for sv_idx, p_ids in enumerate(sv2point):
        if not sv_pre:
            sv_pnums[sv_idx] = len(p_ids)
            sv_centers[sv_idx] = query_points[p_ids].mean(0)
        sv_interds[sv_idx] = interd_points[p_ids].mean()
        sv_interes[sv_idx] = intere_points[p_ids].mean()

    if not sv_pre:
        return sv_id, sv_interds, sv_interes, sv_pnums, sv_centers
    else:
        return sv_id, sv_interds, sv_interes


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = 'Active selection with pseudo labels \
                                    in the super voxel level based on inter-scan divergence and inter-scan entropy')
    parser.add_argument('--dataset_name', type = str, required = True,
                        help = 'name of the used dataset')  
    parser.add_argument('--model_name', type = str, required = True,
                        help = 'name of the trained model providing prob inference')
    parser.add_argument('--r_id', type = int, required = True,
                        help = 'current training r_id')
    args = parser.parse_args()

    # Number of neighboring frames for score calculation
    nei_num = 24
    # Largest distance for matched points
    dis_thresh = 0.1

    if args.dataset_name == 'SK':
        # Sequences
        train_split = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
        # Overall point num
        train_point_num = 2349559532

    if args.dataset_name == 'NU':
        scene_splits = create_splits_scenes()
        train_split = scene_splits['train']
        train_point_num = 976677792
    
    # Since there are no connections between sequences, we process each sequence separately

    # Current labeled dataset
    sv_flags = np.array([])
    frame_sv_offsets = [0]
    save_paths = []
    
    save_folder = 'Processing_files/{}/sv_flag/KMeans/{}'.format(args.dataset_name, args.model_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_folder = 'Processing_files/{}/sv_flag/KMeans/{}/LiDAL'.format(args.dataset_name, args.model_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder) 
    save_folder = 'Processing_files/{}/sv_flag/KMeans/{}/LiDAL/{}r'.format(args.dataset_name, args.model_name, args.r_id)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    for seq_id in train_split:
        assert args.r_id >= 1
        if args.r_id == 1:
            flag_files = sorted(glob.glob('Processing_files/{}/sv_flag/KMeans/0r/{}/*.npy'.format(args.dataset_name, seq_id)))
        else:
            flag_files = sorted(glob.glob('Processing_files/{}/sv_flag/KMeans/{}/LiDAL/{}r/{}/*.npy'.format(args.dataset_name, args.model_name, args.r_id - 1, seq_id)))
        
        save_folder = 'Processing_files/{}/sv_flag/KMeans/{}/LiDAL/{}r/{}'.format(args.dataset_name, args.model_name, args.r_id, seq_id)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        # Load
        for f_file in flag_files:
            sv_f = np.load(f_file)
            sv_flags = np.append(sv_flags, sv_f)
            frame_sv_offsets += [frame_sv_offsets[-1] + sv_f.shape[0]]
            save_paths += [save_folder + '/' + f_file.split('/')[-1]]

    # inter-scan divergence for each super voxel in the dataset
    sv_interds = np.zeros_like(sv_flags, dtype=np.float32)
    # inter-scan entropy for each super voxel in the dataset
    sv_interes = np.zeros_like(sv_flags, dtype=np.float32)
    # Pre-processed stats
    sv_pre = False
    if os.path.exists('Processing_files/{}/super_voxel/KMeans/sv_pnums.npy'.format(args.dataset_name)):
        sv_pnums = np.load('Processing_files/{}/super_voxel/KMeans/sv_pnums.npy'.format(args.dataset_name))
        sv_centers = np.load('Processing_files/{}/super_voxel/KMeans/sv_centers.npy'.format(args.dataset_name))
        sv_pre = True
    else:
        # contained point num for each super voxel
        sv_pnums = np.zeros_like(sv_flags, dtype=int)
        # weight center of each super voxel
        sv_centers = np.zeros((len(sv_flags), 3), dtype=np.float32)
    
    for idx, seq_id in enumerate(train_split):
        
        # Load prob files
        if args.r_id == 1:
            prob_files = sorted(glob.glob('Processing_files/{}/prob_map/{}/fr/0r/{}/*.npy'.format(args.dataset_name, args.model_name, seq_id)))
        else:
            prob_files = sorted(glob.glob('Processing_files/{}/prob_map/{}/sv/LiDAL/{}r/{}/*.npy'.format(args.dataset_name, args.model_name, args.r_id - 1, seq_id)))
                    
        # Load kdtree files
        kdtree_files = sorted(glob.glob('Processing_files/{}/kdtree/{}/*.pickle'.format(args.dataset_name, seq_id)))
        
        # Load sv info files
        sv_info_files = sorted(glob.glob('Processing_files/{}/super_voxel/KMeans/{}/*.pickle'.format(args.dataset_name, seq_id)))
        
        # Calculate inter-frame divergence and entropy for each super voxel   
        # multi-processing
        assert len(prob_files) == len(kdtree_files)
        assert len(kdtree_files) == len(sv_info_files)
        ids = np.arange(len(prob_files))
        with Pool(processes=24, initializer=init_worker, initargs=(sv_pre, nei_num, dis_thresh, seq_id, prob_files, kdtree_files, sv_info_files)) as pool:
            # View divergence of svs
            re_values = pool.map(worker_func, ids)  
        
        if sv_pre:
            for sv_id_frame, sv_interds_frame, sv_interes_frame in re_values:
                sv_interds[sv_id_frame] = sv_interds_frame
                sv_interes[sv_id_frame] = sv_interes_frame
        else:
            for sv_id_frame, sv_interds_frame, sv_interes_frame, sv_pnums_frame, sv_centers_frame in re_values:
                sv_interds[sv_id_frame] = sv_interds_frame
                sv_interes[sv_id_frame] = sv_interes_frame
                sv_pnums[sv_id_frame] = sv_pnums_frame
                # Add a offset to distinguish frames of different sequences
                sv_centers[sv_id_frame] = sv_centers_frame + idx * 1000.0

    if not sv_pre:
        np.save('Processing_files/{}/super_voxel/KMeans/sv_pnums.npy'.format(args.dataset_name), sv_pnums)
        np.save('Processing_files/{}/super_voxel/KMeans/sv_centers.npy'.format(args.dataset_name), sv_centers)


    # AL Strategy : Select super voxels with the largest divergence scores to the label dataset, 
    #               but the distance between the centers of each pair of added super voxels 
    #               must be larger than sv_dis_thresh : 5m
    #               For the set of added super voxels representing the same 3D region, only the one with
    #               the highest inter-frame entropy score stays 
    sv_dis_thresh = 5.0
    # Sort
    sv_flags = sv_flags.astype(int)
    unlabeled_ids = np.where(sv_flags == 0)[0]
    unlabeled_interds = sv_interds[unlabeled_ids]
    sorted_ids = np.argsort(unlabeled_interds)

    added_ids = set()

    # Select sv with the highest inter-frame divergence (1% point limit)
    point_limit = round(0.01 * train_point_num)
    count = 0
    for idx in reversed(sorted_ids):
        print('AL point_limit: {}'.format(point_limit))
        count += 1
        print(count)
        sv_id = unlabeled_ids[idx]
        # Check whether a labeled correspondance exists in labeled set
        sv_c = sv_centers[sv_id]
        flag = True
        for l_sv_id in added_ids:
            l_sv_c = sv_centers[l_sv_id]
            dist = np.sqrt(np.square(sv_c - l_sv_c).sum())
            # If find sv representing the same region, select the one with the higher inter-frame entropy score
            if dist < sv_dis_thresh:
                flag = False
                if sv_interes[l_sv_id] < sv_interes[sv_id]:
                    # Switch
                    sv_flags[sv_id] = 1
                    sv_flags[l_sv_id] = 0
                    added_ids.add(sv_id)
                    added_ids.remove(l_sv_id)
                    point_limit = point_limit + sv_pnums[l_sv_id] - sv_pnums[sv_id]
                break      
        if flag:
            point_limit -= sv_pnums[sv_id]
            if point_limit < 0:
                break
            # update
            sv_flags[sv_id] = 1
            added_ids.add(sv_id)


    # SL Strategy : Select super voxels with the lowest inter-frame divergence scores to the label dataset, 
    #               but the distance between the centers of each pair of selected super voxels 
    #               must be larger than sv_dis_thresh : 5m
    #               For the set of selected super voxels representing the same 3D region, only the one with
    #               the lowest inter-scan entropy score stays    
    # Alternating schedule:
    #               In each AL iteration i, pick top svs that are not already in Pi−1 from the last iteration, to form Pi for the current iteration.    
    # Sort
    unlabeled_ids = np.where(sv_flags == 0)[0]
    unlabeled_interds = sv_interds[unlabeled_ids]
    sorted_ids = np.argsort(unlabeled_interds)
    
    # Reset the pseudo labels
    sv_flags[sv_flags == 2] = 0
    
    added_ids = set()

    # Select sv with the lowest inter-frame divergence (1% point limit)
    point_limit = round(0.01 * train_point_num)
    count = 0
    for idx in sorted_ids:
        print('SL point_limit: {}'.format(point_limit))
        count += 1
        print(count)
        
        if unlabeled_interds[idx] == 0:
            continue
        
        sv_id = unlabeled_ids[idx]
        # Check whether a labeled correspondance exists in labeled set
        sv_c = sv_centers[sv_id]
        flag = True
        for l_sv_id in added_ids:
            l_sv_c = sv_centers[l_sv_id]
            dist = np.sqrt(np.square(sv_c - l_sv_c).sum())
            # If find sv representing the same region, select the one with the lower inter-frame entropy score
            if dist < sv_dis_thresh:
                flag = False
                if sv_interes[l_sv_id] > sv_interes[sv_id]:
                    # Switch
                    sv_flags[sv_id] = 2
                    sv_flags[l_sv_id] = 0
                    added_ids.add(sv_id)
                    added_ids.remove(l_sv_id)
                    point_limit = point_limit + sv_pnums[l_sv_id] - sv_pnums[sv_id]
                break      
        if flag:
            point_limit -= sv_pnums[sv_id]
            if point_limit < 0:
                break
            # update
            sv_flags[sv_id] = 2
            added_ids.add(sv_id)
            
    # Save
    for idx in range(len(frame_sv_offsets) - 1):
        sv_f = sv_flags[frame_sv_offsets[idx] : frame_sv_offsets[idx + 1]]
        np.save(save_paths[idx], sv_f)