import os
import argparse
import numpy as np
import glob
import pickle
from multiprocessing import Pool

from nuscenes.utils.splits import create_splits_scenes

# basic
from sklearn.cluster import KMeans

# Config
num_clusters = 150
alpha = 1.0
beta = 0.0
gamma = 0.05
decay_rate = 0.95
trim_region = True
trim_rate = 0.1
ft_dim = 96



# A global dictionary storing the variables passed from the initializer.
var_dict = {}

def init_worker(sv_pre, seq_id, prob_files, outfeat_files, curvature_files, sv_info_files):
    var_dict['sv_pre'] = sv_pre
    var_dict['seq_id'] = seq_id
    var_dict['prob_files'] = prob_files
    var_dict['outfeat_files'] = outfeat_files
    var_dict['curvature_files'] = curvature_files
    var_dict['sv_info_files'] = sv_info_files
    
# Calculate region information scores and feats for idth frame
def worker_func(id):
    
    # Load from buffer
    sv_pre = var_dict['sv_pre']
    seq_id = var_dict['seq_id']
    prob_files = var_dict['prob_files']
    outfeat_files = var_dict['outfeat_files']
    curvature_files = var_dict['curvature_files']
    sv_info_files = var_dict['sv_info_files']
    
    print("Processing frame {}_{}".format(seq_id, id))
    
    # Load prob
    prob = np.load(prob_files[id])
    
    # Load outfeat
    outfeat = np.load(outfeat_files[id])
    
    # Load sv info
    with open(sv_info_files[id], 'rb') as f:
        sv_id, sv2point = pickle.load(f)
    
    # Load curvature
    curvature = np.load(curvature_files[id]).astype(np.float32)

    # [Uncertainty Calculation]
    uncertain = np.mean(-prob * np.log2(prob + 1e-12), axis=1)


    # Per-point information score. (Refer to the equation 4 in the main paper.)
    point_score = alpha * uncertain + gamma * curvature

    # Calculate  for each super voxel in the frame
    sv_scores = np.zeros_like(sv_id, dtype=np.float32)
    sv_feats = np.zeros((sv_id.shape[0], ft_dim), dtype=np.float32)
    if not sv_pre:
        sv_pnums = np.zeros_like(sv_id, dtype=int)
    for sv_idx, p_ids in enumerate(sv2point):
        if not sv_pre:
            sv_pnums[sv_idx] = len(p_ids)
        sv_scores[sv_idx] = point_score[p_ids].mean()
        sv_feats[sv_idx] = outfeat[p_ids].mean(0)
    assert sv_feats.shape[1] == ft_dim
    
    if not sv_pre:
        return sv_id, sv_scores, sv_feats, sv_pnums
    else:
        return sv_id, sv_scores, sv_feats

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
    
    save_folder = 'Processing_files/{}/sv_flag/VCCS/{}'.format(args.dataset_name, args.model_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_folder = 'Processing_files/{}/sv_flag/VCCS/{}/ReDAL'.format(args.dataset_name, args.model_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder) 
    save_folder = 'Processing_files/{}/sv_flag/VCCS/{}/ReDAL/{}r'.format(args.dataset_name, args.model_name, args.r_id)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    for seq_id in train_split:
        assert args.r_id >= 1
        if args.r_id == 1:
            flag_files = sorted(glob.glob('Processing_files/{}/sv_flag/VCCS/0r/{}/*.npy'.format(args.dataset_name, seq_id)))
        else:
            flag_files = sorted(glob.glob('Processing_files/{}/sv_flag/VCCS/{}/ReDAL/{}r/{}/*.npy'.format(args.dataset_name, args.model_name, args.r_id - 1, seq_id)))
        
        save_folder = 'Processing_files/{}/sv_flag/VCCS/{}/ReDAL/{}r/{}'.format(args.dataset_name, args.model_name, args.r_id, seq_id)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        # Load
        for f_file in flag_files:
            sv_f = np.load(f_file)
            sv_flags = np.append(sv_flags, sv_f)
            frame_sv_offsets += [frame_sv_offsets[-1] + sv_f.shape[0]]
            save_paths += [save_folder + '/' + f_file.split('/')[-1]]
    
    # Region information scores
    sv_scores = np.zeros_like(sv_flags, dtype=np.float32)
    # Region feats
    sv_feats = np.zeros((sv_flags.shape[0], ft_dim), dtype=np.float32)
    # Pre-processed stats
    sv_pre = False
    if os.path.exists('Processing_files/{}/super_voxel/VCCS/sv_pnums.npy'.format(args.dataset_name)):
        sv_pnums = np.load('Processing_files/{}/super_voxel/VCCS/sv_pnums.npy'.format(args.dataset_name))
        sv_pre = True
    else:
        # contained point num for each super voxel
        sv_pnums = np.zeros_like(sv_flags, dtype=int)
    for idx, seq_id in enumerate(train_split):
        
        # Load prob files
        if args.r_id == 1:
            prob_files = sorted(glob.glob('Processing_files/{}/prob_map/{}/fr/0r/{}/*.npy'.format(args.dataset_name, args.model_name, seq_id)))
        else:
            prob_files = sorted(glob.glob('Processing_files/{}/prob_map/{}/sv/ReDAL/{}r/{}/*.npy'.format(args.dataset_name, args.model_name, args.r_id - 1, seq_id)))
                    
        # Load outfeat files
        if args.r_id == 1:
            outfeat_files = sorted(glob.glob('Processing_files/{}/outfeat/{}/fr/0r/{}/*.npy'.format(args.dataset_name, args.model_name, seq_id)))
        else:
            outfeat_files = sorted(glob.glob('Processing_files/{}/outfeat/{}/sv/ReDAL/{}r/{}/*.npy'.format(args.dataset_name, args.model_name, args.r_id - 1, seq_id)))        
        
        # Load curvature files
        curvature_files = sorted(glob.glob('Processing_files/{}/boundary/{}/*.npy'.format(args.dataset_name, seq_id)))
        
        # Load sv info files
        sv_info_files = sorted(glob.glob('Processing_files/{}/super_voxel/VCCS/{}/*.pickle'.format(args.dataset_name, seq_id)))
        
        # Calculate information score and feat for each super voxel   
        # multi-processing
        assert len(prob_files) == len(outfeat_files)
        assert len(prob_files) == len(sv_info_files)
        ids = np.arange(len(prob_files))
        with Pool(processes=24, initializer=init_worker, initargs=(sv_pre, seq_id, prob_files, outfeat_files, curvature_files, sv_info_files)) as pool:
            re_values = pool.map(worker_func, ids)  

        if sv_pre:
            for sv_id_frame, sv_scores_frame, sv_feats_frame in re_values:
                sv_scores[sv_id_frame] = sv_scores_frame
                sv_feats[sv_id_frame] = sv_feats_frame
        else:
            for sv_id_frame, sv_scores_frame, sv_feats_frame, sv_pnums_frame in re_values:
                sv_scores[sv_id_frame] = sv_scores_frame
                sv_feats[sv_id_frame] = sv_feats_frame
                sv_pnums[sv_id_frame] = sv_pnums_frame
    
    if not sv_pre:
        np.save('Processing_files/{}/super_voxel/VCCS/sv_pnums.npy'.format(args.dataset_name), sv_pnums)
        
    # The core implementation of our diversity-aware selection algorithm.
    # importance_reweight
    sv_flags = sv_flags.astype(int)
    unlabeled_ids = np.where(sv_flags == 0)[0]
    unlabeled_scores = sv_scores[unlabeled_ids]
    unlabeled_feats = sv_feats[unlabeled_ids]
    
    # sorted (first time)
    sorted_ids = np.argsort(unlabeled_scores)[::-1]
    unlabeled_ids_sorted = unlabeled_ids[sorted_ids]
    unlabeled_scores_sorted = unlabeled_scores[sorted_ids]
    unlabeled_feats_sorted = unlabeled_feats[sorted_ids]

    if trim_region is True:
        N = unlabeled_feats_sorted.shape[0] * trim_rate
        N = int(N)
        unlabeled_feats_sorted = unlabeled_feats_sorted[:N]
        unlabeled_scores_sorted = unlabeled_scores_sorted[:N]
        unlabeled_ids_sorted = unlabeled_ids_sorted[:N]
    
    # clustering
    m = KMeans(n_clusters=num_clusters, random_state=0)
    m.fit(unlabeled_feats_sorted)
    clusters = m.labels_  

    # importance re-weighting
    N = unlabeled_feats_sorted.shape[0]
    importance_arr = [1 for _ in range(num_clusters)]
    for i in range(N):
        cluster_i = clusters[i]
        cluster_importance = importance_arr[cluster_i]
        unlabeled_scores_sorted[i] *= cluster_importance
        importance_arr[cluster_i] *= decay_rate
        
    # sorted (second time)
    sorted_ids = np.argsort(unlabeled_scores_sorted)[::-1]
    unlabeled_ids_sorted = unlabeled_ids_sorted[sorted_ids]
    
    point_limit = round(0.01 * train_point_num)
    for sv_id in unlabeled_ids_sorted:
        point_limit -= sv_pnums[sv_id]
        if point_limit < 0:
            break
        # update
        sv_flags[sv_id] = 1
    
    # Save
    for idx in range(len(frame_sv_offsets) - 1):
        sv_f = sv_flags[frame_sv_offsets[idx] : frame_sv_offsets[idx + 1]]
        np.save(save_paths[idx], sv_f)