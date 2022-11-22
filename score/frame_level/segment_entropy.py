import os
import argparse
import numpy as np
import glob
import pickle
from multiprocessing import Pool
from nuscenes.utils.splits import create_splits_scenes


####################################### Score ###############################################

# A global dictionary storing the variables passed from the initializer.
var_dict = {}

def init_worker(class_num, seq_id, pred_files, sv_info_files):
    var_dict['class_num'] = class_num
    var_dict['seq_id'] = seq_id
    var_dict['pred_files'] = pred_files
    var_dict['sv_info_files'] = sv_info_files
    

# Calculate segment entropy for idth frame
def worker_func(id):
    
    # Load from buffer
    class_num = var_dict['class_num']
    seq_id = var_dict['seq_id']
    pred_files = var_dict['pred_files']
    sv_info_files = var_dict['sv_info_files']
    
    print("Processing frame {}_{}".format(seq_id, id))
    
    # Load pred
    pred = np.load(pred_files[id])
    
    # Load sv info
    with open(sv_info_files[id], 'rb') as f:
        sv_id, sv2point = pickle.load(f)

    # Calculate segment entropy for each super voxel in the frame
    frame_sege = 0.0
    for sv_idx, p_ids in enumerate(sv2point):
        sv_preds = pred[p_ids]
        sv_sege = 0.0
        for class_id in range(class_num):
            q_c = (sv_preds == class_id).sum() / sv_preds.shape[0]
            sv_sege += -q_c * np.log2(q_c + 1e-12)
        frame_sege += sv_sege * sv_preds.shape[0] / pred.shape[0]
    
    return frame_sege


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = 'Active selection in the frame level based on segment entropy')
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
        class_num = 19
    if args.dataset_name == 'NU':
        scene_splits = create_splits_scenes()
        train_split = scene_splits['train']
        class_num = 16

    # Since there are no connections between sequences, we process each sequence separately

    # Current labeled dataset
    all_frame_flag = np.array([], dtype=bool)
    seq_offsets = [0]
    
    save_folder = 'Processing_files/{}/frame_flag/{}'.format(args.dataset_name, args.model_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_folder = 'Processing_files/{}/frame_flag/{}/SEGENT'.format(args.dataset_name, args.model_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder) 
    save_folder = 'Processing_files/{}/frame_flag/{}/SEGENT/{}r'.format(args.dataset_name, args.model_name, args.r_id)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for seq_id in train_split:
        assert args.r_id >= 1
        if args.r_id == 1:
            frame_flag = np.load('Processing_files/{}/frame_flag/0r/{}.npy'.format(args.dataset_name, seq_id))
        else:
            frame_flag = np.load('Processing_files/{}/frame_flag/{}/SEGENT/{}r/{}.npy'.format(args.dataset_name, args.model_name, args.r_id - 1, seq_id))
        all_frame_flag = np.append(all_frame_flag, frame_flag)
        seq_offsets.append(seq_offsets[-1] + frame_flag.shape[0])
    
    # Segment entropy for each frame in the dataset
    all_seges = np.zeros_like(all_frame_flag, dtype=np.float32)

    for seq_id in train_split:
        
        # Load prediction files
        # Load prob files
        if args.r_id == 1:
            pred_files = sorted(glob.glob('Processing_files/{}/pred/{}/fr/0r/{}/*.npy'.format(args.dataset_name, args.model_name, seq_id)))
        else:
            pred_files = sorted(glob.glob('Processing_files/{}/pred/{}/fr/SEGENT/{}r/{}/*.npy'.format(args.dataset_name, args.model_name, args.r_id - 1, seq_id)))
        
        # Load sv info files
        sv_info_files = sorted(glob.glob('Processing_files/{}/super_voxel/{}/*.pickle'.format(args.dataset_name, seq_id)))
        
        # Calculate segment entropy for each frame   
        # multi-processing
        ids = np.arange(len(pred_files))
        with Pool(processes=24, initializer=init_worker, initargs=(class_num, seq_id, pred_files, sv_info_files)) as pool:
            frame_seges = pool.map(worker_func, ids)  
            frame_seges = np.array(frame_seges, dtype=np.float32)
        
        # Save
        all_seges = np.append(all_seges, frame_seges)
        

    # Select frames with the largest 1% sege in all sequences
    # Sort
    unlabeled_ids = np.where(all_frame_flag == False)[0]
    unlabeled_seges = all_seges[unlabeled_ids]
    # Select frames with the largest 1% segment entropy
    num_add = round(0.01 * all_frame_flag.shape[0])
    selected_ids = np.argpartition(unlabeled_seges, -num_add)[-num_add:]
    selected_ids = unlabeled_ids[selected_ids]
    # Update
    all_frame_flag[selected_ids] = True
    # Save
    for idx, seq_id in enumerate(train_split):
        frame_flag = all_frame_flag[seq_offsets[idx] : seq_offsets[idx + 1]]
        np.save('Processing_files/{}/frame_flag/{}/SEGENT/{}r/{}.npy'.format(args.dataset_name, args.model_name, args.r_id, seq_id), frame_flag)

    

