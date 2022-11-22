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

def init_worker(seq_id, prob_files):
    var_dict['seq_id'] = seq_id
    var_dict['prob_files'] = prob_files
    

# Calculate margin sampling for idth frame
def worker_func(id):
    
    # Load from buffer
    seq_id = var_dict['seq_id']
    prob_files = var_dict['prob_files']
    
    print("Processing frame {}_{}".format(seq_id, id))
    
    # Load prob
    prob = np.load(prob_files[id])
    
    # Calculate margin sampling for point in the frame
    prob = np.sort(prob, axis=-1)
    frame_mar = np.mean(prob[:, -1] - prob[:, -2])
    
    return frame_mar


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = 'Active selection in the frame level based on margin sampling')
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
    save_folder = 'Processing_files/{}/frame_flag/{}/MAR'.format(args.dataset_name, args.model_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder) 
    save_folder = 'Processing_files/{}/frame_flag/{}/MAR/{}r'.format(args.dataset_name, args.model_name, args.r_id)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for seq_id in train_split:
        assert args.r_id >= 1
        if args.r_id == 1:
            frame_flag = np.load('Processing_files/{}/frame_flag/0r/{}.npy'.format(args.dataset_name, seq_id))
        else:
            frame_flag = np.load('Processing_files/{}/frame_flag/{}/MAR/{}r/{}.npy'.format(args.dataset_name, args.model_name, args.r_id - 1, seq_id))
        all_frame_flag = np.append(all_frame_flag, frame_flag)
        seq_offsets.append(seq_offsets[-1] + frame_flag.shape[0])
    
    # Margin sampling for each frame in the dataset
    all_mars = np.zeros_like(all_frame_flag, dtype=np.float32)

    for seq_id in train_split:
        
        # Load prob files
        if args.r_id == 1:
            prob_files = sorted(glob.glob('Processing_files/{}/prob_map/{}/fr/0r/{}/*.npy'.format(args.dataset_name, args.model_name, seq_id)))
        else:
            prob_files = sorted(glob.glob('Processing_files/{}/prob_map/{}/fr/MAR/{}r/{}/*.npy'.format(args.dataset_name, args.model_name, args.r_id - 1, seq_id)))
                
        # Calculate margin sampling for each frame   
        # multi-processing
        ids = np.arange(len(prob_files))
        with Pool(processes=24, initializer=init_worker, initargs=(seq_id, prob_files)) as pool:
            frame_mars = pool.map(worker_func, ids)  
            frame_mars = np.array(frame_mars, dtype=np.float32)
        
        # Save
        all_mars = np.append(all_mars, frame_mars)
        

    # Select frames with the largest 1% mar in all sequences
    # Sort
    unlabeled_ids = np.where(all_frame_flag == False)[0]
    unlabeled_mars = all_mars[unlabeled_ids]
    # Select frames with the largest 1% mar
    num_add = round(0.01 * all_frame_flag.shape[0])
    selected_ids = np.argpartition(unlabeled_mars, -num_add)[-num_add:]
    selected_ids = unlabeled_ids[selected_ids]
    # Update
    all_frame_flag[selected_ids] = True
    # Save
    for idx, seq_id in enumerate(train_split):
        frame_flag = all_frame_flag[seq_offsets[idx] : seq_offsets[idx + 1]]
        np.save('Processing_files/{}/frame_flag/{}/MAR/{}r/{}.npy'.format(args.dataset_name, args.model_name, args.r_id, seq_id), frame_flag)

    

