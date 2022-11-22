import os
import numpy as np
import argparse
import glob
import pickle

from nuscenes.utils.splits import create_splits_scenes


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = 'Random selection in the frame level')
    parser.add_argument('--dataset_name', type = str, required = True,
                        help = 'name of the used dataset')  
    parser.add_argument('--r_id', type = int, required = True,
                        help = 'current training r_id')

    args = parser.parse_args()

    if args.dataset_name == 'SK':
        # Sequences
        train_split = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
    if args.dataset_name == 'NU':
        scene_splits = create_splits_scenes()
        train_split = scene_splits['train']

    # Load previous labeled frames and randomly add 1%
    frame_flag_all = np.array([])
    offsets = [0]
    for i_folder in train_split:
        if args.r_id == 1:
            frame_flag = np.load('Processing_files/{}/frame_flag/0r/{}.npy'.format(args.dataset_name, i_folder))
        else:
            frame_flag = np.load('Processing_files/{}/frame_flag/RAND/{}r/{}.npy'.format(args.dataset_name, args.r_id - 1, i_folder))
        offsets += [offsets[-1] + len(frame_flag)]
        frame_flag_all = np.append(frame_flag_all, frame_flag)
        
    # Randomly add 1% fully labeled frames
    unlabeled = np.where(frame_flag_all == False)[0] 
    selected = np.random.choice(unlabeled, int(np.round(0.01 * len(frame_flag_all))))
    # Update frame_flag
    frame_flag_all[selected] = True   
    
    # Save 
    if not os.path.exists('Processing_files/{}/frame_flag/RAND'.format(args.dataset_name)):
        os.makedirs('Processing_files/{}/frame_flag/RAND'.format(args.dataset_name))
    if not os.path.exists('Processing_files/{}/frame_flag/RAND/{}r'.format(args.dataset_name, args.r_id)):
        os.makedirs('Processing_files/{}/frame_flag/RAND/{}r'.format(args.dataset_name, args.r_id))    
    for idx, i_folder in enumerate(train_split):
        np.save('Processing_files/{}/frame_flag/RAND/{}r/{}.npy'.format(args.dataset_name, args.r_id, i_folder), frame_flag_all[offsets[idx] : offsets[idx + 1]])
    