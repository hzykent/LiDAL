import os
import glob
import numpy as np
import multiprocessing as mp
from functools import partial
from k_means_constrained import KMeansConstrained
import pickle

from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import create_splits_scenes


def process_frame(id, train_scenes, nusc):
    
    scene = train_scenes[id]
    scene_name = scene['name']
    print('Processing ' + scene_name)
    sample_token = scene['first_sample_token']
    count = 0
    while(sample_token):
        sample = nusc.get('sample', sample_token)
        sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        pc = LidarPointCloud.from_file('nuScenes/' + sample_data['filename'])  
        pc = pc.points.transpose()[:, :3]
        
        # Constrained KMeans Clustering
        clf = KMeansConstrained(n_clusters = 20, size_min = int(pc.shape[0] / 20 * 0.95), size_max = int(pc.shape[0] / 20 * 1.05), n_init=1, max_iter=1, random_state=0)
        clf.fit_predict(pc)

        # Save sv label
        frame_id_str = str(count)
        while len(frame_id_str) < 6:
            frame_id_str = '0' + frame_id_str
        label_path = 'Processing_files/NU/super_voxel/KMeans/' + scene_name + '/' + frame_id_str + '.npy'
        np.save(label_path, clf.labels_)

        sample_token = sample['next']
        count += 1    
    

if __name__ == "__main__":

    if not os.path.exists('Processing_files/NU'):
        os.makedirs('Processing_files/NU')
    if not os.path.exists('Processing_files/NU/super_voxel'):
        os.makedirs('Processing_files/NU/super_voxel')
    if not os.path.exists('Processing_files/NU/super_voxel/KMeans'):
        os.makedirs('Processing_files/NU/super_voxel/KMeans')

    nusc = NuScenes(version='v1.0-trainval', dataroot='nuScenes', verbose=True)
    scene_splits = create_splits_scenes()
    train_split = scene_splits['train']
    val_split = scene_splits['val']

    train_scenes = []

    for scene in nusc.scene:

        scene_name = scene['name']
        
        if scene_name in train_split:

            if not os.path.exists('Processing_files/NU/super_voxel/KMeans/' + scene_name):
                os.makedirs('Processing_files/NU/super_voxel/KMeans/' + scene_name)

            train_scenes += [scene]
    
    # Calculate super voxels
    # multi-processing
    pf_pool = mp.Pool(processes=12)
    process_frame_p = partial(process_frame, train_scenes=train_scenes, nusc=nusc)
    ids = np.arange(len(train_scenes))
    pf_pool.map(process_frame_p, ids)
    pf_pool.close()
    pf_pool.join()
    
    # Extract infos for super voxels
    sv_files = []
    for scene in train_scenes:
        scene_name = scene['name']
        sv_files += sorted(glob.glob('Processing_files/NU/super_voxel/KMeans/' + scene_name + '/*.npy'))

    sv_id_count = 0
    id2sv = []
    for file_path in sv_files:
        
        # Load super voxel labels
        sv_label = np.load(file_path)
        
        # Calculate sv2point
        sv2point = []
        for sv_l in np.unique(sv_label):
            p_ids = np.where(sv_label == sv_l)[0]
            sv2point += [p_ids]
        
        # sv_id
        sv_id = np.arange(len(sv2point)) + sv_id_count
        sv_id_count += len(sv2point)
        
        # Save
        save_path = file_path[:-3] + 'pickle'
        with open(save_path, 'wb') as fjson:
            pickle.dump((sv_id, sv2point), fjson) 
        
        # Construct mapping from sv_id to super voxels    
        id2sv += [(file_path.split('/')[-2], file_path.split('/')[-1][:-4], id) for id in np.arange(len(sv2point))]

    with open('Processing_files/NU/super_voxel/KMeans/id2sv.pickle', 'wb') as fjson:
        pickle.dump(id2sv, fjson)