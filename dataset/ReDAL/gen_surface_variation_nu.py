import os
import pickle
import numpy as np
import multiprocessing as mp
from functools import partial

from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

import pandas as pd
from pyntcloud import PyntCloud

def boundary_extractor(xyz, rgb, threshold=None):
    # define hyperparameters
    k_n = 50
    clmns = ['x', 'y', 'z', 'red', 'green', 'blue']
    pcd_pd = pd.DataFrame(data=np.concatenate([xyz, rgb], axis=1), columns=clmns)
    pcd1 = PyntCloud(pcd_pd)

    # find neighbors
    kdtree_id = pcd1.add_structure("kdtree")
    k_neighbors = pcd1.get_neighbors(k=k_n, kdtree=kdtree_id)

    # calculate eigenvalues
    pcd1.add_scalar_field("eigen_values", k_neighbors=k_neighbors)

    e1 = pcd1.points['e3('+str(k_n+1)+')'].values
    e2 = pcd1.points['e2('+str(k_n+1)+')'].values
    e3 = pcd1.points['e1('+str(k_n+1)+')'].values

    sum_eg = np.add(np.add(e1, e2), e3)
    sigma = np.divide(e1, sum_eg)
    sigma_value = sigma
    if threshold is not None:
        sigma_value[sigma_value > threshold] = threshold

    return sigma_value


def process_frame(id, train_scenes, nusc):
    
    scene = train_scenes[id]
    scene_name = scene['name']
    print('Processing ' + scene_name)
    sample_token = scene['first_sample_token']
    count = 0
    while(sample_token):
        sample = nusc.get('sample', sample_token)
        sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])

        xyz = np.fromfile('nuScenes/' + sample_data['filename'], dtype=np.float32).reshape(-1, 5)[:, :3]
        rgb = np.zeros_like(xyz)
        edge_npy = boundary_extractor(xyz, rgb, threshold=0.1)
        
        # Save
        frame_id_str = str(count)
        while len(frame_id_str) < 6:
            frame_id_str = '0' + frame_id_str
        boundary_file = 'Processing_files/NU/boundary/' + scene_name + '/' + frame_id_str + '.npy'
        np.save(boundary_file, edge_npy)     
        
        sample_token = sample['next']
        count += 1    
    

if __name__ == "__main__":

    if not os.path.exists('Processing_files/NU/boundary'):
        os.makedirs('Processing_files/NU/boundary')

    nusc = NuScenes(version='v1.0-trainval', dataroot='nuScenes', verbose=True)
    scene_splits = create_splits_scenes()
    train_split = scene_splits['train']
    val_split = scene_splits['val']

    train_scenes = []

    for scene in nusc.scene:

        scene_name = scene['name']
        
        if scene_name in train_split:

            if not os.path.exists('Processing_files/NU/boundary/' + scene_name):
                os.makedirs('Processing_files/NU/boundary/' + scene_name)

            train_scenes += [scene]
        
    # multi-processing
    pf_pool = mp.Pool(processes=12)
    process_frame_p = partial(process_frame, train_scenes=train_scenes, nusc=nusc)
    ids = np.arange(len(train_scenes))
    pf_pool.map(process_frame_p, ids)
    pf_pool.close()
    pf_pool.join()



