from itertools import count
import os
import pickle
import numpy as np
import multiprocessing as mp
from functools import partial

from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion
from sklearn.neighbors import KDTree


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
        
        # First step: transform the point cloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Second step: transform to the global frame.
        poserecord = nusc.get('ego_pose', sample_data['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))
        
        # Construct KDTree
        search_tree = KDTree(pc.points.transpose()[:, :3])
        
        # Save KDTree
        frame_id_str = str(count)
        while len(frame_id_str) < 6:
            frame_id_str = '0' + frame_id_str
        KDTree_file = 'Processing_files/NU/kdtree/' + scene_name + '/' + frame_id_str + '.pickle'
        with open(KDTree_file, 'wb') as fjson:
            pickle.dump(search_tree, fjson)

        sample_token = sample['next']
        count += 1    
    

if __name__ == "__main__":

    if not os.path.exists('Processing_files/NU'):
        os.makedirs('Processing_files/NU')
    if not os.path.exists('Processing_files/NU/kdtree'):
        os.makedirs('Processing_files/NU/kdtree')

    nusc = NuScenes(version='v1.0-trainval', dataroot='nuScenes', verbose=True)
    scene_splits = create_splits_scenes()
    train_split = scene_splits['train']
    val_split = scene_splits['val']

    train_scenes = []

    for scene in nusc.scene:

        scene_name = scene['name']
        
        if scene_name in train_split:

            if not os.path.exists('Processing_files/NU/kdtree/' + scene_name):
                os.makedirs('Processing_files/NU/kdtree/' + scene_name)

            train_scenes += [scene]
        
    # multi-processing
    pf_pool = mp.Pool(processes=12)
    process_frame_p = partial(process_frame, train_scenes=train_scenes, nusc=nusc)
    ids = np.arange(len(train_scenes))
    pf_pool.map(process_frame_p, ids)
    pf_pool.close()
    pf_pool.join()



