import os
import glob
import numpy as np
import multiprocessing as mp
import utils.pypcd as pypcd
import pickle
from functools import partial

from nuscenes import NuScenes
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

        file_path = 'nuScenes/' + sample_data['filename']
        coords_p = np.fromfile(file_path, dtype=np.float32).reshape(-1, 5)[:, :3]
        # Save as pcd
        coords_arr = coords_p.ravel().view(dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        pc = pypcd.PointCloud.from_array(coords_arr)
        pcd_path = file_path[:-3] + 'pcd'
        pc.save(pcd_path)

        # Supervoxel clustering
        os.system("./pcl_related/supervoxel_clustering -p " + pcd_path + " -o " + pcd_path)
        pc_label = pypcd.PointCloud.from_path(pcd_path)
        sv_label = np.array(pc_label.pc_data['label'], dtype=np.int)

        # Save sv label
        frame_id_str = str(count)
        while len(frame_id_str) < 6:
            frame_id_str = '0' + frame_id_str
        label_path = 'Processing_files/NU/super_voxel/VCCS/' + scene_name + '/' + frame_id_str + '.npy'
        np.save(label_path, sv_label)
        
        # Clear
        os.system("rm " + pcd_path)   
        
        sample_token = sample['next']
        count += 1  
    

if __name__ == "__main__":

    if not os.path.exists('Processing_files/NU'):
        os.makedirs('Processing_files/NU')
    if not os.path.exists('Processing_files/NU/super_voxel'):
        os.makedirs('Processing_files/NU/super_voxel')
    if not os.path.exists('Processing_files/NU/super_voxel/VCCS'):
        os.makedirs('Processing_files/NU/super_voxel/VCCS')

    nusc = NuScenes(version='v1.0-trainval', dataroot='nuScenes', verbose=True)
    scene_splits = create_splits_scenes()
    train_split = scene_splits['train']
    val_split = scene_splits['val']

    train_scenes = []

    for scene in nusc.scene:

        scene_name = scene['name']
        
        if scene_name in train_split:

            if not os.path.exists('Processing_files/NU/super_voxel/VCCS/' + scene_name):
                os.makedirs('Processing_files/NU/super_voxel/VCCS/' + scene_name)

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
    for i_folder in train_split:
        sv_files += sorted(glob.glob('Processing_files/NU/super_voxel/VCCS/{}/*.npy'.format(i_folder)))

    sv_id_count = 0
    id2sv = []
    for file_path in sv_files:
        
        # Load super voxel labels
        sv_label = np.load(file_path)
        
        # Calculate sv2point
        sv2point = []
        for sv_l in np.unique(sv_label):
            if sv_l != 0:
                # Prune < 100 points
                p_ids = np.where(sv_label == sv_l)[0]
                if len(p_ids) > 100:
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

    with open('Processing_files/NU/super_voxel/VCCS/id2sv.pickle', 'wb') as fjson:
        pickle.dump(id2sv, fjson)