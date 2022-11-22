import os
import glob
import pickle
import numpy as np
import multiprocessing as mp
from functools import partial
from sklearn.neighbors import KDTree


def parse_poses(filename, calibration):
    """ read poses file with per-scan poses from given filename

        Returns
        -------
        list
            list of poses as 4x4 numpy arrays.
    """
    file = open(filename)

    poses = []

    Tr = calibration["Tr"]
    Tr_inv = np.linalg.inv(Tr)

    for line in file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

    return poses


def parse_calibration(filename):
    """ read calibration file with given filename

        Returns
        -------
        dict
            Calibration matrices as 4x4 numpy arrays.
    """
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()

    return calib


def process_frame(id, files, poses):
    
    file_path = files[id]
    pose = poses[id]
    print('Processing ' + file_path)
    
    # Load point data
    raw_data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    coords = raw_data[:, :3]
    
    # Affine transformation to the initial coordinate system
    hcoords = np.hstack((coords, np.ones_like(coords[:, :1])))
    hcoords = np.sum(np.expand_dims(hcoords, 2) * pose.T, axis=1)
    registered_coords = hcoords[:, :3]
    
    # Construct KDTree
    search_tree = KDTree(registered_coords)
    
    # Save KDTree
    KDTree_file = 'Processing_files/SK/kdtree/' + file_path.split('/')[-3] + '/' + file_path.split('/')[-1][:-3] + 'pickle'
    with open(KDTree_file, 'wb') as fjson:
        pickle.dump(search_tree, fjson)
    
    

if __name__ == "__main__":

    seqeunces = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']

    if not os.path.exists('Processing_files/SK'):
        os.makedirs('Processing_files/SK')
    if not os.path.exists('Processing_files/SK/kdtree'):
        os.makedirs('Processing_files/SK/kdtree')

    for i_folder in seqeunces:

        if not os.path.exists('Processing_files/SK/kdtree/' + i_folder):
            os.makedirs('Processing_files/SK/kdtree/' + i_folder)
        
        folder_path = 'Semantic_kitti/dataset/sequences/{}'.format(i_folder)
        
        # Load scene files
        scene_files = sorted(glob.glob(folder_path + '/velodyne/*.bin'))

        # Load calib poses
        calibration = parse_calibration(folder_path + '/calib.txt')
        poses_f64 = parse_poses(filename = folder_path + '/poses.txt', calibration = calibration)
        poses = [pose.astype(np.float32) for pose in poses_f64]
        assert len(poses) == len(scene_files)
        
        # multi-processing
        pf_pool = mp.Pool(processes=12)
        process_frame_p = partial(process_frame, files=scene_files, poses=poses)
        ids = np.arange(len(scene_files))
        pf_pool.map(process_frame_p, ids)
        pf_pool.close()
        pf_pool.join()