"""
Data preprocessing: generate surface variation for S3DIS/SemanticKITTI dataset
"""

import os
import argparse
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud


# Training settings
seq_ids = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]


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

for seq_id in seq_ids:
    coords_dir = 'Semantic_kitti/dataset/sequences/{}/velodyne'.format(seq_id)
    boundary_dir = 'Processing_files/SK/boundary/{}'.format(seq_id)
    os.makedirs(boundary_dir, exist_ok=True)

    for fname in os.listdir(coords_dir):
        coords_path = os.path.join(coords_dir, fname)
        print('Processing ' + coords_path)
        xyz = np.fromfile(coords_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        rgb = np.zeros_like(xyz)
        edge_npy = boundary_extractor(xyz, rgb, threshold=0.1)
        new_path = os.path.join(boundary_dir, f'{fname[:-4]}.npy')
        np.save(new_path, edge_npy)
