import os
import pickle
import math
import numpy as np
import torch
from torch.utils import data


####################################### Meta ###############################################
# labels:
#   0: 'noise'
#   1: 'animal'
#   2: 'human.pedestrian.adult'
#   3: 'human.pedestrian.child'
#   4: 'human.pedestrian.construction_worker'
#   5: 'human.pedestrian.personal_mobility'
#   6: 'human.pedestrian.police_officer'
#   7: 'human.pedestrian.stroller'
#   8: 'human.pedestrian.wheelchair'
#   9: 'movable_object.barrier'
#   10: 'movable_object.debris'
#   11: 'movable_object.pushable_pullable'
#   12: 'movable_object.trafficcone'
#   13: 'static_object.bicycle_rack'
#   14: 'vehicle.bicycle'
#   15: 'vehicle.bus.bendy'
#   16: 'vehicle.bus.rigid'
#   17: 'vehicle.car'
#   18: 'vehicle.construction'
#   19: 'vehicle.emergency.ambulance'
#   20: 'vehicle.emergency.police'
#   21: 'vehicle.motorcycle'
#   22: 'vehicle.trailer'
#   23: 'vehicle.truck'
#   24: 'flat.driveable_surface'
#   25: 'flat.other'
#   26: 'flat.sidewalk'
#   27: 'flat.terrain'
#   28: 'static.manmade'
#   29: 'static.other'
#   30: 'static.vegetation'
#   31: 'vehicle.ego'
# labels_16:
#   255: 'noise'
#   0: 'barrier'
#   1: 'bicycle'
#   2: 'bus'
#   3: 'car'
#   4: 'construction_vehicle'
#   5: 'motorcycle'
#   6: 'pedestrian'
#   7: 'traffic_cone'
#   8: 'trailer'
#   9: 'truck'
#   10: 'driveable_surface'
#   11: 'other_flat'
#   12: 'sidewalk'
#   13: 'terrain'
#   14: 'manmade'
#   15: 'vegetation'
learning_map = {
  1: 255,
  5: 255,
  7: 255,
  8: 255,
  10: 255,
  11: 255,
  13: 255,
  19: 255,
  20: 255,
  0: 255,
  29: 255,
  31: 255,
  9: 0,
  14: 1,
  15: 2,
  16: 2,
  17: 3,
  18: 4,
  21: 5,
  2: 6,
  3: 6,
  4: 6,
  6: 6,
  12: 7,
  22: 8,
  23: 9,
  24: 10,
  25: 11,
  26: 12,
  27: 13,
  28: 14,
  30: 15
}


class NU_Dataset(data.Dataset):

    def __init__(self, mode, lidar_files, label_files = None, pseudo_files = None, sv_flag_files = None, sv_info_files = None, seq_frame = None, scale = 20, full_scale = [8192, 8192, 8192]):
        
        self.mode = mode
        self.lidar_files = lidar_files
        self.label_files = label_files
        self.pseudo_files = pseudo_files
        self.sv_flag_files = sv_flag_files
        self.sv_info_files = sv_info_files
        self.seq_frame = seq_frame
        self.scale = scale
        self.full_scale = full_scale
        
        self.label_map = np.ones(100, dtype=np.int64) * 255
        for key in learning_map:
            self.label_map[key] = learning_map[key]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.lidar_files)

    def __getitem__(self, idx):
            
        # Load point data
        raw_data = np.fromfile(self.lidar_files[idx], dtype=np.float32).reshape(-1, 5)
        raw_data = raw_data[:, :4]
        feats_p = np.zeros_like(raw_data)
        coords_p = raw_data[:, :3]
        feats_p[:, 3] = raw_data[:, 3]
        
        if not self.mode == 'score':
            # Load annotated labels
            labels_anno_p = np.fromfile(self.label_files[idx], dtype=np.uint8).reshape(-1)
            # Label mapping
            labels_anno_p = self.label_map[labels_anno_p].astype(np.int64)
            
            labels_p = labels_anno_p
            
            if 'pseudo' in self.mode:
                # Load pseudo labels
                labels_pseudo_p = np.load(self.pseudo_files[idx])  
                assert labels_pseudo_p.shape[0] == labels_anno_p.shape[0]

        if 'train_sv' in self.mode:
            # Load sv data
            sv_flag = np.load(self.sv_flag_files[idx])
            with open(self.sv_info_files[idx], 'rb') as f:
                _, sv2point = pickle.load(f)
            
            # Label mask (annotated)
            label_anno_mask = np.ones_like(labels_anno_p, dtype=bool)
            for sv_id_frame in np.where(sv_flag == 1)[0]:
                p_ids = sv2point[sv_id_frame]
                label_anno_mask[p_ids] = False
            labels_p[label_anno_mask] = 255
            
            if 'pseudo' in self.mode:
                # Label mask (pseudo)
                label_pseudo_mask = np.zeros_like(labels_pseudo_p, dtype=bool)
                for sv_id_frame in np.where(sv_flag == 2)[0]:
                    p_ids = sv2point[sv_id_frame]
                    label_pseudo_mask[p_ids] = True
                labels_p[label_pseudo_mask] = labels_pseudo_p[label_pseudo_mask]

        # Affine linear transformation
        trans_m = np.eye(3) + np.random.randn(3, 3) * 0.1
        trans_m[0][0] *= np.random.randint(0, 2) * 2 - 1
        theta = np.random.rand() * 2 * math.pi
        trans_m = np.matmul(trans_m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
        
        coords_p = np.matmul(coords_p, trans_m)
        feats_p[:, :3] = coords_p
        coords_p *= self.scale

        # Random translation
        coords_min = coords_p.min(0)
        coords_max = coords_p.max(0)
        offset = -coords_min + np.clip(self.full_scale - coords_max + coords_min - 0.001, 0, None) * np.random.rand(3) + np.clip(self.full_scale - coords_max + coords_min + 0.001, None, 0) * np.random.rand(3)
        coords_p += offset
        
        # Clip valid positions
        valid_idxs = (coords_p.min(1) >= 0) * (coords_p.max(1) < self.full_scale[0])
        assert sum(valid_idxs) == len(valid_idxs), 'input voxels are not valid'

        # Voxelization
        coords_v = coords_p.astype(int)

        # Remove duplicate items
        _, unique_idxs, inverse_idxs = np.unique(coords_v, axis=0, return_index=True, return_inverse=True)
        coords_v = coords_v[unique_idxs]
        feats_v = feats_p[unique_idxs]
        if 'train' in self.mode:
            labels_v = labels_p[unique_idxs]

        if 'train' in self.mode:
            return {'coords_v': coords_v, 'feats_v': feats_v, 'labels_v': labels_v}
        elif self.mode == 'val':
            return {'coords_v': coords_v, 'feats_v': feats_v, 'labels_p': labels_p, 
                    'inverse_idxs': inverse_idxs}
        elif self.mode == 'score':
            return {'coords_v': coords_v, 'feats_v': feats_v, 'inverse_idxs': inverse_idxs,
                    'seq_frame': self.seq_frame[idx]}


    def collate_fn(self, inputs):

        # Data in batch
        coords_v_b = []         # N X 4(x,y,z,B)        
        feats_v_b = []          # N X 4(x,y,z,sig)
        if 'train' in self.mode:
            labels_v_b = []         # N
        if self.mode == 'val':
            labels_p_b = []

        # From voxels to points
        if not 'train' in self.mode:
            inverse_indices_b = [[np.array(-1)]]
        
        # Put into containers
        for idx, sample in enumerate(inputs):
            coords_v = torch.from_numpy(sample['coords_v']).int()
            coords_v_b += [torch.cat([coords_v, torch.IntTensor(coords_v.shape[0], 1).fill_(idx)], 1)]
            feats_v_b += [torch.from_numpy(sample['feats_v']).float()]
            if 'train' in self.mode:
                labels_v_b += [torch.from_numpy(sample['labels_v']).long()]
            if self.mode == 'val':
                labels_p_b += [torch.from_numpy(sample['labels_p']).long()]
            if not 'train' in self.mode:
                inverse_offset = max(inverse_indices_b[-1]) + 1
                inverse_idxs = torch.from_numpy(sample['inverse_idxs'])
                inverse_indices_b += [inverse_idxs + inverse_offset]

        # Concatenation
        coords_v_b = torch.cat(coords_v_b, 0)
        feats_v_b = torch.cat(feats_v_b, 0)
        if 'train' in self.mode:
            labels_v_b = torch.cat(labels_v_b, 0)
        if self.mode == 'val':
            labels_p_b = torch.cat(labels_p_b, 0)
        if not 'train' in self.mode:
            inverse_indices_b = torch.cat(inverse_indices_b[1:], 0).long()
 
        if 'train' in self.mode:
            return {'coords_v_b': coords_v_b, 'feats_v_b': feats_v_b, 'labels_v_b': labels_v_b}
        elif self.mode == 'val':
            return {'coords_v_b': coords_v_b, 'feats_v_b': feats_v_b, 'labels_p_b': labels_p_b, 
                    'inverse_indices_b': inverse_indices_b}
        elif self.mode == 'score':
            return {'coords_v_b': coords_v_b, 'feats_v_b': feats_v_b, 'inverse_indices_b': inverse_indices_b,
                    'seq_frame': inputs[0]['seq_frame']}
