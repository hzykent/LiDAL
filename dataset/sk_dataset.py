import os
import pickle
import math
import numpy as np
import torch
from torch.utils import data


####################################### Meta ###############################################
label_name_mapping = {
    0: 'unlabeled',
    1: 'outlier',
    10: 'car',
    11: 'bicycle',
    13: 'bus',
    15: 'motorcycle',
    16: 'on-rails',
    18: 'truck',
    20: 'other-vehicle',
    30: 'person',
    31: 'bicyclist',
    32: 'motorcyclist',
    40: 'road',
    44: 'parking',
    48: 'sidewalk',
    49: 'other-ground',
    50: 'building',
    51: 'fence',
    52: 'other-structure',
    60: 'lane-marking',
    70: 'vegetation',
    71: 'trunk',
    72: 'terrain',
    80: 'pole',
    81: 'traffic-sign',
    99: 'other-object',
    252: 'moving-car',
    253: 'moving-bicyclist',
    254: 'moving-person',
    255: 'moving-motorcyclist',
    256: 'moving-on-rails',
    257: 'moving-bus',
    258: 'moving-truck',
    259: 'moving-other-vehicle'
}

kept_labels = [
    'road', 'sidewalk', 'parking', 'other-ground', 'building', 'car', 'truck',
    'bicycle', 'motorcycle', 'other-vehicle', 'vegetation', 'trunk', 'terrain',
    'person', 'bicyclist', 'motorcyclist', 'fence', 'pole', 'traffic-sign'
]


class SK_Dataset(data.Dataset):

    def __init__(self, mode, lidar_files, pseudo_files = None, sv_flag_files = None, sv_info_files = None, scale = 20, full_scale = [8192, 8192, 8192]):
        
        self.mode = mode
        self.lidar_files = lidar_files
        self.pseudo_files = pseudo_files
        self.sv_flag_files = sv_flag_files
        self.sv_info_files = sv_info_files
        self.scale = scale
        self.full_scale = full_scale
        
        # Construct label mapping
        if os.path.exists('Processing_files/SK/label_map.npy'):
            self.label_map = np.load('Processing_files/SK/label_map.npy')
        else:
            reverse_label_name_mapping = {}
            label_map = np.zeros(260)
            cnt = 0
            for label_id in label_name_mapping:
                if label_id > 250:
                    if label_name_mapping[label_id].replace('moving-',
                                                            '') in kept_labels:
                        label_map[label_id] = reverse_label_name_mapping[
                            label_name_mapping[label_id].replace('moving-', '')]
                    else:
                        label_map[label_id] = 255
                elif label_id == 0:
                    label_map[label_id] = 255
                else:
                    if label_name_mapping[label_id] in kept_labels:
                        label_map[label_id] = cnt
                        reverse_label_name_mapping[
                            label_name_mapping[label_id]] = cnt
                        cnt += 1
                    else:
                        label_map[label_id] = 255
            np.save("Processing_files/SK/label_map.npy", label_map)
            self.label_map = label_map

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.lidar_files)

    def __getitem__(self, idx):
            
        # Load point data
        raw_data = np.fromfile(self.lidar_files[idx], dtype=np.float32).reshape(-1, 4)
        feats_p = np.zeros_like(raw_data)
        coords_p = raw_data[:, :3]
        feats_p[:, 3] = raw_data[:, 3]
        
        if not self.mode == 'score':
            # Load annotated labels
            labels_anno_p = np.fromfile(self.lidar_files[idx].replace('velodyne', 'labels')[:-3] + 'label',
                                            dtype=np.uint32).reshape(-1)
            # Delete high 16 digits binary
            labels_anno_p = labels_anno_p & 0xFFFF
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

        # # frame_offsets: Offsets of scenes in the collection of points
        # if 'val' in self.mode:
        #     point_ids = np.nonzero(valid_idxs)[0] + self.frame_offsets[idx]

        if 'train' in self.mode:
            return {'coords_v': coords_v, 'feats_v': feats_v, 'labels_v': labels_v}
        elif self.mode == 'val':
            return {'coords_v': coords_v, 'feats_v': feats_v, 'labels_p': labels_p, 
                    'inverse_idxs': inverse_idxs}
                    # , 'point_ids': point_ids}
        elif self.mode == 'score':
            return {'coords_v': coords_v, 'feats_v': feats_v, 'inverse_idxs': inverse_idxs,
                    'lidar_file': self.lidar_files[idx]}


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
        
        # # id in the collection of all points
        # if self.mode == 'val':
        #     point_ids_b = []
        
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
            # if self.mode == 'val':
            #     point_ids_b += [torch.from_numpy(sample['point_ids'])]

        # Concatenation
        coords_v_b = torch.cat(coords_v_b, 0)
        feats_v_b = torch.cat(feats_v_b, 0)
        if 'train' in self.mode:
            labels_v_b = torch.cat(labels_v_b, 0)
        if self.mode == 'val':
            labels_p_b = torch.cat(labels_p_b, 0)
        if not 'train' in self.mode:
            inverse_indices_b = torch.cat(inverse_indices_b[1:], 0).long()
        # if self.mode == 'val':
        #         point_ids_b = torch.cat(point_ids_b, 0)
 
        if 'train' in self.mode:
            return {'coords_v_b': coords_v_b, 'feats_v_b': feats_v_b, 'labels_v_b': labels_v_b}
        elif self.mode == 'val':
            return {'coords_v_b': coords_v_b, 'feats_v_b': feats_v_b, 'labels_p_b': labels_p_b, 
                    'inverse_indices_b': inverse_indices_b}
                    # , 'point_ids_b': point_ids_b}
        elif self.mode == 'score':
            return {'coords_v_b': coords_v_b, 'feats_v_b': feats_v_b, 'inverse_indices_b': inverse_indices_b,
                    'lidar_file': inputs[0]['lidar_file']}
