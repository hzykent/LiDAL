import os
import numpy as np
import torch
import glob
import pickle

import torch.utils.data
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from dataset.nu_dataset import NU_Dataset


class NU_Dataloader():
    def __init__(self, gpu_num = None, gpu_rank = None, scale = 20, full_scale = [8192, 8192, 8192], batch_size = 15, num_workers = 4):
        
        self.gpu_num = gpu_num
        self.gpu_rank = gpu_rank
        self.scale = scale
        self.full_scale = full_scale
        self.batch_size = batch_size
        self.num_workers = num_workers

        if self.gpu_rank == 0:

            if not os.path.exists('Processing_files/NU'):
                os.mkdir('Processing_files/NU')           

            if not os.path.isfile('Processing_files/NU/lidar_files_train.pickle'):
                
                nusc = NuScenes(version='v1.0-trainval', dataroot='nuScenes', verbose=True)
                scene_splits = create_splits_scenes()
                train_split = scene_splits['train']
                
                # Load samples
                lidar_files = []
                label_files = []
                for scene in nusc.scene:
                    scene_name = scene['name']
                    if scene_name in train_split:
                        print('Load ' + scene_name)
                        sample_token = scene['first_sample_token']
                        while(sample_token):
                            sample = nusc.get('sample', sample_token)
                            sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                            lidar_files += ['nuScenes/' + sample_data['filename']]
                            label_files += ['nuScenes/' + nusc.get('lidarseg', sample_data['token'])['filename']]
                            sample_token = sample['next']                
                
                # Save
                with open('Processing_files/NU/lidar_files_train.pickle', 'wb') as fjson:
                    pickle.dump(lidar_files, fjson)
                with open('Processing_files/NU/label_files_train.pickle', 'wb') as fjson:
                    pickle.dump(label_files, fjson)

        if self.gpu_num > 1:
            dist.barrier()
        

    def get_data_loader(self, dataset):
        if self.gpu_num > 1:
            sampler = DistributedSampler(dataset, num_replicas=self.gpu_num, rank=self.gpu_rank)
        else:
            sampler = None
        
        batch_size = self.batch_size
        if dataset.mode == 'val':
            batch_size = 2 * self.batch_size
        
        return sampler, torch.utils.data.DataLoader(
            dataset = dataset,
            batch_size = batch_size,
            collate_fn = dataset.collate_fn,
            sampler = sampler,
            num_workers = self.num_workers,
            shuffle = (sampler is None),
            pin_memory = True
            )

    ###################################################################################################################################
    
    ####################################################### Frame Level ###############################################################
    
    ###################################################################################################################################
    # Fully supervised
    def train_data_loader_full(self):
        
        mode = 'train'
        
        # Load files
        with open('Processing_files/NU/lidar_files_train.pickle', 'rb') as fjson:
            lidar_files = pickle.load(fjson)
        with open('Processing_files/NU/label_files_train.pickle', 'rb') as fjson:
            label_files = pickle.load(fjson)
                        
        dataset = NU_Dataset(mode=mode, lidar_files=lidar_files, label_files=label_files)
        
        sampler, dataloader = self.get_data_loader(dataset)

        return sampler, dataloader
        

    # Intial training
    def train_data_loader_0r(self):
        
        mode = 'train_frame'
        
        # Load files
        with open('Processing_files/NU/lidar_files_train.pickle', 'rb') as fjson:
            lidar_files = pickle.load(fjson)
        with open('Processing_files/NU/label_files_train.pickle', 'rb') as fjson:
            label_files = pickle.load(fjson)

        nusc = NuScenes(version='v1.0-trainval', dataroot='nuScenes', verbose=False)
        scene_splits = create_splits_scenes()
        train_split = scene_splits['train']

        if self.gpu_rank == 0:

            if not os.path.exists('Processing_files/NU/frame_flag'):
                os.mkdir('Processing_files/NU/frame_flag')
                os.mkdir('Processing_files/NU/sv_flag')
                os.mkdir('Processing_files/NU/sv_flag/KMeans')
                os.mkdir('Processing_files/NU/sv_flag/VCCS')
                
            
            if not os.path.exists('Processing_files/NU/frame_flag/0r'):
                os.mkdir('Processing_files/NU/frame_flag/0r')
                os.mkdir('Processing_files/NU/sv_flag/KMeans/0r')
                os.mkdir('Processing_files/NU/sv_flag/VCCS/0r')
                                
                # Randomly select 1% fully labeled frames
                frame_flag = np.zeros(len(lidar_files), dtype=bool)
                selected_ids = np.random.choice(np.arange(len(lidar_files)), int(np.round(0.01 * len(lidar_files))))
                frame_flag[selected_ids] = True
                
                # Save frame_flag and sv_flag
                scene_offset = 0
                for scene in nusc.scene:
                    scene_name = scene['name']
                    if scene_name in train_split:
                        print('Save flag for ' + scene_name)
                        # Save frame_flag
                        frame_num = scene['nbr_samples']
                        scene_flag = frame_flag[scene_offset : scene_offset + frame_num]
                        np.save('Processing_files/NU/frame_flag/0r/{}.npy'.format(scene_name), scene_flag)
                        scene_offset += frame_num
                        # Save sv_flag
                            # KMeans
                        if not os.path.exists('Processing_files/NU/sv_flag/KMeans/0r/' + scene_name):
                            os.mkdir('Processing_files/NU/sv_flag/KMeans/0r/' + scene_name)
                        for idx, flag in enumerate(scene_flag):
                            frame_id_str = str(idx)
                            while len(frame_id_str) < 6:
                                frame_id_str = '0' + frame_id_str
                            sv_info_path = 'Processing_files/NU/super_voxel/KMeans/' + scene_name + '/' + frame_id_str + '.pickle'
                            with open(sv_info_path, 'rb') as f:
                                sv_id, _ = pickle.load(f)
                            if flag:
                                sv_flag = np.ones(len(sv_id), dtype=bool)
                            else:
                                sv_flag = np.zeros(len(sv_id), dtype=bool)
                            np.save('Processing_files/NU/sv_flag/KMeans/0r/{}'.format(scene_name) + '/' + frame_id_str + '.npy', sv_flag) 
                            # VCCS
                        if not os.path.exists('Processing_files/NU/sv_flag/VCCS/0r/' + scene_name):
                            os.mkdir('Processing_files/NU/sv_flag/VCCS/0r/' + scene_name)
                        for idx, flag in enumerate(scene_flag):
                            frame_id_str = str(idx)
                            while len(frame_id_str) < 6:
                                frame_id_str = '0' + frame_id_str
                            sv_info_path = 'Processing_files/NU/super_voxel/VCCS/' + scene_name + '/' + frame_id_str + '.pickle'
                            with open(sv_info_path, 'rb') as f:
                                sv_id, _ = pickle.load(f)
                            if flag:
                                sv_flag = np.ones(len(sv_id), dtype=bool)
                            else:
                                sv_flag = np.zeros(len(sv_id), dtype=bool)
                            np.save('Processing_files/NU/sv_flag/VCCS/0r/{}'.format(scene_name) + '/' + frame_id_str + '.npy', sv_flag) 


        if self.gpu_num > 1:
            dist.barrier()        
        
        # Load flag
        frame_flag = np.array([])
        for scene in nusc.scene:
            scene_name = scene['name']
            if scene_name in train_split:
                frame_f = np.load('Processing_files/NU/frame_flag/0r/{}.npy'.format(scene_name))
                frame_flag = np.append(frame_flag, frame_f)                
            
        frame_flag = frame_flag.astype(bool)
        lidar_files = list(np.array(lidar_files)[frame_flag])
        label_files = list(np.array(label_files)[frame_flag])
                      
        print('Train_0r samples:', len(lidar_files))

        dataset = NU_Dataset(mode=mode, lidar_files=lidar_files, label_files=label_files)
        
        sampler, dataloader = self.get_data_loader(dataset)

        return sampler, dataloader
    

    # Data loader for round r_id training (frame level)
    def train_data_loader_fr(self, model_name, metric_name, r_id):
        
        mode = 'train'
        
        # Load samples
        with open('Processing_files/NU/lidar_files_train.pickle', 'rb') as fjson:
            lidar_files = pickle.load(fjson)   
        with open('Processing_files/NU/label_files_train.pickle', 'rb') as fjson:
            label_files = pickle.load(fjson)     
        
        scene_splits = create_splits_scenes()
        train_split = scene_splits['train']
    
        frame_flag_all = np.array([])

        # Load frames of current r_id
        for i_folder in train_split:  
            if metric_name == 'RAND':
                frame_flag = np.load('Processing_files/NU/frame_flag/RAND/{}r/{}.npy'.format(r_id, i_folder))
            else:
                frame_flag = np.load('Processing_files/NU/frame_flag/{}/{}/{}r/{}.npy'.format(model_name, metric_name, r_id, i_folder))        
            assert r_id > 0
            frame_flag_all = np.append(frame_flag_all, frame_flag)
            
        frame_flag_all = frame_flag_all.astype(bool)
        lidar_files = list(np.array(lidar_files)[frame_flag_all])
        label_files = list(np.array(label_files)[frame_flag_all])        
            
        print('Train_{}r samples:'.format(r_id), len(lidar_files))
        
        dataset = NU_Dataset(mode=mode, lidar_files=lidar_files, label_files=label_files)
        
        sampler, dataloader = self.get_data_loader(dataset) 
        
        return sampler, dataloader


    # Data loader for uncertainty scoring
    # Since there are no connections between sequences, we process each sequence separately
    def score_data_loader(self, inf_reps : int):
        """ seq id in train_split
        """
        mode = 'score'
        
        # Load files
        with open('Processing_files/NU/lidar_files_train.pickle', 'rb') as fjson:
            lidar_files = pickle.load(fjson)
        seq_frame = []
        nusc = NuScenes(version='v1.0-trainval', dataroot='nuScenes', verbose=False)
        scene_splits = create_splits_scenes()
        train_split = scene_splits['train']
        for scene in nusc.scene:
            scene_name = scene['name']
            if scene_name in train_split:
                for idx in range(scene['nbr_samples']):
                    frame_id_str = str(idx)
                    while len(frame_id_str) < 6:
                        frame_id_str = '0' + frame_id_str
                    seq_frame += ['{}/{}'.format(scene_name, frame_id_str)]                    
        assert len(seq_frame) == len(lidar_files)
        
        if self.gpu_num > 1:
            split_num = int(np.ceil(len(lidar_files) / self.gpu_num))
            lidar_files = lidar_files[self.gpu_rank * split_num : (self.gpu_rank + 1) * split_num]
            seq_frame = seq_frame[self.gpu_rank * split_num : (self.gpu_rank + 1) * split_num]
        dataset = NU_Dataset(mode=mode, lidar_files=np.repeat(lidar_files, inf_reps), seq_frame=np.repeat(seq_frame, inf_reps))
        
        return torch.utils.data.DataLoader(
            dataset = dataset,
            batch_size = inf_reps,
            collate_fn = dataset.collate_fn,
            num_workers = self.num_workers,
            shuffle = False,
            drop_last = False,
            pin_memory = True
            )


    # Data loader for validation
    def val_data_loader(self):
        
        mode = 'val'

        if self.gpu_rank == 0:
            
            if not os.path.isfile('Processing_files/NU/lidar_files_val.pickle'):
                
                nusc = NuScenes(version='v1.0-trainval', dataroot='nuScenes', verbose=True)
                scene_splits = create_splits_scenes()
                val_split = scene_splits['val']
                
                # Load samples
                lidar_files = []
                label_files = []
                for scene in nusc.scene:
                    scene_name = scene['name']
                    if scene_name in val_split:
                        print('Load ' + scene_name)
                        sample_token = scene['first_sample_token']
                        while(sample_token):
                            sample = nusc.get('sample', sample_token)
                            sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                            lidar_files += ['nuScenes/' + sample_data['filename']]
                            label_files += ['nuScenes/' + nusc.get('lidarseg', sample_data['token'])['filename']]
                            sample_token = sample['next']                
                
                # Save
                with open('Processing_files/NU/lidar_files_val.pickle', 'wb') as fjson:
                    pickle.dump(lidar_files, fjson)
                with open('Processing_files/NU/label_files_val.pickle', 'wb') as fjson:
                    pickle.dump(label_files, fjson)

        if self.gpu_num > 1:
            dist.barrier()
        
        # Load files
        with open('Processing_files/NU/lidar_files_val.pickle', 'rb') as fjson:
            lidar_files = pickle.load(fjson)
        with open('Processing_files/NU/label_files_val.pickle', 'rb') as fjson:
            label_files = pickle.load(fjson)
                        
        dataset = NU_Dataset(mode=mode, lidar_files=lidar_files, label_files=label_files)
        
        sampler, dataloader = self.get_data_loader(dataset)
        
        return sampler, dataloader



    # #########################################################################################################################################
    
    # ####################################################### Super Voxel Level ###############################################################
    
    # #########################################################################################################################################
    
    
    # Data loader for round r_id training (sv level)
    def train_data_loader_sv(self, model_name, metric_name, r_id):
        
        assert r_id > 0
        
        if 'pseudo' in metric_name:
            mode = 'train_sv_pseudo'
        else:
            mode = 'train_sv'
        
        # Load samples
        with open('Processing_files/NU/lidar_files_train.pickle', 'rb') as fjson:
            lidar_files = pickle.load(fjson)   
        with open('Processing_files/NU/label_files_train.pickle', 'rb') as fjson:
            label_files = pickle.load(fjson)     
        
        scene_splits = create_splits_scenes()
        train_split = scene_splits['train']
        
        sv_flag_files = []
        sv_info_files = []
        pseudo_files = None
        if 'pseudo' in metric_name:
            pseudo_files =[]
        frame_flag_all = np.array([])

        # Load frames with labeled super voxels of current r_id
        for i_folder in train_split:
            if metric_name == 'ReDAL':
                sv_infos = sorted(glob.glob('Processing_files/NU/super_voxel/VCCS/{}/*.pickle'.format(i_folder)))
            else:
                sv_infos = sorted(glob.glob('Processing_files/NU/super_voxel/KMeans/{}/*.pickle'.format(i_folder)))       
            if metric_name == 'RAND':
                sv_flags = sorted(glob.glob('Processing_files/NU/sv_flag/RAND/{}r/{}/*.npy'.format(r_id, i_folder)))
            elif metric_name == 'ReDAL':
                sv_flags = sorted(glob.glob('Processing_files/NU/sv_flag/VCCS/{}/{}/{}r/{}/*.npy'.format(model_name, metric_name, r_id, i_folder))) 
            else:
                sv_flags = sorted(glob.glob('Processing_files/NU/sv_flag/KMeans/{}/{}/{}r/{}/*.npy'.format(model_name, metric_name, r_id, i_folder)))         
            assert len(sv_flags) == len(sv_infos)
            
            if 'pseudo' in metric_name:
                if r_id == 1:
                    pseudos = sorted(glob.glob('Processing_files/NU/pred/{}/fr/0r/{}/*.npy'.format(model_name, i_folder)))
                else:
                    pseudos = sorted(glob.glob('Processing_files/NU/pred/{}/sv/{}/{}r/{}/*.npy'.format(model_name, metric_name, r_id - 1, i_folder)))
                assert len(pseudos) == len(sv_flags)

            # Labeled frames for training
            frame_flag = np.zeros_like(sv_flags, dtype=bool)
            for idx, sv_flag in enumerate(sv_flags):
                flag = np.load(sv_flag)
                if flag.sum() != 0:
                    frame_flag[idx] = True
            frame_flag_all = np.append(frame_flag_all, frame_flag)
            sv_flag_files += sv_flags
            sv_info_files += sv_infos
            if 'pseudo' in metric_name:
                pseudo_files += pseudos
        assert len(lidar_files) == len(sv_flag_files)
        assert frame_flag_all.shape[0] == len(lidar_files)
        frame_flag_all = frame_flag_all.astype(bool)
        lidar_files = list(np.array(lidar_files)[frame_flag_all])
        label_files = list(np.array(label_files)[frame_flag_all])
        sv_flag_files = list(np.array(sv_flag_files)[frame_flag_all])
        sv_info_files = list(np.array(sv_info_files)[frame_flag_all])
        if 'pseudo' in metric_name:
            pseudo_files = list(np.array(pseudo_files)[frame_flag_all])
            
        print('Train_{}r samples:'.format(r_id), len(lidar_files))
        
        dataset = NU_Dataset(mode=mode, lidar_files=lidar_files, label_files=label_files, pseudo_files=pseudo_files, sv_flag_files=sv_flag_files, sv_info_files=sv_info_files)
        
        sampler, dataloader = self.get_data_loader(dataset) 
        
        return sampler, dataloader