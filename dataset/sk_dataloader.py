import os
import numpy as np
import torch
import glob
import pickle
import tqdm

import torch.utils.data
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from dataset.sk_dataset import SK_Dataset


####################################### Meta ###############################################
train_split = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
val_split = ['08']


class SK_Dataloader():
    def __init__(self, gpu_num = None, gpu_rank = None, scale = 20, full_scale = [8192, 8192, 8192], batch_size = 5, num_workers = 4):
        
        self.gpu_num = gpu_num
        self.gpu_rank = gpu_rank
        self.scale = scale
        self.full_scale = full_scale
        self.batch_size = batch_size
        self.num_workers = num_workers

        if self.gpu_rank == 0:

            if not os.path.exists('Processing_files/SK'):
                os.mkdir('Processing_files/SK')           

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

        # Load samples
        lidar_files = []
        for i_folder in train_split:
            lidar_files += sorted(glob.glob('Semantic_kitti/dataset/sequences/{}'.format(i_folder) + '/velodyne/*.bin'))
        
        dataset = SK_Dataset(mode=mode, lidar_files=lidar_files)
        
        sampler, dataloader = self.get_data_loader(dataset)

        return sampler, dataloader
        

    # Intial training
    def train_data_loader_0r(self):
        
        mode = 'train_frame'
        
        if self.gpu_rank == 0:

            if not os.path.exists('Processing_files/SK/frame_flag'):
                os.mkdir('Processing_files/SK/frame_flag')
                os.mkdir('Processing_files/SK/sv_flag')
                os.mkdir('Processing_files/SK/sv_flag/KMeans')
                os.mkdir('Processing_files/SK/sv_flag/VCCS')
                
            
            if not os.path.exists('Processing_files/SK/frame_flag/0r'):
                os.mkdir('Processing_files/SK/frame_flag/0r')
                os.mkdir('Processing_files/SK/sv_flag/KMeans/0r')
                os.mkdir('Processing_files/SK/sv_flag/VCCS/0r')
            
                # Randomly select 1% fully labeled frames
                for i_folder in train_split:
                    frames = sorted(glob.glob('Semantic_kitti/dataset/sequences/{}'.format(i_folder) + '/velodyne/*.bin'))
                    frame_flag = np.zeros(len(frames), dtype=bool)
                    selected_ids = np.random.choice(np.arange(len(frames)), int(np.round(0.01 * len(frames))))
                    frame_flag[selected_ids] = True
                    # Save frame_flag
                    np.save('Processing_files/SK/frame_flag/0r/{}.npy'.format(i_folder), frame_flag)
                    # Save sv_flag
                        # KMeans
                    os.makedirs('Processing_files/SK/sv_flag/KMeans/0r/{}'.format(i_folder))
                    for idx, fr in enumerate(frames):
                        fr_elements = fr.split('/')
                        with open('Processing_files/SK/super_voxel/KMeans/{}'.format(i_folder) + '/' + fr_elements[-1][:-3] + 'pickle', 'rb') as fjson:
                            sv_id, _ = pickle.load(fjson)
                        if idx in selected_ids:
                            sv_flag = np.ones(len(sv_id), dtype=bool)
                        else:
                            sv_flag = np.zeros(len(sv_id), dtype=bool)
                        np.save('Processing_files/SK/sv_flag/KMeans/0r/{}'.format(i_folder) + '/' + fr_elements[-1][:-3] + 'npy', sv_flag)       
                        # VCCS
                    os.makedirs('Processing_files/SK/sv_flag/VCCS/0r/{}'.format(i_folder))
                    for idx, fr in enumerate(frames):
                        fr_elements = fr.split('/')
                        with open('Processing_files/SK/super_voxel/VCCS/{}'.format(i_folder) + '/' + fr_elements[-1][:-3] + 'pickle', 'rb') as fjson:
                            sv_id, _ = pickle.load(fjson)
                        if idx in selected_ids:
                            sv_flag = np.ones(len(sv_id), dtype=bool)
                        else:
                            sv_flag = np.zeros(len(sv_id), dtype=bool)
                        np.save('Processing_files/SK/sv_flag/VCCS/0r/{}'.format(i_folder) + '/' + fr_elements[-1][:-3] + 'npy', sv_flag)       

        if self.gpu_num > 1:
            dist.barrier()                     

        # Load samples
        lidar_files = [] 
        for i_folder in train_split:
            frames = sorted(glob.glob('Semantic_kitti/dataset/sequences/{}'.format(i_folder) + '/velodyne/*.bin'))
            frame_flag = np.load('Processing_files/SK/frame_flag/0r/{}.npy'.format(i_folder))             
            selected_frames = list(np.array(frames)[frame_flag])
            lidar_files += selected_frames
        print('Train_0r samples:', len(lidar_files))

        dataset = SK_Dataset(mode=mode, lidar_files=lidar_files)
        
        sampler, dataloader = self.get_data_loader(dataset)

        return sampler, dataloader
    

    # Data loader for round r_id training (frame level)
    def train_data_loader_fr(self, model_name, metric_name, r_id):
        
        mode = 'train'
        
        # Load samples
        lidar_files = []
        for i_folder in train_split:
            lidar_files += sorted(glob.glob('Semantic_kitti/dataset/sequences/{}'.format(i_folder) + '/velodyne/*.bin'))

        frame_flag_all = np.array([])

        # Load frames of current r_id
        for i_folder in train_split:  
            if metric_name == 'RAND':
                frame_flag = np.load('Processing_files/SK/frame_flag/RAND/{}r/{}.npy'.format(r_id, i_folder))
            else:
                frame_flag = np.load('Processing_files/SK/frame_flag/{}/{}/{}r/{}.npy'.format(model_name, metric_name, r_id, i_folder))        
            assert r_id > 0
            frame_flag_all = np.append(frame_flag_all, frame_flag)
            
        frame_flag_all = frame_flag_all.astype(bool)
        lidar_files = list(np.array(lidar_files)[frame_flag_all])  
                      
        print('Train_{}r samples:'.format(r_id), len(lidar_files))
        
        dataset = SK_Dataset(mode=mode, lidar_files=lidar_files)
        
        sampler, dataloader = self.get_data_loader(dataset)
        
        return sampler, dataloader


    # Data loader for uncertainty scoring
    # Since there are no connections between sequences, we process each sequence separately
    def score_data_loader(self, inf_reps : int):
        """ seq id in train_split
        """
        mode = 'score'
        
        # Load samples
        lidar_files = []
        for i_folder in train_split:
            lidar_files += sorted(glob.glob('Semantic_kitti/dataset/sequences/{}'.format(i_folder) + '/velodyne/*.bin'))
        print('Score samples:', len(lidar_files)) 
        
        if self.gpu_num > 1:
            split_num = int(np.ceil(len(lidar_files) / self.gpu_num))
            lidar_files = lidar_files[self.gpu_rank * split_num : (self.gpu_rank + 1) * split_num]
        dataset = SK_Dataset(mode=mode, lidar_files=np.repeat(lidar_files, inf_reps))
        
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

        # Load samples
        lidar_files = []
        for i_folder in val_split:
            lidar_files += sorted(glob.glob('Semantic_kitti/dataset/sequences/{}'.format(i_folder) + '/velodyne/*.bin'))
        print('Validation samples:', len(lidar_files))    
        
        dataset = SK_Dataset(mode=mode, lidar_files=lidar_files)
        
        sampler, dataloader = self.get_data_loader(dataset)
        
        return sampler, dataloader



    #########################################################################################################################################
    
    ####################################################### Super Voxel Level ###############################################################
    
    #########################################################################################################################################
    
    
    # Data loader for round r_id training (sv level)
    def train_data_loader_sv(self, model_name, metric_name, r_id):

        assert r_id > 0
        
        if 'pseudo' in metric_name:
            mode = 'train_sv_pseudo'
        else:
            mode = 'train_sv'
        
        # Load samples
        lidar_files = []
        sv_flag_files = []
        sv_info_files = []
        pseudo_files = None
        if 'pseudo' in metric_name:
            pseudo_files =[]

        # Load frames with labeled super voxels of current r_id
        for i_folder in train_split:
            frames = sorted(glob.glob('Semantic_kitti/dataset/sequences/{}'.format(i_folder) + '/velodyne/*.bin'))
            if metric_name == 'ReDAL':
                sv_infos = sorted(glob.glob('Processing_files/SK/super_voxel/VCCS/{}/*.pickle'.format(i_folder)))
            else:
                sv_infos = sorted(glob.glob('Processing_files/SK/super_voxel/KMeans/{}/*.pickle'.format(i_folder)))       
            if metric_name == 'RAND':
                sv_flags = sorted(glob.glob('Processing_files/SK/sv_flag/KMeans/RAND/{}r/{}/*.npy'.format(r_id, i_folder)))
            elif metric_name == 'ReDAL':
                sv_flags = sorted(glob.glob('Processing_files/SK/sv_flag/VCCS/{}/{}/{}r/{}/*.npy'.format(model_name, metric_name, r_id, i_folder)))
            else:
                sv_flags = sorted(glob.glob('Processing_files/SK/sv_flag/KMeans/{}/{}/{}r/{}/*.npy'.format(model_name, metric_name, r_id, i_folder)))        
            assert len(sv_flags) == len(sv_infos)
            assert len(frames) == len(sv_flags)
            
            if 'pseudo' in metric_name:
                if r_id == 1:
                    pseudos = sorted(glob.glob('Processing_files/SK/pred/{}/fr/0r/{}/*.npy'.format(model_name, i_folder)))
                else:
                    pseudos = sorted(glob.glob('Processing_files/SK/pred/{}/sv/{}/{}r/{}/*.npy'.format(model_name, metric_name, r_id - 1, i_folder)))
                assert len(pseudos) == len(sv_flags)

            # Labeled frames for training
            frame_flag = np.zeros_like(frames, dtype=bool)
            for idx, sv_flag in enumerate(sv_flags):
                flag = np.load(sv_flag)
                if flag.sum() != 0:
                    frame_flag[idx] = True
            labeled_frames = list(np.array(frames)[frame_flag])
            lidar_files += labeled_frames
            sv_flag_files += list(np.array(sv_flags)[frame_flag])
            sv_info_files += list(np.array(sv_infos)[frame_flag])
            if 'pseudo' in metric_name:
                pseudo_files += list(np.array(pseudos)[frame_flag])
                      
        print('Train_{}r samples:'.format(r_id), len(lidar_files))
        
        dataset = SK_Dataset(mode=mode, lidar_files=lidar_files, pseudo_files=pseudo_files, sv_flag_files=sv_flag_files, sv_info_files=sv_info_files)
        
        sampler, dataloader = self.get_data_loader(dataset) 
        
        return sampler, dataloader