from http.client import ImproperConnectionState
import os
import random
import numpy as np
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchsparse import SparseTensor

from nuscenes.utils.splits import create_splits_scenes

from dataset.sk_dataloader import SK_Dataloader
from dataset.nu_dataloader import NU_Dataloader
from network.spvcnn import SPVCNN
from network.minkunet import MinkUNet


def inference(rank, world_size, save_folder_prob, save_folder_pred, save_folder_outfeat, args):

    ####################################### Traininig ###############################################
    # set random seed
    random.seed(1 + rank)
    np.random.seed(1 + rank)
    torch.manual_seed(7122)
    
    # Initialize DDP
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:{}'.format(args.host_num),
                                world_size=world_size, rank=rank)

    # Set device
    if world_size > 1:
        torch.cuda.set_device(rank)
        pytorch_device = torch.device('cuda', rank)
    else:
        pytorch_device = torch.device('cuda:0')
    
    # Network
    if 'SPVCNN' in args.model_name:
        if args.dataset_name == 'SK':
            model = SPVCNN(class_num=19)
        if args.dataset_name == 'NU':
            model = SPVCNN(class_num=16)
    if 'Mink' in args.model_name:
        if args.dataset_name == 'SK':
            model = MinkUNet(class_num=19)
        if args.dataset_name == 'NU':
            model = MinkUNet(class_num=16)
    model.to(pytorch_device)
    if world_size > 1:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[rank],
                                                        output_device=rank)    


    # Load training statics
    if args.r_id == 0:
        directory = 'check_points/{}/{}/{}/0r'.format(args.dataset_name, args.model_name, args.label_unit)
    else:
        directory = 'check_points/{}/{}/{}/{}/{}r'.format(args.dataset_name, args.model_name, args.label_unit, args.metric_name, args.r_id)
    PATH = directory + '/current.pt'
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}  
    checkpoint = torch.load(PATH, map_location=map_location)
    if world_size > 1:
        model.module.load_state_dict(checkpoint['model_state_dict'], strict=True)
    else:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    if rank == 0:
        print("Restored from: {}".format(PATH))
    if world_size > 1:
        dist.barrier()

    # Dataset
    if args.dataset_name == 'SK':
        score_data_loader = SK_Dataloader(gpu_num = world_size, gpu_rank = rank).score_data_loader(inf_reps=args.inf_reps)
    if args.dataset_name == 'NU':
        score_data_loader = NU_Dataloader(gpu_num = world_size, gpu_rank = rank).score_data_loader(inf_reps=args.inf_reps)        

    # Evaluation process
    with torch.no_grad():            

        model.eval()

        if rank == 0:
            print("*****************************Inference*************************************")
                
        for i, batch in enumerate(score_data_loader):
            
            # Load data
            coords_v_b = batch['coords_v_b'].cuda()
            feats_v_b = batch['feats_v_b'].cuda()       
                
            logits_v_b, out_feat_v_b = model(SparseTensor(feats_v_b, coords_v_b)) 

            # Project to original points
            logits_v_b = logits_v_b.cpu()
            inverse_indices_b = batch["inverse_indices_b"]
            logits_p_b = logits_v_b[inverse_indices_b]
            if args.r_id == 0 or args.metric_name in ['ReDAL', 'CSET']:
                out_feat_v_b = out_feat_v_b.cpu()
                out_feat_p_b = out_feat_v_b[inverse_indices_b]

            # Probability map
            prob_map = torch.softmax(logits_p_b, dim=1)
            prob_map = prob_map.numpy().reshape(args.inf_reps, -1, prob_map.shape[-1])
            prob_map_mean = np.mean(prob_map, axis=0)
            
            # Pred
            pred = np.argmax(prob_map_mean, axis=1)
            
            # Output features
            if args.r_id == 0 or args.metric_name in ['ReDAL', 'CSET']:
                out_feat = out_feat_p_b.numpy().reshape(args.inf_reps, -1, out_feat_p_b.shape[-1])
                out_feat = np.mean(out_feat, axis=0)
                            
            # Save
            if args.dataset_name == 'SK':
                lidar_file = batch['lidar_file']
                seq_id = lidar_file.split('/')[-3]
                frame_id = lidar_file.split('/')[-1][:-4]
            if args.dataset_name == 'NU':
                seq_frame = batch['seq_frame']
                seq_id = seq_frame.split('/')[-2]
                frame_id = seq_frame.split('/')[-1]
            np.save(save_folder_prob + '/{}' '/{}.npy'.format(seq_id, frame_id), prob_map_mean)
            np.save(save_folder_pred + '/{}' '/{}.npy'.format(seq_id, frame_id), pred)
            if args.r_id == 0 or args.metric_name in ['ReDAL', 'CSET']:
                np.save(save_folder_outfeat + '/{}' '/{}.npy'.format(seq_id, frame_id), out_feat)
            print('Processing {}/{}'.format(seq_id, frame_id))
        
    if world_size > 1:
        dist.destroy_process_group()
    

def main(args):

    world_size = torch.cuda.device_count()

    # Save folder
    save_folder_prob = 'Processing_files/{}/prob_map/{}'.format(args.dataset_name, args.model_name)
    if not os.path.exists(save_folder_prob):
        os.makedirs(save_folder_prob)
    save_folder_prob = 'Processing_files/{}/prob_map/{}/{}'.format(args.dataset_name, args.model_name, args.label_unit)
    if not os.path.exists(save_folder_prob):
        os.makedirs(save_folder_prob)
    if args.r_id == 0:
        save_folder_prob = 'Processing_files/{}/prob_map/{}/{}/0r'.format(args.dataset_name, args.model_name, args.label_unit)
        if not os.path.exists(save_folder_prob):
            os.makedirs(save_folder_prob)
    else:
        save_folder_prob = 'Processing_files/{}/prob_map/{}/{}/{}'.format(args.dataset_name, args.model_name, args.label_unit, args.metric_name)
        if not os.path.exists(save_folder_prob):
            os.makedirs(save_folder_prob)        
        save_folder_prob = 'Processing_files/{}/prob_map/{}/{}/{}/{}r'.format(args.dataset_name, args.model_name, args.label_unit, args.metric_name, args.r_id)
        if not os.path.exists(save_folder_prob):
            os.makedirs(save_folder_prob)       

    save_folder_pred = 'Processing_files/{}/pred/{}'.format(args.dataset_name, args.model_name)
    if not os.path.exists(save_folder_pred):
        os.makedirs(save_folder_pred)
    save_folder_pred = 'Processing_files/{}/pred/{}/{}'.format(args.dataset_name, args.model_name, args.label_unit)
    if not os.path.exists(save_folder_pred):
        os.makedirs(save_folder_pred)
    if args.r_id == 0:
        save_folder_pred = 'Processing_files/{}/pred/{}/{}/0r'.format(args.dataset_name, args.model_name, args.label_unit)
        if not os.path.exists(save_folder_pred):
            os.makedirs(save_folder_pred)
    else:
        save_folder_pred = 'Processing_files/{}/pred/{}/{}/{}'.format(args.dataset_name, args.model_name, args.label_unit, args.metric_name)
        if not os.path.exists(save_folder_pred):
            os.makedirs(save_folder_pred)        
        save_folder_pred = 'Processing_files/{}/pred/{}/{}/{}/{}r'.format(args.dataset_name, args.model_name, args.label_unit, args.metric_name, args.r_id)
        if not os.path.exists(save_folder_pred):
            os.makedirs(save_folder_pred) 
    
    save_folder_outfeat = None
    if args.r_id == 0 or args.metric_name in ['ReDAL', 'CSET']:
        save_folder_outfeat = 'Processing_files/{}/outfeat/{}'.format(args.dataset_name, args.model_name)
        if not os.path.exists(save_folder_outfeat):
            os.makedirs(save_folder_outfeat)
        save_folder_outfeat = 'Processing_files/{}/outfeat/{}/{}'.format(args.dataset_name, args.model_name, args.label_unit)
        if not os.path.exists(save_folder_outfeat):
            os.makedirs(save_folder_outfeat)
        if args.r_id == 0:
            save_folder_outfeat = 'Processing_files/{}/outfeat/{}/{}/0r'.format(args.dataset_name, args.model_name, args.label_unit)
            if not os.path.exists(save_folder_outfeat):
                os.makedirs(save_folder_outfeat)
        else:
            save_folder_outfeat = 'Processing_files/{}/outfeat/{}/{}/{}'.format(args.dataset_name, args.model_name, args.label_unit, args.metric_name)
            if not os.path.exists(save_folder_outfeat):
                os.makedirs(save_folder_outfeat)        
            save_folder_outfeat = 'Processing_files/{}/outfeat/{}/{}/{}/{}r'.format(args.dataset_name, args.model_name, args.label_unit, args.metric_name, args.r_id)
            if not os.path.exists(save_folder_outfeat):
                os.makedirs(save_folder_outfeat)       
        
    if args.dataset_name == 'SK':
        train_split = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
    if args.dataset_name == 'NU':
        scene_splits = create_splits_scenes()
        train_split = scene_splits['train']
    for seq_id in train_split:    
        seq_folder_prob = save_folder_prob + '/' + seq_id
        if not os.path.exists(seq_folder_prob):
            os.makedirs(seq_folder_prob)
    for seq_id in train_split:    
        seq_folder_pred = save_folder_pred + '/' + seq_id
        if not os.path.exists(seq_folder_pred):
            os.makedirs(seq_folder_pred)
    if args.r_id == 0 or args.metric_name in ['ReDAL', 'CSET']:
        for seq_id in train_split:    
            seq_folder_outfeat = save_folder_outfeat + '/' + seq_id
            if not os.path.exists(seq_folder_outfeat):
                os.makedirs(seq_folder_outfeat)          

    if world_size > 1:
        mp.spawn(inference,
            args=(world_size, save_folder_prob, save_folder_pred, save_folder_outfeat, args,),
            nprocs=world_size,
            join=True)    
    else:
        inference(0, world_size, save_folder_prob, save_folder_pred, save_folder_outfeat, args)    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Probability map inference')
    parser.add_argument('--dataset_name', type = str, required = True,
                        help = 'name of the used dataset')  
    parser.add_argument('--model_name', type = str, required = True,
                        help = 'name of the trained model to be loaded')
    parser.add_argument('--label_unit', type = str, required = True,
                        help = 'fr for frame-based and sv for supervoxel-based')
    parser.add_argument('--metric_name', type = str, required = True,
                        help = 'name of the active selection metric used for the trained model')
    parser.add_argument('--r_id', type = int, required = True,
                        help = 'current trained r_id')
    parser.add_argument('--inf_reps', default = 8, type = int, required = False,
                    help = 'Number of inference views, 1 or more')
    parser.add_argument('--host_num', type = str, default = 7112)  
    args = parser.parse_args()
    
    use_cuda = torch.cuda.is_available()
    print("use_cuda: {}".format(use_cuda))
    if use_cuda is False:
        raise ValueError("CUDA is not available!")
    
    main(args)