import os
import argparse
import numpy as np
import random
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torchsparse import SparseTensor

from dataset.sk_dataloader import SK_Dataloader
from dataset.nu_dataloader import NU_Dataloader
from network.spvcnn import SPVCNN
from network.minkunet import MinkUNet


def train(rank, world_size, MAX_ITER, directory, args):

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

    # Optimizer
    optimizer = optim.Adam(model.parameters())

    # Load training statics
    curr_iter = 0
    ep_id = 0
    PATH = directory + '/current.pt'
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    if os.path.exists(PATH):    
        checkpoint = torch.load(PATH, map_location=map_location)
        if world_size > 1:
            model.module.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        curr_iter = checkpoint['iteration']
        ep_id = checkpoint['ep_id']
        if rank == 0:
            print("Restored from: {}".format(PATH))
    elif args.r_id > 0:
        # Load weights of last trained model
        if args.r_id == 1:
            PATH = 'check_points/{}/{}/0r/current.pt'.format(args.dataset_name, args.model_name)
        else:
            PATH = 'check_points/{}/{}/{}/{}/{}r/current.pt'.format(args.dataset_name, args.model_name, args.label_unit, args.metric_name, args.r_id - 1)   
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
    if args.r_id == 0:
        if args.dataset_name == 'SK':
            sampler, train_data_loader = SK_Dataloader(gpu_num = world_size, gpu_rank = rank).train_data_loader_0r()
        if args.dataset_name == 'NU':
            sampler, train_data_loader = NU_Dataloader(gpu_num = world_size, gpu_rank = rank).train_data_loader_0r()
    elif args.metric_name == 'full':
        if args.dataset_name == 'SK':
            sampler, train_data_loader = SK_Dataloader(gpu_num = world_size, gpu_rank = rank).train_data_loader_full()
        if args.dataset_name == 'NU':
            sampler, train_data_loader = NU_Dataloader(gpu_num = world_size, gpu_rank = rank).train_data_loader_full()    
    elif args.label_unit == 'fr':
        if args.dataset_name == 'SK':
            sampler, train_data_loader = SK_Dataloader(gpu_num = world_size, gpu_rank = rank).train_data_loader_fr(model_name=args.model_name, metric_name=args.metric_name, r_id=args.r_id)
        if args.dataset_name == 'NU':
            sampler, train_data_loader = NU_Dataloader(gpu_num = world_size, gpu_rank = rank).train_data_loader_fr(model_name=args.model_name, metric_name=args.metric_name, r_id=args.r_id)  
    elif args.label_unit == 'sv':
        if args.dataset_name == 'SK':
            sampler, train_data_loader = SK_Dataloader(gpu_num = world_size, gpu_rank = rank).train_data_loader_sv(model_name=args.model_name, metric_name=args.metric_name, r_id=args.r_id)
        if args.dataset_name == 'NU':
            sampler, train_data_loader = NU_Dataloader(gpu_num = world_size, gpu_rank = rank).train_data_loader_sv(model_name=args.model_name, metric_name=args.metric_name, r_id=args.r_id)            

    # Training process
    is_training = True
    
    while(is_training):
        
        model.train()
        
        if sampler:
            sampler.set_epoch(ep_id)
        
        for i, batch in enumerate(train_data_loader):
            
            if rank == 0:
                print("Iteration: {}".format(curr_iter))
            
            # Load data
            coords_v_b = batch['coords_v_b'].cuda()
            feats_v_b = batch['feats_v_b'].cuda()
            labels_v_b = batch['labels_v_b'].cuda()

            optimizer.zero_grad()
            torch.cuda.synchronize()
            
            logits_v_b, _ = model(SparseTensor(feats_v_b, coords_v_b)) 
            
            loss = torch.nn.functional.cross_entropy(logits_v_b, labels_v_b, ignore_index=255,  reduction='mean')
            loss.backward()

            # Update    
            optimizer.step()

            if curr_iter >= MAX_ITER:
                is_training = False
                break
            
            curr_iter += 1
            
            if rank == 0:
                print('loss: {}'.format(loss.item()))
                if curr_iter % 500 == 0:
                    torch.save({
                        'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                        'iteration': curr_iter,
                        'ep_id': ep_id,
                        }, directory + '/current.pt') 
            torch.cuda.synchronize()

        ep_id += 1
    if world_size > 1:    
        dist.destroy_process_group()
    

def main(args):

    # Training statics
    MAX_ITER = 20000
    
    world_size = torch.cuda.device_count()

    directory = 'check_points/{}/{}'.format(args.dataset_name, args.model_name)
    if not os.path.exists(directory):
        os.makedirs(directory)  
       
    directory = 'check_points/{}/{}/{}'.format(args.dataset_name, args.model_name, args.label_unit)
    if not os.path.exists(directory):
        os.makedirs(directory)        
        
    if args.r_id == 0:
        directory = 'check_points/{}/{}/0r'.format(args.dataset_name, args.model_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    elif args.metric_name == 'full':
        directory = 'check_points/{}/{}/full'.format(args.dataset_name, args.model_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    else:
        directory = 'check_points/{}/{}/{}/{}'.format(args.dataset_name, args.model_name, args.label_unit, args.metric_name)    
        if not os.path.exists(directory):
            os.makedirs(directory)

        directory = 'check_points/{}/{}/{}/{}/{}r'.format(args.dataset_name, args.model_name, args.label_unit, args.metric_name, args.r_id)    
        if not os.path.exists(directory):
            os.makedirs(directory)

    if world_size > 1:
        mp.spawn(train,
            args=(world_size, MAX_ITER, directory, args,),
            nprocs=world_size,
            join=True)    
    else:
        train(0, world_size, MAX_ITER, directory, args)    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Train')
    parser.add_argument('--dataset_name', type = str, required = True,
                        help = 'name of the used dataset, [SK: semantic-kitti, NU: nuScenes]')  
    parser.add_argument('--model_name', type = str, required = True,
                        help = 'name of the current model to be trained, [SPVCNN, Mink]')  
    parser.add_argument('--label_unit', type = str, required = True,
                        help = '[fr: frame-based, sv: supervoxel-based]')
    parser.add_argument('--metric_name', type = str, required = True,
                        help = 'name of the active selection metric used for training')
    parser.add_argument('--r_id', type = int, required = True,
                        help = 'current training r_id, -1 for fully-supervised setting.')    
    parser.add_argument('--host_num', type = str, default = 7112)    
    args = parser.parse_args()
    
    use_cuda = torch.cuda.is_available()
    print("use_cuda: {}".format(use_cuda))
    if use_cuda is False:
        raise ValueError("CUDA is not available!")
    
    main(args)