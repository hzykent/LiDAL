import argparse
import numpy as np
import random
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchsparse import SparseTensor
import time

import utils.iou_sk as iou_sk
import utils.iou_nu as iou_nu
from dataset.sk_dataloader import SK_Dataloader
from dataset.nu_dataloader import NU_Dataloader
from network.spvcnn import SPVCNN
from network.minkunet import MinkUNet


def eval(rank, world_size, args):

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
    elif 'Mink' in args.model_name:
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
        directory = 'check_points/{}/{}/fr/0r'.format(args.dataset_name, args.model_name)
    elif args.metric_name == 'full':
        directory = 'check_points/{}/{}/{}/full'.format(args.dataset_name, args.model_name, args.label_unit)
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
        sampler, val_data_loader = SK_Dataloader(gpu_num = world_size, gpu_rank = rank).val_data_loader()
    if args.dataset_name == 'NU':
        sampler, val_data_loader = NU_Dataloader(gpu_num = world_size, gpu_rank = rank).val_data_loader()
    start = time.time()
    # Evaluation process
    with torch.no_grad():            

        model.eval()

        if args.dataset_name == 'SK':
            c_matrix = np.zeros((19, 19)).astype(np.int32)
        if args.dataset_name == 'NU':
            c_matrix = np.zeros((16, 16)).astype(np.int32)

        if rank == 0:
            print("*****************************Validation*************************************")
   
        for i, batch in enumerate(val_data_loader):
            
            # Load data
            coords_v_b = batch['coords_v_b'].cuda()
            feats_v_b = batch['feats_v_b'].cuda() 
            labels_p_b = batch['labels_p_b']      
                
            logits_v_b, _ = model(SparseTensor(feats_v_b, coords_v_b)) 

            # Project to original points
            logits_v_b = logits_v_b.cpu()
            inverse_indices_b = batch["inverse_indices_b"]
            logits_p_b = logits_v_b[inverse_indices_b]

            if args.dataset_name == 'SK':
                c_matrix += iou_sk.confusion_matrix(logits_p_b.max(1)[1].numpy(), labels_p_b.numpy()) 
            if args.dataset_name == 'NU':
                c_matrix += iou_nu.confusion_matrix(logits_p_b.max(1)[1].numpy(), labels_p_b.numpy())
        
        if world_size > 1:
            dist.barrier()

        c_matrix = torch.from_numpy(c_matrix).cuda()
        if world_size > 1:
            dist.all_reduce(c_matrix, op=dist.ReduceOp.SUM)
        if rank == 0:
            if args.dataset_name == 'SK':
                iou_sk.evaluate(confusion=c_matrix.cpu().numpy())
            if args.dataset_name == 'NU':
                iou_nu.evaluate(confusion=c_matrix.cpu().numpy())
    end = time.time()
    print(end - start)
    if world_size > 1:
        dist.destroy_process_group()
    

def main(args):

    world_size = torch.cuda.device_count()

    if world_size > 1:
        mp.spawn(eval,
            args=(world_size, args,),
            nprocs=world_size,
            join=True)    
    else:
        eval(0, world_size, args)    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Trained model evaluation')
    parser.add_argument('--dataset_name', type = str, required = True,
                        help = 'name of the used dataset')  
    parser.add_argument('--model_name', type = str, required = True,
                        help = 'name of the trained model to be loaded')
    parser.add_argument('--label_unit', type = str, required = True,
                        help = 'fr for frame-based and sv for supervoxel-based')
    parser.add_argument('--metric_name', type = str, required = True,
                        help = 'name of the active selection metric used for the trained model')
    parser.add_argument('--r_id', type = int, required = True,
                        help = 'current trained round')
    parser.add_argument('--host_num', type = str, default = 7112)  
    args = parser.parse_args()
    
    use_cuda = torch.cuda.is_available()
    print("use_cuda: {}".format(use_cuda))
    if use_cuda is False:
        raise ValueError("CUDA is not available!")
    
    main(args)