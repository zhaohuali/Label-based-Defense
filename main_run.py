
from datetime import datetime
import random
import builtins
import os
import sys
import time
import argparse

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from config import load_checkpoint, build_history
from config import set_BN_regularization
from kernels.superparameters_search import set_superparameters
from metric.GInfoR import get_gir
from kernels import init_x_pseudo, get_y_pseudo, reconstruction
from distributed import set_distributed, get_rank_samples
from kernels.superparameters_search import ray_tune_main
from metric.image_similarity import save_results_with_metric

parser = argparse.ArgumentParser(description='E2EGI')
parser.add_argument('--id', default='test', type=str, help='code ID')
parser.add_argument('--root', default='/data/lzh/code/E2EGI', type=str)

''' Configuration of Hardware'''
parser.add_argument('--world-size', default=-1, type=int,
                    help='Number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='Node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='Url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='Distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

''' Configuration of the target environment '''
parser.add_argument('--checkpoint', type=str,
                    help='Path of checkpoint')
parser.add_argument('--seed', default=0, type=int,
                    help='seed of initializing training. ')
parser.add_argument('-p', '--print-freq', default=500, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--exact-bn', action='store_true',
                    help='bn statistics from target samples, rather than ' 
                         'from all datasets')

''' Configuration of basic GI '''
parser.add_argument('--n-seed', default=1, type=int,
                    help='number of groups of pseudo-samples.')
parser.add_argument('--optim', default='Adam')
parser.add_argument('--gradient-loss-fun', default='sim', type=str,
                    choices=['sim', 'L2'], 
                    help='fun for measuring loss between pseudo-grads and true-grads, \
                        sim for negative cosine similarity')
parser.add_argument('--min-grads-loss', action='store_true',
                    help='The final result takes the sample with ' 
                         'the smallest gradient loss')

''' configuration of superparameters '''
parser.add_argument('--epochs', default=15000, type=int,
                    help='total epochs for running E2EGI.')
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial update step size for running gi.')
parser.add_argument('--grads-weight', default=1, type=float, 
                    help='weights of gradient loss')
parser.add_argument('--TV', default=0.1, type=float,
                    help='weight of total variation regularization.') 
parser.add_argument('--BN', default=0, type=float,
                    help='weights of bn loss item.')
parser.add_argument('--input-norm', default=0, type=float)  
parser.add_argument('--target-tv', default=0.28, type=float)

''' GI component selection '''
parser.add_argument('--pseudo-label-init', default='llg', type=str,
                    choices=['from_grads', 'known', 'llg'],
                    help='way for initializing pseudo labels')
parser.add_argument('--superparameters-search', action='store_true')
parser.add_argument('--MinCombine', action='store_true',
                    help='Whether to use the minimum loss combinatorial optimization')

''' Configuration of E2EGI '''
parser.add_argument('--Group', default=0, type=float,
                    help='weight of group regularization.')
parser.add_argument('--T-init-rate', default=0.3, type=float, 
                    help='epochs of Initial Reconstruction')
parser.add_argument('--total-T-in-rate', default=0.3, type=float, 
                    help='total epochs of the minimum loss combinatorial optimization')
parser.add_argument('--T-in', default=1000, type=int, 
                    help='interval of constructing a new group consistency regularization.')
parser.add_argument('--T-end-rate', default=0.4, type=float,
                    help='epochs of Final Reconstruction')
parser.add_argument('--input-noise', default=0, type=float,
                    help='the degree to which random noise is introduced into' 
                         'the pseudo-input')

''' Configuration of Superparameters Search'''
parser.add_argument('--simulation-checkpoint', type=str, 
                    help='path of simulation data for Superparameters Search')
parser.add_argument('--max-concurrent', default=8, type=int, 
                    help='The number of parallel hyperparameter search experiments')
parser.add_argument('--num-samples', default=16, type=int,
                    help='Total number of hyperparameter search experiments')
parser.add_argument('--ngpus-per-trial', default=0.5, 
                    help='The number of GPU resources spent on each hyperparameter search')
parser.add_argument('--epochs-tune', action='store_true', 
                    help='Whether to search epochs setup')
parser.add_argument('--lr-tune', action='store_true', 
                    help='Whether to search update step size setup')
parser.add_argument('--TV-tune', action='store_true', 
                    help='Whether to search weight of total variation regularization')
parser.add_argument('--BN-tune', action='store_true', 
                    help='Whether to search weight of BN loss item')
parser.add_argument('--input-norm-tune', action='store_true', 
                    help='Whether to search weight of input norm regularization')
parser.add_argument('--grads-weight-tune', action='store_true', 
                    help='Whether to search weight of gradent loss')
parser.add_argument('--verbose', default=3, type=int, 
                    help='Verbosity mode. 0 = silent, 1 = only status updates, '
                         '2 = status and brief trial results, '
                         '3 = status and detailed trial result')
parser.add_argument('--max-t', default=3000, type=int,
                    help='max epochs of hyperparameter search')
parser.add_argument('--min-t', default=2000, type=int,
                    help='It starts after min_t time')
parser.add_argument('--HP-epochs', default=5000, type=int,
                    help='epochs of GI with HP ')

''' metric (just for test, the target samples is known) '''
parser.add_argument('--metric', action='store_true', 
                    help='Whether to able metric')
parser.add_argument('--one-to-one-similarity', default=True)
parser.add_argument('--GInfoR', action='store_true',
                    help='get the gradient information ratio of each sample')


args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2" # available GPUs

def main(param_config=None):

    args.clock = datetime.now()

    if args.superparameters_search:
        set_superparameters(args, param_config)

    if args.gpu is not None:
        print('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    args.sys_print = builtins.print

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    args.ngpus_per_node = ngpus_per_node

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f'torch seed: {torch.initial_seed()}')

    build_history(args)
    model, bn_mean_list, bn_var_list, dm, ds, target_gradient, metric_dict = load_checkpoint(args)

    if args.metric and args.GInfoR:
        GInfoR = get_gir(model, metric_dict, args.model_eval)
        print(f'GInfoR: \n{GInfoR}')

    model = set_distributed(model, args)
    
    bn_loss_layers = None
    if args.BN > 0:
        bn_loss_layers = set_BN_regularization(
                            bn_mean_list, 
                            bn_var_list, 
                            model,
                            args)

    # initialize fake-samples
    x_pseudo_list = init_x_pseudo(args)
    y_pseudo = get_y_pseudo(args, metric_dict, target_gradient)

    if args.distributed:
        x_pseudo_list, y_pseudo = get_rank_samples(x_pseudo_list, y_pseudo, args)

    # set gpu
    x_pseudo_list = x_pseudo_list.cuda(args.gpu)
    y_pseudo = y_pseudo.cuda(args.gpu)    
    target_gradient = list((grad.cuda(args.gpu) for grad in target_gradient))
    dm = dm.cuda(args.gpu)
    ds = ds.cuda(args.gpu)

    args.init_time = time.time()
    x_recon, cache = reconstruction(
                        x_pseudo_list, 
                        y_pseudo,
                        target_gradient,
                        model,
                        dm,
                        ds,
                        args, 
                        metric_dict=metric_dict, 
                        bn_loss_layers=bn_loss_layers)

    # save results and print metric
    if not args.superparameters_search:
        save_results_with_metric(x_recon, y_pseudo, dm, ds, metric_dict, args) 

def save_args(args):

    config_bk = dict(world_size=args.world_size, 
                        rank=args.rank,
                        multiprocessing_distributed=args.multiprocessing_distributed, 
                        epochs=args.epochs, 
                        print_freq=args.print_freq, 
                        GInfoR=args.GInfoR)
    args.world_size = -1
    args.rank = -1
    args.multiprocessing_distributed = False
    args.epochs = args.HP_epochs
    args.print_freq = sys.maxsize
    args.GInfoR = False

    return config_bk

def update_args(args, HP_config, config_bk):

    if config_bk is not None:
        for key, value in config_bk.items():
            args.__dict__[key] = value

    for key, value in HP_config.items():
        args.__dict__[key] = value

    args.superparameters_search = False

if __name__=='__main__':

    if args.superparameters_search:
        config_bk = save_args(args)
        HP_config = ray_tune_main(main, args)
        update_args(args, HP_config, config_bk)
    
    main()