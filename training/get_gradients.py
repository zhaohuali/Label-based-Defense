import argparse
import os
from pickletools import optimize
import random
import builtins
from turtle import forward
import warnings
import math
from datetime import datetime
import sys
import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

import apex

from preprocessing import dataset_config, get_target_samples, get_mean_std
from utils import kaiming_uniform, BNForwardFeatureHook, save_results
from RGAP import get_resnet_18_rank

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Training')

''' Configuration of the target environment '''
parser.add_argument('--id', default='test', type=str, help='code ID')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')             
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', type=int, default=1, 
                    help="number of local iterations to train")
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', type=float, default=0.1, 
                    help="learning rate of local iteration")
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--data-name', default='imagenet', type=str)
parser.add_argument('--target-idx', default=0, type=int,
                     help='The index of the target sample')
parser.add_argument('--pretrained', default=None, type=str,
                    help='path to pretrained model')
parser.add_argument('--outlayer-state', default='None', type=str,
                    choices=['None', 'kaiming_uniform', 'normal'],
                    help='Set the initialization method of the classification layer of the model')
parser.add_argument('--model-eval', action='store_true',
                    help='True: model.eval(), False:model.train()')
parser.add_argument('--syncbn', action='store_true',
                    help='enable syncbn of apex')
parser.add_argument('--ra', action='store_true', 
                    help='enable rank analysis of R-GAP')
parser.add_argument('--results', default='', type=str,
                    help='path to store results')
parser.add_argument('--enable-dp', action="store_true",
                    help="ensable privacy training and dont just train with vanilla SGD")
parser.add_argument('--sigma', type=float, default=None, help="Noise multiplier")
parser.add_argument('--max-per-sample-grad_norm', type=float, default=None, 
                    help="Clip per-sample gradients to this norm")
parser.add_argument('--delta', type=float, default=None, help="Target delta")
parser.add_argument('--duplicate-label', action='store_true', 
                    help='for testing "duplicate labels')
parser.add_argument('--kernel-size-of-maxpool', default=3)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,4"

args = parser.parse_args()

def main():

    sys.stdout = Logger(args.results)
    args.clock = f'{datetime.now()}'
    print(f'\n========== {args.clock} ========')
    for arg in vars(args):
        print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        print(f'torch seed: {torch.initial_seed()}')

    x_true, y_true = get_target_samples(args)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, x_true, y_true, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, x_true, y_true, args)


def main_worker(gpu, ngpus_per_node, x_true, y_true, args):
    args.gpu = gpu

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
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
   
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        print(f'torch seed: {torch.initial_seed()}')
    
    # create model
    model = models.__dict__[args.arch](pretrained=False)

    if args.pretrained is not None:
        if os.path.isfile(args.pretrained):
            if 'moco' not in args.pretrained:
                if args.pretrained.endswith('tar'):
                    checkpoint = torch.load(args.pretrained, map_location='cpu')
                    state_dict = checkpoint['state_dict']
                    for k in list(state_dict.keys()):
                        if k.startswith('module.'):
                            state_dict[k[len("module."):]] = state_dict[k]
                        del state_dict[k]
                    model.load_state_dict(state_dict)
                elif args.pretrained.endswith('pth'):
                    model.load_state_dict(torch.load(args.pretrained))
                else:
                    raise ValueError('args.pretrained file naming format is incorrect, \
                                     should be .pth (parameters) or .tar (checkpoint)')
            else:
                print("=> loading MoCoV2 checkpoint '{}'".format(args.pretrained))
                checkpoint = torch.load(args.pretrained, map_location="cpu")
                # rename moco pre-trained keys
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                        # remove prefix
                        state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

                msg = model.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no model found at '{}'".format(args.pretrained))

    n_channels, W, H, n_classes = dataset_config(args.data_name)
    
    if 'resnet' in args.arch or 'regnet' in args.arch:
        if n_classes != model.fc.out_features:
            model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=n_classes)
            print('adjust model.fc.in_features')
            
    if 'vgg' in args.arch:
        if n_classes != model.classifier[6].out_features:
            model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features, out_features=n_classes)
            print('adjust model.classifier[6].in_features')

    if args.outlayer_state == 'normal':
        if 'resnet' in args.arch or 'regnet' in args.arch:
            model.fc.weight.data.normal_(mean=0.0, std=0.01)
            model.fc.bias.data.zero_()
        elif 'vgg' in args.arch:
            model.classifier[6].weight.data.normal_(mean=0.0, std=0.01)
            model.classifier[6].bias.data.zero_()
    elif args.outlayer_state == 'kaiming_uniform':
        if 'resnet' in args.arch or 'regnet' in args.arch:
            kaiming_uniform(model.fc)
        elif 'vgg' in args.arch:
            kaiming_uniform(model.classifier[6])
    elif args.outlayer_state == 'None':
        pass

    if args.scheme == 'RA':
        model.maxpool.kernel_size = args.kernel_size_of_maxpool
        print(model)
        print('enable RA!')

    if args.ra:
        input_size = (args.batch_size, n_channels, W, H)
        ra_i = get_resnet_18_rank(model, input_size)

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
        # save target samples
        print(f'x_true.shape: {x_true.shape}')
        print(f'y_true.shape: {y_true.shape}')
        save_results(x_true, y_true, args)

        # save target model
        model_path = os.path.join(args.results, 'model.pth')
        torch.save(model.state_dict(), model_path)
        print(f'save model: {model_path}')
    
    if args.syncbn:
        model = apex.parallel.convert_syncbn_model(model)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    cudnn.benchmark = True

    if args.syncbn:
        bn_mean_var_layers = []
        for module in model.modules():
            if isinstance(module, apex.parallel.SyncBatchNorm):
                bn_mean_var_layers.append(BNForwardFeatureHook(module, args.distributed))

    
    # load data
    if args.distributed:
        images, target = get_rank_samples(x_true, y_true)
    else:
        images, target = x_true, y_true

    # switch to mode
    if args.model_eval:
        model.eval()
    else:
        model.train()

    images = images.split(len(images) // args.epochs, dim=0)
    target = target.split(len(target) // args.epochs, dim=0)
    print(f'label: {target}')

    old_state_dict = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):

        x = images[epoch].cuda(args.gpu)
        y = target[epoch].cuda(args.gpu)

        # compute output
        output = model(x)
        loss = criterion(output, y)

        # compute gradient and save
        model.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        
        if args.enable_dp:
            total_norm = torch.nn.utils.clip_grad.clip_grad_norm(model.parameters(), max_norm=args.max_per_sample_grad_norm)
            print(f'run differential privacy, max_norm={args.max_per_sample_grad_norm}')
            print(f'orig norm is {total_norm}.')

        optimizer.step()

        with torch.no_grad():
            grads = []
            for param in model.parameters():
                if param.requires_grad:
                    grads.append(param.grad)
            
            if args.enable_dp:
                total_norm = torch.norm(torch.stack([torch.norm((g.detach()), 2).cuda(args.gpu) for g in grads]), 2)
                print(f'clipped norm is {total_norm}.')

    new_state_dict = copy.deepcopy(model.state_dict())

    mean_list = None
    var_list = None
    if args.syncbn:
        mean_list = [mod.mean.detach().clone() for mod in bn_mean_var_layers]
        var_list = [mod.var.detach().clone() for mod in bn_mean_var_layers]

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):

        grads = []
        for key, param in model.named_parameters():
            if param.requires_grad:
                g = (old_state_dict[key] - new_state_dict[key]) / (args.lr * args.epochs)
                grads.append(g)
        true_grads = list(g.detach().clone() for g in grads)
        total_norm = torch.norm(torch.stack([torch.norm((g.detach()), 2).cuda(args.gpu) for g in true_grads]), 2)
        print(f'Num of grads: {len(true_grads)}')
        print(f'Full gradient norm is {total_norm:e}.')

        if args.enable_dp:
            for i, g in enumerate(true_grads):
                noise = torch.normal(
                            mean=0,
                            std=args.sigma,
                            size=g.size(),
                            device=g.device)
                true_grads[i] += noise
            print(f'run differential privacy, sigma={args.sigma}')

        mean, std = get_mean_std(args.data_name)
        dataset_setup = dict(mean=mean, 
                             std=std,
                             size=(n_channels, W, H), 
                             n_classes=n_classes)
        save_checkpoint({
                'clock': args.clock,
                'batch_size': args.batch_size,
                'dataset_setup': dataset_setup,
                'arch': args.arch,
                'model_eval': args.model_eval,
                'state_dict': torch.load(model_path),
                'free_last_layers_list': ['all'],
                'target_gradient': true_grads,
                'bn_mean_list': mean_list,
                'bn_var_list': var_list,
                'x_true': x_true,
                'y_true': y_true,
                'args': args,
                'syncbn': args.syncbn,
            }, args)
    

def save_checkpoint(state, args):

    if args.pretrained is not None:
        model_file = os.path.basename(args.pretrained)
    else:
        model_file = 'init'
    filename = f'id{args.id}_{args.arch}_{model_file}_{args.data_name}_b{args.batch_size}_i{args.target_idx}-checkpoint.pth.tar'
    filepath = os.path.join(args.results, filename)
    torch.save(state, filepath)

    print(f'save checkpoint: {os.path.join(os.getcwd(), filename)}')    

def get_rank_samples(x_true, y_true):
    
    num_replicas = dist.get_world_size()
    rank = dist.get_rank()

    if len(x_true) % num_replicas != 0:
        raise ValueError('number of samples % world_size != 0')
    else:
        num_samples = math.ceil(len(x_true) / num_replicas)

    start_idx = int(rank * num_samples)
    end_idx = int( (rank + 1) * num_samples)
    
    return x_true[start_idx:end_idx,:,:,:], y_true[start_idx:end_idx]

class Logger(object):
    def __init__(self, results, filename='history.log', stream=sys.stdout):
        
        self.terminal = stream
        filepath = os.path.join(results, filename)
        self.log = open(filepath, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
	    pass


if __name__ == '__main__':
    
    main()
