

import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image


def get_target_samples(args):

    mean, std = get_mean_std(args.data_name)
    normalize = transforms.Normalize(mean=mean,
                                     std=std)
    n_channels, W, H, n_classes = dataset_config(args.data_name)

    if args.data_name == 'imagenet':
        
        loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(args.data, transforms.Compose([
                transforms.Resize(W),
                transforms.CenterCrop(W),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=False)

    elif args.data_name == 'svhn':

        dataset = datasets.SVHN(args.data)
        transform = transforms.Compose([
            transforms.Resize(W),
            transforms.CenterCrop(W),
            transforms.ToTensor(),
            normalize,])
        dataloader = torch_dataset(dataset, transform)
        loader = torch.utils.data.DataLoader(
                    dataloader, 
                    batch_size=args.batch_size, 
                    shuffle=True, 
                    num_workers=args.workers, 
                    pin_memory=False)

    elif args.data_name == 'cifar100':
        dataset = datasets.CIFAR100(args.data)
        transform = transforms.Compose([transforms.ToTensor(),
                                        normalize,])
        dataloader = torch_dataset(dataset, transform)
        loader = torch.utils.data.DataLoader(
                    dataloader, 
                    batch_size=args.batch_size, 
                    shuffle=True, 
                    num_workers=args.workers, 
                    pin_memory=False)

    elif args.data_name == 'celeba':
        dataset = celeba_root(args.data)
        transform = transforms.Compose([transforms.Resize(W),
                                        transforms.CenterCrop(W),
                                        transforms.ToTensor(),
                                        normalize,])
        dataloader = celeba_dataset(dataset, transform)
        loader = torch.utils.data.DataLoader(
                    dataloader, 
                    batch_size=args.batch_size, 
                    shuffle=True,
                    num_workers=args.workers, 
                    pin_memory=False)
    
    # load target samples
    for i in range(args.target_idx + 1):
        x_true, y_true = next(iter(loader)) 

    if y_true.dim() == 2:
        y_true = y_true.view(-1,)

    if args.duplicate_label:
        dype = (y_true.device, y_true.dtype)
        y_true = torch.as_tensor([1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3, 4,4,4,4,4,4,4,4]).to(*dype)
        print(f'duplicate label: {y_true}')

    return x_true, y_true

class torch_dataset:
    
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img = self.dataset[index][0]
        label = self.dataset[index][1]
        label = torch.as_tensor(label, dtype=torch.long).view(1,)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)

def celeba_root(path):

    x_train_dir_path = os.path.join(path, 'x_train')
    y_train_file =  os.path.join(path, 'y_train.txt')
    dataset = dict(x_train_dir_path=x_train_dir_path,
                   y_train_file=y_train_file)   
    return dataset

class celeba_dataset:

    def __init__(self, dataset, transform):
        self.x_train_dir_path = dataset['x_train_dir_path']
        self.y_train_file = dataset['y_train_file']
        self.transform = transform

        self.y_list = []
        self.img_name_list = []
        with open(self.y_train_file, 'r') as f:
            for line in f:
                _img_name, _y = line.split(' ')
                self.img_name_list.append(_img_name)
                self.y_list.append(int(_y)-1) # celeba数据从1开始

    def __getitem__(self, index):
        img_name = self.img_name_list[index]
        img_path = os.path.join(self.x_train_dir_path, img_name)
        img = Image.open(img_path)
        y = self.y_list[index]
        label = torch.Tensor([y]).long().view(1,)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.y_list)

def get_mean_std(data_name):

    if data_name == 'imagenet':
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
    elif data_name == 'cifar100':
        mean=[0.5071598291397095, 0.4866936206817627, 0.44120192527770996]
        std=[0.2673342823982239, 0.2564384639263153, 0.2761504650115967]
    elif data_name == 'celeba':
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
    elif data_name == 'svhn':
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

    return mean, std 

def dataset_config(data_name):

    if data_name == 'imagenet':
        n_channels = 3
        W = H = 224
        n_classes = 1000
    elif data_name == 'cifar100':
        n_channels = 3
        W = H = 32
        n_classes = 100
    elif data_name == 'celeba':
        n_channels = 3
        W = H = 224
        n_classes = 10177
    elif data_name == 'svhn':
        n_channels = 3
        W = H = 32
        n_classes = 10

    return n_channels, W, H, n_classes