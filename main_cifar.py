#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import numpy as np
import torch, gzip, os, argparse, glob
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from torch.optim.lr_scheduler import StepLR
from torch.nn.modules import Module

from dist_train_update_center_per_epoch import *
from utils import *

import densenet

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score

import matplotlib.pyplot as plt
import pandas as pd
# from networks import *


# In[2]:


parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
parser.add_argument('--batchsize', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 5)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--step-size', type=float, default=1, metavar='M',
                    help='Learning rate step-size (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=5041, metavar='S',
                    help='random seed (default: 5041)')
parser.add_argument('--ID', type=str, default='cifar10', help='the ID data')
parser.add_argument('--net', type=str, default='densenet', 
                    help='the network, densenet or wideresnet')
parser.add_argument('--data_root', default='/4T/ml_dataset/torch_data', 
                    help='path to dataset')
parser.add_argument('--save_root', default='/4T/ood/chencang1/ours/checkpoint', 
                    help='path to save model')
args = parser.parse_args(args=[])
print (args)


# In[3]:


use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# In[4]:


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((125.3/255, 123.0/255, 113.9/255), 
                         (63.0/255, 62.1/255.0, 66.7/255.0)),
])


# In[5]:


class CIFAR8020(Dataset):
    def __init__(self, root, train=True, download=False, transform=transform, num=80):
        if train:
            mode = 'train'
        else:
            mode = 'test'
        self.root = os.path.join(root, 'cifar_{}_pt'.format(num), mode)
        self.img_list = glob.glob(os.path.join(self.root, '*/*.pt'))
        
        
    def __getitem__(self, index):
        img = torch.load(self.img_list[index])
        target = int(self.img_list[index].split('/')[-2])
        return img, target
 
    def __len__(self):
        return len(self.img_list)


# In[6]:


if args.ID == 'cifar10':
    cifar = torchvision.datasets.CIFAR10
    args.n_classes = 10
if args.ID == 'cifar100':
    cifar = torchvision.datasets.CIFAR100
    args.n_classes = 100
if args.ID == 'cifar80':
    cifar = CIFAR8020
    args.n_classes = 80

trainset = cifar(root=args.data_root, train=True, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize,
                                 shuffle=True, num_workers=0)

testset = cifar(root=args.data_root, train=False, download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize,
                                 shuffle=False, num_workers=0)


# In[7]:


model = torch.load('./backbone/{}{}.pth'.format(args.net, args.n_classes))


# In[8]:


def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module

for i, (name, module) in enumerate(model._modules.items()):
    module = recursion_change_bn(model)
# model.eval()


# In[ ]:





# In[9]:


model.fc = nn.Linear(342, 10)
model.cuda()


# In[10]:


# model.eval()
# n_true = 0
# for x, y in test_loader:
#     x = x.cuda()
#     y = y.cuda()
#     pred = model(x)
#     _, pred = pred.max(1)
#     n_true += int((y == pred).sum())
# accuracy = n_true/len(test_loader.dataset)
# print ('Accuracy:', accuracy)


# In[11]:


train_center = dist_train_main(model, device, train_loader, args)


# In[12]:


OOD_list = ['Imagenet', 'Imagenet_resize', 'LSUN', 'LSUN_resize', 'iSUN', 'Uniform_noise', 'Gaussian_noise']


# In[13]:


result_npy = np.zeros(shape=(len(OOD_list), 5))
for i, ood_name in enumerate(OOD_list):
    os.path.join(args.data_root, '{}'.format(ood_name))
    ood_data = torchvision.datasets.ImageFolder(os.path.join(args.data_root, '{}'.format(ood_name)), transform=transform)
    ood_loader = torch.utils.data.DataLoader(
        ood_data, 
        batch_size=args.batchsize,
        shuffle=False, 
        num_workers=2)
    result = dist_test_main(model, train_center[0], device, test_loader, ood_loader, args)
    for j in range(5):
        result_npy[i, j] = result[j]


# In[14]:


if args.ID == 'cifar80':
    cifar20 = CIFAR8020(root=args.data_root, train=False, download=False, transform=transform, num=20)
    cifar20_loader = torch.utils.data.DataLoader(cifar20, batch_size=args.batchsize,
                                     shuffle=False, num_workers=0)
    cifar20_result = dist_test_main(model, train_center[0], device, test_loader, ood_loader, args)
    result_npy = np.concatenate((np.array(cifar20_result).reshape(1, -1), result_npy), axis=0)


# In[15]:


df = pd.DataFrame(result_npy).apply(lambda x: x).round(1) 
df.to_excel('./result/ID_{}_{}.xls'.format(args.ID, args.net), header=False, index=False)


# In[ ]:




