#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
import numpy as np
import torch, gzip, os, argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from torch.optim.lr_scheduler import StepLR
from torch.nn.modules import Module
import matplotlib.pyplot as plt
from method import *
from pre_train import *
from dist_train_update_center_per_epoch import *
from utils import *

import sys
sys.path.append("..")
import Data_loader

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score

parser = argparse.ArgumentParser(description='PyTorch XMNIST Example')
parser.add_argument('--batchsize', type=int, default=1000, metavar='N',
                    help='input batch size for training (default: 1000)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epoch', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--step-size', type=float, default=1, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=5041, metavar='S',
                    help='random seed (default: 5041)')
parser.add_argument('--resize', type=bool, default=False, help='resize to 32')
parser.add_argument('--ID', type=str, default='MNIST', help='the ID data')
parser.add_argument('--pre_train', type=bool, default=True,
                    help='Load the pre-trained model')
parser.add_argument('--patience', type=int, default=1000,
                    help='patience for earlystop.')
parser.add_argument('--half', type=bool, default=False, help='the half data')
parser.add_argument('--root', default='/4T/ml_dataset/torch_data', help='path to dataset')
parser.add_argument('--save_model', action='store_true', default=False,
                    help='For Saving the current Model')
args = parser.parse_args()
print (args)


use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


if args.half:
    args.num_classes = 5
    train_loader, test_loader, test_loader_59 = Data_loader.ID_loader(args.half, 
                                                                  args.root, 
                                                                  args.ID, 
                                                                  args.batchsize, 
                                                                  args.resize)
else:
    args.num_classes = 10
    train_loader, test_loader = Data_loader.ID_loader(args.half, 
                                                                  args.root, 
                                                                  args.ID, 
                                                                  args.batchsize, 
                                                                  args.resize)


if args.pre_train:
    save_path = './pre_models/'
    encoder = torch.load(os.path.join(save_path, 'ID_pre_{}_0-4.pth'.format(args.ID)))
    print ('Load pre-trained model')
else:
    encoder = Encoder().to(device)
    print ('Not load pre-trained model')


# encoder = torch.load('/4T/ood/nlood/torch/xmnist/pre_models/mnist0-9pre_model_1.pth')


train_center = dist_train_main(encoder, args.epoch, device, args.lr, args.step_size, args.gamma, 
            train_loader, args.save_model, args.patience)


train_center = train_center[0]


encoder.eval()


n_samples=len(test_loader.dataset)


OOD_loader = Data_loader.OOD_loader(args.half, 
                                    args.root, 
                                    args.ID, 
                                    args.batchsize, 
                                    args.resize, 
                                    n_samples)


OOD_list = list(OOD_loader)
if args.half:
    OOD_list.insert(0, test_loader_59)
print (len(OOD_list))

result_npy = np.zeros(shape=(len(OOD_list), 5))
for i, ood_loader in enumerate(OOD_list):
    print ('-'*10, ood_loader, '-'*10)
    result = dist_test_main(encoder, train_center, args.epoch, device, args.lr, 
                            args.step_size, args.gamma, test_loader, ood_loader,
                            args.save_model)
    for j in range(len(result)):
        result_npy[i, j] = result[j]