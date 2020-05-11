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

from utils import *

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
        
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
        
        nn.MaxPool2d(2),
        nn.Dropout(0.5, inplace=True)
    )

class Encoder(nn.Module):
    def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            # conv_block(128, 256),
        )
        self.flatten = nn.Flatten()
        self.embedding = nn.Linear(1152, 64)

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.embedding(x)
        return x

class FC_classifier(nn.Module):
    def __init__(self, n_classes, embedding_dim=64):
        super(FC_classifier, self).__init__()
        self.out = nn.Linear(embedding_dim, n_classes)

    def forward(self, x):
        x = self.out(x)
        return x

def dist_loss(input, target, center):
    
    target_cpu = target
    input_cpu = input
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')
    del target, input
    
    batch_size = target_cpu.shape[0]

    n_classes = len(torch.unique(target_cpu))
    # batch_center = compute_center(input_cpu, target_cpu)
    dists = euclidean_dist(input_cpu, center)
    del center, input_cpu

    log_p_y = F.log_softmax(-dists, dim=1)
    
    target_one_hot = torch.zeros(batch_size, n_classes).scatter_(1, target_cpu.view(batch_size, 1), 1)
    del target_cpu

    d_near = dists.mul(target_one_hot).sum(1)
    d_far = dists.mul(1-target_one_hot).sum(1)
    a = 0.01
    b = a
    # loss = (d_near/(a*d_far)+b*d_far).mean()
    loss = -log_p_y.mul(target_one_hot).sum(1).mean() + (0.01*d_near).mean()
    return loss