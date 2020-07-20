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

def euclidean_dist(x, y):
    x = x.to('cpu')
    y = y.to('cpu')
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def cosine_dist(x, y):
    x = x.to('cpu')
    y = y.to('cpu')
    if x.size(1) != y.size(1):
        raise Exception
    return x.mm(y.t())/torch.norm(x,p=2,dim=1,keepdim=True).mm((torch.norm(y,p=2,dim=1,keepdim=True)).t())

def compute_center(embedding, target):
    target_cpu = target
    embedding_cpu = embedding
    target_cpu = target.to('cpu')
    embedding_cpu = embedding.to('cpu')
    del target, embedding

    def c_idxs(c):
        return target_cpu.eq(c).nonzero().squeeze(1)

    classes = torch.unique(target_cpu)
    n_classes = len(classes)

    class_idxs = list(map(c_idxs, classes))
    del target_cpu
    center = torch.stack([embedding_cpu[idx_list].mean(0) for idx_list in class_idxs])
    return center