#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
import cv2
import numpy as np  
import os, glob
from PIL import Image 


# In[2]:


data_root = '/4T/ml_dataset/torch_data/'
cifar80_pt_root = os.path.join(data_root, 'cifar_100_pt')
cifar20_pt_root = os.path.join(data_root, 'cifar_20_pt')


# In[3]:


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((125.3/255, 123.0/255, 113.9/255), 
                         (63.0/255, 62.1/255.0, 66.7/255.0)),
])


# In[4]:


trainset = datasets.CIFAR100(root=data_root, 
                             train=True,
                             download=False, 
                             transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, 
                                           batch_size=1,
                                           shuffle=False, 
                                           num_workers=2)

testset = datasets.CIFAR100(root=data_root, 
                            train=False, 
                            download=False, 
                            transform=transform)
test_loader = torch.utils.data.DataLoader(testset, 
                                          batch_size=1,
                                          shuffle=False, 
                                          num_workers=2)


# In[6]:


for i, (x, y) in enumerate(train_loader):
    x = x[0]
    y = int(y)
    if y < 80:
        save_root = os.path.join(cifar80_pt_root, 'train', str(y))
    else:
        save_root = os.path.join(cifar20_pt_root, 'train', str(y))
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    torch.save(x, os.path.join(save_root, '{}.pt'.format(i)))


# In[7]:


for i, (x, y) in enumerate(test_loader):
    x = x[0]
    y = int(y)
    if y < 80:
        save_root = os.path.join(cifar80_pt_root, 'test', str(y))
    else:
        save_root = os.path.join(cifar20_pt_root, 'test', str(y))
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    torch.save(x, os.path.join(save_root, '{}.pt'.format(i)))


# In[ ]:




