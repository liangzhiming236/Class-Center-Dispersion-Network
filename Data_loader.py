#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, glob
import torch, torchvision
import torch.utils.data as Data
from torch.utils.data.dataset import Dataset
from PIL import Image 


# In[2]:


def XMNIST_Dataloader(xmnist, root, tr, resize_32, num_samples, batch_size):
    use_cuda = True
    num_workers = 7
    kwargs = {'num_workers': num_workers, 'pin_memory': False} if use_cuda else {}
    if resize_32:
        trans = torchvision.transforms.Compose([
                torchvision.transforms.Resize((32, 32)),
                # torchvision.transforms.Pad(2),
                torchvision.transforms.ToTensor()])
    else: 
        trans = torchvision.transforms.ToTensor()
    
    if xmnist == 'MNIST':
        data = torchvision.datasets.MNIST(root=root, train=tr, transform=trans, download=False)
    elif xmnist == 'KMNIST':
        data = torchvision.datasets.KMNIST(root=root, train=tr, transform=trans, download=False)
    elif xmnist == 'QMNIST':
        data = torchvision.datasets.QMNIST(root=root, train=tr, transform=trans, download=False)
    elif xmnist == 'FashionMNIST':
        data = torchvision.datasets.FashionMNIST(root=root, train=tr, transform=trans, download=False)    
    else:
        raise ValueError("Only support: MNIST, KMNIST, QMNIST, FashionMNIST")
        
    if num_samples > 0:
        loader = Data.DataLoader(dataset=data, 
                                 batch_size=batch_size, 
                                 shuffle=False,
                                 sampler=Data.sampler.RandomSampler(data, num_samples=num_samples, replacement=True))
    else:
        loader = Data.DataLoader(dataset=data, 
                                 batch_size=batch_size, 
                                 shuffle=True)
    return loader


# In[3]:


def EMNIST_dataloader(xmnist, root, resize_32, num_samples, batch_size):
    use_cuda = True
    num_workers = 7
    kwargs = {'num_workers': num_workers, 'pin_memory': False} if use_cuda else {}
    
    if resize_32:
        trans = torchvision.transforms.Compose([
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.ToTensor()])
    else: 
        trans = torchvision.transforms.ToTensor()
        
    data = torchvision.datasets.EMNIST(root=root,
                                       split=xmnist,
                                       transform=trans)
    
    loader = Data.DataLoader(dataset=data, 
                                 batch_size=batch_size, 
                                 shuffle=False,
                                 sampler=Data.sampler.RandomSampler(data, num_samples=num_samples, replacement=True))
    return loader


# In[4]:


def Omniglot_dataloader(root, resize_32, num_samples, batch_size):
    use_cuda = True
    num_workers = 7
    kwargs = {'num_workers': num_workers, 'pin_memory': False} if use_cuda else {}
    
    if resize_32:
        trans = torchvision.transforms.Compose([
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.ToTensor()])
    else: 
        trans = torchvision.transforms.Compose([
                torchvision.transforms.Resize((28, 28)),
                torchvision.transforms.ToTensor()])
        
    data = torchvision.datasets.Omniglot(root=root,
                                         transform=trans)
    
    loader = Data.DataLoader(dataset=data, 
                                 batch_size=batch_size, 
                                 shuffle=False,
                                 sampler=Data.sampler.RandomSampler(data, num_samples=num_samples, replacement=True))
    return loader


class Noise_Dataset(Dataset):
    def __init__(self, img_size, distibution):
        self.transform = torchvision.transforms.ToTensor()
        self.img_size = img_size
        if distibution == 'Uniform':
            self.img_list = glob.glob(os.path.join('/4T/ml_dataset/torch_data/Uniform_noise', str(img_size), '*.png'))
        elif distibution == 'Gaussian':
            self.img_list = glob.glob(os.path.join('/4T/ml_dataset/torch_data/Gaussian_noise', str(img_size), '*.png'))
        else:
            raise ValueError("Only support: Uniform, Gaussian")


    def __getitem__(self, index):
        img = Image.open(self.img_list[index])
        img = self.transform(img)
        target = 10
        return img, target
 
    def __len__(self):
        return len(self.img_list)


def Noise_dataloader(img_size, distribution, batch_size):
    use_cuda = True
    num_workers = 7
    kwargs = {'num_workers': num_workers, 'pin_memory': False} if use_cuda else {}

    loader = Data.DataLoader(dataset=Noise_Dataset(img_size, distribution), 
                             batch_size=batch_size, 
                             **kwargs
                             )
    return loader



class Half_XMNIST_Dataset(Dataset):
    def __init__(self, root, data_name, mode, mini, resize_32):
        if data_name == 'NotMNIST':
          self.root = '/4T/ml_dataset/torch_data/NotMNIST/MNIST/raw'
        else:
          self.root = os.path.join(root, data_name, 'raw')
        self.mode = mode
        self.mini = mini
        self.resize_32 = resize_32
        if self.resize_32:
            self.transform = torchvision.transforms.Compose([
                             torchvision.transforms.Resize((32, 32)),
                             torchvision.transforms.ToTensor()])
        else:
            self.transform = torchvision.transforms.ToTensor()

        if self.mini == 40:
            self.img_list = glob.glob(os.path.join(self.root, self.mode, '[0-4]/*.png'))
        elif self.mini == 59:
            self.img_list = glob.glob(os.path.join(self.root, self.mode, '[5-9]/*.png'))
        else:
            raise ValueError("Only support: 40, 59")
        
    def __getitem__(self, index):
        img = Image.open(self.img_list[index])
        target = int(self.img_list[index].split('/')[-2])
        return self.transform(img), target
 
    def __len__(self):
        return len(self.img_list)


# In[7]:


def Half_XMNIST_Dataloader(root, data_name, batch_size, mode, mini, resize_32):
    
    num_workers = 7
    use_cuda = True
    kwargs = {'num_workers': num_workers, 'pin_memory': False} if use_cuda else {}
    dataset = Half_XMNIST_Dataset(root, data_name, mode, mini, resize_32)
    
    loader = Data.DataLoader(dataset,
                             batch_size=batch_size, 
                             shuffle=True,
                             **kwargs)
    return loader


# In[8]:


def NotMNIST_dataloader(tr, resize_32, num_samples, batch_size):
    use_cuda = True
    num_workers = 7
    kwargs = {'num_workers': num_workers, 'pin_memory': False} if use_cuda else {}
    if resize_32:
        trans = torchvision.transforms.Compose([
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.ToTensor()])
    else: 
        trans = torchvision.transforms.ToTensor()
    

    data = torchvision.datasets.MNIST(root='/4T/ml_dataset/torch_data/NotMNIST', 
                                      train=tr, 
                                      transform=trans, 
                                      download=False)

    if num_samples > 0:
        loader = Data.DataLoader(dataset=data, 
                                 batch_size=batch_size, 
                                 shuffle=False,
                                 sampler=Data.sampler.RandomSampler(data, num_samples=num_samples, replacement=True))
    else:
        loader = Data.DataLoader(dataset=data, 
                                 batch_size=batch_size, 
                                 shuffle=True)
    return loader


def ID_loader(half, root, ID, batchsize, resize_32):
    if half:
        train_loader = Half_XMNIST_Dataloader(
            root=root,
            data_name=ID,
            mode='train', 
            mini=40,
            batch_size=batchsize,
            resize_32=resize_32)

        test_loader = Half_XMNIST_Dataloader(
            root=root,
            data_name=ID,
            mode='test', 
            mini=40,
            batch_size=batchsize,
            resize_32=resize_32)

        test_loader_59 = Half_XMNIST_Dataloader(
            root=root,
            data_name=ID,
            mode='test', 
            mini=59,
            batch_size=batchsize,
            resize_32=resize_32)
        return train_loader, test_loader, test_loader_59
    
    else:
        if ID == 'NotMNIST':
            train_loader = NotMNIST_dataloader(
                tr=True,
                resize_32=resize_32,
                num_samples=-1,
                batch_size=batchsize,
            )

            test_loader = NotMNIST_dataloader(
                tr=False,
                resize_32=resize_32,
                num_samples=-1,
                batch_size=batchsize,
            )
        else:
            train_loader = XMNIST_Dataloader(
                ID,
                root=root,
                tr=True,
                resize_32=resize_32,
                num_samples=-1,
                batch_size=batchsize,
            )

            test_loader = XMNIST_Dataloader(
                ID,
                root=root,
                tr=False,
                resize_32=resize_32,
                num_samples=-1,
                batch_size=batchsize,
            )
        return train_loader, test_loader

def OOD_loader(half, root, ID, batchsize, resize_32, n_samples=-1):
    if resize_32:
        img_size = 32
    else:
        img_size = 28
    Gaussian_noise_loader = Noise_dataloader(img_size=img_size,
                                                         distribution='Gaussian', 
                                                         batch_size=batchsize)

    Uniform_noise_loader = Noise_dataloader(img_size=img_size,
                                                         distribution='Uniform', 
                                                         batch_size=batchsize)

    MNIST_loader = XMNIST_Dataloader(
        'MNIST',
        root=root,
        tr=False,
        resize_32=resize_32,
        num_samples=n_samples,
        batch_size=batchsize,
    )
    
    FashionMNIST_loader = XMNIST_Dataloader(
        'FashionMNIST',
        root=root,
        tr=False,
        resize_32=resize_32,
        num_samples=n_samples,
        batch_size=batchsize,
    )

    KMNIST_loader = XMNIST_Dataloader(
        'KMNIST',
        root=root,
        tr=False,
        resize_32=resize_32,
        num_samples=n_samples,
        batch_size=batchsize,
    )

    NotMNIST_loader = NotMNIST_dataloader(tr=False, 
                                          resize_32=resize_32, 
                                          num_samples=n_samples, 
                                          batch_size=batchsize)
    if n_samples == -1:
        n_samples = 10000


    Omniglot_loader = Omniglot_dataloader(root=root, 
                                          resize_32=resize_32, 
                                          num_samples=n_samples, 
                                          batch_size=batchsize)

    EMNIST_letter_loader = EMNIST_dataloader(
        xmnist='letters', 
        root=root, 
        resize_32=resize_32, 
        num_samples=n_samples,
        batch_size=batchsize)
    
    return MNIST_loader, FashionMNIST_loader, \
    KMNIST_loader, NotMNIST_loader, \
    EMNIST_letter_loader, Omniglot_loader, \
    Gaussian_noise_loader, Uniform_noise_loader