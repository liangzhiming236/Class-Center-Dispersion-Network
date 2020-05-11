from __future__ import print_function
import numpy as np
import torch, gzip, os, glob
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler, Sampler
from torch.utils.data import DataLoader
from PIL import Image 


class my_RandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False
    """

    def __init__(self, data_source, replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return len(self.data_source)


class omniglot_Dataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.img_list = glob.glob(os.path.join(root, '*/*/*.png'))
        self.transform = transforms.Compose([transforms.ToTensor()])
        
    def __getitem__(self, index):
        img = Image.open(self.img_list[index])
        return self.transform(img)
 
    def __len__(self):
        return len(self.img_list)

class mnist_Dataset(Dataset):
    def __init__(self, root, mode, ood, mini):
        self.root = root
        self.mode = mode
        self.ood = ood
        self.mini = mini
        self.transform = transforms.Compose([transforms.ToTensor()])
        if self.mini:
            if self.mode == 'train' and self.ood == False:
                self.img_list = glob.glob(os.path.join(self.root, 'train', '[0-4]/*.png'))
            elif self.mode == 'test' and self.ood == False:
                self.img_list = glob.glob(os.path.join(self.root, 'test', '[0-4]/*.png'))
            elif self.mode == 'test' and self.ood == True:
                self.img_list = glob.glob(os.path.join(self.root, 'test', '[5-9]/*.png'))
        else:
            self.img_list = glob.glob(os.path.join(self.root, mode, '*/*.png'))
        
    def __getitem__(self, index):
        img = Image.open(self.img_list[index])
        if not self.ood:
            target = int(self.img_list[index].split('/')[-2])
            return self.transform(img), target
        else:
            return self.transform(img)
 
    def __len__(self):
        return len(self.img_list)


def ood_Dataloader(data_name, root, batch_size, num_samples, num_workers, use_cuda):
    kwargs = {'num_workers': num_workers, 'pin_memory': False} if use_cuda else {}
    if data_name == 'omniglot':
        dataset = omniglot_Dataset(root)
    elif data_name == 'mnist' or data_name == 'f-mnist' or data_name == 'emnist-letters' or data_name == 'not-mnist':
        dataset = mnist_Dataset(root=root, mode='test', ood=True, mini=False)
    elif data_name == 'mnist5-9' or data_name == 'f-mnist5-9':
        dataset = mnist_Dataset(root=root, mode='test', ood=True, mini=True)
    else:
        raise ValueError("Only support: omniglot, mnist, f-mnist, emnist-letters, not-mnist, mnist5-9, f-mnist5-9")
    loader_ood = DataLoader(dataset,
                     batch_size=batch_size, 
                     sampler=my_RandomSampler(dataset, num_samples=num_samples, replacement=False),
                     **kwargs)
    return loader_ood

def id_Dataloader(data_name, root, train_batch_size, test_batch_size, num_workers, use_cuda):
    kwargs = {'num_workers': num_workers, 'pin_memory': False} if use_cuda else {}
    if data_name == 'mnist' or data_name == 'f-mnist':
        train_dataset = mnist_Dataset(root=root, mode='train', ood=False, mini=False)
        test_dataset = mnist_Dataset(root=root, mode='test', ood=False, mini=False)
    elif data_name == 'mnist0-4' or data_name == 'f-mnist0-4':
        train_dataset = mnist_Dataset(root=root, mode='train', ood=False, mini=True)
        test_dataset = mnist_Dataset(root=root, mode='test', ood=False, mini=True)
    else:
        raise ValueError("Only support: mnist, f-mnist, mnist0-4, f-mnist0-4")
    train_loader_id = DataLoader(train_dataset,
                     batch_size=train_batch_size, 
                     shuffle=True,
                     **kwargs)
    test_loader_id = DataLoader(test_dataset,
                     batch_size=test_batch_size, 
                     shuffle=False,
                     **kwargs)
    return train_loader_id, test_loader_id

def noise_Dataloader(data_name, batch_size, num_samples, num_workers, use_cuda):
    kwargs = {'num_workers': num_workers, 'pin_memory': False} if use_cuda else {}
    if data_name == 'uniform':
        dataset = torch.rand(size=(10000, 1, 28, 28))
    elif data_name == 'Gaussian':
        dataset = torch.randn(size=(10000, 1, 28, 28)) + 0.5
        dataset = clip_by_tensor(dataset, 0, 1)
    else:
        raise ValueError("Only support: uniform, Gaussian")

    loader_ood = DataLoader(dataset,
                     batch_size=batch_size, 
                     sampler=my_RandomSampler(dataset, num_samples=num_samples, replacement=False),
                     **kwargs)
    return loader_ood

def clip_by_tensor(t,t_min,t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    # t=t.float()
    # t_min=t_min.float()
    # t_max=t_max.float()
 
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result