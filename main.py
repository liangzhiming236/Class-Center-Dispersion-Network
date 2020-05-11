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

from method import *
from allDataset import *
from pre_train import *
from dist_train_update_center_per_batch import *
from utils import *

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
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
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1234)')
    # parser.add_argument('--log-interval', type=int, default=100, metavar='N',
    #                     help='how many batches to wait before logging training status')
    # parser.add_argument('--save-model', action='store_true', default=False,
    #                     help='For Saving the current Model')
    parser.add_argument('--pre_train', action='store_true', default=2,
                        help='')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience for earlystop.')
    args = parser.parse_args()
    
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # -------------------------------
    #            ID: mnist09
    # ------------------------------- 


    print ('-'*10, 'ID: mnist09', '-'*10)
    root = '../data/MNIST/raw/'
    train_loader_ID, test_loader_ID = id_Dataloader('mnist', root, args.batch_size, 
                                       args.test_batch_size, num_workers=7, use_cuda=use_cuda)
    encoder = Encoder().to(device)

    pre_model_name = 'mnist0-9'
    if args.pre_train == 0:
        pass
    elif args.pre_train == 1:
        encoder = torch.load('./pre_models/'+pre_model_name+'pre_model_1.pth')
    elif args.pre_train == 2:
        encoder = pre_train_main(50, device, args.lr, args.step_size, args.gamma, 
                                 train_loader_ID, test_loader_ID, pre_model_name, 10)
    else:
        raise ValueError("Only support: 0, 1, 2. 0 for no pre-train. 1 for loading pre-trained model. 2 for training a new pre-trained model")
    
    train_center = dist_train_main(encoder, args.epoch, device, args.lr, args.step_size, args.gamma, 
                train_loader_ID, args.save_model, args.patience)
    num_samples=len(test_loader_ID.dataset)
    # Test Stage
    ood_data = {
             'f-mnist': '../data/FashionMNIST/raw/', 
             'emnist-letters': '../data/EMNIST-letter/raw/',
             'not-mnist': '../data/NotMNIST/raw/', 
             'omniglot': '../data/omniglot_resized/'
             }
    for i in range(len(ood_data)):
        print('-'*10, list(ood_data.keys())[i])
        test_loader_OOD = ood_Dataloader(list(ood_data.keys())[i], list(ood_data.values())[i], 
                               args.test_batch_size, num_samples=num_samples, num_workers=7, use_cuda=use_cuda)
        dist_test_main(encoder, train_center, args.epoch, device, args.lr, args.step_size, args.gamma, 
                        test_loader_ID, test_loader_OOD, args.save_model)
    stop        
    print('-'*10, 'Gaussian')
    test_loader_OOD = noise_Dataloader('Gaussian', args.test_batch_size, num_samples=num_samples, num_workers=7, use_cuda=use_cuda)
    dist_test_main(encoder, train_center, args.epoch, device, args.lr, args.step_size, args.gamma, 
                        test_loader_ID, test_loader_OOD, args.save_model)

    print('-'*10, 'uniform')
    test_loader_OOD = noise_Dataloader('uniform', args.test_batch_size, num_samples=num_samples, num_workers=7, use_cuda=use_cuda)
    dist_test_main(encoder, train_center, args.epoch, device, args.lr, args.step_size, args.gamma, 
                        test_loader_ID, test_loader_OOD, args.save_model)
    

    # -------------------------------
    #        ID: fashion mnist09
    # ------------------------------- 


    print ('-'*10, 'ID: f-mnist0-9', '-'*10)
    root = '../data/FashionMNIST/raw/'
    train_loader_ID, test_loader_ID = id_Dataloader('f-mnist', root, args.batch_size, 
                                       args.test_batch_size, num_workers=7, use_cuda=use_cuda)
    encoder = Encoder().to(device)

    pre_model_name = 'f-mnist0-9'
    if args.pre_train == 0:
        pass
    elif args.pre_train == 1:
        encoder = torch.load('./pre_models/'+pre_model_name+'pre_model_1.pth')
    elif args.pre_train == 2:
        encoder = pre_train_main(50, device, args.lr, args.step_size, args.gamma, 
                                 train_loader_ID, test_loader_ID, pre_model_name, 10)
    else:
        raise ValueError("Only support: 0, 1, 2. 0 for no pre-train. 1 for loading pre-trained model. 2 for training a new pre-trained model")

    train_center = dist_train_main(encoder, args.epoch, device, args.lr, args.step_size, args.gamma, 
                    train_loader_ID, args.save_model, args.patience)
    
    num_samples=len(test_loader_ID.dataset)
    # Test Stage
    ood_data = {
             'mnist': '../data/MNIST/raw/', 
             'emnist-letters': '../data/EMNIST-letter/raw/',
             'not-mnist': '../data/NotMNIST/raw/', 
             'omniglot': '../data/omniglot_resized/'}
    for i in range(len(ood_data)):
        print('-'*10, list(ood_data.keys())[i])
        test_loader_OOD = ood_Dataloader(list(ood_data.keys())[i], list(ood_data.values())[i], 
                               args.test_batch_size, num_samples=num_samples, num_workers=7, use_cuda=use_cuda)
        dist_test_main(encoder, train_center, args.epoch, device, args.lr, args.step_size, args.gamma, 
                        test_loader_ID, test_loader_OOD, args.save_model)

    print('-'*10, 'Gaussian')
    test_loader_OOD = noise_Dataloader('Gaussian', args.test_batch_size, num_samples=num_samples, num_workers=7, use_cuda=use_cuda)
    dist_test_main(encoder, train_center, args.epoch, device, args.lr, args.step_size, args.gamma, 
                        test_loader_ID, test_loader_OOD, args.save_model)

    print('-'*10, 'uniform')
    test_loader_OOD = noise_Dataloader('uniform', args.test_batch_size, num_samples=num_samples, num_workers=7, use_cuda=use_cuda)
    dist_test_main(encoder, train_center, args.epoch, device, args.lr, args.step_size, args.gamma, 
                        test_loader_ID, test_loader_OOD, args.save_model)


    # -------------------------------
    #        ID: mnist04
    # -------------------------------  


    print ('-'*10, 'ID: mnist0-4', '-'*10)
    root = '../data/MNIST/raw/'
    train_loader_ID, test_loader_ID = id_Dataloader('mnist0-4', root, args.batch_size, 
                                       args.test_batch_size, num_workers=7, use_cuda=use_cuda)
    encoder = Encoder().to(device)

    pre_model_name = 'mnist0-4'
    if args.pre_train == 0:
        pass
    elif args.pre_train == 1:
        encoder = torch.load('./pre_models/'+pre_model_name+'pre_model_1.pth')
    elif args.pre_train == 2:
        encoder = pre_train_main(50, device, args.lr, args.step_size, args.gamma, 
                                 train_loader_ID, test_loader_ID, pre_model_name, num_classes=5)
    else:
        raise ValueError("Only support: 0, 1, 2. 0 for no pre-train. 1 for loading pre-trained model. 2 for training a new pre-trained model")
    train_center = dist_train_main(encoder, args.epoch, device, args.lr, args.step_size, args.gamma, 
                    train_loader_ID, args.save_model, args.patience)

    num_samples=len(test_loader_ID.dataset)
    # Test Stage
    ood_data = {
             'mnist5-9': '../data/MNIST/raw/', 
             'f-mnist': '../data/FashionMNIST/raw/', 
             'emnist-letters': '../data/EMNIST-letter/raw/',
             'not-mnist': '../data/NotMNIST/raw/', 
             'omniglot': '../data/omniglot_resized/'}
    for i in range(len(ood_data)):
        print('-'*10, list(ood_data.keys())[i])
        test_loader_OOD = ood_Dataloader(list(ood_data.keys())[i], list(ood_data.values())[i], 
                               args.test_batch_size, num_samples=num_samples, num_workers=7, use_cuda=use_cuda)
        dist_test_main(encoder, train_center, args.epoch, device, args.lr, args.step_size, args.gamma, 
                        test_loader_ID, test_loader_OOD, args.save_model)

    print('-'*10, 'Gaussian')
    test_loader_OOD = noise_Dataloader('Gaussian', args.test_batch_size, num_samples=num_samples, num_workers=7, use_cuda=use_cuda)
    dist_test_main(encoder, train_center, args.epoch, device, args.lr, args.step_size, args.gamma, 
                        test_loader_ID, test_loader_OOD, args.save_model)

    print('-'*10, 'uniform')
    test_loader_OOD = noise_Dataloader('uniform', args.test_batch_size, num_samples=num_samples, num_workers=7, use_cuda=use_cuda)
    dist_test_main(encoder, train_center, args.epoch, device, args.lr, args.step_size, args.gamma, 
                        test_loader_ID, test_loader_OOD, args.save_model)


    # -------------------------------
    #       ID: fashion mnist04
    # -------------------------------  


    print ('-'*10, 'f-mnist0-4', '-'*10)
    root = '../data/FashionMNIST/raw/'
    train_loader_ID, test_loader_ID = id_Dataloader('f-mnist0-4', root, args.batch_size, 
                                       args.test_batch_size, num_workers=7, use_cuda=use_cuda)
    encoder = Encoder().to(device)

    pre_model_name = 'f-mnist0-4'
    if args.pre_train == 0:
        pass
    elif args.pre_train == 1:
        encoder = torch.load('./pre_models/'+pre_model_name+'pre_model_1.pth')
    elif args.pre_train == 2:
        encoder = pre_train_main(50, device, args.lr, args.step_size, args.gamma, 
                                 train_loader_ID, test_loader_ID, pre_model_name, num_classes=5)
    else:
        raise ValueError("Only support: 0, 1, 2. 0 for no pre-train. 1 for loading pre-trained model. 2 for training a new pre-trained model")
    train_center = dist_train_main(encoder, args.epoch, device, args.lr, args.step_size, args.gamma, 
                    train_loader_ID, args.save_model, args.patience)
    
    num_samples=len(test_loader_ID.dataset)
    # Test Stage
    ood_data = {
             'f-mnist5-9': '../data/FashionMNIST/raw/', 
             'mnist': '../data/MNIST/raw/', 
             'emnist-letters': '../data/EMNIST-letter/raw/',
             'not-mnist': '../data/NotMNIST/raw/', 
             'omniglot': '../data/omniglot_resized/'}
    for i in range(len(ood_data)):
        print('-'*10, list(ood_data.keys())[i])
        test_loader_OOD = ood_Dataloader(list(ood_data.keys())[i], list(ood_data.values())[i], 
                               args.test_batch_size, num_samples=num_samples, num_workers=7, use_cuda=use_cuda)
        dist_test_main(encoder, train_center, args.epoch, device, args.lr, args.step_size, args.gamma, 
                        test_loader_ID, test_loader_OOD, args.save_model)

    print('-'*10, 'Gaussian')
    test_loader_OOD = noise_Dataloader('Gaussian', args.test_batch_size, num_samples=num_samples, num_workers=7, use_cuda=use_cuda)
    dist_test_main(encoder, train_center, args.epoch, device, args.lr, args.step_size, args.gamma, 
                        test_loader_ID, test_loader_OOD, args.save_model)

    print('-'*10, 'uniform')
    test_loader_OOD = noise_Dataloader('uniform', args.test_batch_size, num_samples=num_samples, num_workers=7, use_cuda=use_cuda)
    dist_test_main(encoder, train_center, args.epoch, device, args.lr, args.step_size, args.gamma, 
                        test_loader_ID, test_loader_OOD, args.save_model)


if __name__ == '__main__':
    main()