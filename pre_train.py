from __future__ import print_function
import numpy as np
import torch, gzip, os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from torch.optim.lr_scheduler import StepLR
from torch.nn.modules import Module

from method import *
from allDataset import *
from utils import *

def pre_train(epoch, encoder, fc_classifier, encoder_optimizer, fc_classifier_optimizer, datasetloader, device, loss_func, verbose=1):
    encoder.train()
    fc_classifier.train()
    for batch_idx, (data, target) in enumerate(datasetloader):
        data, target = data.to(device), target.to(device)
        encoder_optimizer.zero_grad()
        fc_classifier_optimizer.zero_grad()

        data = data.squeeze().unsqueeze(1)
        target = target.view(-1).long()
        embedding = encoder(data)
        output = fc_classifier(embedding)
        loss = loss_func(output, target)

        loss.backward()
        encoder_optimizer.step()
        fc_classifier_optimizer.step()
        if verbose != 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(datasetloader.dataset),
                100. * batch_idx / len(datasetloader), loss.item()))
    
def pre_test(encoder, fc_classifier, datasetloader, device):
    encoder.eval()
    fc_classifier.eval()
#     test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in datasetloader:
            data, target = data.to(device), target.to(device)
            data = data.squeeze().unsqueeze(1)
            target = target.view(-1).long()
            embedding = encoder(data)
            output = fc_classifier(embedding)
#             test_loss += F.cross_entropy(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    print('Accuracy: {}/{} ({:.4f}%)'.format(correct, len(datasetloader.dataset), 100.*correct/len(datasetloader.dataset)))


def pre_train_main(epochs, device, lr, step_size, gamma, traindatasetloader, testdatasetloader, model_name, num_classes):
    encoder = Encoder().to(device)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    encoder_scheduler = StepLR(encoder_optimizer, step_size=step_size, gamma=gamma)
    fc_classifier = FC_classifier(num_classes).to(device)
    fc_classifier_optimizer = optim.Adam(fc_classifier.parameters(), lr=lr)
    fc_classifier_scheduler = StepLR(fc_classifier_optimizer, step_size=step_size, gamma=gamma)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        pre_train(epoch, encoder, fc_classifier, encoder_optimizer, fc_classifier_optimizer, traindatasetloader, device, loss_func)
        encoder_scheduler.step()
        fc_classifier_scheduler.step()

    if not os.path.exists('./pre_models/'):
        os.mkdir('./pre_models/')    
    pre_test(encoder, fc_classifier, testdatasetloader, device)
    torch.save(encoder, './pre_models/'+model_name+'pre_model_1.pth')
    torch.save(fc_classifier, './pre_models/'+model_name+'pre_model_2.pth')
    print ("saved")
    return encoder