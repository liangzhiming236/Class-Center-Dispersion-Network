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
from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score

from method import *
from allDataset import *
from utils import *
        
def dist_train(epoch, encoder, center, encoder_optimizer, datasetloader, device, verbose=1):   
    encoder.train()
    for batch_idx, (data, target) in enumerate(datasetloader):
        data, target = data.to(device), target.to(device)
        encoder_optimizer.zero_grad()

        data = data.squeeze().unsqueeze(1)
        target = target.view(-1).long()
        embedding = encoder(data)
        loss = dist_loss(embedding, target, center) # calculate loss
        num_data = len(data)
        del embedding

        loss.backward() # bp
        encoder_optimizer.step() # update weights
        embedding = encoder(data) #
        center = compute_center(embedding, target)# update center

        if verbose != 0:
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * num_data, len(datasetloader.dataset),
                    100. * batch_idx / len(datasetloader), loss.item()))
    return center, loss

def dist_test(encoder, datasetloader, device):
    encoder.eval()
    embedding = []
    target_ = []
    with torch.no_grad():
        for data_list in datasetloader:
            if type(data_list) == torch.Tensor:
                data = data_list.to(device)
                data = data.squeeze().unsqueeze(1)
            elif type(data_list) == list:
                data = data_list[0].to(device)
                data = data.squeeze().unsqueeze(1)
                target = data_list[1].view(-1).long()
                target_.append(target)
                del target
            embedding.append(encoder(data))
            del data, data_list

        embedding = torch.cat(embedding)
        if len(target_):
            target_ = torch.cat(target_)
    return embedding, target_

def dist_train_main(encoder, epochs, device, lr, step_size, gamma, traindatasetloader, save_model, patience):
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    encoder_scheduler = StepLR(encoder_optimizer, step_size=step_size, gamma=gamma)
    # initialize center
    train_embedding, train_target = dist_test(encoder, traindatasetloader, device)
    center = compute_center(train_embedding, train_target)
    loss_max = 10e5
    count = 0
    for epoch in range(1, epochs + 1):
        center, loss = dist_train(epoch, encoder, center, encoder_optimizer, traindatasetloader, device)
        encoder_scheduler.step()
        if loss_max - loss > 0.01:
            loss_max = loss
            count = 0
        else:
            count = count + 1
        if count == patience:
            print ('Early stop')
            break  
    
    train_embedding, train_target = dist_test(encoder, traindatasetloader, device)
    train_center = compute_center(train_embedding, train_target)
    # train_center = center.detach()
    return train_center

def dist_test_main(encoder, train_center, epochs, device, lr, step_size, gamma, testdatasetloader_id, testdatasetloader_ood, save_model):
    
    test_embedding_ID, label = dist_test(encoder, testdatasetloader_id, device)
    test_embedding_OOD, _ = dist_test(encoder, testdatasetloader_ood, device)
    num_ID = test_embedding_ID.shape[0]
    num_OOD = test_embedding_OOD.shape[0]

    # np.save('./id.npy', test_embedding_ID.cpu().numpy())
    # np.save('./ood.npy', test_embedding_OOD.cpu().numpy())
    # np.save('./center.npy', train_center)
    # np.save('./id_label.npy', label)

    index = torch.randint(low=0, high=num_OOD, size=(num_ID, ))
    test_embedding_OOD = test_embedding_OOD[index]

    test_dist_ID = euclidean_dist(test_embedding_ID, train_center)
    test_dist_OOD = euclidean_dist(test_embedding_OOD, train_center)
    del test_embedding_ID, test_embedding_OOD
    test_dist = torch.cat([test_dist_ID, test_dist_OOD], dim=0)
    test_target = torch.cat([torch.ones(test_dist_ID.shape[0]), torch.zeros(test_dist_OOD.shape[0])],dim=0)
    del test_dist_ID, test_dist_OOD
    
    pred_prob, pred = test_dist.min(1)
    del test_dist
    pred_prob = pred_prob.cpu().numpy()
    pred = pred.cpu().numpy()
    true = test_target.cpu().numpy()
    del test_target
    
    # ROC_AUC
    fpr, tpr, thresholds = roc_curve(true+1, pred_prob, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    # fpr at tpr=0.95
    tpr_index = np.argmin(np.abs(tpr-0.95))
    fpr95 = fpr[tpr_index]
    
    # best binary classification accuracy
    acc = 0
    for threshold in thresholds:
        if accuracy_score(true, pred_prob<threshold) > acc:
            acc = accuracy_score(true, pred_prob<threshold)
    error = 1 - acc
    
    # AUPR_in   
    precision, recall, _ = precision_recall_curve(true, np.max(pred_prob)-pred_prob, pos_label=1)
    AUPR_in = auc(recall, precision)
    
    # AUPR_out
    precision, recall, _ = precision_recall_curve(1-true, pred_prob, pos_label=1)
    AUPR_out = auc(recall, precision)
            
    print('FPR(95%tpr): {:.4f} \t Detection Error: {:.4f} \t ROC_AUC:  {:.4f} \t AUPR_out: {:.4f}\t AUPR_in: {:.4f}'.format(
        fpr95*100, error*100, roc_auc*100, AUPR_out*100, AUPR_in*100))