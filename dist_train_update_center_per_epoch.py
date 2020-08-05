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
import random
from utils import *
        
def dist_train(epoch, encoder, center, encoder_optimizer, datasetloader, device, n_classes, verbose=1):   
    encoder.train()

    def recursion_change_bn(module):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.track_running_stats = 1
        else:
            for i, (name, module1) in enumerate(module._modules.items()):
                module1 = recursion_change_bn(module1)
            return module
    for i, (name, module) in enumerate(encoder._modules.items()):
            module = recursion_change_bn(encoder)
            encoder.eval()

    for batch_idx, (data, target) in enumerate(datasetloader):
        data, target = data.to(device), target.to(device)
        encoder_optimizer.zero_grad()

        # data = data.squeeze().unsqueeze(1)

        target = target.view(-1).long()
        embedding = encoder(data)
        loss, dists, target_one_hot = dist_loss(embedding, target, center, n_classes) # calculate loss
        num_data = len(data)
        del embedding

        loss.backward() # bp
        encoder_optimizer.step() # update weights
        embedding = encoder(data) #
        # center = compute_center(embedding, target)# update center
        if verbose != 0:
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * num_data, len(datasetloader.dataset),
                    100. * batch_idx / len(datasetloader), loss.item()))
    return center, loss, dists, target_one_hot

def dist_test(encoder, datasetloader, device):
    encoder.eval()
    embedding = []
    target_ = []
    with torch.no_grad():
        for data_list in datasetloader:
            if type(data_list) == torch.Tensor:
                data = data_list.to(device)
                # data = data.squeeze().unsqueeze(1)
            elif type(data_list) == list:
                data = data_list[0].to(device)
                # data = data.squeeze().unsqueeze(1)
                target = data_list[1].view(-1).long()
                target_.append(target)
                del target
            embedding.append(encoder(data))
            del data, data_list

        embedding = torch.cat(embedding)
        if len(target_):
            target_ = torch.cat(target_)
    return embedding, target_

def dist_train_main(encoder, device, traindatasetloader, args):
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
    encoder_scheduler = StepLR(encoder_optimizer, step_size=args.step_size, gamma=args.gamma)
    # initialize center
    # train_embedding, train_target = dist_test(encoder, traindatasetloader, device)
    # center = compute_center(train_embedding, train_target)
    for epoch in range(1, args.epochs + 1):
        train_embedding, train_target = dist_test(encoder, traindatasetloader, device)
        center = compute_center(train_embedding, train_target)
        center, loss, dists, target_one_hot = dist_train(epoch, encoder, center, encoder_optimizer, traindatasetloader, device, args.n_classes)
        encoder_scheduler.step()
        path = os.path.join(args.save_root, args.ID, args.net)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(encoder, os.path.join(path, 'model_{}.pth'.format(epoch)))
        torch.save(center, os.path.join(path, 'center_{}.pt'.format(epoch)) )
        torch.save(loss.cpu().detach(), os.path.join(path, 'loss_{}.pt'.format(epoch)))
        torch.save(dists.cpu().detach(), os.path.join(path, 'dists_{}.pt'.format(epoch)))
        torch.save(target_one_hot.cpu().detach(), os.path.join(path, 'target_one_hot_{}.pt'.format(epoch)))
    train_embedding, train_target = dist_test(encoder, traindatasetloader, device)
    train_center = compute_center(train_embedding, train_target)


    return train_center, loss, dists, target_one_hot

def dist_test_main(encoder, train_center, device, testdatasetloader_id, testdatasetloader_ood, args):
    
    test_embedding_ID, _ = dist_test(encoder, testdatasetloader_id, device)
    test_embedding_OOD, _ = dist_test(encoder, testdatasetloader_ood, device)
    num_ID = test_embedding_ID.shape[0]
    num_OOD = test_embedding_OOD.shape[0]
    print (num_ID, num_OOD)

    # Make sure that the numbers of ID and OOD samples are equal.
    if num_OOD >= num_ID:
        index = random.sample(range(0, num_OOD), num_ID)
    else:
        index = torch.randint(low=0, high=num_OOD, size=(num_ID, ))
    test_embedding_OOD = test_embedding_OOD[index]

    test_dist_ID = euclidean_dist(test_embedding_ID, train_center)
    test_dist_OOD = euclidean_dist(test_embedding_OOD, train_center)
    del test_embedding_ID, test_embedding_OOD
    test_dist = torch.cat([test_dist_ID, test_dist_OOD], dim=0)
    test_target = torch.cat([torch.ones(test_dist_ID.shape[0]), torch.zeros(test_dist_OOD.shape[0])],dim=0)
    del test_dist_ID, test_dist_OOD
    
    # normalize = True
    # if normalize:
    #     c_max = torch.max(test_dist, axis=1)[0].reshape(-1, 1)
    #     test_dist = test_dist/c_max

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
    
    return fpr95*100, error*100, roc_auc*100, AUPR_out*100, AUPR_in*100