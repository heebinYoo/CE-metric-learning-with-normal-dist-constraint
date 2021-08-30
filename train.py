import argparse

import numpy as np
import pandas as pd
import torch
from thop import profile, clever_format
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn.functional as F
from model import Model
from utils import recall, ImageReader, MPerClassSampler
from torch.distributions import normal
from losses import ProxyNCA_prob
# for reproducibility
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def LargestEig(x, center=True, scale=True):
    with torch.no_grad():
        n, p = x.size()
        ones = torch.ones(n).view([n, 1]).to(device)
        h = ((1 / n) * torch.mm(ones, ones.t())) if center else torch.zeros(n * n).view([n, n])
        H = torch.eye(n).to(device)  - h
        X_center = torch.mm(H.double(), x.double())
        covariance = 1 / (n - 1) * torch.mm(X_center.t(), X_center).view(p, p)
        scaling = torch.sqrt(1 / torch.diag(covariance)).double() if scale else torch.ones(p).to(device) .double()
        scaled_covariance = torch.mm(torch.diag(scaling).view(p, p), covariance)
        eigenvalues, eigenvectors = torch.linalg.eigh(scaled_covariance, 'U')
        """
        total = eigenvalues.sum()
        if k>=1:
            index = 511-k
            components = (eigenvectors[:, index:])
        else :
            eigsum = 0
            index = 0
            for i in range(512):
                eigsum = eigsum + eigenvalues[511-i]
                if eigsum >= total*k:
                    index = 511-i
                    break;
            components = (eigenvectors[:, index:])
        """

    return eigenvectors[:,1] ,scaled_covariance


def train(net, optim, feature_dim, batch_size, num_sample, num_class, threshold, eig_para):
    net.train()
    total_loss, total_correct, total_num, data_bar = 0.0, 0.0, 0, tqdm(train_data_loader, dynamic_ncols=True)
    for inputs, labels in data_bar:
        inputs, labels = inputs.to(device) , labels.to(device)
        features, classes, classes_high = net(inputs, False)
        """
        여기에 제안하는 방법이 추가되는 부분.
        """
        #get first eigenvector
        labels_set = list(set(labels.cpu().numpy()))
        label_to_indices = {label: np.where(labels.cpu().numpy() == label)[0] for label in labels_set}

        eig_vecs=torch.zeros((num_class,feature_dim)).to(device)
        low_confidence_sample =  torch.zeros((batch_size))
        # 새로운 샘플의 임베딩위치 :

        for i in range(len(labels_set)):
            num_sample =len(label_to_indices[labels_set[i]])
            if i==0 :
                new_samples_target = torch.full((num_sample,1),labels_set[0]+num_class,dtype=int)
            emp_center = features[label_to_indices[labels_set[i]]].mean(0)
            if num_sample >1 :
                eig_vecs[i],scaled_covariance = LargestEig(features[label_to_indices[labels_set[i]]])

            else : eig_vecs[i] = features[label_to_indices[labels_set[i]]]

            #샘플의 방향으로 아이겐벡터방향 정해줘야함
            if torch.dot(emp_center,eig_vecs[i]) <0 :
                eig_vecs[i]=eig_vecs[i]*-1
            with torch.no_grad():
                aug_weight = (1-eig_para)* net.fc.weight.data[labels_set[i]] + eig_para * eig_vecs[i]

            new_sample_centroid = emp_center + (aug_weight * torch.norm(emp_center) * 2)

            #샘플의 크기가 emprical mean을 mean으로 갖는 1d gaussian이라고 가정해서 기준이되는 sigma 값에 따라 low confidence sample을 뽑음
            #sigma=0.2
            length_std = torch.std(torch.norm(features[label_to_indices[labels_set[i]]],dim=1))
            inds = torch.where(torch.norm(features[label_to_indices[labels_set[i]]],dim=1)<torch.norm(emp_center)+ threshold*length_std)[0].cpu().detach()
            if len(inds) != 0:
                low_confidence_sample[label_to_indices[labels_set[i]][inds]] = 1

            new_sample_distribution = normal.Normal(new_sample_centroid, torch.norm(emp_center)/8)
            new_samples_emb = new_sample_distribution.sample([num_sample])
            features = torch.cat((features,new_samples_emb.float()),0)
            if i != 0:
                new_samples_target = torch.cat((new_samples_target.flatten(),torch.full((num_sample,1),labels_set[i]+num_class,dtype=int).flatten()),0)

        aug_features, aug_classes, aug_classes_high = net(features[batch_size:], True)
        loss_aug = loss_criterion(aug_classes,new_samples_target.to(device) )
        loss_high =0
        loss_low = 0
        if len(torch.where(low_confidence_sample == 0)[0]) != 0:
            loss_high = loss_criterion(classes_high[torch.where(low_confidence_sample == 0)[0]], labels[torch.where(low_confidence_sample == 0)[0]].to(device) )
        if len(torch.where(low_confidence_sample == 1)[0]) != 0:
            loss_low = loss_criterion(classes[torch.where(low_confidence_sample == 1)[0]], labels[torch.where(low_confidence_sample == 1)[0]].to(device) )
        loss =  loss_high +  loss_low + 0.1 * loss_aug



        #loss = loss_criterion(classes / temperature, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        pred = torch.argmax(classes, dim=-1)
        total_loss += loss.item() * inputs.size(0)
        total_correct += torch.sum(pred == labels).item()
        total_num += inputs.size(0)
        data_bar.set_description('Train Epoch {}/{} - Loss:{:.4f} - Acc:{:.2f}%'
                                 .format(epoch, num_epochs + 1, total_loss / total_num,
                                         total_correct / total_num * 100))

    return total_loss / total_num, total_correct / total_num * 100


def test(net, recall_ids):
    net.eval()
    with torch.no_grad():
        # obtain feature vectors for all data
        for key in eval_dict.keys():
            eval_dict[key]['features'] = []
            for inputs, labels in tqdm(eval_dict[key]['data_loader'], desc='processing {} data'.format(key),
                                       dynamic_ncols=True):
                features, classes, classes_high = net(inputs.to(device) , False)
                features = F.normalize(features, dim=-1)
                eval_dict[key]['features'].append(features)
            eval_dict[key]['features'] = torch.cat(eval_dict[key]['features'], dim=0)

        test_features = torch.sign(eval_dict['test']['features']).cpu()
        # compute recall metric
        if data_name == 'isc':
            dense_acc_list = recall(eval_dict['test']['features'].cpu(), test_data_set.labels, recall_ids,
                                    eval_dict['gallery']['features'].cpu(), gallery_data_set.labels)
            gallery_features = torch.sign(eval_dict['gallery']['features']).cpu()
            binary_acc_list = recall(test_features, test_data_set.labels, recall_ids,
                                     gallery_features, gallery_data_set.labels, binary=True)
        else:
            dense_acc_list = recall(eval_dict['test']['features'].cpu(), test_data_set.labels, recall_ids)
            binary_acc_list = recall(test_features, test_data_set.labels, recall_ids, binary=True)
    desc = 'Test Epoch {}/{} '.format(epoch, num_epochs + 1)
    for index, rank_id in enumerate(recall_ids):
        desc += 'R@{}:{:.2f}%[{:.2f}%] '.format(rank_id, dense_acc_list[index] * 100, binary_acc_list[index] * 100)
        results['test_dense_recall@{}'.format(rank_id)].append(dense_acc_list[index] * 100)
        results['test_binary_recall@{}'.format(rank_id)].append(binary_acc_list[index] * 100)
    print(desc)
    return dense_acc_list[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--data_path', default='D:\MetricLearning\data', type=str, help='datasets path')
    parser.add_argument('--data_name', default='CUB_200_2011', type=str, choices=['cars196', 'CUB_200_2011', 'Stanford_Online_Products', 'isc'],
                        help='dataset name')
    parser.add_argument('--crop_type', default='cropped', type=str, choices=['uncropped', 'cropped'],
                        help='crop data or not, it only works for car or cub dataset')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--feature_dim', default=2048, type=int, help='feature dim')
    parser.add_argument('--temperature', default=0.05, type=float, help='temperature used in softmax')
    parser.add_argument('--recalls', default='1,2,4,8', type=str, help='selected recall')
    parser.add_argument('--batch_size', default=75, type=int, help='train batch size')
    parser.add_argument('--num_sample', default=25, type=int, help='samples within each class')
    parser.add_argument('--num_epochs', default=30, type=int, help='train epoch number')
    parser.add_argument('--threshold', default=0.0, type=float, help='threshold for low confidence samples')
    parser.add_argument('--eigvec_para', default=0.2, type=float, help='ratio of former weight : eigenvector')
    parser.add_argument('--gpu_id', default=0, type=int, help='gpu id')

    opt = parser.parse_args()
    # args parse
    data_path, data_name, crop_type, lr = opt.data_path, opt.data_name, opt.crop_type, opt.lr
    feature_dim, temperature, batch_size, num_epochs,  gpu_id = opt.feature_dim, opt.temperature, opt.batch_size, opt.num_epochs, opt.gpu_id
    num_sample, threshold, eig_para, recalls = opt.num_sample, opt.threshold, opt.eigvec_para, [int(k) for k in opt.recalls.split(',')]
    save_name_pre = '{}_{}_{}'.format(data_name, crop_type, feature_dim)

    device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
    results = {'train_loss': [], 'train_accuracy': []}
    for recall_id in recalls:
        results['test_dense_recall@{}'.format(recall_id)] = []
        results['test_binary_recall@{}'.format(recall_id)] = []

    # dataset loader
    train_data_set = ImageReader(data_path, data_name, 'train', crop_type)
    train_sample = MPerClassSampler(train_data_set.labels, batch_size, num_sample)
    train_data_loader = DataLoader(train_data_set, batch_sampler=train_sample, num_workers=8)
    test_data_set = ImageReader(data_path, data_name, 'query' if data_name == 'isc' else 'test', crop_type)
    test_data_loader = DataLoader(test_data_set, batch_size, shuffle=False, num_workers=8)
    eval_dict = {'test': {'data_loader': test_data_loader}}
    if data_name == 'isc':
        gallery_data_set = ImageReader(data_path, data_name, 'gallery', crop_type)
        gallery_data_loader = DataLoader(gallery_data_set, batch_size, shuffle=False, num_workers=8)
        eval_dict['gallery'] = {'data_loader': gallery_data_loader}

    # model setup, model profile, optimizer config and loss definition
    model = Model(feature_dim, 2*len(train_data_set.class_to_idx)).to(device) # modify
    flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224).to(device) ,False,))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer_init = SGD([{'params': model.refactor.parameters()}, {'params': model.fc.parameters()}],
                         lr=lr, momentum=0.9, weight_decay=1e-4)
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = StepLR(optimizer, step_size=num_epochs // 2, gamma=0.1)
    #loss_criterion = ProxyNCA_prob(len(train_data_set.class_to_idx),feature_dim,scale=1).cuda()
    loss_criterion = nn.CrossEntropyLoss()
    best_recall = 0.0
    for epoch in range(1, num_epochs + 2):
        if epoch == 1:
            train_loss, train_accuracy = train(model, optimizer_init,feature_dim,batch_size,num_sample,len(train_data_set.class_to_idx), threshold,eig_para)
        else:
            train_loss, train_accuracy = train(model, optimizer,feature_dim,batch_size,num_sample,len(train_data_set.class_to_idx), threshold, eig_para)
        results['train_loss'].append(train_loss)
        results['train_accuracy'].append(train_accuracy)
        rank = test(model, recalls)
        if epoch >= 2:
            lr_scheduler.step()

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        # save database and model
        data_base = {}
        if rank > best_recall:
            best_recall = rank
            data_base['test_images'] = test_data_set.images
            data_base['test_labels'] = test_data_set.labels
            data_base['test_features'] = eval_dict['test']['features']
            if data_name == 'isc':
                data_base['gallery_images'] = gallery_data_set.images
                data_base['gallery_labels'] = gallery_data_set.labels
                data_base['gallery_features'] = eval_dict['gallery']['features']
            torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
            torch.save(data_base, 'results/{}_data_base.pth'.format(save_name_pre))
