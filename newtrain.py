import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#"None"#"0"

import argparse

import numpy as np
import pandas as pd
import torch
from thop import profile, clever_format
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn.functional as F
from model import ConfidenceControl, ConvAngularPenCC
from utils import recall, ImageReader, MPerClassSampler
from torch.distributions import normal


# for reproducibility
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train(net, optim, feature_dim, batch_size, num_class, threshold, eig_para, multi_gpu):
    net.train()
    total_loss, total_correct, total_num, data_bar = 0.0, 0.0, 0, tqdm(train_data_loader, dynamic_ncols=True)
    for inputs, labels in data_bar:

        inputs, labels = inputs.to(device), labels.to(device)
        features = net(inputs, embed=True)

        # 존재하는 레이블, 존재하는 레이블 리스트에서의 인덱스
        # 1,2,3 & 0,2,1,2 <- 1,3,2,3
        # labels_set, label_to_indices = torch.unique(labels, return_inverse=True)

        # for i in range(len(labels_set)):
        #
        #     num_sample = torch.sum(label_to_indices == i)
        #     # if i == 0:
        #     #    new_samples_target[:] = torch.full((num_sample, 1), labels_set[0] + num_class, dtype=int)
        #     chosen_features_indices = (label_to_indices == i)
        #     chosen_features = features[0:batch_size][chosen_features_indices]
        #     emp_center = chosen_features.mean(0)

        loss = net(features, labels=labels, sample_type='high')
        loss = loss.mean()

        # loss = loss_criterion(classes / temperature, labels)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss')
            break
        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.item() * inputs.size(0)
        total_num += inputs.size(0)
        data_bar.set_description('Train Epoch {}/{} - Loss:{:.4f} '
                                 .format(epoch, num_epochs + 1, total_loss / total_num))
    return total_loss / total_num


def validation(net, recall_ids):
    net.eval()
    with torch.no_grad():
        # obtain feature vectors for all data
        key = 'valid'
        eval_dict[key]['features'] = []
        for inputs, labels in tqdm(eval_dict[key]['data_loader'], desc='processing {} data'.format(key),
                                   dynamic_ncols=True):
            features = net(inputs.to(device), embed=True)
            features = F.normalize(features, dim=-1)
            eval_dict[key]['features'].append(features)
        eval_dict[key]['features'] = torch.cat(eval_dict[key]['features'], dim=0)

        valid_features = torch.sign(eval_dict['valid']['features']).cpu()
        # compute recall metric

        dense_acc_list = recall(eval_dict['valid']['features'].cpu(), val_subset_labels, recall_ids)
        binary_acc_list = recall(valid_features, val_subset_labels, recall_ids, binary=True)

    desc = 'Validation Epoch {}/{} '.format(epoch, num_epochs + 1)

    for index, rank_id in enumerate(recall_ids):
        desc += 'R@{}:{:.2f}%[{:.2f}%] '.format(rank_id, dense_acc_list[index] * 100, binary_acc_list[index] * 100)
        results['valid_dense_recall@{}'.format(rank_id)].append(dense_acc_list[index] * 100)
        results['valid_binary_recall@{}'.format(rank_id)].append(binary_acc_list[index] * 100)
    print(desc)
    return dense_acc_list[0]


def test(net, recall_ids):
    net.eval()
    with torch.no_grad():
        # obtain feature vectors for all data
        for key in eval_dict.keys():
            eval_dict[key]['features'] = []
            for inputs, labels in tqdm(eval_dict[key]['data_loader'], desc='processing {} data'.format(key),
                                       dynamic_ncols=True):
                features = net(inputs.to(device), embed=True)
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

    ############################ cuda ############################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    if device.type != 'cpu':
        print('Current cuda device:', torch.cuda.current_device())
        print('Count of using GPUs:', torch.cuda.device_count())
        device_count = torch.cuda.device_count()
    else:
        device_count = 1


    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--data_path', default='../data', type=str, help='datasets path')
    parser.add_argument('--data_name', default='cars196', type=str,
                        choices=['cars196', 'CUB_200_2011', 'sop', 'isc'],
                        help='dataset name')
    parser.add_argument('--crop_type', default='uncropped', type=str, choices=['uncropped', 'cropped'],
                        help='crop data or not, it only works for car or cub dataset')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--feature_dim', default=2048, type=int, help='feature dim')
    parser.add_argument('--temperature', default=0.05, type=float, help='temperature used in softmax')
    parser.add_argument('--recalls', default='1,2,4,8', type=str, help='selected recall')
    parser.add_argument('--batch_size', default=75, type=int, help='train batch size')
    parser.add_argument('--num_sample', default=25, type=int, help='samples within each class')
    parser.add_argument('--num_epochs', default=1, type=int, help='train epoch number')
    parser.add_argument('--threshold', default=-1.0, type=float, help='threshold for low confidence samples')
    parser.add_argument('--eigvec_para', default=0.1, type=float, help='ratio of former weight : eigenvector')
    parser.add_argument('--model_angular_penalty', default='None', type=str,
                        choices=['cosface', 'arcface', 'sphereface', 'None'], help='add angular penalty')
    parser.add_argument('--lr_gamma', default=0.1, type=float, help='learning rate scheduler gamma')

    opt = parser.parse_args()
    # args parse
    data_path, data_name, crop_type, lr = opt.data_path, opt.data_name, opt.crop_type, opt.lr
    feature_dim, temperature, batch_size, num_epochs = opt.feature_dim, opt.temperature, opt.batch_size, opt.num_epochs
    num_sample, threshold, eig_para, model_angular_penalty, lr_gamma, recalls = opt.num_sample, opt.threshold, opt.eigvec_para, opt.model_angular_penalty, opt.lr_gamma, [
        int(k) for k in
        opt.recalls.split(',')]
    save_name_pre = '{}_{}_{}'.format(data_name, crop_type, feature_dim)


    results = {'train_loss': [], 'train_accuracy': []}
    for recall_id in recalls:
        results['valid_dense_recall@{}'.format(recall_id)] = []
        results['valid_binary_recall@{}'.format(recall_id)] = []


    # dataset loader
    train_data_set = ImageReader(data_path, data_name, 'train', crop_type)
    valid_num = int(len(train_data_set) * 0.15)
    train_subset, val_subset = torch.utils.data.random_split(
        train_data_set, [len(train_data_set)-valid_num, valid_num], generator=torch.Generator().manual_seed(1))
    train_subset_labels = [train_data_set.labels[x] for x in train_subset.indices]
    val_subset_labels = [train_data_set.labels[x] for x in val_subset.indices]

    train_sample = MPerClassSampler(train_subset_labels, batch_size, num_sample)
    train_data_loader = DataLoader(train_subset, batch_sampler=train_sample, num_workers=8)

    val_data_loader = DataLoader(val_subset, batch_size, shuffle=False, num_workers=8)
    # test data, Do not use test data in training
    test_data_set = ImageReader(data_path, data_name, 'query' if data_name == 'isc' else 'test', crop_type)
    test_data_loader = DataLoader(test_data_set, batch_size, shuffle=False, num_workers=8)


    eval_dict = {'valid': {'data_loader': val_data_loader}, 'test': {'data_loader': test_data_loader}}

    if data_name == 'isc':
        gallery_data_set = ImageReader(data_path, data_name, 'gallery', crop_type)
        gallery_data_loader = DataLoader(gallery_data_set, batch_size, shuffle=False,
                                         num_workers=4 * device_count)
        eval_dict['gallery'] = {'data_loader': gallery_data_loader}



    ############################ model ############################

    multi_gpu = False
    if (device.type == 'cuda') and torch.cuda.device_count() > 1:
        print("multi GPU activate")
        multi_gpu = True
        if model_angular_penalty in ['arcface', 'sphereface', 'cosface']:
            #
            model = ConvAngularPenCC(feature_dim, 2 * len(train_data_set.class_to_idx), model_angular_penalty)
        else:
            model = ConfidenceControl(feature_dim, 2 * len(train_data_set.class_to_idx))
        model = nn.DataParallel(model)
        model.to(f'cuda:{model.device_ids[0]}')
    elif (device.type == 'cuda') and torch.cuda.device_count() == 1:
        print("single GPU activate")
        if model_angular_penalty in ['arcface', 'sphereface', 'cosface']:
            print("angular penalty")
            model = ConvAngularPenCC(feature_dim, 2 * len(train_data_set.class_to_idx), model_angular_penalty)
        else:
            model = ConfidenceControl(feature_dim, 2 * len(train_data_set.class_to_idx))
        model = model.to(device)
    else:
        print("cpu mode")
        if model_angular_penalty in ['arcface', 'sphereface', 'cosface']:
            model = ConvAngularPenCC(feature_dim, 2 * len(train_data_set.class_to_idx), model_angular_penalty)
        else:
            model = ConfidenceControl(feature_dim, 2 * len(train_data_set.class_to_idx))


    ############################ optimizer ############################

    if (device.type == 'cuda') and torch.cuda.device_count() > 1:
        optimizer_init = SGD(
            [{'params': model.module.convlayers.refactor.parameters()}, {'params': model.module.cc_loss.weight}],
            lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer_init = SGD(
            [{'params': model.convlayers.refactor.parameters()}, {'params': model.cc_loss.weight}],
            lr=lr, momentum=0.9, weight_decay=1e-4)

    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    ############################ learning rate ############################

    lr_scheduler = StepLR(optimizer, step_size=15, gamma=lr_gamma)

    ############################ loss ############################

    loss_criterion = nn.CrossEntropyLoss()

    ############################ training ############################

    best_recall = 0.0
    for epoch in range(1, num_epochs + 2):
        #에포크 단위로 train 함수를 부르고 train 함수에서는 모든 데이터셋을 iter하며 훈련하는 구조
        if epoch == 1:
            train_loss = train(model, optimizer_init, feature_dim, batch_size,
                               len(train_data_set.class_to_idx), threshold, eig_para, multi_gpu)
        else:
            train_loss = train(model, optimizer, feature_dim, batch_size,
                               len(train_data_set.class_to_idx), threshold, eig_para, multi_gpu)
        results['train_loss'].append(train_loss)
        results['train_accuracy'].append(0)
        rank = validation(model, recalls)
        if epoch >= 2:
            lr_scheduler.step()

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        # save database and model
        data_base = {}
        if rank > best_recall:
            best_recall = rank
            data_base['valid_images'] = [train_data_set.images[x] for x in val_subset.indices]
            data_base['valid_labels'] = [train_data_set.labels[x] for x in val_subset.indices]
            data_base['valid_features'] = eval_dict['valid']['features']

            torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
            torch.save(data_base, 'results/{}_data_base.pth'.format(save_name_pre))

    for recall_id in recalls:
        results['test_dense_recall@{}'.format(recall_id)] = []
        results['test_binary_recall@{}'.format(recall_id)] = []
    print(test(model, recalls))
