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
from model import ConfidenceControl, ConvAngularPenCC
from utils import recall, ImageReader, MPerClassSampler
from torch.distributions import normal
from losses import ProxyNCA_prob

# for reproducibility
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def LargestEig(x, center=True, scale=True):
    only_high=False
    with torch.no_grad():
        n, p = x.size()
        ones = torch.ones(n, dtype=torch.float).view([n, 1]).to(device)
        h = ((1 / n) * torch.mm(ones, ones.t())) if center else torch.zeros(n * n, dtype=torch.float).view([n, n]).to(
            device)
        H = torch.eye(n, dtype=torch.float).to(device) - h
        X_center = torch.mm(H, x)
        covariance = 1 / (n - 1) * torch.mm(X_center.t(), X_center).view(p, p)
        scaling = torch.sqrt(1 / torch.diag(covariance)) if scale else torch.ones(p, dtype=torch.float).to(device)
        scaled_covariance = torch.mm(torch.diag(scaling).view(p, p), covariance)
        try:
            eigenvalues, eigenvectors = torch.lobpcg(scaled_covariance, k=1, method="ortho", niter=50)
        except:
            print("can't find stable eigenvector")
            only_high=True
            return x[0],only_high
        else :
            return torch.flatten(eigenvectors), only_high
    #return eigenvectors[:, 1], ones


def train(net, optim, feature_dim, batch_size, num_sample, num_class, threshold, eig_para, multi_gpu):
    net.train()
    total_loss, total_correct, total_num, data_bar = 0.0, 0.0, 0, tqdm(train_data_loader, dynamic_ncols=True)
    for inputs, labels in data_bar:
        loss_aug= 0
        loss_low = 0
        loss_high = 0
        inputs, labels = inputs.to(device), labels.to(device)
        features = net(inputs, embed=True)

        # get first eigenvector
        labels_set, label_to_indices = torch.unique(labels, return_inverse=True)
        # label_to_indices = {label: np.where(labels.cpu().numpy() == label)[0] for label in labels_set}

        eig_vecs = torch.zeros((num_class, feature_dim)).to(device)
        low_confidence_sample = torch.zeros((batch_size))
        new_samples_target = torch.zeros((batch_size), dtype=int).to(device)
        # 새로운 샘플의 임베딩위치 :
        accumulated_num_sample = 0
        for i in range(len(labels_set)):

            num_sample = torch.sum(label_to_indices == i)
            # if i == 0:
            #    new_samples_target[:] = torch.full((num_sample, 1), labels_set[0] + num_class, dtype=int)
            chosen_features_indices = (label_to_indices == i)
            chosen_features = features[0:batch_size][chosen_features_indices]
            emp_center = chosen_features.mean(0)
            if num_sample > 1:
                eig_vecs[labels_set[i]], only_high = LargestEig(chosen_features)
            else:
                eig_vecs[labels_set[i]] = chosen_features
            if not only_high :

                # 샘플의 방향으로 아이겐벡터방향 정해줘야함
                if torch.dot(emp_center, eig_vecs[labels_set[i]]) < 0:
                    eig_vecs[labels_set[i]] = - eig_vecs[labels_set[i]]
                with torch.no_grad():
                    if multi_gpu :
                        aug_weight = (1 - eig_para) * net.module.cc_loss.weight.data[labels_set[i]] + eig_para * eig_vecs[labels_set[i]]
                    else :
                        aug_weight = (1 - eig_para) * net.cc_loss.weight.data[labels_set[i]] + eig_para * eig_vecs[labels_set[i]]

                # TBD
                new_sample_centroid = emp_center + (aug_weight * torch.norm(emp_center) * 2)

                # 샘플의 크기가 emprical mean을 mean으로 갖는 1d gaussian이라고 가정해서 기준이되는 sigma 값에 따라 low confidence sample을 뽑음
                # sigma=0.2
                normed_chosen_features = torch.norm(chosen_features, dim=1)
                length_std = torch.std(normed_chosen_features)
                inds = torch.where(normed_chosen_features < normed_chosen_features.mean() + threshold * length_std)[0]
                if inds.size()[0] != 0:
                    low_confidence_sample[torch.where(chosen_features_indices)[0][inds]] = 1

                new_sample_distribution = normal.Normal(new_sample_centroid, torch.norm(emp_center) / 8)
                new_samples_emb = new_sample_distribution.sample([num_sample])
                features = torch.cat((features, new_samples_emb.float()), 0)

                new_samples_target[accumulated_num_sample:accumulated_num_sample + num_sample] = labels_set[i] + num_class
                accumulated_num_sample += num_sample


        if features.size()[0]>batch_size :
            loss_aug = net(features[batch_size:], labels=new_samples_target, sample_type='aug')
        # loss_aug = loss_criterion(aug_classes,new_samples_target.to(device) )

        high_confidence_ind = torch.where(low_confidence_sample == 0)[0]
        low_confidence_ind = torch.where(low_confidence_sample == 1)[0]
        if high_confidence_ind.size()[0] != 0:
            loss_high = net(features[high_confidence_ind],
                            labels=labels[high_confidence_ind], sample_type='high')
        if low_confidence_ind.size()[0] != 0:
            loss_low = net(features[low_confidence_ind],
                           labels=labels[low_confidence_ind], sample_type='low')
        loss = loss_high + loss_low + 0.1 * loss_aug
        loss = loss.mean()

        # loss = loss_criterion(classes / temperature, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        # pred = torch.argmax(classes, dim=-1)
        total_loss += loss.item() * inputs.size(0)
        # total_correct += torch.sum(pred == labels).item()
        total_num += inputs.size(0)
        data_bar.set_description('Train Epoch {}/{} - Loss:{:.4f} '
                                 .format(epoch, num_epochs + 1, total_loss / total_num))

    return total_loss / total_num


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
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--data_path', default='../data', type=str, help='datasets path')
    parser.add_argument('--data_name', default='CUB_200_2011', type=str,
                        choices=['cars196', 'CUB_200_2011', 'sop', 'isc'],
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
    parser.add_argument('--eigvec_para', default=0.1, type=float, help='ratio of former weight : eigenvector')
    parser.add_argument('--model_angular_penalty', default=False, type=bool, help='add angular penalty')

    opt = parser.parse_args()
    # args parse
    data_path, data_name, crop_type, lr = opt.data_path, opt.data_name, opt.crop_type, opt.lr
    feature_dim, temperature, batch_size, num_epochs = opt.feature_dim, opt.temperature, opt.batch_size, opt.num_epochs
    num_sample, threshold, eig_para, model_angular_penalty, recalls = opt.num_sample, opt.threshold, opt.eigvec_para, opt.model_angular_penalty, [int(k) for k in
                                                                                                opt.recalls.split(',')]
    save_name_pre = '{}_{}_{}'.format(data_name, crop_type, feature_dim)

    results = {'train_loss': [], 'train_accuracy': []}
    for recall_id in recalls:
        results['test_dense_recall@{}'.format(recall_id)] = []
        results['test_binary_recall@{}'.format(recall_id)] = []

    # dataset loader
    train_data_set = ImageReader(data_path, data_name, 'train', crop_type)
    train_sample = MPerClassSampler(train_data_set.labels, batch_size, num_sample)
    train_data_loader = DataLoader(train_data_set, batch_sampler=train_sample,
                                   num_workers=4 * torch.cuda.device_count())
    test_data_set = ImageReader(data_path, data_name, 'query' if data_name == 'isc' else 'test', crop_type)
    test_data_loader = DataLoader(test_data_set, batch_size, shuffle=False, num_workers=4 * torch.cuda.device_count())
    eval_dict = {'test': {'data_loader': test_data_loader}}
    if data_name == 'isc':
        gallery_data_set = ImageReader(data_path, data_name, 'gallery', crop_type)
        gallery_data_loader = DataLoader(gallery_data_set, batch_size, shuffle=False,
                                         num_workers=4 * torch.cuda.device_count())
        eval_dict['gallery'] = {'data_loader': gallery_data_loader}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())

    # model setup, model profile, optimizer config and loss definition
    multi_gpu = False
    if (device.type == 'cuda') and torch.cuda.device_count() > 1:
        print("multi GPU activate")
        multi_gpu=True
        if model_angular_penalty :
        #
            model = ConvAngularPenCC(feature_dim, 2 * len(train_data_set.class_to_idx),'arcface')
        else:
            model = ConfidenceControl(feature_dim, 2 * len(train_data_set.class_to_idx))
        model = nn.DataParallel(model)
        model.to(f'cuda:{model.device_ids[0]}')
    elif (device.type == 'cuda') and torch.cuda.device_count() == 1:
        print("single GPU activate")
        if model_angular_penalty:
           model = ConvAngularPenCC(feature_dim, 2 * len(train_data_set.class_to_idx),'arcface')
        else:
            model = ConfidenceControl(feature_dim, 2 * len(train_data_set.class_to_idx))
        model = model.to(device)
    else:
        print("cpu mode")
        if model_angular_penalty:
            model = ConvAngularPenCC(feature_dim, 2 * len(train_data_set.class_to_idx),'arcface')
        else:
            model = ConfidenceControl(feature_dim, 2 * len(train_data_set.class_to_idx))

    #flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224).to(device), True, None))
    #flops, params = clever_format([flops, params])
    #print('# Model Params: {} FLOPs: {}'.format(params, flops))
    if (device.type == 'cuda') and torch.cuda.device_count() > 1:
        optimizer_init = SGD([{'params': model.module.convlayers.refactor.parameters()}, {'params': model.module.cc_loss.weight}],
                         lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer_init = SGD([{'params': model.convlayers.refactor.parameters()}, {'params': model.cc_loss.weight}],
                             lr=lr, momentum=0.9, weight_decay=1e-4)

    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = StepLR(optimizer, step_size=num_epochs // 2, gamma=0.1)
    # loss_criterion = ProxyNCA_prob(len(train_data_set.class_to_idx),feature_dim,scale=1).cuda()
    loss_criterion = nn.CrossEntropyLoss()
    best_recall = 0.0
    for epoch in range(1, num_epochs + 2):
        if epoch == 1:
            train_loss = train(model, optimizer_init, feature_dim, batch_size, num_sample,
                               len(train_data_set.class_to_idx), threshold, eig_para,multi_gpu )
        else:
            train_loss = train(model, optimizer, feature_dim, batch_size, num_sample, len(train_data_set.class_to_idx),
                               threshold, eig_para,multi_gpu)
        results['train_loss'].append(train_loss)
        results['train_accuracy'].append(0)
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
