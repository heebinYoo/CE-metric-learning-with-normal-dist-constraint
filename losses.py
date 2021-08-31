import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import copy
import math


class ConfidenceControlLoss(nn.Module):
    def __init__(self, in_features, out_features):
        '''
        Confidence Contol Softmax Loss
        Three 'sample_type' available: ['high', 'low', 'aug']
        임베딩 스페이스에서의 위치에 따라 컨피던스가 작은 애들이 low, augmented sample은 aug
        '''
        super(ConfidenceControlLoss, self).__init__()

        self.loss = nn.CrossEntropyLoss()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))


    def forward(self, x, labels,sample_type):

        assert sample_type in  ['high', 'low', 'aug', None]
        if sample_type == 'high':
            weight_from = int(0)
            weight_to = int(self.weight.data.shape[0] /2)
        if sample_type == 'low':
            weight_from = int(0)
            weight_to = int(self.weight.data.shape[0])
        if sample_type == 'aug':
            weight_from = int(0)
            weight_to = int(self.weight.data.shape[0])

        #x = self.convlayers(x)
        #output = x.matmul(F.normalize(self.weight, dim=-1).t())
        output = x.matmul(self.weight.t())
        output= output[:,weight_from:weight_to]
        cc_loss = self.loss(output,labels)

        return cc_loss

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

class AngularPenaltySMLoss(nn.Module):

    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers:

        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in  ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)

def pairwise_distance(a, squared=False):
    """Computes the pairwise distance matrix with numerical stability."""
    pairwise_distances_squared = torch.add(
        a.pow(2).sum(dim=1, keepdim=True).expand(a.size(0), -1),
        torch.t(a).pow(2).sum(dim=0, keepdim=True).expand(a.size(0), -1)
    ) - 2 * (
                                     torch.mm(a, torch.t(a))
                                 )

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.clamp(
        pairwise_distances_squared, min=0.0
    )

    # Get the mask where the zero distances are at.
    error_mask = torch.le(pairwise_distances_squared, 0.0)
    #print(error_mask.sum())
    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = torch.sqrt(
            pairwise_distances_squared + error_mask.float() * 1e-16
        )

    # Undo conditionally adding 1e-16.
    #pairwise_distances = torch.mul(pairwise_distances,(error_mask == False).float())

    # Explicitly set diagonals to zero.
    mask_offdiagonals = 1 - torch.eye(
        *pairwise_distances.size(),
        device=pairwise_distances.device
    )
    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

    return pairwise_distances



def binarize_and_smooth_labels(T, nb_classes, smoothing_const = 0):
    import sklearn.preprocessing
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    T = torch.FloatTensor(T).cuda()

    return T


class ProxyNCA_prob(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale, **kwargs):
        torch.nn.Module.__init__(self)
        self.nb_classes = nb_classes
        print(nb_classes)
        self.proxies = torch.nn.Parameter(torch.randn(2*nb_classes, sz_embed) / 8)
        self.scale = scale

    def forward(self, X, T, low):
        P = self.proxies
        #note: self.scale is equal to sqrt(1/T)
        # in the paper T = 1/9, therefore, scale = sart(1/(1/9)) = sqrt(9) = 3
        #  we need to apply sqrt because the pairwise distance is calculated as norm^2
        """
        P = self.scale * F.normalize(P, p = 2, dim = -1)
        X = self.scale * F.normalize(X, p = 2, dim = -1)
        """



        if low ==True:
            D = pairwise_distance(
                torch.cat(
                    [X, P]
                ),
                squared = True
            )[:X.size()[0], X.size()[0]:]
            T = binarize_and_smooth_labels(
                T = T, nb_classes = 2 * self.nb_classes, smoothing_const = 0
            )

        else:
            D = pairwise_distance(
                torch.cat(
                    [X, P[:self.nb_classes]]
                ),
                squared = True
            )[:X.size()[0], X.size()[0]:]
            T = binarize_and_smooth_labels(
                T = T, nb_classes = self.nb_classes, smoothing_const = 0
            )




        loss = torch.sum(- T * F.log_softmax(-D, -1), -1)
        loss = loss.mean()
        return loss