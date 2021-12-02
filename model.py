import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50
from losses import AngularPenaltyCCLoss, ConfidenceControlLoss


class ConvAngularPenCC(nn.Module):
    def __init__(self, in_features, out_features, loss_type='arcface'):
        super(ConvAngularPenCC, self).__init__()
        self.convlayers = ConvNet(in_features)
        self.cc_loss = AngularPenaltyCCLoss(in_features, out_features, loss_type=loss_type)

    def forward(self, x,embed=False ,labels=None, sample_type=None ):

        if embed:
            x = self.convlayers(x)
            return x
        L = self.cc_loss(x, labels, sample_type)
        return L



class ConfidenceControl(nn.Module):
    def __init__(self, in_features, out_features):
        super(ConfidenceControl, self).__init__()
        self.convlayers = ConvNet(in_features)
        self.cc_loss = ConfidenceControlLoss(in_features, out_features)

    def forward(self, x, embed=False, labels=None, sample_type=None):

        # 처음에 임베드 true로 통과시킨 애만 임베드 false하고 통과시킬 수 있다.
        if embed:
            x = self.convlayers(x)
            return x
        L = self.cc_loss(x, labels, sample_type)
        return L



class ConvNet(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.feature = []
        for name, module in resnet50(pretrained=True).named_children():
            if isinstance(module, nn.Linear):
                continue
            self.feature.append(module)
        self.feature = nn.Sequential(*self.feature)

        # Refactor Layer
        self.refactor = nn.Linear(2048, feature_dim)
        # Classification Layer
        #self.fc = ProxyLinear(feature_dim, num_classes)

    def forward(self, x):
            # x is feature
        feature = self.feature(x)
        global_feature = torch.flatten(feature, start_dim=1)
        global_feature = F.layer_norm(global_feature, [global_feature.size(-1)])
        #feature = F.normalize(self.refactor(global_feature), dim=-1)

        feature = self.refactor(global_feature)
        #classes,classes_high = self.fc(feature)

        return feature
        #return feature, classes, classes_high

class ProxyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProxyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        #output = x.matmul(F.normalize(self.weight, dim=-1).t())
        output = x.matmul(self.weight.t())
        weight_size=int(self.weight.data.shape[0] /2)
        output_high = output[:weight_size]
        return output, output_high

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)
