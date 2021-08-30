import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import copy
import math


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