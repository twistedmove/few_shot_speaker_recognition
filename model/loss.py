import torch.nn.functional as F
import torch
from torch.nn.modules import Module


def nll_loss(output, target):
    return F.nll_loss(output, target)

def ce_loss(output, target):
    return F.cross_entropy(output, target)

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def prototypical_loss_test(input, target, prototypes):
    
    dists = euclidean_dist(input, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1)
    target = target.view(target.size()[0],1)

    loss = -log_p_y.gather(1,target).squeeze().view(-1).mean()
    _,y_hat = log_p_y.max(1)
    acc = y_hat.eq(target.squeeze()).float().mean()

    return loss, acc