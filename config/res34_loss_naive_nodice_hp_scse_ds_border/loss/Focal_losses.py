import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average
#
#     def forward(self, input, target):
#         if input.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1,1)
#
#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())
#
#         if self.alpha is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0,target.data.view(-1))
#             logpt = logpt * Variable(at)
#
#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average: return loss.mean()
#         else: return loss.sum()


# class FocalLoss(nn.Module):
#     def __init__(self,
#                  alpha=0.25,
#                  gamma=2,
#                  reduction='mean',
#                  ignore_lb=0):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#         self.ignore_lb = ignore_lb
#
#     def forward(self, logits, label):
#         '''
#         args: logits: tensor of shape (N, C, H, W)
#         args: label: tensor of shape(N, H, W)
#         '''
#         # overcome ignored label
#         ignore = label.data.cpu() == self.ignore_lb
#         n_valid = (ignore == 0).sum()
#         label[ignore] = 0
#
#         ignore = ignore.nonzero()
#         _, M = ignore.size()
#         a, *b = ignore.chunk(M, dim=1)
#         mask = torch.ones_like(logits)
#         mask[[a, torch.arange(mask.size(1)), *b]] = 0
#
#         # compute loss
#         probs = torch.sigmoid(logits)
#         lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
#         pt = torch.where(lb_one_hot == 1, probs, 1-probs)
#         alpha = self.alpha*lb_one_hot + (1-self.alpha)*(1-lb_one_hot)
#         loss = -alpha*((1-pt)**self.gamma)*torch.log(pt + 1e-12)
#         loss[mask == 0] = 0
#         print(loss.size())
#         if self.reduction == 'mean':
#             loss = loss.sum(dim=1).sum()/n_valid
#         return loss




#https://github.com/marvis/pytorch-yolo2/blob/master/FocalLoss.py
#https://github.com/unsky/focal-loss
class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, size_average=True):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit, target, class_weight=None, type='softmax'):
        target = target.view(-1, 1).long()
        if type=='sigmoid':
            if class_weight is None:
                class_weight = [1]*2 #[0.5, 0.5]
            prob   = F.sigmoid(logit)
            prob   = prob.view(-1, 1)
            prob   = torch.cat((1-prob, prob), 1)
            select = Variable(torch.FloatTensor(len(prob), 2).zero_().cuda())
            select.scatter_(1, target, 1.)
        elif  type=='softmax':
            B,C,H,W = logit.size()
            if class_weight is None:
                class_weight =[1]*C #[1/C]*C
            logit   = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob    = F.softmax(logit,1)
            select  = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)
        class_weight = Variable(torch.FloatTensor(class_weight).cuda().view(-1,1))
        class_weight = torch.gather(class_weight, 0, target)
        prob       = (prob*select).sum(1).view(-1,1)
        prob       = torch.clamp(prob,1e-8,1-1e-8)
        batch_loss = - class_weight *(torch.pow((1-prob), self.gamma))*prob.log()
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss
        return loss