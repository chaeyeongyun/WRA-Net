from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import models


def dice_coefficient(pred:Tensor, target:Tensor, num_classes:int, ignore_idx=None, reduction='mean'):
    assert pred.shape[0] == target.shape[0]
    epsilon = 1e-6
    if num_classes == 2:
        dice = 0
        pred = torch.sigmoid(pred)
        # if both a and b are 1-D arrays, it is inner product of vectors(without complex conjugation)
        for batch in range(pred.shape[0]):
            pred_1d = pred[batch].view(-1)
            target_1d = target[batch].view(-1)
            inter = (pred_1d * target_1d).sum() # GT가 1인 것들의 score의 합 (TP)
            sum_sets = pred_1d.sum() + target_1d.sum() # 예측 score합 + 1의 개수 합(TP+FP + TP+FN) # binary 에서 dice coefficient는 F1 score와 동일
            dice += (2*inter+epsilon) / (sum_sets + epsilon)
            return dice / pred.shape[0]

    else:
        pred = torch.softmax(pred, dim=1).float()
        dice = []
        for c in range(num_classes):
            if c==ignore_idx:
                continue
            dice += [dice_coefficient(pred[:, c, :, :], torch.where(target==c, 1, 0), 2, ignore_idx)] # class별 dice coefficient (batch 내 평균)
        if reduction == 'mean':
            return sum(dice) / len(dice)
        if reduction == 'none':
            return dice
        

def dice_loss(pred, target, num_classes, weighted=False, ignore_idx=None):
    if not isinstance(pred, torch.Tensor) :
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(pred)}")
    if weighted:
        dice = torch.tensor(dice_coefficient(pred, target, num_classes, ignore_idx, reduction='none'))
        weights = torch.tensor(calc_weights(target, num_classes)) # nparray shape (num_classes, )
        return torch.mean((torch.tensor([1., 1., 1.]) - dice) * weights) #TODO: 제대로 되는지 test 
        
    else:
        dice = dice_coefficient(pred, target, num_classes, ignore_idx, reduction='mean')
    return 1 - dice

class DiceLoss(nn.Module):
    def __init__(self, num_classes, weighted=False, ignore_idx=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_idx = ignore_idx
        self.weighted = weighted
    
    def forward(self, pred, target):
        return dice_loss(pred, target, self.num_classes, self.weighted, self.ignore_idx)
    