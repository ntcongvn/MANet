from . import *
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

"""Reference: https://github.com/ReaFly/ACSNet/blob/master/utils/loss.py"""

"""BCE loss"""

class BCELoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss(weight=weight, reduction=reduction)

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        loss = self.bceloss(pred_flat, target_flat)

        return loss


"""Dice loss"""


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1

        size = pred.size(0)

        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


"""BCE + DICE Loss"""


class BceDiceLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss(weight, reduction)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = diceloss + bceloss

        return loss


""" Deep Supervision Loss"""

class IOULoss(nn.Module):
    def __init__(self):
        super(IOULoss, self).__init__()

    def forward(self, pred, target):
        #prob = torch.sigmoid(pred)
        prob=pred
        alpha = 0.5
        beta  = 0.5

        p0 = prob
        p1 = 1 - prob
        g0 = target
        g1 = 1 - target

        num = torch.sum(p0 * g0)
        den = num + alpha * torch.sum(p0 * g1) + beta * torch.sum(p1 * g0)
        
        loss = 1 - num / (den + 1e-5)
        return loss


def DeepSupervisionLoss(pred, gt):
    
    gt=gt.detach().clone()
    gt[gt<0.5]=0
    gt[gt>=0.5]=1
    #d0=torch.sigmoid(pred[0])
    d1=torch.sigmoid(pred[1])
    d2=torch.sigmoid(pred[2])
    d3=torch.sigmoid(pred[3])
    #d4=torch.sigmoid(pred[4])

    criterion = BceDiceLoss()

    #loss0 = criterion(d0, gt)
    
    gt = F.interpolate(gt, scale_factor=0.5, mode='trilinear', align_corners=True,recompute_scale_factor= False)
    gt[gt<0.5]=0
    gt[gt>=0.5]=1
    loss1 = criterion(d1, gt)
    gt = F.interpolate(gt, scale_factor=0.5, mode='trilinear', align_corners=True,recompute_scale_factor= False)
    gt[gt<0.5]=0
    gt[gt>=0.5]=1
    loss2 = criterion(d2, gt)
    gt = F.interpolate(gt, scale_factor=0.5, mode='trilinear', align_corners=True,recompute_scale_factor= False)
    gt[gt<0.5]=0
    gt[gt>=0.5]=1
    loss3 = criterion(d3, gt)
    
    #gt = F.interpolate(gt, scale_factor=0.5, mode='trilinear', align_corners=True,recompute_scale_factor= False)
    #loss4 = criterion(d4, gt)

    #loss0
    return  loss1 + loss2 + loss3 #+ loss4


