import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from .utils import weight_reduce_loss
from ..registry import LOSSES
from .cross_entropy_loss import cross_entropy, _expand_binary_labels, binary_cross_entropy, partial_cross_entropy
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.nn import Parameter
import math
import os
import sys
from collections import OrderedDict


@LOSSES.register_module
class PGLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 dataset='voc',
                 reduction='mean',
                 loss_weight=1.0,
                 partial=False,
                 focal=dict(
                     focal=True,
                     balance_param=2.0,
                     gamma=2,
                     apaf=False
                 ),
                 CB_loss=dict(
                     CB_beta=0.9,
                     CB_mode='average_w'
                 ),
                 map_param=dict(
                     alpha=10.0,
                     beta=0.2,
                     gamma=0.1
                 ),
                 logit_reg=dict(
                     neg_scale=5.0,
                     init_bias=0.1
                 ),
                 reweight_func=None,
                 weight_norm=None,
                 freq_file='./class_freq.pkl',
                 class_split='./class_split.pkl',
                 apr=False):
        super(PGLoss, self).__init__()

        assert (use_sigmoid is True) or (partial is False)
        self.use_sigmoid = use_sigmoid
        self.partial = partial
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.use_sigmoid:
            if self.partial:
                self.cls_criterion = partial_cross_entropy
            else:
                self.cls_criterion = binary_cross_entropy
        else:
            self.cls_criterion = cross_entropy

        self.dataset = dataset
        self.apr = apr

        # reweighting function
        self.reweight_func = reweight_func

        # normalization (optional)
        self.weight_norm = weight_norm

        # focal loss params
        self.focal = focal['focal']
        self.gamma = focal['gamma']
        self.balance_param = focal['balance_param']
        self.apaf = focal.get('apaf', False)

        # mapping function params
        self.map_alpha = map_param['alpha']
        self.map_beta = map_param['beta']
        self.map_gamma = map_param['gamma']

        # CB loss params (optional)
        self.CB_beta = CB_loss['CB_beta']
        self.CB_mode = CB_loss['CB_mode']

        self.class_freq = torch.from_numpy(np.asarray(
            mmcv.load(freq_file)['class_freq'])).float().cuda()
        self.neg_class_freq = torch.from_numpy(
            np.asarray(mmcv.load(freq_file)['neg_class_freq'])).float().cuda()
        self.num_classes = self.class_freq.shape[0]
        self.train_num = self.class_freq[0] + self.neg_class_freq[0]

        # regularization params
        self.logit_reg = logit_reg
        self.neg_scale = logit_reg.get('neg_scale', 1.0)
        init_bias = logit_reg.get('init_bias', 0.0)
        self.init_bias = - torch.log(
            self.train_num / self.class_freq - 1) * init_bias / self.neg_scale

        self.freq_inv = torch.ones(self.class_freq.shape).cuda() / self.class_freq
        self.propotion_inv = self.train_num / self.class_freq

        # adaptive focal parameters
        self.af_pos_scale = torch.Tensor([focal['gamma']] * self.num_classes).cuda().detach()
        self.af_neg_scale = torch.Tensor([focal['gamma']] * self.num_classes).cuda().detach()

        # probability rebalancing parameters
        self.logit_pos_scale = torch.Tensor([1.] * self.num_classes).cuda().detach()
        self.logit_neg_scale = torch.Tensor([1.] * self.num_classes).cuda().detach()

        # Load and validate class split
        self.split = mmcv.load(class_split)

        # Validate split
        all_classes = set()
        for key in ['head', 'middle', 'tail']:
            if key not in self.split:
                raise ValueError(f"class_split must contain '{key}' key")
            all_classes.update(self.split[key])

        if len(all_classes) != self.num_classes:
            raise ValueError(f"class_split covers {len(all_classes)} classes, expected {self.num_classes}")

        print('\033[1;35m [PG Loss] Dataset: {} \033[0;0m'.format(self.dataset))
        print('\033[1;35m [PG Loss] Class split - Head:{} middle:{} Tail:{} \033[0;0m'.format(
            len(self.split['head']), len(self.split['middle']), len(self.split['tail'])))
        print('\033[1;35m [PG Loss] Reweight: {} | Logit reg: {} \033[0;0m'.format(
            reweight_func, logit_reg))

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.apr:
            # 设置针对不同数据集的参数
            if self.dataset == 'coco':
                # COCO-LT: 动态获取 neg_scale
                _, neg_scale_pr = self.get_scale(cls_score, label,
                                                 self.logit_pos_scale,
                                                 self.logit_neg_scale)
                pos_lambda1 = 2.0
                pos_lambda2 = 1.0
            else:
                # VOC-LT: 只对 tail 类应用特殊处理
                _, neg_scale_pr = self.get_scale(cls_score, label,
                                                 self.logit_pos_scale,
                                                 self.logit_neg_scale)
                pos_lambda1 = 1.0
                pos_lambda2 = torch.ones(self.num_classes).cuda()
                pos_lambda2[list(self.split['tail'])] = 2.0
        else:
            if self.dataset == 'coco':
                neg_scale_pr = self.logit_neg_scale.clone()
                pos_lambda1 = 2.0
                pos_lambda2 = 1.0
            else:
                neg_scale_pr = torch.ones(self.num_classes).cuda()
                neg_scale_pr[list(self.split['tail'])] = 4.0
                pos_lambda1 = 1.0
                pos_lambda2 = torch.ones(self.num_classes).cuda()
                pos_lambda2[list(self.split['tail'])] = 2.0

        # Re-balanced weighting (DB Loss 的重加权部分)
        weight = self.reweight_functions(label)

        # Probability rebalancing (PG Loss 的核心创新)
        cls_score, weight = self.logit_reg_functions(
            label.float(), cls_score, weight,
            neg_scale_pr, pos_lambda1, pos_lambda2)

        # Focal loss
        if self.focal:
            logpt = - self.cls_criterion(
                cls_score.clone(), label, weight=None, reduction='none',
                avg_factor=avg_factor)
            pt = torch.exp(logpt)
            loss = self.cls_criterion(cls_score, label.float(),
                                      weight=weight, reduction='none')

            # Adaptive focal parameters (可选)
            if self.apaf:
                pos_scale, neg_scale_focal = self.get_scale(
                    cls_score, label,
                    self.af_pos_scale, self.af_neg_scale)
                pos_scale = pos_scale.detach()
                neg_scale_focal = neg_scale_focal.detach()
            else:
                pos_scale = self.gamma
                neg_scale_focal = self.gamma

            # Focal weighting
            focal_weight = ((1 - pt) ** neg_scale_focal) * (1 - label) + \
                           ((1 - pt) ** pos_scale) * label
            loss = focal_weight * loss
            loss = self.balance_param * loss
            loss = loss.mean()
        else:
            loss = self.cls_criterion(cls_score, label.float(), weight,
                                      reduction=reduction)

        loss = self.loss_weight * loss
        return loss

    def get_scale(self, logit, label, Pos_scale, Neg_scale):
        """动态计算不同类别组的scale参数"""
        neg_scale = Neg_scale.clone()
        pos_scale = Pos_scale.clone()

        pt = torch.sigmoid(logit)
        num_samples = label.shape[0]
        pos_nums = label.sum(axis=0) + 1e-6  # 避免除零

        for k, v in self.split.items():
            v_list = list(v)
            if label[:, v_list].sum() == 0:
                continue

            pos_cls_avg = ((pt * label).sum(dim=0) / pos_nums)[v_list].cpu().detach().numpy()
            neg_cls_avg = (((1 - pt) * (1 - label)).sum(dim=0) /
                           (num_samples - pos_nums + 1e-6))[v_list].cpu().detach().numpy()

            pos_avg = np.nanmean(pos_cls_avg) + 1e-6
            neg_avg = np.nanmean(neg_cls_avg) + 1e-6

            if pos_avg > 0:
                negVpos_cls = neg_cls_avg / (pos_cls_avg + 1e-6)
                negVpos_avg = neg_avg / (pos_avg + 1e-6)

                for i in range(len(v_list)):
                    negVpos = negVpos_cls[i]
                    if np.isnan(negVpos) or negVpos <= 0:
                        negVpos = negVpos_avg if negVpos_avg > 0 else 1.0

                    # 限制scale的范围，避免过大
                    negVpos = np.clip(negVpos, 0.5, 2.0)

                    neg_scale[v_list[i]] = neg_scale[v_list[i]] * negVpos
                    pos_scale[v_list[i]] = pos_scale[v_list[i]] / negVpos

        return pos_scale, neg_scale

    def logit_reg_functions(self, labels, logits, weight,
                            neg_scale, pos_lambda1, pos_lambda2):
        """概率再平衡：调整 logits 和 weight"""
        if not self.logit_reg:
            return logits, weight

        # 添加初始bias
        if 'init_bias' in self.logit_reg:
            logits = logits + self.init_bias

        # 应用 neg_scale
        if 'neg_scale' in self.logit_reg:
            if self.dataset == 'voc':
                # VOC-LT: 更保守的策略
                # neg_scale 是 tensor，对tail类是4，其他是1
                logits = logits * ((1 - labels) * neg_scale.view(1, -1) +
                                   labels * pos_lambda1)

                if weight is not None:
                    weight = weight * ((1 - labels) / 3.0 +
                                       labels / pos_lambda2.view(1, -1))
            else:
                # COCO-LT: 动态 neg_scale
                logits = logits * ((1 - labels) * neg_scale.view(1, -1) +
                                   labels * pos_lambda1)

                if weight is not None:
                    weight = weight * ((1 - labels) / 2.0 +
                                       labels / pos_lambda2)

        return logits, weight

    def reweight_functions(self, label):
        """样本重加权"""
        if self.reweight_func is None:
            return None
        elif self.reweight_func in ['inv', 'sqrt_inv']:
            weight = self.RW_weight(label.float())
        elif self.reweight_func in 'rebalance':
            weight = self.rebalance_weight(label.float())
        elif self.reweight_func in 'CB':
            weight = self.CB_weight(label.float())
        else:
            return None

        if self.weight_norm is not None:
            if 'by_instance' in self.weight_norm:
                max_by_instance, _ = torch.max(weight, dim=-1, keepdim=True)
                weight = weight / (max_by_instance + 1e-6)
            elif 'by_batch' in self.weight_norm:
                weight = weight / (torch.max(weight) + 1e-6)

        return weight

    def rebalance_weight(self, gt_labels):
        repeat_rate = torch.sum(gt_labels.float() * self.freq_inv,
                                dim=1, keepdim=True) + 1e-6
        pos_weight = self.freq_inv.clone().detach().unsqueeze(0) / repeat_rate
        weight = torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + \
                 self.map_alpha
        return weight

    def CB_weight(self, gt_labels):
        if 'by_class' in self.CB_mode:
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, self.class_freq) + 1e-6).cuda()
        elif 'average_n' in self.CB_mode:
            avg_n = torch.sum(gt_labels * self.class_freq, dim=1, keepdim=True) / \
                    (torch.sum(gt_labels, dim=1, keepdim=True) + 1e-6)
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, avg_n) + 1e-6).cuda()
        elif 'average_w' in self.CB_mode:
            weight_ = torch.tensor((1 - self.CB_beta)).cuda() / \
                      (1 - torch.pow(self.CB_beta, self.class_freq) + 1e-6).cuda()
            weight = torch.sum(gt_labels * weight_, dim=1, keepdim=True) / \
                     (torch.sum(gt_labels, dim=1, keepdim=True) + 1e-6)
        elif 'min_n' in self.CB_mode:
            min_n, _ = torch.min(gt_labels * self.class_freq +
                                 (1 - gt_labels) * 100000, dim=1, keepdim=True)
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, min_n) + 1e-6).cuda()
        else:
            raise NameError(f"Unknown CB_mode: {self.CB_mode}")
        return weight

    def RW_weight(self, gt_labels, by_class=True):
        if 'sqrt' in self.reweight_func:
            weight = torch.sqrt(self.propotion_inv)
        else:
            weight = self.propotion_inv
        if not by_class:
            sum_ = torch.sum(weight * gt_labels, dim=1, keepdim=True)
            weight = sum_ / (torch.sum(gt_labels, dim=1, keepdim=True) + 1e-6)
        return weight