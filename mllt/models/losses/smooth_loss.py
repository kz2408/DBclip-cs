import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import LOSSES


@LOSSES.register_module
class LabelSmoothingLoss(nn.Module):
    """标签平滑损失函数

    对标签进行软化处理，防止模型过度自信
    """

    def __init__(self,
                 smoothing=0.1,
                 loss_weight=1.0,
                 use_sigmoid=True,
                 reduction='mean'):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.loss_weight = loss_weight
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """计算标签平滑损失

        Args:
            pred: 预测值
            target: 目标值
            weight: 样本权重
            avg_factor: 平均因子
            reduction_override: 覆盖默认的reduction方式

        Returns:
            torch.Tensor: 计算得到的损失
        """
        reduction = reduction_override if reduction_override else self.reduction

        if self.use_sigmoid:
            # 在多标签分类中应用标签平滑
            target_smooth = target * (1 - self.smoothing) + self.smoothing * 0.5

            # 计算二元交叉熵损失
            loss = F.binary_cross_entropy_with_logits(
                pred, target_smooth, reduction='none')
        else:
            # 单标签分类中的标签平滑
            num_classes = pred.size(-1)
            target_onehot = F.one_hot(target, num_classes=num_classes).float()

            # 应用标签平滑
            target_smooth = target_onehot * (1 - self.smoothing) + self.smoothing / num_classes

            # 计算交叉熵损失
            log_prob = F.log_softmax(pred, dim=-1)
            loss = -torch.sum(target_smooth * log_prob, dim=-1)

        # 应用样本权重
        if weight is not None:
            loss = loss * weight

        # 应用reduction
        if reduction == 'mean':
            if avg_factor is not None:
                loss = loss.sum() / avg_factor
            else:
                loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return self.loss_weight * loss