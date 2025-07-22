import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import NECKS


@NECKS.register_module
class MultiScalePFC(nn.Module):
    """用于多尺度特征的参数化特征连接器

    Args:
        in_channels_list (list): 输入特征的通道数列表。
        out_channels (int): 输出通道数。
        dropout (float): Dropout率。默认: 0.0。
    """

    def __init__(self, in_channels_list, out_channels, dropout=0.0):
        super(MultiScalePFC, self).__init__()
        self.dropout = dropout
        self.num_levels = len(in_channels_list)

        # 为每个尺度特征创建降维层
        self.reduction_convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for in_channels in in_channels_list:
            self.reduction_convs.append(nn.Conv2d(in_channels, out_channels, 1))
            self.bns.append(nn.BatchNorm2d(out_channels))

        # 全局平均池化后的融合层
        self.fc = nn.Linear(out_channels * self.num_levels, out_channels)
        self.bn_fc = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        """初始化权重"""
        assert init_linear in ['normal', 'kaiming'], \
            "Undefined init_linear: {}".format(init_linear)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_linear == 'normal':
                    nn.init.normal_(m.weight, 0, std)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        """前向传播函数

        Args:
            inputs: 多尺度特征的元组

        Returns:
            x: 形状为(B, out_channels)的特征
        """
        assert isinstance(inputs, tuple) and len(inputs) == self.num_levels

        outs = []
        for i in range(self.num_levels):
            # 应用降维卷积和BN
            x = self.reduction_convs[i](inputs[i])
            x = self.bns[i](x)
            x = F.relu(x)

            # 全局平均池化
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.view(x.size(0), -1)
            outs.append(x)

        # 拼接多个尺度的特征
        x = torch.cat(outs, dim=1)

        # 最终全连接层
        x = self.fc(x)
        x = self.bn_fc(x)
        x = self.drop(x)

        return x