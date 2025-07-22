import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from collections import OrderedDict
import os
from ..builder import BACKBONES
from ..utils.clip_utils import CLIPImageProcessor


class Bag(nn.Module):
    def __init__(self):
        super(Bag, self).__init__()

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return edge_att * p + (1 - edge_att) * i


class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups=1
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups=groups)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x


class DASI(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.bag = Bag()
        self.tail_conv = nn.Sequential(
            conv_block(in_features=out_features,
                       out_features=out_features,
                       kernel_size=(1, 1),
                       padding=(0, 0),
                       norm_type=None,
                       activation=False)
        )
        self.conv = nn.Sequential(
            conv_block(in_features=out_features // 2,
                       out_features=out_features // 4,
                       kernel_size=(1, 1),
                       padding=(0, 0),
                       norm_type=None,
                       activation=False)
        )
        self.bns = nn.BatchNorm2d(out_features)

        self.skips = conv_block(in_features=in_features,
                                out_features=out_features,
                                kernel_size=(1, 1),
                                padding=(0, 0),
                                norm_type=None,
                                activation=False)
        self.skips_2 = conv_block(in_features=in_features * 2,
                                  out_features=out_features,
                                  kernel_size=(1, 1),
                                  padding=(0, 0),
                                  norm_type=None,
                                  activation=False)
        self.skips_3 = nn.Conv2d(in_features // 2, out_features,
                                 kernel_size=3, stride=2, dilation=2, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x, x_low, x_high):
        if x_high is not None:
            x_high = self.skips_3(x_high)
            x_high = torch.chunk(x_high, 4, dim=1)
        if x_low is not None:
            x_low = self.skips_2(x_low)
            x_low = F.interpolate(x_low, size=[x.size(2), x.size(3)], mode='bilinear', align_corners=True)
            x_low = torch.chunk(x_low, 4, dim=1)
        x_skip = self.skips(x)
        x = self.skips(x)
        x = torch.chunk(x, 4, dim=1)
        if x_high is None:
            x0 = self.conv(torch.cat((x[0], x_low[0]), dim=1))
            x1 = self.conv(torch.cat((x[1], x_low[1]), dim=1))
            x2 = self.conv(torch.cat((x[2], x_low[2]), dim=1))
            x3 = self.conv(torch.cat((x[3], x_low[3]), dim=1))
        elif x_low is None:
            x0 = self.conv(torch.cat((x[0], x_high[0]), dim=1))
            x1 = self.conv(torch.cat((x[1], x_high[1]), dim=1))
            x2 = self.conv(torch.cat((x[2], x_high[2]), dim=1))
            x3 = self.conv(torch.cat((x[3], x_high[3]), dim=1))
        else:
            x0 = self.bag(x_low[0], x_high[0], x[0])
            x1 = self.bag(x_low[1], x_high[1], x[1])
            x2 = self.bag(x_low[2], x_high[2], x[2])
            x3 = self.bag(x_low[3], x_high[3], x[3])

        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)
        x += x_skip
        x = self.bns(x)
        x = self.relu(x)

        return x


class eca_layer_2d(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer_2d, self).__init__()
        padding = k_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        out = self.avg_pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x


class ElementScale(nn.Module):
    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


class MultiOrderDWConv(nn.Module):
    def __init__(self,
                 embed_dims,
                 dw_dilation=[1, 2, 3],
                 channel_split=[1, 3, 4],
                 ):
        super(MultiOrderDWConv, self).__init__()

        self.split_ratio = [i / sum(channel_split) for i in channel_split]

        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2

        self.embed_dims = embed_dims

        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0

        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,
            groups=self.embed_dims,
            stride=1,
            dilation=dw_dilation[0],
        )
        self.DW_conv1 = nn.Conv2d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1,
            dilation=dw_dilation[1],
        )
        self.DW_conv2 = nn.Conv2d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1,
            dilation=dw_dilation[2],
        )
        self.PW_conv = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1
        )

    def forward(self, x):
        x_0 = self.DW_conv0(x)

        x_1 = self.DW_conv1(x_0[:, self.embed_dims_0: self.embed_dims_0 + self.embed_dims_1, ...])

        x_2 = self.DW_conv2(x_0[:, self.embed_dims - self.embed_dims_2:, ...])

        x = torch.cat([x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)

        x = self.PW_conv(x)
        return x


class MultiOrderGatedAggregation(nn.Module):
    def __init__(self,
                 embed_dims,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_force_fp32=False,
                 ):
        super(MultiOrderGatedAggregation, self).__init__()

        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32
        self.proj_1 = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.gate = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(
            embed_dims=embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )

        self.proj_2 = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        self.act_value = nn.SiLU()
        self.act_gate = nn.SiLU()

        self.sigma = ElementScale(embed_dims, init_value=1e-5, requires_grad=True)

    def feat_decompose(self, x):
        x = self.proj_1(x)
        x_d = F.adaptive_avg_pool2d(x, output_size=1)
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)
        return x

    def forward(self, x):
        shortcut = x.clone()

        x = self.feat_decompose(x)

        F_branch = self.gate(x)
        G_branch = self.value(x)

        x = self.proj_2(self.act_gate(F_branch) * self.act_gate(G_branch))
        x = x + shortcut

        return x


class VitFeaturePyramid(nn.Module):
    """创建ViT特征的多尺度特征金字塔"""

    def __init__(self, feature_dim):
        super().__init__()
        # 下采样模块
        self.downsample = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(inplace=True)
        )

        # 上采样模块
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, feature_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True)
        )

        self.feature_dim = feature_dim

    def forward(self, x):
        # 输入x是已重排的ViT特征
        x_down = self.downsample(x)  # 下采样
        x_up = self.upsample(x)  # 上采样

        return x_up, x, x_down


@BACKBONES.register_module
class CLIPVisionTransformerEnhanced(nn.Module):
    """带有DASI和MultiOrderGatedAggregation模块的CLIP ViT-B/16模型"""

    def __init__(self,
                 img_size=224,
                 clip_model="ViT-B/16",
                 frozen_stages=-1):
        super().__init__()
        self.frozen_stages = frozen_stages
        self.img_size = img_size

        # 加载CLIP模型
        print(f"Loading CLIP model: {clip_model}")
        model, _ = clip.load(clip_model, device="cpu")
        self.clip_model = model.visual

        # 获取特征维度 (CLIP ViT-B/16 为768)
        self.num_features = self.clip_model.output_dim
        print(f"CLIP visual encoder output dimension: {self.num_features}")

        # 创建特征金字塔
        self.feature_pyramid = VitFeaturePyramid(self.num_features)

        # 特征增强模块
        self.Cross_en = True
        if self.Cross_en:
            self.DASI1 = DASI(self.num_features // 2, self.num_features // 2)
            self.DASI2 = DASI(self.num_features, self.num_features)
            self.DASI3 = DASI(self.num_features * 2, self.num_features * 2)

            self.FeedForward1 = MultiOrderGatedAggregation(self.num_features // 2)
            self.FeedForward2 = MultiOrderGatedAggregation(self.num_features)
            self.FeedForward3 = MultiOrderGatedAggregation(self.num_features * 2)

        # 冻结阶段
        self._freeze_stages()

    def _freeze_stages(self):
        """根据frozen_stages参数冻结模型的相应部分"""
        if self.frozen_stages >= 0:
            # 冻结部分或全部CLIP ViT模型
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def _reshape_vit_features(self, features):
        """将ViT的序列特征转换为空间特征图"""
        B, N, C = features.shape
        h = w = int((self.img_size // 16))  # 对于ViT-B/16，patch大小是16

        # 移除[CLS]标记
        patch_features = features[:, 1:, :]

        # 重新排列为特征图 [B, C, H, W]
        spatial_features = patch_features.transpose(1, 2).reshape(B, C, h, w)

        return spatial_features

    def init_weights(self, pretrained=None):
        # CLIP已经预训练，无需额外初始化
        pass

    def forward(self, x):
        # 获取CLIP特征
        features = self.clip_model(x)  # 输出形状为[B, 512] 对于ViT-B/16

        # 将特征重排为序列形式，以便后续处理
        if features.dim() == 2:
            B, C = features.shape
            features = features.unsqueeze(1)  # 添加序列维度 [B, 1, C]
            # 创建伪序列以便重塑为空间特征
            features = features.repeat(1, (self.img_size // 16) ** 2 + 1, 1)  # [B, N+1, C]

        # 将特征重排为空间形式
        spatial_features = self._reshape_vit_features(features)  # [B, C, H/16, W/16]

        # 创建多尺度特征金字塔
        x_up, x_mid, x_down = self.feature_pyramid(spatial_features)

        outs = [x_up, x_mid, x_down]  # 多尺度特征列表

        # 应用DASI和MultiOrderGatedAggregation增强特征
        if self.Cross_en:
            x0 = outs[0]  # 上采样特征
            x1 = outs[1]  # 原始特征
            x2 = outs[2]  # 下采样特征

            x_0 = self.DASI1(x0, x1, None)
            x_1 = self.DASI2(x1, x2, x0)
            x_2 = self.DASI3(x2, None, x1)

            outs[0] = self.FeedForward1(x_0)
            outs[1] = self.FeedForward2(x_1)
            outs[2] = self.FeedForward3(x_2)

        return tuple(outs)

    def train(self, mode=True):
        super(CLIPVisionTransformerEnhanced, self).train(mode)
        self._freeze_stages()