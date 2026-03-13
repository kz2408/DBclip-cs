import logging

import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

from mllt.models.plugins import GeneralizedAttention

from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer
import torch
import clip
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 gen_attention=None):
        super(BasicBlock, self).__init__()
        assert gen_attention is None, "Not implemented yet."

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 gen_attention=None):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert gen_attention is None or isinstance(gen_attention, dict)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.gen_attention = gen_attention
        self.with_gen_attention = gen_attention is not None
        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            planes,
            planes,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        # gen_attention
        if self.with_gen_attention:
            self.gen_attention_block = GeneralizedAttention(
                planes, **gen_attention)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_gen_attention:
                out = self.gen_attention_block(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   with_cp=False,
                   conv_cfg=None,
                   norm_cfg=dict(type='BN'),
                   gen_attention=None,
                   gen_attention_blocks=[]):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            build_norm_layer(norm_cfg, planes * block.expansion)[1],
        )

    layers = []
    layers.append(
        block(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            style=style,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            gen_attention=gen_attention if
            (0 in gen_attention_blocks) else None))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=1,
                dilation=dilation,
                style=style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                gen_attention=gen_attention if
                (i in gen_attention_blocks) else None))

    return nn.Sequential(*layers)


@BACKBONES.register_module
class ResNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
    """

    arch_settings = {
        10: (BasicBlock, (1, 1, 1, 1)),
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 gen_attention=None,
                 stage_with_gen_attention=((), (), (), ()),
                 with_cp=False,
                 zero_init_residual=True):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.gen_attention = gen_attention
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64

        self._make_stem_layer()

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = 64 * 2 ** i
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                gen_attention=gen_attention,
                gen_attention_blocks=stage_with_gen_attention[i])
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * 64 * 2 ** (
                len(self.stage_blocks) - 1)

        self.clip_en = False
        if self.clip_en:
            self.clip, self.pre = clip.load("ViT-B/32", device="cuda")
            self.clip_proj = nn.Linear(512, 2048)
            self.conv = nn.Conv2d(4096, 2048, kernel_size=1)

        self.Cross_en = False
        if self.Cross_en:
            self.DASI1 = DASI(256, 256)
            self.DASI2 = DASI(512, 512)
            self.DASI3 = DASI(1024, 1024)
            self.DASI4 = DASI(2048, 2048)
            self.FeedForward1 = MultiOrderGatedAggregation(256)
            self.FeedForward2 = MultiOrderGatedAggregation(512)
            self.FeedForward3 = MultiOrderGatedAggregation(1024)
            self.FeedForward4 = MultiOrderGatedAggregation(2048)
        '''
        self.MOGA_en = True
        if self.MOGA_en:
            self.FeedForward1 = MultiOrderGatedAggregation(256)
            self.FeedForward2 = MultiOrderGatedAggregation(512)
            self.FeedForward3 = MultiOrderGatedAggregation(1024)
            self.FeedForward4 = MultiOrderGatedAggregation(2048)
        '''

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self):
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        if self.clip_en:
            image_features = self.clip.encode_image(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)

            if i in self.out_indices:
                outs.append(x)

        if self.Cross_en:
            x0 = outs[0]
            x1 = outs[1]
            x2 = outs[2]
            x3 = outs[3]
            x_0 = self.DASI1(x0, x1, None)
            x_1 = self.DASI2(x1, x2, x0)
            x_2 = self.DASI3(x2, x3, x1)
            x_3 = self.DASI4(x3, None, x2)

            outs[0] = self.FeedForward1(x_0)
            outs[1] = self.FeedForward2(x_1)
            outs[2] = self.FeedForward3(x_2)
            outs[3] = self.FeedForward4(x_3)
            outs[0] = x_0
            outs[1] = x_1
            outs[2] = x_2
            outs[3] = x_3
        '''
        if self.MOGA_en:
            outs[0] = self.FeedForward1(outs[0])
            outs[1] = self.FeedForward2(outs[1])
            outs[2] = self.FeedForward3(outs[2])
            outs[3] = self.FeedForward4(outs[3])
        '''
        if self.clip_en:
            resnet_layer4 = outs[-1]
            clip_proj = self.clip_proj(image_features.float())
            clip_proj = clip_proj.unsqueeze(-1).unsqueeze(-1)
            clip_proj = clip_proj.expand(-1, -1, 7, 7)
            fused_feature = resnet_layer4 * clip_proj
            fused_feature = torch.cat([resnet_layer4, clip_proj], dim=1)  # [32, 4096, 7, 7]
            fused_feature = self.conv(fused_feature)
            outs[-1] = fused_feature

        return tuple(outs)

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


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
        # self.skips_3 = nn.Conv2d(in_features//2, out_features,
        #                          kernel_size=3, stride=2, dilation=1, padding=1)
        self.relu = nn.ReLU()

        self.gelu = nn.GELU()

    def forward(self, x, x_low, x_high):
        if x_high != None:
            x_high = self.skips_3(x_high)
            x_high = torch.chunk(x_high, 4, dim=1)
        if x_low != None:
            x_low = self.skips_2(x_low)
            x_low = F.interpolate(x_low, size=[x.size(2), x.size(3)], mode='bilinear', align_corners=True)
            x_low = torch.chunk(x_low, 4, dim=1)
        x_skip = self.skips(x)
        x = self.skips(x)
        x = torch.chunk(x, 4, dim=1)
        if x_high == None:
            x0 = self.conv(torch.cat((x[0], x_low[0]), dim=1))
            x1 = self.conv(torch.cat((x[1], x_low[1]), dim=1))
            x2 = self.conv(torch.cat((x[2], x_low[2]), dim=1))
            x3 = self.conv(torch.cat((x[3], x_low[3]), dim=1))
        elif x_low == None:
            x0 = self.conv(torch.cat((x[0], x_high[0]), dim=1))
            x1 = self.conv(torch.cat((x[0], x_high[1]), dim=1))
            x2 = self.conv(torch.cat((x[0], x_high[2]), dim=1))
            x3 = self.conv(torch.cat((x[0], x_high[3]), dim=1))
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
            # self.relu = nn.GELU()
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
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


# Complementary Feed-forward Network (CFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.06, bias=False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                   groups=hidden_features,
                                   bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2,
                                   groups=hidden_features,
                                   bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)
        self.eca = eca_layer_2d(dim)

    def forward(self, x):
        x_3, x_5 = self.project_in(x).chunk(2, dim=1)
        x1_3 = self.relu3(self.dwconv3x3(x_3))
        x1_5 = self.relu5(self.dwconv5x5(x_5))
        x = torch.cat([x1_3, x1_5], dim=1)
        x = self.project_out(x)
        x = self.eca(x)
        return x


class ElementScale(nn.Module):

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()  # 调用父类构造函数初始化模块

        # 初始化一个可训练的缩放参数self.scale，用于对输入张量进行逐元素加权。
        # 例如，当embed_dims为64且初始值为0.5时，self.scale的形状为(1, 64, 1, 1)，所有值均为0.5。
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),  # 初始缩放因子
            requires_grad=requires_grad  # 决定该参数是否参与梯度更新
        )

    def forward(self, x):
        return x * self.scale  # 返回输入与缩放因子相乘的结果


class MultiOrderDWConv(nn.Module):
    """基于膨胀深度卷积实现多阶特征提取模块。

    参数:
        embed_dims (int): 输入特征图的通道数。
        dw_dilation (list): 三个深度卷积层对应的膨胀因子。
        channel_split (list): 不同分支的通道分配比例。
    """

    def __init__(self,
                 embed_dims,
                 dw_dilation=[1, 2, 3],
                 channel_split=[1, 3, 4],
                 ):
        super(MultiOrderDWConv, self).__init__()

        # 根据channel_split计算每个分支的通道比例，例如：1/8, 3/8, 4/8
        self.split_ratio = [i / sum(channel_split) for i in channel_split]

        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)  # 第二部分通道数：约3/8 * embed_dims
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)  # 第三部分通道数：约4/8 * embed_dims
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2  # 第一部分通道数：剩余部分

        self.embed_dims = embed_dims

        # 检查参数长度和合理性：确保dw_dilation和channel_split均为3个元素，并且膨胀率在1到3之间，同时embed_dims能被分割比例整除
        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0

        # 基础深度卷积：对整个输入进行5x5卷积操作，填充值依据膨胀率计算
        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,  # 根据膨胀因子计算所需填充
            groups=self.embed_dims,  # 分组数等于通道数，实现深度卷积
            stride=1,
            dilation=dw_dilation[0],  # 膨胀率为dw_dilation[0]（通常为1）
        )
        # 第二个深度卷积：对第一分支（通道占比约3/8）应用5x5卷积
        self.DW_conv1 = nn.Conv2d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1,
            dilation=dw_dilation[1],  # 膨胀率为2
        )
        # 第三个深度卷积：对第二分支（通道占比约4/8）应用7x7卷积
        self.DW_conv2 = nn.Conv2d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1,
            dilation=dw_dilation[2],  # 膨胀率为3
        )
        # 逐点卷积，用于融合各分支特征
        self.PW_conv = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1
        )

    def forward(self, x):
        x_0 = self.DW_conv0(x)  # 对整个输入应用第一个5x5深度卷积

        # 从x_0中截取第二部分通道，输入到第二个深度卷积
        x_1 = self.DW_conv1(x_0[:, self.embed_dims_0: self.embed_dims_0 + self.embed_dims_1, ...])

        # 从x_0中截取最后一部分通道，输入到第三个深度卷积
        x_2 = self.DW_conv2(x_0[:, self.embed_dims - self.embed_dims_2:, ...])

        # 在通道维度上拼接第一部分、第二部分和第三部分的输出
        x = torch.cat([x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)

        x = self.PW_conv(x)  # 使用1x1卷积融合拼接后的多分支特征
        return x


class MultiOrderGatedAggregation(nn.Module):
    """
    实现多阶门控聚合的空间模块。

    参数:
        embed_dims (int): 输入特征图的通道数。
        attn_dw_dilation (list): 三个深度卷积层的膨胀因子。
        attn_channel_split (list): 各分支的通道分配比例。
        attn_force_fp32 (bool): 是否强制以FP32计算，默认为False。
    """

    def __init__(self,
                 embed_dims,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_force_fp32=False,
                 ):
        super(MultiOrderGatedAggregation, self).__init__()

        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32
        # 第一个1x1卷积，用于初步特征投影
        self.proj_1 = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        # 门控分支：生成门控系数
        self.gate = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        # 值分支：通过多阶深度卷积提取特征
        self.value = MultiOrderDWConv(
            embed_dims=embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )

        # 最终融合的1x1卷积层
        self.proj_2 = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        # 使用SiLU激活函数分别激活门控和值分支
        self.act_value = nn.SiLU()
        self.act_gate = nn.SiLU()

        # 使用ElementScale对特征进行微调分解
        self.sigma = ElementScale(embed_dims, init_value=1e-5, requires_grad=True)

    def feat_decompose(self, x):
        # 通过1x1卷积先进行特征投影
        x = self.proj_1(x)

        # 对投影后的特征进行全局平均池化，获得全局统计信息
        x_d = F.adaptive_avg_pool2d(x, output_size=1)

        # 利用sigma对原始特征与全局均值的差异进行微调，并将结果加回原始特征中
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)  # 应用激活函数，此处设计可根据需求调整
        return x

    def forward(self, x):
        shortcut = x.clone()  # 保存输入以便后续残差连接

        # 蓝色框部分：通过特征分解模块调整特征
        x = self.feat_decompose(x)

        # 灰色框部分：分别生成门控系数F和值特征G
        F_branch = self.gate(x)
        G_branch = self.value(x)

        # 分别对F和值特征应用SiLU激活后逐元素相乘，并通过proj_2融合
        x = self.proj_2(self.act_gate(F_branch) * self.act_gate(G_branch))
        x = x + shortcut  # 添加残差连接以保留原始信息

        return x
