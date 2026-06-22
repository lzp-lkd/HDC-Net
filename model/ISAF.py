# 文件名: vit_seg_modeling_resnet_skip_isaf.py
# coding=utf-8
import math
from os.path import join as pjoin
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== ISAF 模块 START ====================

class ISAFSpatialAttention(nn.Module):
    """ISAF中的空间注意力模块"""

    def __init__(self, kernel_size=7):
        super(ISAFSpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ISAFModule(nn.Module):
    """
    ISAF (Interactive Spatial-Aware Fusion)
    双向门控注意力融合模块（论文最终版）
    """

    def __init__(self, channels_in, reduction=16, activation=nn.ReLU(inplace=True)):
        super(ISAFModule, self).__init__()

        # 1. 交互式通道注意力
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channels_in * 2, channels_in * 2 // reduction, kernel_size=1, bias=False),
            activation,
            nn.Conv2d(channels_in * 2 // reduction, channels_in * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 2. 交互式空间注意力
        self.spatial_attention = ISAFSpatialAttention(kernel_size=7)

        # 3. 分支特定融合
        self.fusion_conv_rgb = nn.Conv2d(channels_in, channels_in, kernel_size=1, bias=True)
        self.fusion_conv_depth = nn.Conv2d(channels_in, channels_in, kernel_size=1, bias=True)

    def forward(self, rgb, depth):
        # ===== Stage 1: Channel Interaction =====
        combined = torch.cat((rgb, depth), dim=1)
        weights = F.adaptive_avg_pool2d(combined, 1)
        weights = self.channel_attention(weights)
        att_rgb, att_depth = torch.chunk(weights, 2, dim=1)

        rgb_refined = rgb * att_rgb
        depth_refined = depth * att_depth

        # ===== Stage 2: Spatial Interaction =====
        combined_feat = rgb_refined + depth_refined
        spatial_map = self.spatial_attention(combined_feat)

        # ===== Stage 3: Gated Fusion =====
        gated_rgb = rgb_refined * spatial_map
        gated_depth = depth_refined * spatial_map

        shared_info = gated_rgb + gated_depth

        fused_rgb = self.fusion_conv_rgb(shared_info)
        fused_depth = self.fusion_conv_depth(shared_info)

        # ===== Stage 4: Residual Enhancement =====
        out_rgb = rgb + fused_rgb
        out_depth = depth + fused_depth

        return out_rgb, out_depth


# ==================== ISAF 模块 END ====================


def np2th(weights, conv=False):
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))
        y = self.relu(residual + y)
        return y


class FuseResNetV2(nn.Module):
    """ResNetV2 + ISAF"""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width
        self.activation = nn.ReLU(inplace=True)

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        self.rootd = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(1, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        # ===== ISAF 层 =====
        self.isaf_layer0 = ISAFModule(width, activation=self.activation)
        self.isaf_layer1 = ISAFModule(width * 4, activation=self.activation)
        self.isaf_layer2 = ISAFModule(width * 8, activation=self.activation)
        self.isaf_layer3 = ISAFModule(width * 16, activation=self.activation)

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width * 4, cmid=width))] +
                [(f'unit{i}', PreActBottleneck(width * 4, width * 4, width))
                 for i in range(2, block_units[0] + 1)]
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(width * 4, width * 8, width * 2, stride=2))] +
                [(f'unit{i}', PreActBottleneck(width * 8, width * 8, width * 2))
                 for i in range(2, block_units[1] + 1)]
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(width * 8, width * 16, width * 4, stride=2))] +
                [(f'unit{i}', PreActBottleneck(width * 16, width * 16, width * 4))
                 for i in range(2, block_units[2] + 1)]
            ))),
        ]))

        self.bodyd = self.body  # 深度分支结构相同

    def forward(self, x, y):
        features = []
        b, c, in_size, _ = x.size()

        x = self.root(x)
        y = self.rootd(y)

        # ===== ISAF =====
        x, y = self.isaf_layer0(x, y)
        features.append(x)

        x = nn.MaxPool2d(3, 2)(x)
        y = nn.MaxPool2d(3, 2)(y)

        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            y = self.bodyd[i](y)

            if i == 0:
                x, y = self.isaf_layer1(x, y)
            if i == 1:
                x, y = self.isaf_layer2(x, y)

            features.append(x)

        x = self.body[-1](x)
        y = self.bodyd[-1](y)

        x, y = self.isaf_layer3(x, y)

        return x, y, features[::-1]