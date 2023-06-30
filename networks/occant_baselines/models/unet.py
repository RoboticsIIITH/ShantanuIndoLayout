#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tmodels

from networks.occant_baselines.models.utils import ConvLSTMCell

norm_layer = lambda x : nn.BatchNorm2d(x, affine=True, track_running_stats=True)


# =========================== Sub-parts of the U-Net model ============================


class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            norm_layer(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            norm_layer(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class up_lstm(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up_lstm, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv1 = ConvLSTMCell(in_ch, out_ch, (3, 3), True)
        self.conv2 = ConvLSTMCell(out_ch, out_ch, (3, 3), True)

    def init_hidden(self, batch_size, image_size):
        h1, c1 = self.conv1.init_hidden(batch_size=batch_size, image_size=image_size)
        h2, c2 = self.conv2.init_hidden(batch_size=batch_size, image_size=image_size)

        return h1, c1, h2, c2

    def forward(self, x1, x2, h1, h2, c1, c2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        h1, c1 = self.conv1(x, [h1, c1])
        h2, c2 = self.conv2(h1, [h2, c2])
        return h1, c1, h2, c2


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


# =================================== Component modules ===============================


class UNetEncoder(nn.Module):
    def __init__(self, n_channels, nsf=16):
        super().__init__()
        self.inc = inconv(n_channels, nsf)
        self.down1 = down(nsf, nsf * 2)
        self.down2 = down(nsf * 2, nsf * 4)
        self.down3 = down(nsf * 4, nsf * 8)
        self.down4 = down(nsf * 8, nsf * 8)

    def forward(self, x):
        x1 = self.inc(x)  # (bs, nsf, ..., ...)
        x2 = self.down1(x1)  # (bs, nsf*2, ... ,...)
        x3 = self.down2(x2)  # (bs, nsf*4, ..., ...)
        x4 = self.down3(x3)  # (bs, nsf*8, ..., ...)
        x5 = self.down4(x4)  # (bs, nsf*8, ..., ...)

        return {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}


class UNetDecoder(nn.Module):
    def __init__(self, n_classes, nsf=16):
        super().__init__()
        self.up1 = up(nsf * 16, nsf * 4)
        self.up2 = up(nsf * 8, nsf * 2)
        self.up3 = up(nsf * 4, nsf)
        self.up4 = up(nsf * 2, nsf)
        self.outc = outconv(nsf, n_classes)

    def forward(self, xin):
        """
        xin is a dictionary that consists of x1, x2, x3, x4, x5 keys
        from the UNetEncoder
        """
        x1 = xin["x1"]  # (bs, nsf, ..., ...)
        x2 = xin["x2"]  # (bs, nsf*2, ..., ...)
        x3 = xin["x3"]  # (bs, nsf*4, ..., ...)
        x4 = xin["x4"]  # (bs, nsf*8, ..., ...)
        x5 = xin["x5"]  # (bs, nsf*8, ..., ...)

        x = self.up1(x5, x4)  # (bs, nsf*4, ..., ...)
        x = self.up2(x, x3)  # (bs, nsf*2, ..., ...)
        x = self.up3(x, x2)  # (bs, nsf, ..., ...)
        x = self.up4(x, x1)  # (bs, nsf, ..., ...)
        x = self.outc(x)  # (bs, n_classes, ..., ...)

        return x

class FPNDecoder(nn.Module):
    def __init__(self, n_classes, nsf=16):
        super().__init__()
        self.up1 = up(nsf * 16, nsf * 4)
        self.up2 = up(nsf * 8, nsf * 2)
        self.outc1 = double_conv(nsf * 8, n_classes)
        self.outc2 = double_conv(nsf * 4, n_classes)
        self.outc3 = double_conv(nsf * 2, n_classes)
        self.scale1 = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True)
        self.scale2 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        self.scale3 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

    def forward(self, xin):
        """
        xin is a dictionary that consists of x1, x2, x3, x4, x5 keys
        from the UNetEncoder
        """
        x3 = xin["x3p"]  # (bs, nsf*4, ..., ...)
        x4 = xin["x4p"]  # (bs, nsf*8, ..., ...)
        x5 = xin["x5p"]  # (bs, nsf*8, ..., ...)

        y5 = self.outc1(x5)
        y5 = self.scale1(y5)

        x = self.up1(x5, x4)  # (bs, nsf*4, ..., ...)

        y4 = self.outc2(x)
        y4 = self.scale2(y4)

        x = self.up2(x, x3)  # (bs, nsf*2, ..., ...)
        y3 = self.outc3(x)
        y3 = self.scale3(y3)

        return  {"y3p": y3, "y4p": y4, "y5p": y5}


class UNetDecoderLSTM(nn.Module):
    def __init__(self, n_classes, nsf=16):
        super().__init__()
        self.up1 = up_lstm(nsf * 16, nsf * 4)
        self.up2 = up_lstm(nsf * 8, nsf * 2)
        self.up3 = up(nsf * 4, nsf)
        self.up4 = up(nsf * 2, nsf)
        self.outc = outconv(nsf, n_classes)

    def forward(self, xin):
        """
        xin is a dictionary that consists of x1, x2, x3, x4, x5 keys
        from the UNetEncoder
        """
        x1 = xin["x1"]  # (bs, seq_len, nsf, ..., ...)
        x2 = xin["x2"]  # (bs, seq_len, nsf*2, ..., ...)
        x3 = xin["x3"]  # (bs, nsf*4, ..., ...)
        x4 = xin["x4"]  # (bs, nsf*8, ..., ...)
        x5 = xin["x5"]  # (bs, nsf*8, ..., ...)

        b, _, h, w = x4.shape
        hc1 = self.up1.init_hidden(batch_size=b, image_size=(h,w))

        b, _, h, w = x3.shape
        hc2 = self.up2.init_hidden(batch_size=b, image_size=(h,w))

        outputs = []
        for idx in range(x1.shape[1]):
            hc1 = self.up1(x5, x4, *hc1)  # h1,c1,h2,c2 # h2 shape - (bs, nsf*4, ..., ...)
            hc2 = self.up2(hc1[2], x3, *hc2)  # same as above # h1 shape - (bs, nsf*2, ..., ...)
            x = self.up3(hc2[2], x2[:, idx, ...])  # (bs, nsf, ..., ...)
            x = self.up4(x, x1[:, idx, ...])  # (bs, nsf, ..., ...)
            x = self.outc(x)  # (bs, n_classes, ..., ...)

            outputs += [x]

        return torch.stack(outputs, dim=1)


class MiniUNetEncoder(nn.Module):
    """
    Used to encode complex feature projections
    """

    def __init__(self, n_channels, feat_size):
        super().__init__()
        self.inc = inconv(n_channels, feat_size // 2)
        self.down3p = down(feat_size // 2, feat_size)
        self.down4p = down(feat_size, feat_size)

    def forward(self, x):
        x3p = self.inc(x)
        x4p = self.down3p(x3p)
        x5p = self.down4p(x4p)

        return {"x3p": x3p, "x4p": x4p, "x5p": x5p}


class LearnedRGBProjection(nn.Module):
    def __init__(self, mtype="downsample", infeats=768):
        super().__init__()
        if mtype == "downsample":
            self.projection = nn.Sequential(  # (bs, infeats, H, W)
                nn.Conv2d(infeats, infeats, 3, stride=1, padding=1),
                norm_layer(infeats),
                nn.ReLU(),
                nn.Conv2d(infeats, infeats, 5, stride=1, padding=2),
                norm_layer(infeats),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(
                    infeats, infeats, 5, stride=1, padding=2
                ),  # (bs, 768, H/2, W/2)
            )
        elif mtype == "upsample":
            self.projection = nn.Sequential(  # (bs, infeats, H, W)
                nn.Conv2d(infeats, infeats, 3, stride=1, padding=1),
                norm_layer(infeats),
                nn.ReLU(),
                nn.Conv2d(infeats, infeats, 5, stride=1, padding=2),
                norm_layer(infeats),
                nn.ReLU(),
                nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=True
                ),  # (bs, infeats, H*2, W*2)
                nn.Conv2d(
                    infeats, infeats, 5, stride=1, padding=2
                ),  # (bs, infeats, H*2, W*2),
            )
        else:
            raise ValueError(f"LearnedRGBProjection: Undefined model type {mtype}!")

    def forward(self, img_feats):
        return self.projection(img_feats)


class MergeMultimodal(nn.Module):
    """
    Merges features from multiple modalities (say RGB, projected occupancy) into a single
    feature map.
    """

    def __init__(self, nfeats, nmodes=2):
        super().__init__()
        self._nmodes = nmodes
        self.merge = nn.Sequential(
            nn.Conv2d(nmodes * nfeats, nfeats, 3, stride=1, padding=1),
            norm_layer(nfeats),
            nn.ReLU(),
            nn.Conv2d(nfeats, nfeats, 3, stride=1, padding=1),
            norm_layer(nfeats),
            nn.ReLU(),
            nn.Conv2d(nfeats, nfeats, 3, stride=1, padding=1),
        )

    def forward(self, *inputs):
        """
        Inputs:
            xi - (bs, nfeats, H, W)
        """
        x = torch.cat(inputs, dim=1)
        return self.merge(x)


class MergeMultiView(nn.Module):
    """
    Merges features from multiple modalities (say RGB, projected occupancy) into a single
    feature map.
    """

    def __init__(self, nfeats, nviews=2):
        super().__init__()
        self._nviews = nviews
        self.dropout_p = 0.5
        self.merge = nn.Sequential(
            nn.Conv2d(nviews * nfeats, nfeats, 3, stride=1, padding=1),
            norm_layer(nfeats),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_p),
            nn.Conv2d(nfeats, nfeats, 3, stride=1, padding=1),
            norm_layer(nfeats),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_p),
            nn.Conv2d(nfeats, nfeats, 3, stride=1, padding=1),
            norm_layer(nfeats),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_p),
            nn.Conv2d(nfeats, nfeats * nviews, 3, stride=1, padding=1),
        )

    def forward(self, x):
        """
        Inputs:
            xi - (bs, nfeats, H, W)
        """
        return self.merge(x)


class MergeMultimodalLSTM(nn.Module):
    """
    Merges features from multiple modalities (say RGB, projected occupancy) for multiple timesteps
    into a single feature map.
    """

    def __init__(self, nfeats, nmodes=2):
        super().__init__()
        self._nmodes = nmodes
        self.convlstm1 = ConvLSTMCell(nmodes * nfeats, nfeats, (3, 3), True)
        self.convlstm2 = ConvLSTMCell(nfeats, nfeats, (3, 3), True)
        self.convlstm3 = ConvLSTMCell(nfeats, nfeats, (3, 3), True)

    def forward(self, *inputs):
        """
        Inputs:
            xi - (bs, nfeats, H, W)
        """
        x = torch.cat(inputs, dim=2)
        b, t, _, h, w = x.shape

        h1, c1 = self.convlstm1.init_hidden(batch_size=b, image_size=(h, w))
        h2, c2 = self.convlstm2.init_hidden(batch_size=b, image_size=(h, w))
        h3, c3 = self.convlstm3.init_hidden(batch_size=b, image_size=(h, w))

        for idx in range(t):
            h1, c1 = self.convlstm1(input_tensor=x[:, idx, ...], cur_state=[h1, c1])
            h2, c2 = self.convlstm2(input_tensor=h1, cur_state=[h2, c2])
            h3, c3 = self.convlstm3(input_tensor=h2, cur_state=[h3, c3])

        return h3


class ResNetRGBEncoder(nn.Module):
    """
    Encodes RGB image via ResNet block1, block2 and merges them.
    """

    def __init__(self, resnet_type="resnet50"):
        super().__init__()
        if resnet_type == "resnet50":
            resnet = tmodels.resnet50(pretrained=True, norm_layer=norm_layer)
        elif resnet_type == "resnet18":
            resnet = tmodels.resnet18(pretrained=True, norm_layer=norm_layer)
        else:
            raise ValueError(f"ResNet type {resnet_type} not defined!")

        self.resnet_base = nn.Sequential(  # (B, 3, H, W)
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.resnet_block1 = resnet.layer1  # (256, H/4, W/4)
        self.resnet_block2 = resnet.layer2  # (512, H/8, W/8)

    def forward(self, x):
        """
        Inputs:
            x - RGB image of size (bs, 3, H, W)
        """
        x_base = self.resnet_base(x)
        x_block1 = self.resnet_block1(x_base)
        x_block2 = self.resnet_block2(x_block1)
        x_block1_red = F.avg_pool2d(
            x_block1, 3, stride=2, padding=1
        )  # (bs, 256, H/8, W/8)
        x_feat = torch.cat([x_block1_red, x_block2], dim=1)  # (bs, 768, H/8, W/8)

        return x_feat