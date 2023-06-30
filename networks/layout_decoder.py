from collections import OrderedDict

import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import networks
from networks.layers import DoubleConv, Down, Up, OutConv

class LayoutDecoder(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, **kwargs):
        super(LayoutDecoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # self.inc = DoubleConv(n_channels, 32)
        # self.down1 = Down(32, 64)
        # self.down2 = Down(64, 128)
        # self.down3 = Down(128, 128)
        # factor = 2 if bilinear else 1
        # self.up1 = Up(256, 128 // factor, bilinear)
        # self.up2 = Up(128, 64 // factor, bilinear)
        # self.up3 = Up(64, 32 // factor, bilinear)
        # self.outc = OutConv(64, n_classes)

        self.encoder = networks.feature_extraction_conv(32, 2, n_channels)
        self.outc = OutConv(64, n_classes)

    def forward(self, inputs, features):
        side = 'l'
        # x1 = self.inc(x)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x = self.up1(x4, x3)
        # x = self.up2(x, x2)
        # x = self.up3(x, x1)
        # logits = self.outc(x)
        x = features[('rgbd_features', side, 0)]  # B x C x 480 x 640
        x = x[:, :self.n_channels, :, :] # B x n_channels x 480 x 640
        f = self.encoder(x)
        logits = self.outc(f[-1])  # B x n_classes x 120 x 160
        logits = logits[:, :, :, 20:-20]

        outputs = {}
        outputs[('bev', side, 0)] = logits
        return outputs


    def get_params(self, opt):
        encoder_dict = {
            'name': 'bev_encoder',
            'params': list(self.encoder.parameters())
        }
        if hasattr(opt, 'encoder_lr'):
            encoder_dict['lr'] = opt.encoder_lr

        decoder_params = []
        decoder_params += list(self.outc.parameters())

        decoder_dict = {
            'name': 'bev_decoder',
            'params': decoder_params
        }
        if hasattr(opt, 'decoder_lr'):
            encoder_dict['lr'] = opt.decoder_lr

        bev_params = [encoder_dict, decoder_dict]
        return bev_params