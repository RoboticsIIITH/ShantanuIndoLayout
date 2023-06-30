from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import networks
from networks.layers import DoubleConv, Down, Up, OutConv

def ConvReLU(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True)
    )

class LayoutResnetDecoder(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, num_layers=18, 
        weights_init='pretrained', dropout=0.5, **kwargs):
        super(LayoutResnetDecoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dropout = dropout

        self.encoder = networks.ResnetEncoder(num_layers, weights_init == "pretrained",
                num_input_images=1, in_channels=n_channels)
        
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # self.layer3_1x1 = ConvReLU(256, 256, 1, 0)
        # self.conv_up3 = ConvReLU(256 + 512, 256, 3, 1)

        # self.layer2_1x1 = ConvReLU(128, 128, 1, 0)
        # self.conv_up2 = ConvReLU(128 + 256, 128, 3, 1)

        # self.layer1_1x1 = ConvReLU(64, 64, 1, 0)
        # self.conv_up1 = ConvReLU(64 + 128, 64, 3, 1)

        self.up3 = networks.layers.Up(256 + 512, 256)
        self.dropout3 = nn.Dropout2d(p=dropout)
        self.up2 = networks.layers.Up(128 + 256, 128)
        self.dropout2 = nn.Dropout2d(p=dropout)
        self.up1 = networks.layers.Up(64 + 128, 64)
        self.dropout1 = nn.Dropout2d(p=dropout)

        self.out3 = OutConv(256, n_classes)
        self.out2 = OutConv(128, n_classes)
        self.out1 = OutConv(64, n_classes)

    def forward(self, inputs, features):
        side = 'l'
        x = features[('rgbd_features', side, 0)]  # B x C x 480 x 640
        x = x[:, :self.n_channels, :, :] # B x n_channels x 480 x 640
        f = self.encoder(x)  # output of last 5 layers.

        # x = f[4] # B x 512 x 15 x 20
        # x = self.upsample(x)  # B x 512 x 30 x 40
        # l3 = self.layer3_1x1(f[3])  # B x 256 x 30 x 40
        # x = torch.cat([x, l3], dim=1) # B x 768 x 30 x 40
        # x = self.conv_up3(x)  # B x 256 x 30 x 40

        # x = self.upsample(x)  # B x 256 x 60 x 80
        # l2 = self.layer2_1x1(f[2]) # B x 128 x 60 x 80
        # x = torch.cat([x, l2], dim=1) # B x 384 x 60 x 80
        # x = self.conv_up2(x) # B x 128 x 60 x 80

        # x = self.upsample(x) # B x 128 x 120 x 160
        # l3 = self.layer1_1x1(f[1]) # B x 64 x 120 x 160
        # x = torch.cat([x, l3], dim=1) # B x 192 x 120 x 160
        # x = self.conv_up1(x) # B x 64 x 120 x 160

        x = f[4] # B x 512 x 15 x 20
        x = self.up3(x, f[3]) # B x 256 x 30 x 40
        x = self.dropout3(x)

        pred3 = self.out3(x)
        pred3 = networks.layers.upsample(pred3, 4)[:, :, :, 20:-20]

        x = self.up2(x, f[2]) # B x 128 x 60 x 80
        x = self.dropout2(x)

        pred2 = self.out2(x)
        pred2 = networks.layers.upsample(pred2, 2)[:, :, :, 20:-20]

        x = self.up1(x, f[1]) # B x 64 x 120 x 160
        x = self.dropout1(x)

        pred1 = self.out1(x)  # B x n_classes x 120 x 160
        pred1 = pred1[:, :, :, 20:-20]

        outputs = {}
        outputs[('bev', side, 0)] = pred1
        outputs[('bev', side, 1)] = pred2
        outputs[('bev', side, 2)] = pred3
        return outputs

    def compute_loss(self, inputs, outputs, *args, **kwargs):
        side = 'l'
        output = {}
        output["bev_loss/bce"] = {}
        output["bev_loss/ssim"] = {}
        output["bev_loss/smoothness"] = {}
        
        bev_loss = 0
        for scale in range(0, 3):
            loss = 0

            pred = outputs[("bev", side, scale)]
            pred = pred.reshape((-1, *pred.shape[2:]))

            # if ("bev_gt", side) not in inputs:
            #     inputs[("bev_gt", side)] = self.generate_gt_bev(inputs, outputs, side)
            # outputs[("bev_proj", side, scale)] = self.generate_gt_bev(inputs, outputs, side, scale)

            target = inputs[("bev_gt", side)]
            target = target.reshape((-1, *target.shape[2:]))
            
            ce_loss =  self.bev_ce_loss(pred, target).mean()
            loss += self.opt.BEV_DECODER.bce_loss * ce_loss

            pred_sm = F.softmax(pred, dim=1)
            pred_sm = (pred_sm[:, 1, ...] * 1 + pred_sm[:, 2, ...] * 2).unsqueeze(dim=1)
            target = target.unsqueeze(dim=1).float()

            ssim_loss = self.ssim(pred_sm, target).mean()
            loss += self.opt.BEV_DECODER.ssim_loss * ssim_loss 

            smoothness_loss = networks.layers.get_smooth_loss(pred_sm, target)
            loss += self.opt.BEV_DECODER.smoothness_loss * smoothness_loss 

            bev_loss += loss/ 3

            output["bev_loss/bce"][str(scale)] = ce_loss
            output["bev_loss/ssim"][str(scale)] = ssim_loss
            output["bev_loss/smoothness"][str(scale)] = smoothness_loss
        
        output["bev_loss"] = bev_loss

        return output

    def get_params(self, opt):
        encoder_dict = {
            'name': 'bev_encoder',
            'params': list(self.encoder.parameters())
        }
        if hasattr(opt, 'encoder_lr'):
            encoder_dict['lr'] = opt.encoder_lr

        decoder_params = []
        decoder_params += list(self.up3.parameters())
        decoder_params += list(self.up2.parameters())
        decoder_params += list(self.up1.parameters())
        decoder_params += list(self.out3.parameters())
        decoder_params += list(self.out2.parameters())
        decoder_params += list(self.out1.parameters())

        decoder_dict = {
            'name': 'bev_decoder',
            'params': decoder_params
        }
        if hasattr(opt, 'decoder_lr'):
            encoder_dict['lr'] = opt.decoder_lr

        bev_params = [encoder_dict, decoder_dict]
        return bev_params
