from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.layers import upsample

import networks
from networks.layers import DoubleConv, Down, Up, OutConv

def ConvReLU(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True)
    )

class DepthResnetDecoder(nn.Module):
    def __init__(self, args):
        super(DepthResnetDecoder, self).__init__()

        num_layers = 18
        weights_init = 'pretrained'

        self.num_output_channels = 120

        ## TODO: Fix the hard coded values.
        self.res = 0.025
        self.min_depth = 0.1
        self.max_depth = self.res * self.num_output_channels + self.min_depth
        self.baseline = 0.2
        self.focal_length = 320

        self.use_skips = True
        self.upsample_mode = 'nearest'
        self.scales = range(2)

        self.encoder = networks.ResnetEncoder(num_layers, weights_init == "pretrained",
                num_input_images=1, in_channels=3)
        self.num_ch_enc = self.encoder.num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        
        # decoder
        self.convs = nn.ModuleDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[str(("upconv", i, 0))] = networks.layers.ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[str(("upconv", i, 1))] = networks.layers.ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[str(("depthconv", s))] = networks.layers.Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        self.disp_ce_loss = nn.CrossEntropyLoss(reduction='none').cuda()

    def forward(self, inputs, features):
        imgL = inputs[("color_aug", 'l', 0)]
        imgL = imgL.reshape((-1, *imgL.shape[2:]))
        imgR = inputs[("color_aug", 'r', 0)]
        imgR = imgR.reshape((-1, *imgR.shape[2:]))

        outputs = {}

        for side, img in [('l', imgL), ('r', imgR)]:
            img_features = self.encoder(img)  # output of last 5 layers.

            for i in range(len(img_features)):
                outputs[('disp_encoder_feats', side, i)] = img_features[i]
                if torch.any(torch.isnan(img_features[i])).item():
                    print('img has nans', side)

            # decoder
            x = img_features[-1]
            for i in range(4, -1, -1):
                x = self.convs[str(("upconv", i, 0))](x)
                x = [networks.layers.upsample(x)]
                if self.use_skips and i > 0:
                    x += [img_features[i - 1]]
                x = torch.cat(x, 1)
                x = self.convs[str(("upconv", i, 1))](x)
                if i in self.scales:
                    depth_logits = self.sigmoid(self.convs[str(("depthconv", i))](x))
                    depth_logits = networks.layers.upsample(depth_logits, 2**i, self.upsample_mode)
                    outputs[('depth_logits', side, i)] = depth_logits

                    depth_probs = F.softmax(depth_logits, dim=1)

                    depth_vals = torch.arange(self.min_depth, self.max_depth, self.res, requires_grad=False).view(1, -1, 1, 1).float().to(depth_probs.device)
                    depth_vals = depth_vals.repeat(depth_probs.size()[0], 1, depth_probs.size()[2], depth_probs.size()[3])
                    depth = torch.sum(depth_probs * depth_vals, 1, keepdim=True)

                    disp = self.focal_length * self.baseline / depth
                    outputs[('disp', side, i)] = disp

        return outputs

    def compute_loss(self, inputs, outputs, *args, **kwargs):
        all_losses = []

        depth_L = inputs[("depth_gt", "l")].float().cuda()
        depth_L = depth_L.reshape((-1, *depth_L.shape[2:]))

        # Using -1 for inf,-inf to filter it using mask.
        depth_L = torch.nan_to_num(depth_L, -1, -1, -1)

        mask = (depth_L > self.min_depth) & (depth_L < self.max_depth)
        # mask.detach_()

        depth_L = (depth_L / self.res).type(torch.int64)
        depth_L = torch.clamp(depth_L, 0, self.num_output_channels-1)

        for idx in self.scales:
            pred = outputs[('depth_logits', 'l', idx)]
            pred = pred.reshape((-1, *pred.shape[2:]))
            loss = self.disp_ce_loss(pred, depth_L)
            loss *= mask
            all_losses.append(loss.mean())

        depth_R = inputs[("depth_gt", "r")].float().cuda()
        depth_R = depth_R.reshape((-1, *depth_R.shape[2:]))

        # Using -1 for inf,-inf to filter it using mask.
        depth_R = torch.nan_to_num(depth_R, -1, -1, -1)

        mask = (depth_R > self.min_depth) & (depth_R < self.max_depth)
        # mask.detach_()

        depth_R = (depth_R / self.res).type(torch.int64)
        depth_R = torch.clamp(depth_R, 0, self.num_output_channels-1)

        for idx in self.scales:
            pred = outputs[('depth_logits', 'r', idx)]
            pred = pred.reshape((-1, *pred.shape[2:]))
            loss = self.disp_ce_loss(pred, depth_R)
            loss *= mask
            all_losses.append(loss.mean())

        return sum(all_losses)

    def get_params(self, opt):
        encoder_dict = {
            'name': 'disp_encoder',
            'params': list(self.encoder.parameters())
        }
        if hasattr(opt, 'encoder_lr'):
            encoder_dict['lr'] = opt.encoder_lr

        decoder_params = list(self.decoder.parameters())
        # for i in range(4, -1, -1):
        #     decoder_params += list(self.convs[("upconv", i, 0)].parameters())
        #     decoder_params += list(self.convs[("upconv", i, 1)].parameters())

        # for i in self.scales:
        #     decoder_params += list(self.convs[("depthconv", i)].parameters())

        decoder_dict = {
            'name': 'disp_decoder',
            'params': decoder_params
        }
        if hasattr(opt, 'decoder_lr'):
            decoder_dict['lr'] = opt.decoder_lr

        disp_params = [encoder_dict, decoder_dict]
        return disp_params