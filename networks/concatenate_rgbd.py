from collections import OrderedDict

import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from misc.layers import *

class ConcatenateRGBD(nn.Module):
    """ Merges RGB Input with the depth channel.

    Attributes
    ----------
    normalize_depth: bool
        whether to normalize the depth channel

    Methods
    -------
    forward(x, ):
        Concatenates RGB and Depth Map and returns RGBD tensor.
    """

    def __init__(self, normalize_depth=False, **kwargs):
        super(ConcatenateRGBD, self).__init__()
        self.num_ch_enc = 4 #rgbd
        self.normalize_depth = normalize_depth
        self.depth_mean = 5
        self.depth_std_dev = 1.667 # Range of 10m mapped to 6*sigma

    def forward(self, inputs, features):
        outputs = {}
        for side in ['l', 'r']:
            img = inputs[('color_aug', side, 0)]
            img = img.reshape((-1, *img.shape[2:]))

            depth = features[('depth', side, 0)]
            if self.normalize_depth:
                depth = (depth - self.depth_mean)/self.depth_std_dev
                
            rgbd = torch.cat([img, depth], dim=1)  # (batch, num_ch_enc, height, width)
            outputs[('rgbd_features', side, 0)] = rgbd
        return outputs