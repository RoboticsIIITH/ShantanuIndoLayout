from collections import OrderedDict

import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from misc.layers import *

class Discriminator(nn.Module):
    """
    A patch discriminator used to regularize the decoder
    in order to produce layouts close to the true data distribution
    """

    def __init__(self, n_channels, **kwargs):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(n_channels, 8, 3, 2, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(8, 16, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(16, 32, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(32, 8, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(8, 1, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """

        Parameters
        ----------
        x : torch.FloatTensor
            Batch of output Layouts
            | Shape: (batch_size, 2, occ_map_size, occ_map_size)

        Returns
        -------
        x : torch.FloatTensor
            Patch output of the Discriminator
            | Shape: (batch_size, 1, occ_map_size/16, occ_map_size/16)
        """

        return self.main(x)
