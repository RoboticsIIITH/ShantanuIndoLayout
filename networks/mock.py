import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from misc.layers import *

class MockDecoder(nn.Module):
    """ Returns an input item as the output. Only for testing

    Attributes
    ----------
    input_key: str
        input key to read data and return

    Methods
    -------
    forward(x, ):
        Reads the key from input and returns it a clone of it.
    """

    def __init__(self, type, **kwargs):
        super(MockDecoder, self).__init__()
        self.type = type # which module to mock

    def forward(self, inputs, features):
        #TODO fix this reshape. Shouldn't happen here
        # tgt = inputs[self.input_key].clone()
        # tgt = tgt.reshape((-1, *tgt.shape[2:]))  
        # tgt = F.one_hot(tgt,num_classes=self.n_classes).permute(0,3,1,2).float() * self.scale_factor
        # tgt = Variable(tgt, requires_grad=False).to(tgt.device)
        # output = {}
        # output[("bev", "l", 0)] = tgt
        if self.type == 'bev':
            output = {}
        elif self.type == 'disp':
            output = {}
            for side in ['l', 'r']:
                tgt = inputs[('depth_gt', side)]
                output[('depth', side, 0)] = Variable(tgt, requires_grad=False).to(tgt.device)
        else:
            raise NotImplementedError()

        return output

    def get_params(self, opt):
        params = []
        return params

    def compute_loss(self, inputs, outputs, *args, **kwargs):
        if self.type == 'bev':
            return {"bev_loss": 0}
        elif self.type == 'disp':
            return {'disparity_loss': 0}
        else:
            raise NotImplementedError()
