from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodules import post_3dconvs,feature_extraction_conv
import sys


class AnyNet(nn.Module):
    def __init__(self, args, **kwargs):
        super(AnyNet, self).__init__()

        self.init_channels = args.init_channels
        self.maxdisplist = args.maxdisplist
        self.spn_init_channels = args.spn_init_channels
        self.nblocks = args.nblocks
        self.layers_3d = args.layers_3d
        self.channels_3d = args.channels_3d
        self.growth_rate = args.growth_rate
        self.with_spn = args.with_spn
        self.start_epoch_for_spn = args.start_epoch_for_spn
        self.loss_weights = args.loss_weights

        self.baseline = 0.2
        self.focal_length = 320

        if self.with_spn:
            try:
                # from .spn.modules.gaterecurrent2dnoind import GateRecurrent2dnoind
                from .spn_t1.modules.gaterecurrent2dnoind import GateRecurrent2dnoind
            except:
                print('Cannot load spn model')
                sys.exit()
            self.spn_layer = GateRecurrent2dnoind(True,False)
            spnC = self.spn_init_channels
            self.refine_spn = [nn.Sequential(
                nn.Conv2d(3, spnC*2, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(spnC*2, spnC*2, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(spnC*2, spnC*2, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(spnC*2, spnC*3, 3, 1, 1, bias=False),
            )]
            self.refine_spn += [nn.Conv2d(1,spnC,3,1,1,bias=False)]
            self.refine_spn += [nn.Conv2d(spnC,1,3,1,1,bias=False)]
            self.refine_spn = nn.ModuleList(self.refine_spn)
        else:
            self.refine_spn = None

        self.feature_extraction = feature_extraction_conv(self.init_channels,
                                      self.nblocks)

        self.volume_postprocess = []

        for i in range(3):
            net3d = post_3dconvs(self.layers_3d, self.channels_3d*self.growth_rate[i])
            self.volume_postprocess.append(net3d)
        self.volume_postprocess = nn.ModuleList(self.volume_postprocess)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def warp(self, x, disp):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W, device='cuda').view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, device='cuda').view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        vgrid = torch.cat((xx, yy), 1).float()

        # vgrid = Variable(grid)
        vgrid[:,:1,:,:] = vgrid[:,:1,:,:] - disp

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        return output


    def _build_volume_2d(self, feat_l, feat_r, maxdisp, stride=1):
        assert maxdisp % stride == 0  # Assume maxdisp is multiple of stride
        cost = torch.zeros((feat_l.size()[0], maxdisp//stride, feat_l.size()[2], feat_l.size()[3]), device='cuda')
        for i in range(0, maxdisp, stride):
            cost[:, i//stride, :, :i] = feat_l[:, :, :, :i].abs().sum(1)
            if i > 0:
                cost[:, i//stride, :, i:] = torch.norm(feat_l[:, :, :, i:] - feat_r[:, :, :, :-i], 1, 1)
            else:
                cost[:, i//stride, :, i:] = torch.norm(feat_l[:, :, :, :] - feat_r[:, :, :, :], 1, 1)

        return cost.contiguous()

    def _build_volume_2d3(self, feat_l, feat_r, maxdisp, disp, stride=1):
        size = feat_l.size()
        batch_disp = disp[:,None,:,:,:].repeat(1, maxdisp*2-1, 1, 1, 1).view(-1,1,size[-2], size[-1])
        batch_shift = torch.arange(-maxdisp+1, maxdisp, device='cuda').repeat(size[0])[:,None,None,None] * stride
        batch_disp = batch_disp - batch_shift.float()
        batch_feat_l = feat_l[:,None,:,:,:].repeat(1,maxdisp*2-1, 1, 1, 1).view(-1,size[-3],size[-2], size[-1])
        batch_feat_r = feat_r[:,None,:,:,:].repeat(1,maxdisp*2-1, 1, 1, 1).view(-1,size[-3],size[-2], size[-1])
        cost = torch.norm(batch_feat_l - self.warp(batch_feat_r, batch_disp), 1, 1)
        cost = cost.view(size[0],-1, size[2],size[3])
        return cost.contiguous()


    def forward(self, inputs, features):
        imgL = inputs[("color_aug", 'l', 0)]
        imgL = imgL.reshape((-1, *imgL.shape[2:]))
        imgR = inputs[("color_aug", 'r', 0)]
        imgR = imgR.reshape((-1, *imgR.shape[2:]))

        outputs = {}

        for side in ['l', 'r']:
            if side == 'l':
                left = imgL
                right = imgR
            else:
                left = torch.flip(imgR, dims=[-1])
                right = torch.flip(imgL, dims=[-1])

            img_size = left.size()

            feats_l = self.feature_extraction(left)
            feats_r = self.feature_extraction(right)
            pred = []
            for scale in range(len(feats_l)):
                if scale > 0:
                    wflow = F.upsample(pred[scale-1], (feats_l[scale].size(2), feats_l[scale].size(3)),
                                    mode='bilinear') * feats_l[scale].size(2) / img_size[2]
                    cost = self._build_volume_2d3(feats_l[scale], feats_r[scale],
                                            self.maxdisplist[scale], wflow, stride=1)
                else:
                    cost = self._build_volume_2d(feats_l[scale], feats_r[scale],
                                                self.maxdisplist[scale], stride=1)

                cost = torch.unsqueeze(cost, 1)
                cost = self.volume_postprocess[scale](cost)
                cost = cost.squeeze(1)
                if scale == 0:
                    pred_low_res = disparityregression2(0, self.maxdisplist[0])(F.softmax(-cost, dim=1))
                    pred_low_res = pred_low_res * img_size[2] / pred_low_res.size(2)
                    disp_up = F.upsample(pred_low_res, (img_size[2], img_size[3]), mode='bilinear')
                    pred.append(disp_up)
                else:
                    pred_low_res = disparityregression2(-self.maxdisplist[scale]+1, self.maxdisplist[scale], stride=1)(F.softmax(-cost, dim=1))
                    pred_low_res = pred_low_res * img_size[2] / pred_low_res.size(2)
                    disp_up = F.upsample(pred_low_res, (img_size[2], img_size[3]), mode='bilinear')
                    pred.append(disp_up+pred[scale-1])


            if self.refine_spn:
                spn_out = self.refine_spn[0](nn.functional.upsample(left, (img_size[2]//4, img_size[3]//4), mode='bilinear'))
                G1, G2, G3 = spn_out[:,:self.spn_init_channels,:,:], spn_out[:,self.spn_init_channels:self.spn_init_channels*2,:,:], spn_out[:,self.spn_init_channels*2:,:,:]
                sum_abs = G1.abs() + G2.abs() + G3.abs()
                G1 = torch.div(G1, sum_abs + 1e-8)
                G2 = torch.div(G2, sum_abs + 1e-8)
                G3 = torch.div(G3, sum_abs + 1e-8)
                pred_flow = nn.functional.upsample(pred[-1], (img_size[2]//4, img_size[3]//4), mode='bilinear')
                refine_flow = self.spn_layer(self.refine_spn[1](pred_flow), G1, G2, G3)
                refine_flow = self.refine_spn[2](refine_flow)
                pred.append(nn.functional.upsample(refine_flow, (img_size[2] , img_size[3]), mode='bilinear'))

            for idx in reversed(range(len(pred))):
                if side == 'l':
                    outputs[('disp', side, idx)] = pred[idx]
                else:
                    outputs[('disp', side, idx)] = torch.flip(pred[idx], dims=[-1])
            
            # for idx in range(len(feats_l)):
            #     outputs[('disp_feats', side, idx)] = feats_l[idx]

        ## TODO : Fix this to work at every scale
        dispL = torch.abs(outputs[('disp', 'l', 0)])
        depthL = torch.nan_to_num((self.opt.baseline * self.opt.focal_length)/dispL, 
                        self.opt.max_depth, self.opt.max_depth)
        depthL = torch.clamp(depthL, self.opt.min_depth, self.opt.max_depth)

        dispR = torch.abs(outputs[('disp', 'r', 0)])
        depthR = torch.nan_to_num((self.opt.baseline * self.opt.focal_length)/dispR, 
                        self.opt.max_depth, self.opt.max_depth)
        depthR = torch.clamp(depthR, self.opt.min_depth, self.opt.max_depth)

        outputs[('depth', 'l', 0)] = depthL
        outputs[('depth', 'r', 0)] = depthR

        return outputs

    def compute_loss(self, inputs, outputs, epoch, **kwargs):

        all_losses = []

        if self.with_spn and epoch >= self.start_epoch_for_spn:
            num_out = self.nblocks + 2
        else:
            num_out = self.nblocks + 1

        depth_L = inputs[("depth_gt", "l")].float().cuda()
        depth_L = depth_L.reshape((-1, depth_L.shape[-2], depth_L.shape[-1]))

        # Using -1 for inf,-inf to filter it using mask.
        disp_L = torch.nan_to_num((self.baseline * self.focal_length)/depth_L, -1, -1) 

        mask = disp_L > 0
        mask.detach_()

        for idx in range(num_out):
            pred = outputs[('disp', 'l', idx)]
            pred = pred.reshape((-1, pred.shape[-2], pred.shape[-1]))
            loss = self.loss_weights[idx] * \
                F.smooth_l1_loss(pred[mask], disp_L[mask], size_average=True)
            all_losses.append(loss)

        depth_R = inputs[("depth_gt", "r")].float().cuda()
        depth_R = depth_R.reshape((-1, depth_R.shape[-2], depth_R.shape[-1]))

        # Using -1 for inf,-inf to filter it using mask.
        disp_R = torch.nan_to_num((self.baseline * self.focal_length)/depth_R, -1, -1) 

        mask = disp_R > 0
        mask.detach_()

        for idx in range(num_out):
            pred = outputs[('disp', 'r', idx)]
            pred = pred.reshape((-1, pred.shape[-2], pred.shape[-1]))
            loss = self.loss_weights[idx] * \
                F.smooth_l1_loss(pred[mask], disp_R[mask], size_average=True)
            all_losses.append(loss)

        return sum(all_losses)

    def get_params(self, opt):
        encoder_dict = {
            'name': 'disp_encoder',
            'params': list(self.encoder.parameters())
        }
        if hasattr(opt, 'encoder_lr'):
            encoder_dict['lr'] = opt.encoder_lr

        decoder_params = []
        for i in range(4, -1, -1):
            decoder_params += list(self.convs[("upconv", i, 0)].parameters())
            decoder_params += list(self.convs[("upconv", i, 1)].parameters())

        for i in self.scales:
            decoder_params += list(self.convs[("depthconv", i)].parameters())

        decoder_dict = {
            'name': 'disp_decoder',
            'params': decoder_params
        }
        if hasattr(opt, 'decoder_lr'):
            encoder_dict['lr'] = opt.decoder_lr

        disp_params = [encoder_dict, decoder_dict]
        return disp_params

class disparityregression2(nn.Module):
    def __init__(self, start, end, stride=1):
        super(disparityregression2, self).__init__()
        self.disp = torch.arange(start*stride, end*stride, stride, device='cuda', requires_grad=False).view(1, -1, 1, 1).float()

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1, keepdim=True)
        return out