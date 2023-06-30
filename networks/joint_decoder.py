from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.layers import upsample

import networks
from networks.layers import DoubleConv, Down, Up, OutConv
from torchvision.transforms import CenterCrop

def ConvReLU(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True)
    )

class JointDecoder(nn.Module):
    def __init__(self, args, **kwargs):
        super(JointDecoder, self).__init__()

        num_layers = 18
        weights_init = 'pretrained'

        self.num_output_channels = 64

        self.opt = kwargs['opt']

        ## TODO: Fix the hard coded values.
        self.res = 0.05
        self.x_min = -1.6
        self.x_max = 1.6
        self.z_min = 0.1
        self.z_max = 3.3
        self.bev_size = (64, 64)
        self.cam_offset = torch.Tensor([self.x_min, 0, self.z_min]).float()

        self.min_depth = self.z_min + self.res/2
        self.max_depth = self.min_depth + (self.res * self.num_output_channels) - self.res

        self.baseline = 0.2
        self.focal_length = 320

        self.use_skips = True
        self.upsample_mode = 'nearest'
        self.scales = range(2)
        
        # self.backproject = networks.layers.BackprojectDepth(self.opt.batch_size, self.opt.height, self.opt.width)

        self.encoder = networks.ResnetEncoder(num_layers, weights_init == "pretrained",
                num_input_images=1, in_channels=3)
        self.num_ch_enc = self.encoder.num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        
        # Depth decoder
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

        self.depth_decoder = nn.ModuleList(list(self.convs.values()))

        # BEV
        self.n_classes = 2
        # self.bev_ce_weights = [1.0, 2.0, 2.0]
        dropout = 0.5
        self.up3 = networks.layers.Up(256 + 512, 256)
        self.dropout3 = nn.Dropout2d(p=dropout)
        self.up2 = networks.layers.Up(128 + 256, 128)
        self.dropout2 = nn.Dropout2d(p=dropout)
        self.up1 = networks.layers.Up(64 + 128, 64)
        self.dropout1 = nn.Dropout2d(p=dropout)
        self.up0 = networks.layers.Up(64 + 64, 32)
        self.dropout0 = nn.Dropout2d(p=dropout)

        self.out3 = OutConv(256, self.n_classes)
        self.out2 = OutConv(128, self.n_classes)
        self.out1 = OutConv(64, self.n_classes)
        self.out0 = OutConv(32, self.n_classes)

        bev_modules = [self.up0,self.up1,self.up2,self.up3,self.out0,self.out1,self.out2,self.out3]
        self.bev_decoder = nn.ModuleList(bev_modules)

        self.disp_nll_loss = nn.NLLLoss(reduction='none')
        self.bev_bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.combined_nll_loss = nn.NLLLoss(reduction='none')

        self.center_crop = CenterCrop(self.bev_size)

        self.set_pix_coords()

    def set_pix_coords(self):
        meshgrid = np.meshgrid(range(self.opt.width), range(self.opt.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(1, 1, self.opt.height * self.opt.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)


    def forward(self, inputs, features):
        imgL = inputs[("color_aug", 'l', 0)]
        imgL = imgL.reshape((-1, 3, self.opt.height, self.opt.width))
        # imgL = imgL.reshape((-1, *imgL.shape[2:]))
        imgR = inputs[("color_aug", 'r', 0)]
        imgR = imgL.reshape((-1, 3, self.opt.height, self.opt.width))
        # imgR = imgR.reshape((-1, *imgR.shape[2:]))

        ####### For training with RGBD only ##########
        # depth_L = inputs[("depth_gt", "l")].reshape((-1, 1, self.opt.height, self.opt.width))
        # imgL = torch.cat([imgL, depth_L], dim=1)
        # depth_R = inputs[("depth_gt", "r")].reshape((-1, 1, self.opt.height, self.opt.width))
        # imgR = torch.cat([imgR, depth_R], dim=1)

        outputs = {}

        B, N, D = imgL.shape[0], self.opt.height * self.opt.width, self.num_output_channels

        depth_vals = torch.linspace(self.min_depth, self.max_depth, self.num_output_channels, requires_grad=False).view(1, -1, 1, 1).float().cuda()
        depth_vals = depth_vals.repeat(B, 1, self.opt.height, self.opt.width)

        for side, img in [('l', imgL), ('r', imgR)]:
            img_features = self.encoder(img)  # output of last 5 layers.

            for i in range(len(img_features)):
                outputs[('disp_encoder_feats', side, i)] = img_features[i]

            # depth decoder
            x = img_features[-1]
            for i in range(4, -1, -1):
                x = self.convs[str(("upconv", i, 0))](x)
                x = [networks.layers.upsample(x)]
                if self.use_skips and i > 0:
                    x += [img_features[i - 1]]
                x = torch.cat(x, 1)
                x = self.convs[str(("upconv", i, 1))](x)
                if i in self.scales:
                    depth_logits = self.convs[str(("depthconv", i))](x)
                    depth_logits = networks.layers.upsample(depth_logits, 2**i, self.upsample_mode)
                    depth_logprobs = F.log_softmax(depth_logits, dim=1)
                    outputs[('depth_logprobs', side, i)] = depth_logprobs

                    depth_dist = F.softmax(depth_logits, dim=1) * depth_vals

                    depth = torch.sum(depth_dist, 1, keepdim=True)
                    outputs[('depth', side, i)] = depth

                    # disp = self.focal_length * self.baseline / depth
                    # outputs[('disp', side, i)] = disp

            # bev decoder
            x = img_features[4] # B x 512 x 9 x 12
            x = self.up3(x, img_features[3]) # B x 256 x 18 x 24
            x = self.dropout3(x)

            # pred3 = self.out3(x)  
            # pred3 = networks.layers.upsample(pred3, 4) # B x n_classes x 72 x 96

            x = self.up2(x, img_features[2]) # B x 128 x 36 x 48
            x = self.dropout2(x)

            # pred2 = self.out2(x)  
            # pred2 = networks.layers.upsample(pred2, 2) # B x n_classes x 72 x 96

            x = self.up1(x, img_features[1]) # B x 64 x 72 x 96
            x = self.dropout1(x)

            pred1 = self.out1(x)  
            pred1 = networks.layers.upsample(pred1, 2) # B x n_classes x 72 x 96

            x = self.up0(x, img_features[0]) # B x 64 x 144 x 192
            x = self.dropout0(x)

            pred0 = self.out0(x)  # B x n_classes x 144 x 192
            pred0 = networks.layers.upsample(pred0, 1) # B x n_classes x 72 x 96

            outputs[('bev', side, 0)] = self.center_crop(pred0)
            outputs[('bev', side, 1)] = self.center_crop(pred1)
            # outputs[('bev', side, 2)] = self.center_crop(pred2)
            # outputs[('bev', side, 3)] = self.center_crop(pred3)

        # Lifting BEV to Perspective 
        depth_logprobs = outputs[('depth_logprobs', side, 0)]
        depth_logprobs = depth_logprobs.view(*depth_logprobs.shape[:2], -1).permute(0,2,1) # B x N x D
        depth_logprobs = depth_logprobs.unsqueeze(dim=-1).repeat(1,1,1,3)  # B x N x D x 3

        depth_range = depth_vals.view(*depth_vals.shape[:2], -1).unsqueeze(dim=2) # B x D x 1 x N

        inv_K = inputs[("inv_K", 0)]
        inv_K = inv_K.repeat(B//inv_K.shape[0], 1, 1)
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords.repeat(B, 1, 1))
        cam_points = cam_points.unsqueeze(dim=1).repeat(1, self.num_output_channels, 1, 1) # B x D x 3 x N

        cam_points = (depth_range * cam_points).permute(0, 3, 1, 2)  # B x N x D x 3

        cam_points = cam_points.reshape(-1, 3)

        cam_points = ((cam_points - self.cam_offset.to(cam_points.device)) / self.res).long()

        batch_ix = torch.cat([torch.full([N* D, 1], bx, device=cam_points.device, dtype=torch.long) for bx in range(B)])
        cam_points = torch.cat((cam_points, batch_ix), 1)

        valid = (cam_points[:, 0] >= 0) & (cam_points[:, 0] < self.bev_size[1])\
            & (cam_points[:, 2] >= 0) & (cam_points[:, 2] < self.bev_size[0])\
            & (cam_points[:, 1] >= 0)

        valid_idx = torch.arange(B*N*D).long()[valid]

        cam_points = cam_points[valid]

        bev_probs = F.sigmoid(outputs[('bev', side, 0)].reshape((B, self.n_classes, *self.bev_size))) # B x [occ, exp] x Hb x Wb
        bev_probs = bev_probs.permute(0,2,3,1) # B x Hb x Wb x C

        eps = 1e-6
        bev_probs = torch.clamp(bev_probs, eps, 1-eps)

        class_probs = torch.zeros((B, *self.bev_size, 3), device=bev_probs.device, dtype=torch.float32)  # B x Hb x Wb x 3 [unknown, occ, free]
        class_probs[..., 0] = 1 - bev_probs[..., 1]  # p(Unknown) = 1 - p(Explored)
        class_probs[..., 1] = bev_probs[..., 1] * bev_probs[..., 0]  # p(Occ) = p(Explored)p(Occupied)
        class_probs[..., 2] = bev_probs[..., 1] * (1 - bev_probs[..., 0])  # p(Free) = p(Explored)(1 - p(Occupied))

        class_probs = class_probs.permute(0, 3, 1, 2).reshape(-1) # B x 3 x Hb x Wb

        B_lifted = torch.zeros((B*N*D, 3), device=cam_points.device, dtype=torch.float32)

        for C in range(3):
            gather_indices = cam_points[:, 0] + \
                (self.bev_size[1] * cam_points[:, 2]) + \
                (self.bev_size[1] * self.bev_size[0] * C) + \
                (self.bev_size[1] * self.bev_size[0] * self.n_classes * cam_points[:, 3]) 

            idx_probs = torch.gather(class_probs, 0, gather_indices)
            B_lifted[valid_idx, C] = torch.log(idx_probs)  # Assign log probs

        B_lifted = B_lifted.reshape((B, N, D, 3))

        combined_logprobs = (B_lifted + depth_logprobs).view(B, self.opt.height, self.opt.width, D, 3)

        outputs[('combined_db_logprobs', 'l', 0)] = combined_logprobs

        return outputs

    def compute_loss(self, inputs, outputs, *args, **kwargs):
        depth_losses = []

        depth_L = inputs[("depth_gt", "l")].float().cuda()
        depth_L = depth_L.reshape((-1, self.opt.height, self.opt.width))
        # depth_L = depth_L.reshape((-1, *depth_L.shape[2:]))

        # Using -1 for inf,-inf to filter it using mask.
        depth_L = torch.nan_to_num(depth_L, -1, -1, -1)

        mask_L = (depth_L > self.min_depth) & (depth_L < self.max_depth)
        # mask.detach_()

        depth_L = ((depth_L - self.z_min) / self.res).type(torch.int64)
        depth_L = torch.clamp(depth_L, 0, self.num_output_channels-1)

        for idx in self.scales:
            pred = outputs[('depth_logprobs', 'l', idx)]
            pred = pred.reshape((-1, self.num_output_channels, self.opt.height, self.opt.width))
            # pred = pred.reshape((-1, *pred.shape[2:]))
            loss = self.disp_nll_loss(pred, depth_L)
            loss *= mask_L
            depth_losses.append(loss.mean())

        depth_R = inputs[("depth_gt", "r")].float().cuda()
        depth_R = depth_R.reshape((-1, self.opt.height, self.opt.width))
        # depth_R = depth_R.reshape((-1, *depth_R.shape[2:]))

        # Using -1 for inf,-inf to filter it using mask.
        depth_R = torch.nan_to_num(depth_R, -1, -1, -1)

        mask_R = (depth_R > self.min_depth) & (depth_R < self.max_depth)
        # mask.detach_()

        depth_R = ((depth_R - self.z_min) / self.res).type(torch.int64)
        depth_R = torch.clamp(depth_R, 0, self.num_output_channels-1)

        for idx in self.scales:
            pred = outputs[('depth_logprobs', 'r', idx)]
            pred = pred.reshape((-1, self.num_output_channels, self.opt.height, self.opt.width))
            # pred = pred.reshape((-1, *pred.shape[2:]))
            loss = self.disp_nll_loss(pred, depth_R)
            loss *= mask_R
            depth_losses.append(loss.mean())

        depth_losses = sum(depth_losses) / (2 * len(self.scales))

        # BEV
        bce_weight = 1
        bev_losses = 0
        side = 'l'
        for scale in range(0, 2):
            loss = 0

            pred = outputs[("bev", side, scale)]
            pred = pred.reshape((-1, self.n_classes, *self.bev_size))
            # pred = pred.reshape((-1, *pred.shape[2:]))

            target = inputs[("bev_gt", side)]
            target = target.reshape((-1, *self.bev_size))
            target_prob = torch.zeros_like(pred, device=pred.device, dtype=torch.float32, requires_grad=False)
            target_prob[:, 0, ...] = (target == 1) * 1.0  # Occupied
            target_prob[:, 1, ...] = (target != 0) * 1.0  # Explored
            
            ce_loss =  self.bev_bce_loss(pred, target_prob).mean()
            loss += bce_weight * ce_loss

            bev_losses += loss/ 2

        # Lifting BEV to Perspective
        B, N, D = depth_L.shape[0], self.opt.height * self.opt.width, self.num_output_channels

        loss_mask = torch.zeros(B*N, device=depth_L.device, dtype=torch.bool)
        loss_idx = torch.arange(B*N, device=depth_L.device, dtype=torch.long)

        gt_depth = inputs[('depth_gt', 'l')]
        gt_depth = gt_depth.reshape((-1, self.opt.height, self.opt.width))
        # gt_depth = gt_depth.reshape((-1, *gt_depth.shape[2:]))
        inv_K = inputs[('inv_K', 0)]
        inv_K = inv_K.repeat(B//inv_K.shape[0], 1, 1)
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords.repeat(B, 1, 1))
        cam_points = gt_depth.view(B,N,1) * cam_points.permute(0, 2, 1) # B x N x 3
        cam_points = cam_points.reshape(-1, 3)

        cam_points = ((cam_points - self.cam_offset.to(cam_points.device)) / self.res).long()

        batch_ix = torch.cat([torch.full([N, 1], bx, device=cam_points.device, dtype=torch.long) for bx in range(B)])
        cam_points = torch.cat((cam_points, batch_ix), 1)

        valid = (cam_points[:, 0] >= 0) & (cam_points[:, 0] < self.bev_size[1])\
            & (cam_points[:, 2] >= 0) & (cam_points[:, 2] < self.bev_size[0])\
            & (cam_points[:, 1] >= 0)
        valid_idx = torch.arange(B*N).long()[valid]

        cam_points = cam_points[valid]
        loss_idx = loss_idx[valid]

        max_y = self.max_depth // self.res  # hfov = 45, vfov is lesser
        ranks = cam_points[:, 1] + \
            (max_y * cam_points[:, 0]) + \
            (max_y * self.bev_size[1] * cam_points[:, 2]) + \
            (max_y * self.bev_size[1] * self.bev_size[0] * cam_points[:, 3])
            
        sorts = ranks.argsort()

        voxel_ids = cam_points[:, 0] + \
            (self.bev_size[1] * cam_points[:, 2]) + \
            (self.bev_size[1] * self.bev_size[0] * cam_points[:, 3])

        voxel_ids = voxel_ids[sorts]
        min_height_mask = torch.ones(voxel_ids.shape[0], device=voxel_ids.device, dtype=torch.bool)
        min_height_mask[:-1] = (voxel_ids[1:] != voxel_ids[:-1])
        
        loss_idx = loss_idx[sorts]
        loss_idx = loss_idx[min_height_mask]

        loss_mask[loss_idx] = True
        # loss_mask = loss_mask.reshape(B, self.opt.height, self.opt.width)
        outputs[('combined_db_mask', 'l', 0)] = loss_mask.reshape(B, self.opt.height, self.opt.width)

        bev = inputs[('bev_gt', side)].reshape(-1) # Flattened from B x Hb x Wb
        B_lifted = torch.zeros(B*N, device=bev.device, dtype=torch.long)
        gather_indices = cam_points[:, 0] + \
            (self.bev_size[1] * cam_points[:, 2]) + \
            (self.bev_size[1] * self.bev_size[0] * cam_points[:, 3]) 

        bev_vals = torch.gather(bev, 0, gather_indices)
        B_lifted[valid_idx] = bev_vals 
        
        # semantics = inputs[('semantics_gt', 'l')]
        semantics = B_lifted
        tgt_combined = (semantics.reshape(-1) * self.num_output_channels + depth_L.reshape(-1)).long()

        pred_combined = outputs[('combined_db_logprobs', 'l', 0)].reshape((-1, self.num_output_channels * 3))

        combined_loss = self.combined_nll_loss(pred_combined[loss_mask], tgt_combined[loss_mask])
        combined_loss = combined_loss.mean()

        depth_weight = 0.1
        bev_weight = 1
        combined_weight = 0.1
        all_losses = depth_weight * depth_losses + bev_weight * bev_losses + combined_weight * combined_loss

        # import cv2
        # from utils import invnormalize_imagenet
        # loss_mask = loss_mask.reshape(B, self.opt.height, self.opt.width)
        # for idx in range(B):
        #     color = invnormalize_imagenet(inputs[('color', 'l', 0)][idx, 0].cpu().detach()).numpy()
        #     color = np.clip(color * 255, 0, 255).astype(np.uint8)
        #     color = np.transpose(color, (1, 2, 0))
        #     cv2.imwrite(f'/scratch/shantanu/color_{idx}.png', color)

        #     mask = loss_mask[idx].cpu().detach().numpy().astype(np.uint8) * 255
        #     cv2.imwrite(f'/scratch/shantanu/mask_{idx}.png', mask)

        # return all_losses
        losses = {}
        losses[f"joint_decoder_loss/depth {depth_weight}"] = depth_losses
        losses[f"joint_decoder_loss/bev_loss {bev_weight}"] = bev_losses
        losses[f"joint_decoder_loss/combined {combined_weight}"] = combined_loss
        losses["disparity_loss"] = all_losses

        return losses


    def get_params(self, opt):
        encoder_dict = {
            'name': 'disp_encoder',
            'params': list(self.encoder.parameters())
        }
        if hasattr(opt, 'encoder_lr'):
            encoder_dict['lr'] = opt.encoder_lr

        disp_params = list(self.depth_decoder.parameters())

        depth_decoder_dict = {
            'name': 'disp_decoder',
            'params': disp_params
        }
        if hasattr(opt, 'decoder_lr'):
            depth_decoder_dict['lr'] = opt.decoder_lr


        bev_params = list(self.bev_decoder.parameters())

        bev_decoder_dict = {
            'name': 'bev_decoder',
            'params': bev_params
        }
        if hasattr(opt, 'decoder_lr'):
            bev_decoder_dict['lr'] = opt.decoder_lr

        model_params = [encoder_dict, depth_decoder_dict, bev_decoder_dict]
        return model_params