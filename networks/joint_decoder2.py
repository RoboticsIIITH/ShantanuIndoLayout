from collections import OrderedDict

import numpy as np

import torch
from torch.functional import chain_matmul
import torch.nn as nn
import torch.nn.functional as F
from misc.layers import upsample

import networks
from networks.layers import DoubleConv, Down, Up, OutConv
from torchvision.transforms import CenterCrop

import kornia.morphology
import kornia.filters

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

        self.num_output_channels = 128 # 64 #

        self.opt = kwargs['opt']

        ## TODO: Fix the hard coded values.
        self.res = 0.025
        self.x_min = -1.6
        self.x_max = 1.6
        self.z_min = 0.1
        self.z_max = 3.3
        self.bev_size = (128, 128) # (64, 64) #
        self.cam_offset = torch.Tensor([self.x_min, 0, -self.z_max]).float()

        self.min_depth = self.z_min + self.res/2
        self.max_depth = self.min_depth + (self.res * self.num_output_channels) - self.res

        self.baseline = 0.2
        self.focal_length = 85.333

        self.use_skips = True
        self.upsample_mode = 'bilinear'
        self.scales = range(1)

        self.n_classes = 2
        
        # self.backproject = networks.layers.BackprojectDepth(self.opt.batch_size, self.opt.height, self.opt.width)

        self.bev_decoder = networks.occant.OccupancyAnticipator(self.opt)
        self.num_ch_enc = np.array([64, 128, 128])
        self.num_ch_dec = np.array([64, 64, 128])

        
        # Depth decoder
        self.convs = nn.ModuleDict()
        for i in range(2, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 2 else self.num_ch_dec[i + 1]
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

        self.disp_nll_loss = nn.NLLLoss(reduction='none')
        self.bev_bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.combined_nll_loss = nn.NLLLoss(reduction='none')

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
        # imgL = inputs[("color_aug", 'l', 0)]
        # imgL = imgL.reshape((-1, 3, self.opt.height, self.opt.width))
        # imgL = imgL.reshape((-1, *imgL.shape[2:]))
        # imgR = inputs[("color_aug", 'r', 0)]
        # imgR = imgL.reshape((-1, 3, self.opt.height, self.opt.width))
        # imgR = imgR.reshape((-1, *imgR.shape[2:]))

        ####### For training with RGBD only ##########
        # depth_L = inputs[("depth_gt", "l")].reshape((-1, 1, self.opt.height, self.opt.width))
        # imgL = torch.cat([imgL, depth_L], dim=1)
        # depth_R = inputs[("depth_gt", "r")].reshape((-1, 1, self.opt.height, self.opt.width))
        # imgR = torch.cat([imgR, depth_R], dim=1)

        outputs = self.bev_decoder(inputs, features)
        img_features = [outputs[("encoder_features", 0)], outputs[("encoder_features", 1)], outputs[("encoder_features", 2)]]

        B, N, D = img_features[0].shape[0], self.opt.height * self.opt.width, self.num_output_channels

        depth_vals = torch.linspace(self.min_depth, self.max_depth, self.num_output_channels, requires_grad=False).view(1, -1, 1, 1).float().cuda()
        depth_vals = depth_vals.repeat(B, 1, self.opt.height, self.opt.width)

        
        # depth decoder
        x = img_features[-1]
        for i in range(2, -1, -1):
            x = self.convs[str(("upconv", i, 0))](x)
            x = [networks.layers.upsample(x)]
            if self.use_skips and i > 0:
                x += [img_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[str(("upconv", i, 1))](x)
            if i in self.scales:
                depth_logits = self.convs[str(("depthconv", i))](x)
                depth_logits = networks.layers.upsample(depth_logits, 2 * 2**i, self.upsample_mode)
                depth_logprobs = F.log_softmax(depth_logits, dim=1)
                outputs[('depth_logprobs', 'l', i)] = depth_logprobs

                depth_dist = F.softmax(depth_logits, dim=1) * depth_vals

                depth = torch.sum(depth_dist, 1, keepdim=True)

                # depth 
                # depth_dist = F.softmax(depth_logits, dim=1)
                # depth_label = torch.argmax(depth_dist, dim=1)
                # depth_label = torch.unsqueeze(depth_label, dim=1)
    
                # depth = torch.gather(depth_vals, 1, depth_label)

                outputs[('depth', 'l', i)] = depth
     
        # depth = inputs[("depth_gt", "l")].float().cuda()
        # depth = depth.reshape((-1, self.opt.height, self.opt.width)) # B x H x W
        # # Using -1 for inf,-inf to filter it using mask.
        # depth = torch.nan_to_num(depth, -1, -1, -1)
        # depth_idx = ((depth - self.z_min) / self.res).type(torch.int64)
        # depth_idx = torch.clamp(depth_idx, 0, self.num_output_channels-1) # B x H x W
        # depth_prob = F.one_hot(depth_idx, num_classes=self.num_output_channels).float() # B x H x W x D

        # eps = 1e-7
        # depth_prob = torch.clamp(depth_prob, eps, 1-eps)
        # outputs[('depth_logprobs', 'l', 0)] = torch.log(depth_prob).permute(0,3,1,2)  # B x D x H x W

        # Lifting BEV to Perspective 
        depth_logprobs = outputs[('depth_logprobs', 'l', 0)]
        depth_logprobs = depth_logprobs.view(*depth_logprobs.shape[:2], -1).permute(0,2,1) # B x N x D
        depth_logprobs = depth_logprobs.unsqueeze(dim=-1).repeat(1,1,1,3)  # B x N x D x 3

        depth_range = depth_vals.view(*depth_vals.shape[:2], -1).unsqueeze(dim=2) # B x D x 1 x N

        inv_K = inputs[("inv_K", 0)]
        inv_K = inv_K.repeat(B//inv_K.shape[0], 1, 1)
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords.repeat(B, 1, 1))
        cam_points = cam_points.unsqueeze(dim=1).repeat(1, self.num_output_channels, 1, 1) # B x D x 3 x N

        cam_points = (depth_range * cam_points).permute(0, 3, 1, 2)  # B x N x D x 3

        cam_points = cam_points.reshape(-1, 3)

        cam_points[:, 1] *= -1  # Flip the y-axis to point upwards
        cam_points[:, 2] *= -1  # Flip the z-axis to point backwards. This helps to align with direction in bev space.

        cam_points = ((cam_points - self.cam_offset.to(cam_points.device)) / self.res).long()

        batch_ix = torch.cat([torch.full([N* D, 1], bx, device=cam_points.device, dtype=torch.long) for bx in range(B)])
        cam_points = torch.cat((cam_points, batch_ix), 1)

        valid = (cam_points[:, 0] >= 0) & (cam_points[:, 0] < self.bev_size[1])\
            & (cam_points[:, 2] >= 0) & (cam_points[:, 2] < self.bev_size[0])\
            & (cam_points[:, 1] <= 0)

        valid_idx = torch.arange(B*N*D).long()[valid]

        cam_points = cam_points[valid]

        bev_probs = F.sigmoid(outputs[('bev', 'l', 0)].reshape((B, self.n_classes, *self.bev_size))) # B x [occ, exp] x Hb x Wb
        bev_probs = bev_probs.permute(0,2,3,1) # B x Hb x Wb x [occ, exp]

        # target = inputs[("bev_gt", 'l')]
        # target = target.reshape((-1, *self.bev_size))
        # target_prob = torch.zeros((B, self.n_classes, *self.bev_size), device=cam_points.device, dtype=torch.float32, requires_grad=False)
        # target_prob[:, 0, ...] = (target == 1) * 1.0  # Occupied
        # target_prob[:, 1, ...] = (target != 0) * 1.0  # Explored
        # bev_probs = target_prob.permute(0,2,3,1)

        eps = 1e-7
        bev_probs = torch.clamp(bev_probs, eps, 1-eps)

        class_probs = torch.zeros((B, *self.bev_size, 3), device=bev_probs.device, dtype=torch.float32)  # B x Hb x Wb x 3 [unknown, occ, free]
        class_probs[..., 0] = 1 - bev_probs[..., 1]  # p(Unknown) = 1 - p(Explored)
        class_probs[..., 1] = bev_probs[..., 1] * bev_probs[..., 0]  # p(Occ) = p(Explored)p(Occupied)
        class_probs[..., 2] = bev_probs[..., 1] * (1 - bev_probs[..., 0])  # p(Free) = p(Explored)(1 - p(Occupied))

        class_probs = class_probs.permute(0, 3, 1, 2).reshape(-1) # B x 3 x Hb x Wb flattend.

        B_lifted = torch.cat([torch.full([B*N*D, 1], -1.0986, \
            device=cam_points.device, dtype=torch.float32) for _ in range(3)], dim=-1) # By default set prob=0.33 for all.

        for C in range(3):
            gather_indices = cam_points[:, 0] + \
                (self.bev_size[1] * cam_points[:, 2]) + \
                (self.bev_size[1] * self.bev_size[0] * C) + \
                (self.bev_size[1] * self.bev_size[0] * 3 * cam_points[:, 3]) 

            idx_probs = torch.gather(class_probs, 0, gather_indices)
            B_lifted[valid_idx, C] = torch.log(idx_probs)  # Assign log probs

        B_lifted = B_lifted.reshape((B, N, D, 3))

        combined_logprobs = (B_lifted + depth_logprobs).view(B, self.opt.height, self.opt.width, D, 3)

        outputs[('combined_db_logprobs', 'l', 0)] = combined_logprobs

        return outputs

    def compute_loss(self, inputs, outputs, epoch, *args, **kwargs):
        depth_losses = []

        depth_L = inputs[("depth_gt", "l")].float().cuda()
        depth_L = depth_L.reshape((-1, self.opt.height, self.opt.width))
        # depth_L = depth_L.reshape((-1, *depth_L.shape[2:]))

        # Using -1 for inf,-inf to filter it using mask.
        # depth_L = torch.nan_to_num(depth_L, -1, -1, -1)

        # mask_L = (depth_L > self.min_depth) & (depth_L < self.max_depth)
        mask_L = 1 - torch.isnan(depth_L).int()
        # mask.detach_()

        depth_L = ((depth_L - self.z_min) / self.res).type(torch.int64)
        depth_L[depth_L < 0] = self.num_output_channels - 1  # Holes have depth zero. Need to set them to max depth.
        depth_L = torch.clamp(depth_L, 0, self.num_output_channels-1)

        for idx in self.scales:
            pred = outputs[('depth_logprobs', 'l', idx)]
            pred = pred.reshape((-1, self.num_output_channels, self.opt.height, self.opt.width))
            # pred = pred.reshape((-1, *pred.shape[2:]))
            loss = self.disp_nll_loss(pred, depth_L)
            loss *= mask_L
            depth_losses.append(loss.mean())

        depth_losses = sum(depth_losses) / len(self.scales)

        # BEV
        side = 'l'
        pred = outputs[("bev", side, 0)]
        pred = pred.reshape((-1, self.n_classes, *self.bev_size))

        target = inputs[("bev_gt", side)]
        target = target.reshape((-1, *self.bev_size))
        target_prob = torch.zeros_like(pred, device=pred.device, dtype=torch.float32, requires_grad=False)
        target_prob[:, 0, ...] = (target == 1) * 1.0  # Occupied
        target_prob[:, 1, ...] = (target != 0) * 1.0  # Explored
        
        bev_losses =  self.bev_bce_loss(pred, target_prob).mean()

        if "depth_proj_estimate" in outputs:
            pred = outputs["depth_proj_estimate"]
            pred = pred.reshape((-1, *pred.shape[2:]))

            target = inputs[("ego_map_gt", 'l')]
            target = target.reshape((-1, *target.shape[2:]))

            bev_losses += self.bev_bce_loss(pred, target).mean()

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

        cam_points[:, 1] *= -1  # Flip the y-axis to point upwards
        cam_points[:, 2] *= -1  # Flip the z-axis to point backwards. This helps to align with direction in bev space.

        cam_points = ((cam_points - self.cam_offset.to(cam_points.device)) / self.res).long()

        batch_ix = torch.cat([torch.full([N, 1], bx, device=cam_points.device, dtype=torch.long) for bx in range(B)])
        cam_points = torch.cat((cam_points, batch_ix), 1)

        valid = (cam_points[:, 0] >= 0) & (cam_points[:, 0] < self.bev_size[1])\
            & (cam_points[:, 2] >= 0) & (cam_points[:, 2] < self.bev_size[0])\
            & (cam_points[:, 1] <= 0)
        valid_idx = torch.arange(B*N).long()[valid]

        cam_points = cam_points[valid]
        loss_idx = loss_idx[valid]

        max_y = self.max_depth // self.res  # hfov = 45, vfov is lesser
        ranks = -cam_points[:, 1] + \
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
        loss_mask = loss_mask.reshape(B, N)

        # Dilate joint-opt mask
        dilated_mask = loss_mask.reshape(B, self.opt.height, self.opt.width).unsqueeze(1) # B x 1 x H x W
        kernel = torch.ones((3,3)).float().cuda()
        dilated_mask = kornia.morphology.dilation(dilated_mask, kernel).squeeze(1)

        # Prepare dilated depth for filtering points.
        # Idea is to copy depth of selected keypoints to its neighbours
        # Then we subtract that from the original depth and filter based on threshold.
        masked_depth = gt_depth * loss_mask.reshape(B, self.opt.height, self.opt.width)
        kernel = torch.ones((1,3,3)).float().cuda()
        dilated_depth = kornia.filters.filter2d(masked_depth.unsqueeze(1), kernel, border_type='constant').squeeze(1)
        depth_diff = torch.abs(gt_depth - dilated_depth)

        # Now use this depth threshold filter only on new points sampled using dilation.
        # Merge the remaining points with the original one to generate final loss mask.
        org_loss_mask = loss_mask.reshape(B, self.opt.height, self.opt.width)
        new_loss_mask = (depth_diff < 0.025) & (dilated_mask - org_loss_mask.float()).bool()
        loss_mask = (new_loss_mask | org_loss_mask).reshape(B, N)

        outputs[('combined_db_mask', 'l', 0)] = loss_mask.reshape(B, self.opt.height, self.opt.width)

        bev = inputs[('bev_gt', side)].reshape(-1) # Flattened from B x Hb x Wb
        B_lifted = torch.zeros(B*N, device=bev.device, dtype=torch.long)
        gather_indices = cam_points[:, 0] + \
            (self.bev_size[1] * cam_points[:, 2]) + \
            (self.bev_size[1] * self.bev_size[0] * cam_points[:, 3]) 

        bev_vals = torch.gather(bev, 0, gather_indices)
        B_lifted[valid_idx] = bev_vals 

        B_lifted = B_lifted.reshape(B, N)
        
        # semantics = inputs[('semantics_gt', 'l')]
        tgt_combined = (B_lifted + 3 * depth_L.reshape(B, N)).long()

        pred_combined_logprobs = outputs[('combined_db_logprobs', 'l', 0)].reshape(
            (B, N, D * 3)).permute(0, 2, 1)  # B x (DxC) x N


        ## To check where all the bev is sampled
        # bev_sampled = torch.zeros_like(bev)
        # bev_sampled[gather_indices] = 1
        # bev_sampled = bev_sampled.reshape(inputs[('bev_gt', side)].shape)
        # target_prob = bev_sampled.unsqueeze(dim=2).repeat((1, 1, self.n_classes, 1, 1))
        # target_logits = torch.log((target_prob + 1e-7)/(1 - target_prob + 1e-7))
        # outputs[("bev", 'l', 0)] = target_logits.reshape(outputs[("bev", 'l', 0)].shape)

        # pred_combined = torch.argmax(pred_combined_probs, dim=1)
        # mismatch = ((tgt_combined == pred_combined) * loss_mask).reshape(B, self.opt.height, self.opt.width)
        # outputs[('combined_db_mask', 'l', 0)] = mismatch

        # sum_probs = torch.exp(pred_combined_logprobs).sum(dim=1)
        # outputs[('combined_db_mask', 'l', 0)] = ((sum_probs > 1.1) & loss_mask).reshape(B, self.opt.height, self.opt.width)

        combined_loss = self.combined_nll_loss(pred_combined_logprobs, tgt_combined) # B x N
        combined_loss *= loss_mask

        combined_loss = combined_loss.mean()

        depth_weight = 1.0
        bev_weight = 1.0
        combined_weight = 1.0

        # all_losses = bev_losses
        # all_losses = depth_weight * depth_losses + bev_weight * bev_losses
        all_losses = depth_weight * depth_losses + bev_weight * bev_losses # + combined_weight * combined_loss
        # if epoch < 10:
        #     all_losses = depth_losses
        # else:
        #     all_losses = depth_weight * depth_losses + bev_weight * bev_losses + combined_weight * combined_loss
        

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
        # encoder_dict = {
        #     'name': 'disp_encoder',
        #     'params': list(self.encoder.parameters())
        # }
        # if hasattr(opt, 'encoder_lr'):
        #     encoder_dict['lr'] = opt.encoder_lr

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

        model_params = [depth_decoder_dict, bev_decoder_dict]
        return model_params
