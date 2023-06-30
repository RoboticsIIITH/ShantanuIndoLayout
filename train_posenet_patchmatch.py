# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
from os import unlink

from configs.default import get_cfg_defaults

import numpy as np
import time
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import cv2

from utils import *
import networks

import datasets


class Trainer:
    def __init__(self, options):
        self.opt = options
        print("Training mode: {}".format(self.opt.mode))

        assert self.opt.model_name is not None
        os.makedirs(self.opt.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        
        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.POSE.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids first index should be 0"

        self.setup_disparity_model()

        self.setup_rgbd_encoder_model()

        self.setup_pose_model()

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        self.setup_datasets()

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        # SSIM and SSIM sparse
        self.ssim_sparse = networks.layers.SSIM_sparse()
        self.ssim_sparse.to(self.device)
        
        self.ssim = networks.layers.SSIM()
        self.ssim.to(self.device)

        # Backproject and Project3D modules used in photometric reconstruction
        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = networks.layers.BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = networks.layers.Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        self.save_opts()

    # Setup models and datasets.
    def setup_disparity_model(self):
        if self.opt.DISPARITY.module == 'AnyNet':
            disp_module = lambda x: nn.DataParallel(networks.AnyNet(x))
        else:
            raise NotImplementedError()

        self.models["disp_module"] = disp_module(self.opt.DISPARITY)
        self.models["disp_module"].to(self.device)
        if not self.opt.DISPARITY.freeze_weights:
            self.parameters_to_train += list(self.models["disp_module"].parameters())
        else:
            for param in self.models["disp_module"].parameters():
                param.requires_grad = False

    def setup_rgbd_encoder_model(self):
        ## RGBD_ENCODER
        if self.opt.RGBD_ENCODER.module == 'concatenate':
            rgbd_encoder = networks.ConcatenateRGBD
        elif self.opt.RGBD_ENCODER.module == 'feature_extraction_conv':
            rgbd_encoder = networks.feature_extraction_conv
        else:
            raise NotImplementedError()

        self.models["rgbd_encoder"] = rgbd_encoder(**self.opt.RGBD_ENCODER)
        self.models["rgbd_encoder"].to(self.device)
        if not self.opt.RGBD_ENCODER.freeze_weights:
            self.parameters_to_train += list(self.models["rgbd_encoder"].parameters())
        else:
            for param in self.models["rgbd_encoder"].parameters():
                param.requires_grad = False

    def setup_pose_model(self):
        if self.opt.POSE.pose_model_type == "separate_resnet":
            self.models["pose_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers,
                self.opt.weights_init == "pretrained",
                num_input_images=self.num_pose_frames)

            self.models["pose_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())

            self.models["pose"] = networks.PoseDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2)

        elif self.opt.POSE.pose_model_type == "shared":
            self.models["pose"] = networks.PoseDecoder(
                self.models["rgbd_encoder"].num_ch_enc, self.num_pose_frames)

        elif self.opt.POSE.pose_model_type == "posecnn":
            self.models["pose"] = networks.PoseCNN(self.num_pose_frames)
            self.models["pose"].to(self.device)

        else:
            raise NotImplementedError(self.opt.POSE.pose_model_type)
        
        if not self.opt.POSE.freeze_weights:
            self.parameters_to_train += list(self.models["pose"].parameters())
        else:
            for param in self.models["pose"].parameters():
                param.requires_grad = False

    def setup_datasets(self):
        # data
        datasets_dict = {"habitat": datasets.HabitatDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        with open(self.opt.train_split_file, 'r') as f:
            train_filenames = f.read().splitlines()
        with open(self.opt.val_split_file, 'r') as f:
            val_filenames = f.read().splitlines()

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, len(self.opt.scales), is_train=True, 
            extract_keypts=True)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.train_workers, pin_memory=True, drop_last=True)
            
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, len(self.opt.scales), is_train=False, 
            extract_keypts=False)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.val_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

    # Setup training and validation functions based on modes.
    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

        if self.opt.mode == 'dense':
            self.extract_poses = self.predict_poses
            self.generate_pred = self.generate_images_pred
            self.compute_reprojection_loss = self.compute_dense_reprojection_loss
            self.compute_losses = self.compute_dense_losses
            return
        
        if self.opt.mode == 'sparse':
            self.extract_poses = self.predict_poses
            self.generate_pred = self.generate_sparse_pred
            self.compute_reprojection_loss = self.compute_sparse_reprojection_loss
            self.compute_losses = self.compute_patch_losses
            return

        if self.opt.mode == 'debug':
            self.extract_poses = getattr(self, self.opt.DEBUG.extract_poses)
            self.generate_pred = getattr(self, self.opt.DEBUG.generate_pred)
            self.compute_reprojection_loss = getattr(self, self.opt.DEBUG.compute_reprojection_loss)
            self.compute_losses = getattr(self, self.opt.DEBUG.compute_losses)
            return

        raise NotImplementedError(f'{self.opt.mode} not implemented for training. \
            Only dense, sparse and debug modes supported.')

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

        if self.opt.mode == 'dense':
            self.extract_poses = self.predict_poses
            self.generate_pred = self.generate_images_pred
            self.compute_reprojection_loss = self.compute_dense_reprojection_loss
            self.compute_losses = self.compute_dense_losses
            return
        
        if self.opt.mode == 'sparse':
            self.extract_poses = self.predict_poses
            self.generate_pred = self.generate_images_pred
            self.compute_reprojection_loss = self.compute_dense_reprojection_loss
            self.compute_losses = self.compute_dense_losses
            return

        if self.opt.mode == 'debug':
            self.extract_poses = getattr(self, self.opt.DEBUG.extract_poses)
            self.generate_pred = self.generate_images_pred
            self.compute_reprojection_loss = self.compute_dense_reprojection_loss
            self.compute_losses = self.compute_dense_losses
            return

        raise NotImplementedError(f'{self.opt.mode} not implemented for training. \
            Only dense, sparse and debug modes supported.')

    # Implementation
    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_metrics(inputs, outputs, losses)
            
            if "relpose_gt" in inputs:
                self.compute_pose_metrics(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_metrics(inputs, outputs, losses)

                if "relpose_gt" in inputs:
                    self.compute_pose_metrics(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            if not isinstance(ipt, (torch.Tensor)):
                continue
            inputs[key] = ipt.to(self.device)
        
        # DEPTH  
        imgL = inputs[("color_aug", 'l', 0)]
        imgL = imgL.reshape((-1, *imgL.shape[2:]))
        imgR = inputs[("color_aug", 'r', 0)]
        imgR = imgR.reshape((-1, *imgR.shape[2:]))

        disps, features_l, _ = self.models['disp_module'](imgL, imgR)
        disp = torch.abs(disps[-1])
        depth = torch.nan_to_num((self.opt.baseline * self.opt.focal_length)/disp, 
                        self.opt.max_depth, self.opt.max_depth)
        depth = torch.clamp(depth, self.opt.min_depth, self.opt.max_depth)

        # RGBD
        outputs = {}
        outputs[('features', 0)] = features_l
        outputs[('depth', 0)] = depth
        rgbd_features = self.models['rgbd_encoder'](imgL, outputs)

        # Reshape all outputs to (batch, timesteps, output_shape)
        for idx, feature in enumerate(features_l):
            outputs[('features', 0)][idx] = feature.reshape((self.opt.batch_size, -1, *feature.shape[1:]))
        outputs[('depth', 0)] = depth.reshape((self.opt.batch_size, -1, *depth.shape[1:]))
        outputs[('rgbd_features', 0)] = rgbd_features.reshape((self.opt.batch_size, -1, *rgbd_features.shape[1:]))

        # POSE
        if self.num_input_frames > 1:
            outputs.update(self.extract_poses(inputs, outputs))

        # LOSS
        self.generate_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    # Pose Fns
    def get_gt_poses(self, inputs, features):
        outputs = {}
        outputs[("cam_T_cam", 0)] = Variable(inputs["relpose_gt"].clone(), 
                                        requires_grad=True).to(self.device)
        return outputs

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        outputs[("axisangle", 0)] = []
        outputs[("translation", 0)] = []
        outputs[("cam_T_cam", 0)] = []

        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.POSE.pose_model_type == "separate_resnet":
                pose_feats = inputs[("color_aug", 'l', 0)]
            else:
                pose_feats = features[('rgbd_features', 0)]

            for idx in range(1, len(self.opt.frame_ids)):
                # To maintain ordering we always pass frames in temporal order
                f_i = self.opt.frame_ids[idx]
                if f_i < 0:
                    pose_inputs = [pose_feats[:,idx,...], pose_feats[:,0,...]]
                else:
                    pose_inputs = [pose_feats[:,0,...], pose_feats[:,idx,...]]

                if self.opt.POSE.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                elif self.opt.POSE.pose_model_type == "posecnn":
                    pose_inputs = torch.cat(pose_inputs, 1)

                axisangle, translation = self.models["pose"](pose_inputs)
                outputs[("axisangle", 0)].append(axisangle)
                outputs[("translation", 0)].append(translation)

                # Invert the matrix if the frame id is negative
                T = networks.layers.transformation_from_parameters(axisangle[:, 0], 
                        translation[:, 0], invert=(f_i < 0))

                outputs[("cam_T_cam", 0)].append(T)

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.POSE.pose_model_type == "separate_resnet":
                pose_inputs = inputs[("color_aug", 'l', 0)]

                if self.opt.POSE.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type in ["shared", "posecnn"]:
                pose_inputs = features[('rgbd_features', 0)]

            axisangle, translation = self.models["pose"](pose_inputs)

            outputs[("axisangle", 0)] = axisangle
            outputs[("translation", 0)] = translation

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                T = networks.layers.transformation_from_parameters(axisangle[:, i], translation[:, i])
                outputs[("cam_T_cam", 0)].append(T)

        for k,v in outputs.items():
            outputs[k] = torch.stack(v, dim=1)

        return outputs

    # Dense Fns
    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            # outputs[("sample", scale)] = torch.zeros((
            #     self.opt.batch_size, self.num_input_frames-1, 
            #     self.opt.height, self.opt.width, 2)).to(self.device)
            # outputs[("color", scale)] = torch.zeros((
            #     self.opt.batch_size, self.num_input_frames-1, 
            #     3, self.opt.height, self.opt.width)).to(self.device)

            outputs[("sample", scale)] = []
            outputs[("color", scale)] = []

            for idx in range(self.num_input_frames-1):
                T = outputs[("cam_T_cam", 0)][:, idx, ...]

                depth = outputs[('depth', 0)][:,0, ...]

                # # from the authors of https://arxiv.org/abs/1712.00175
                # if self.opt.POSE.pose_model_type == "posecnn":

                #     axisangle = outputs[("axisangle", 0, frame_id)]
                #     translation = outputs[("translation", 0, frame_id)]

                #     inv_depth = 1 / depth
                #     mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                #     T = networks.layers.transformation_from_parameters(
                #         axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[0](
                    depth, inputs[("inv_K", 0)])
                pix_coords = self.project_3d[0](
                    cam_points, inputs[("K", 0)], T)

                # outputs[("sample", scale)][:,idx, ...] = pix_coords
                outputs[("sample", scale)].append(pix_coords)

                # outputs[("color", scale)][:,idx, ...] = F.grid_sample(
                #     inputs[("color", 'l', 0)][:, idx, ...],
                #     outputs[("sample", scale)][:,idx, ...],
                #     padding_mode="border")
                pred_color = F.grid_sample(
                    inputs[("color", 'l', 0)][:, idx+1, ...],
                    outputs[("sample", scale)][idx],
                    padding_mode="border")
                outputs[("color", scale)].append(pred_color)

            if not self.opt.disable_automasking:
                outputs[("color_identity", scale)] = inputs[("color", 'l', scale)]

    def compute_dense_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_dense_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            depth = outputs[("depth", scale)][:,0, ...]
            color = inputs[("color", 'l', scale)][:,0, ...]
            target = inputs[("color", 'l', 0)][:,0, ...]

            for idx in range(self.num_input_frames-1):
                # pred = outputs[("color", scale)][:,idx, ...]
                pred = outputs[("color", scale)][idx]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for idx in range(self.num_input_frames-1):
                    pred = inputs[("color", 'l', 0)][:,idx, ...]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            # TODO: Fix this for finetuning depth prediction
            # smooth_loss = networks.layers.get_smooth_loss(depth, color)
            # loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale) * (not self.opt.DISPARITY.freeze_weights)

            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    # Sparse Fns
    def generate_sparse_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
     
        for scale in self.opt.scales:
            depth = outputs[('depth', 0)][:, 0, ...]

            # sample depth for dso points                                                
            dso_points = inputs['dso_points']
            y0 = dso_points[:, :, 0]
            x0 = dso_points[:, :, 1]
            dso_points = torch.cat((x0.unsqueeze(2), y0.unsqueeze(2)), dim=2)

            flat = (x0 + y0 * self.opt.width).long()
            dso_depth = torch.gather(depth.view(self.opt.batch_size, -1), 1, flat)

            # generate pattern
            meshgrid = np.meshgrid([-2, 0, 2], [-2, 0, 2], indexing='xy')
            meshgrid = np.stack(meshgrid, axis=0).astype(np.float32)
            meshgrid = torch.from_numpy(meshgrid).to(dso_points.device).permute(1, 2, 0).view(1, 1, 9, 2)
            dso_points = dso_points.unsqueeze(2) + meshgrid
            dso_points = dso_points.reshape(self.opt.batch_size, -1, 2)
            dso_depth = dso_depth.view(self.opt.batch_size, -1, 1).expand(-1, -1, 9).reshape(self.opt.batch_size, 1, -1)

            # convert to point cloud
            xy1 = torch.cat((dso_points, torch.ones_like(dso_points[:, :, :1])), dim=2)
            xy1 = xy1.permute(0, 2, 1)
            cam_points = (inputs[("inv_K", 0)][:, :3, :3] @ xy1) * dso_depth
            points = torch.cat((cam_points, torch.ones_like(cam_points[:, :1, :])), dim=1)

            outputs[("dso_mask", scale)] = []
            outputs[("dso_color", scale)] = []

            for idx in range(self.num_input_frames):
                if idx == 0:
                    T = torch.eye(4).view(1, 4, 4).expand(self.opt.batch_size, 4, 4).cuda()
                else:
                    T = outputs[("cam_T_cam", 0)][:,idx-1,...]

                # projects to different frames
                P = torch.matmul(inputs[("K", 0)], T)[:, :3, :]
                cam_points = torch.matmul(P, points)

                pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + 1e-7)
                pix_coords = pix_coords.view(self.opt.batch_size, 2, -1, 9)
                pix_coords = pix_coords.permute(0, 2, 3, 1)
                pix_coords[..., 0] /= self.opt.width - 1
                pix_coords[..., 1] /= self.opt.height - 1
                pix_coords = (pix_coords - 0.5) * 2

                # save mask
                valid = (pix_coords[..., 0] > -1.) & (pix_coords[..., 0] < 1.) & (pix_coords[..., 1] > -1.) & (
                            pix_coords[..., 1] < 1.)
                outputs[("dso_mask", scale)].append(valid.unsqueeze(1).float())

                # sample patch from color images
                pred_color = F.grid_sample(
                    inputs[("color", 'l', 0)][:, idx, ...],
                    pix_coords,
                    padding_mode="border")
                outputs[("dso_color", scale)].append(pred_color)

    def compute_sparse_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target points
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            l1_loss = l1_loss.mean(3, True)
            ssim_loss = self.ssim_sparse(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_patch_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            sparse_reprojection_losses = []

            depth = outputs[('depth', scale)][:, 0, ...]
            color = inputs[("color", 'l', scale)][:, 0, ...]
            dso_target = outputs[("dso_color", scale)][0]

            # dso loss
            for idx in range(1, self.num_input_frames):
                dso_pred = outputs[("dso_color", scale)][idx]
                sparse_reprojection_losses.append(self.compute_reprojection_loss(dso_pred, dso_target))

            dso_combined = torch.cat(sparse_reprojection_losses, dim=1)
            dso_to_optimise, _ = torch.min(dso_combined, dim=1)
            dso_loss = dso_to_optimise.mean()
            loss += dso_loss 

            losses["dso_loss/{}".format(scale)] = dso_loss

            # TODO: Fix this for finetuning depth prediction
            # smooth_loss = networks.layers.get_smooth_loss(depth, color)
            # loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale) * (not self.opt.DISPARITY.freeze_weights)
            # losses["smooth_loss/{}".format(scale)] = smooth_loss

            total_loss += loss

            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    # Metrics Fns
    def compute_depth_metrics(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0)][:,0, ...].squeeze().detach()

        depth_gt = inputs["depth_gt"]
        mask = (depth_gt > self.opt.min_depth) * (depth_gt < self.opt.max_depth)

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]

        depth_errors = networks.layers.compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def compute_pose_metrics(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        gt = inputs["relpose_gt"].cpu().detach()

        pred = outputs[("cam_T_cam", 0)].cpu().detach()

        error = torch.linalg.matrix_norm(pred - gt)

        losses["pose_error"] = torch.mean(error)

        gt = gt.numpy()
        pred = pred.numpy()

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for idx, frame_id in enumerate(self.opt.frame_ids[1:]): 
                err_dict = get_pose_diff(gt[j,idx,...], pred[j,idx,...])
                for k,v in err_dict.items():
                    losses["err_{}_{}/{}".format(k, frame_id, j)] = v

    # Logging Fns
    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            if isinstance(v, dict):
                writer.add_scalars("{}".format(l), v, self.step)
            else:
                writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for idx, frame_id in enumerate(self.opt.frame_ids):
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        invnormalize_imagenet(inputs[("color", 'l', s)][j, idx].data), self.step)
                    if s == 0 and frame_id != 0 and ("color", 0) in outputs:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            invnormalize_imagenet(outputs[("color", s)][idx-1][j].data), self.step)

                writer.add_image(
                    "depth_{}/{}".format(s, j),
                    normalize_image(outputs[("depth", s)][j, 0], 
                        (self.opt.min_depth, self.opt.max_depth)), self.step)

            if "svo_map" in inputs:
                writer.add_image(
                    "svo_{}/{}".format(0, j), 
                    inputs['svo_map'][j].unsqueeze(0).data, self.step)
                writer.add_image(
                    "svo_noise_{}/{}".format(0, j), 
                    inputs['svo_map_noise'][j].unsqueeze(0).data, self.step)

    def dump_raw_data(self, inputs, outputs):
        imgL = inputs[("color_aug", 'l', 0)]
        imgR = inputs[("color_aug", 'r', 0)]
        pred_depth = outputs[("depth", 0)]
        
        for idx in range(self.opt.batch_size):
            line = inputs['filename'][idx].split()
            outdir = os.path.join('/tmp/indoor_layout_estimation/disp_module_output', 
                        os.path.basename(line[0]))
            os.makedirs(outdir, exist_ok=True)

            # write depth
            outpath = os.path.join(outdir, '{}_d.png'.format(line[1]))
            img = pred_depth[idx, 0, ...].squeeze().cpu().detach().numpy()
            img = (img * 65535/10).astype(np.uint16)
            cv2.imwrite(outpath, img)

            # write depth_gt
            outpath = os.path.join(outdir, '{}_gt.png'.format(line[1]))
            img = inputs['depth_gt'][idx].squeeze().cpu().detach().numpy()
            img = (img * 65535/10).astype(np.uint16)
            cv2.imwrite(outpath, img)

            # write imgL
            outpath = os.path.join(outdir, '{}_l.png'.format(line[1]))
            img = invnormalize_imagenet(imgL[idx, 0, ...].cpu().detach())
            img = (img.permute(1,2,0).numpy() * 255).astype(np.uint8)
            cv2.imwrite(outpath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # write imgR
            outpath = os.path.join(outdir, '{}_r.png'.format(line[1]))
            img = invnormalize_imagenet(imgR[idx, 0, ...].cpu().detach())
            img = (img.permute(1,2,0).numpy() * 255).astype(np.uint8)
            cv2.imwrite(outpath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Model and opts Fns
    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        model_dir = os.path.join(self.log_path, "models")
        os.makedirs(model_dir, exist_ok=True)

        with open(os.path.join(model_dir, 'opt.yaml'), 'w') as f:
            f.write(self.opt.dump())

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        os.makedirs(save_folder, exist_ok=True)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'disp_module':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(load_weights_folder), \
            "Cannot find folder {}".format(load_weights_folder)
        print("loading model from folder {}".format(load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(load_weights_folder, "{}.pth".format(n))
            # model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # model_dict.update(pretrained_dict)
            mk, uk = self.models[n].load_state_dict(pretrained_dict, strict=False)
            print('{}, missing keys:{}, unknown keys:{}'.format(n, mk, uk))

        # loading adam state
        optimizer_load_path = os.path.join(load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    if len(sys.argv) < 2:
        raise Exception("Usage: python3 train_posenet_patchmatch.py <config_file> <additional-args>")
    cfg.merge_from_file(sys.argv[1])
    additional_args = sys.argv[2:]
    cfg.merge_from_list(additional_args)
    cfg.freeze()
    trainer = Trainer(cfg)
    trainer.train()
