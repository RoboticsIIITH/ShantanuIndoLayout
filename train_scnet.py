# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import pickle

from os import unlink
from configs.default import get_cfg_defaults

import numpy as np
import time
import sys
import random
import warnings

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

from tqdm import tqdm

from collections import defaultdict

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Trainer:
    def __init__(self, options):
        self.opt = options
        print("Training mode: {}".format(self.opt.mode))
        assert np.all([(k in self.opt.PIPELINE.run) for k in self.opt.PIPELINE.train])

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

        self.setup_bev_decoder_model()

        self.setup_discriminator_model()

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

        # use wandb
        if self.opt.DEBUG.wandb:
            try:
                import wandb
                wandb.init(config=self.opt, sync_tensorboard=True)
            except ImportError:
                warnings.warn("Wandb not installed. Logging only locally...", ImportWarning)

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

        # Homography modules in bev reconstruction
        self.homography_warp = networks.layers.WarpHomography(self.opt.batch_size, self.opt.bev_height, \
            self.opt.bev_width, self.opt.bev_res)
        self.homography_warp.to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        self.bev_ce_loss = nn.CrossEntropyLoss(reduction='none', \
            weight=torch.Tensor(self.opt.bev_ce_weights).to(self.device))

        self.criterion_d = nn.BCEWithLogitsLoss()

        self.save_opts()
        self.epoch = 0
        self.step = 0

    # Setup models and datasets.
    def setup_disparity_model(self):
        if self.opt.DISPARITY.module == 'AnyNet':
            disp_module = networks.AnyNet
        else:
            raise NotImplementedError()

        self.models["disp_module"] = nn.DataParallel(disp_module(self.opt.DISPARITY))
        self.models["disp_module"].cuda()
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

    def setup_bev_decoder_model(self):
        ## BEV_DECODER
        if self.opt.BEV_DECODER.module == 'mock':
            bev_decoder = networks.MockDecoder
        elif self.opt.BEV_DECODER.module == 'layout_decoder':
            bev_decoder = networks.LayoutDecoder
        else:
            raise NotImplementedError()

        self.models["bev_decoder"] = bev_decoder(**self.opt.BEV_DECODER)
        self.models["bev_decoder"].to(self.device)

        if not self.opt.BEV_DECODER.freeze_weights:
            self.parameters_to_train += list(self.models["bev_decoder"].parameters())
        else:
            for param in self.models["bev_decoder"].parameters():
                param.requires_grad = False

    def setup_discriminator_model(self):
        self.models["discriminator"] = networks.Discriminator(**self.opt.DISCRIMINATOR)
        self.models["discriminator"].to(self.device)

        if not self.opt.DISCRIMINATOR.freeze_weights:
            self.parameters_to_train_D = list(self.models["discriminator"].parameters())
            self.model_optimizer_D = optim.Adam(self.parameters_to_train_D,
                self.opt.DISCRIMINATOR.learning_rate)
            self.model_lr_scheduler_D =  optim.lr_scheduler.StepLR(
                self.model_optimizer_D, self.opt.DISCRIMINATOR.scheduler_step_size, 0.1)
        else:
            for param in self.models["discriminator"].parameters():
                param.requires_grad = False
            self.model_optimizer_D = None
            self.model_lr_scheduler_D = None


        self.patch = (1, self.opt.bev_height // 2 **
                    4, self.opt.bev_width // 2**4)

        self.valid = Variable(
            torch.Tensor(
                np.ones(
                    (self.opt.batch_size,
                    *self.patch))),
            requires_grad=False).float().cuda()

        self.fake = Variable(
            torch.Tensor(
                np.zeros(
                    (self.opt.batch_size,
                    *self.patch))),
            requires_grad=False).float().cuda()

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

        load_keys = set()
        if 'DISPARITY' in self.opt.PIPELINE.train:
            load_keys.add('depth')
        if 'RGBD' in self.opt.PIPELINE.train:
            load_keys.add('semantics')
        if 'BEV' in self.opt.PIPELINE.train:
            load_keys.add('bev')
            load_keys.add('depth')
            # load_keys.add('semantics')
        if 'DISCR' in self.opt.PIPELINE.train:
            load_keys.add('discr')
        if 'POSE' in self.opt.PIPELINE.train:
            load_keys.add('pose')

        with open(self.opt.train_split_file, 'r') as f:
            train_filenames = f.read().splitlines()
        with open(self.opt.val_split_file, 'r') as f:
            val_filenames = f.read().splitlines()

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, load_keys,
            self.opt.height, self.opt.width,
            self.opt.frame_ids, len(self.opt.scales), is_train=False, 
            bev_width=self.opt.bev_width, bev_height=self.opt.bev_height, 
            bev_res=self.opt.bev_res, floor_path=self.opt.floor_path)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True, worker_init_fn=seed_worker,
            num_workers=self.opt.val_workers, pin_memory=True, drop_last=True)
        # self.val_iter = iter(self.val_loader)

        # For sparse mode, we need keypoints only for training.
        if self.opt.mode == 'sparse':
            load_keys.add('keypts')

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, load_keys,
            self.opt.height, self.opt.width,
            self.opt.frame_ids, len(self.opt.scales), is_train=True, 
            bev_width=self.opt.bev_width, bev_height=self.opt.bev_height, 
            bev_res=self.opt.bev_res, floor_path=self.opt.floor_path)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True, worker_init_fn=seed_worker,
            num_workers=self.opt.train_workers, pin_memory=True, drop_last=True)

        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

    # Setup training and validation functions based on modes.
    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

        if self.opt.mode == 'dense':
            self.extract_depth = self.generate_disparity_depth
            self.extract_poses = self.predict_poses
            self.generate_reprojection_pred = self.generate_dense_pred
            self.compute_reprojection_losses = self.compute_dense_reprojection_losses
            return
        
        if self.opt.mode == 'sparse':
            self.extract_depth = self.generate_disparity_depth
            self.extract_poses = self.predict_poses
            self.generate_reprojection_pred = self.generate_sparse_pred
            self.compute_reprojection_losses = self.compute_sparse_reprojection_losses
            return

        if self.opt.mode == 'debug':
            self.extract_depth = getattr(self, self.opt.DEBUG.extract_depth)
            self.extract_poses = getattr(self, self.opt.DEBUG.extract_poses)
            self.generate_reprojection_pred = getattr(self, self.opt.DEBUG.generate_reprojection_pred)
            self.compute_reprojection_losses = getattr(self, self.opt.DEBUG.compute_reprojection_losses)
            return

        raise NotImplementedError(f'{self.opt.mode} not implemented for training. \
            Only dense, sparse and debug modes supported.')

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

        if self.opt.mode == 'dense':
            self.extract_depth = self.generate_disparity_depth
            self.extract_poses = self.predict_poses
            self.generate_reprojection_pred = self.generate_dense_pred
            self.compute_reprojection_losses = self.compute_dense_reprojection_losses
            return
        
        if self.opt.mode == 'sparse':
            self.extract_depth = self.generate_disparity_depth
            self.extract_poses = self.predict_poses
            self.generate_reprojection_pred = self.generate_dense_pred
            self.compute_reprojection_losses = self.compute_dense_reprojection_losses
            return

        if self.opt.mode == 'debug':
            self.extract_depth = getattr(self, self.opt.DEBUG.extract_depth)
            self.extract_poses = getattr(self, self.opt.DEBUG.extract_poses)
            self.generate_reprojection_pred = self.generate_dense_pred
            self.compute_reprojection_losses = self.compute_dense_reprojection_losses
            return

        raise NotImplementedError(f'{self.opt.mode} not implemented for training. \
            Only dense, sparse and debug modes supported.')

    # Implementation
    def train(self):
        """Run the entire training pipeline
        """
        self.start_time = time.time()
        self.val()

        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.val()
                self.save_model()

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()

        metrics = defaultdict(lambda : 0)
    
        for inputs in tqdm(self.val_loader):
            with torch.no_grad():
                outputs = self.process_batch(inputs)
                losses = self.compute_losses(inputs, outputs)

                if "DISPARITY" in self.opt.PIPELINE.run:
                    self.compute_depth_metrics(inputs, outputs, losses)
                
                if "POSE" in self.opt.PIPELINE.run:
                    self.compute_pose_metrics(inputs, outputs, losses)

                if "BEV" in self.opt.PIPELINE.run:
                    self.compute_bev_metrics(inputs, outputs, losses)

            if self.opt.script_mode == 'predict':
                self.dump_raw_data(inputs, outputs)
                
            for k, v in losses.items():
                if v is None:
                    continue
                if isinstance(v, dict):
                    if not isinstance(metrics[k], dict):
                        metrics[k] = defaultdict(lambda : 0)
                    for (i, v_i) in v.items():
                        metrics[k][i] += v_i
                else:
                    metrics[k] += v

        for k,v in metrics.items():
            if isinstance(v, dict):
                for i in v.keys():
                    metrics[k][i] /= len(self.val_loader)
                metrics[k] = dict(metrics[k])
            else:
                metrics[k] /= len(self.val_loader)

        metrics = dict(metrics)

        self.log("val", inputs, outputs, metrics)
        del inputs, outputs, losses

        self.set_train()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()
        if self.model_lr_scheduler_D is not None:
            self.model_lr_scheduler_D.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in tqdm(enumerate(self.train_loader)):

            before_op_time = time.time()

            outputs = self.process_batch(inputs)
            losses = self.compute_losses(inputs, outputs)
            discr_backward = "discriminator_loss" in losses

            self.model_optimizer.zero_grad()
            losses["loss"].backward(retain_graph=discr_backward)
            self.model_optimizer.step()

            if discr_backward:
                self.model_optimizer_D.zero_grad()
                losses["discriminator_loss"].backward()
                self.model_optimizer_D.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses)

                if "DISPARITY" in self.opt.PIPELINE.run:
                    self.compute_depth_metrics(inputs, outputs, losses)

                if "POSE" in self.opt.PIPELINE.run:
                    self.compute_pose_metrics(inputs, outputs, losses)

                if "BEV" in self.opt.PIPELINE.run:
                    self.compute_bev_metrics(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            if not isinstance(ipt, (torch.Tensor)):
                continue
            inputs[key] = ipt.to(self.device)
        
        outputs = {}

        # DEPTH
        if 'DISPARITY' in self.opt.PIPELINE.run:
            outputs.update(self.extract_depth(inputs, outputs))
        
        # RGBD
        if 'RGBD' in self.opt.PIPELINE.run:
            rgbd_features = self.models['rgbd_encoder'](inputs, outputs)
            outputs[('rgbd_features', 0)] = rgbd_features

        # BEV
        if 'BEV' in self.opt.PIPELINE.run:
            bev = self.models['bev_decoder'](inputs, outputs)
            outputs[('bev', 0)] = bev

        # Reshape all outputs to (batch, timesteps, output_shape)
        for k, v in outputs.items():
            outputs[k] = v.reshape((self.opt.batch_size, -1, *v.shape[1:]))

        # POSE
        if ('POSE' in self.opt.PIPELINE.run) and (self.num_input_frames > 1):
            outputs.update(self.extract_poses(inputs, outputs))

        return outputs

    def compute_losses(self, inputs, outputs):
        losses = {
            "disparity_loss": None,
            "rgbd_loss": None,
            "bev_loss": None,
            "gan_loss": None,
            "reprojection_loss": None,
            "homography_loss": None
        }

        # DISPARITY LOSS
        if 'DISPARITY' in self.opt.PIPELINE.train:
            losses.update(self.compute_disparity_losses(inputs, outputs))

        # RGBD LOSS
        if 'RGBD' in self.opt.PIPELINE.train:
            losses.update(self.compute_rgbd_losses(inputs, outputs))

        # BEV LOSS
        if 'BEV' in self.opt.PIPELINE.train:
            losses.update(self.compute_bev_losses(inputs, outputs))
        
        # DISCRIMINATOR LOSS
        if 'DISCR' in self.opt.PIPELINE.train and 'BEV' in self.opt.PIPELINE.run:
            losses.update(self.compute_discriminator_losses(inputs, outputs))

        # REPROJECTION LOSS
        if 'POSE' in self.opt.PIPELINE.train:
            self.generate_reprojection_pred(inputs, outputs)
            losses.update(self.compute_reprojection_losses(inputs, outputs))

        # HOMOGRAPHY LOSS
        if 'POSE' in self.opt.PIPELINE.train and 'BEV' in self.opt.PIPELINE.run:
            self.generate_homography_pred(inputs, outputs)
            losses.update(self.compute_homography_losses(inputs, outputs))

        # MERGE LOSSES
        total_loss = 0
        for key, weight in self.opt.loss_weights.items():
            if losses[key] is not None:
                total_loss += weight * losses[key]

        losses["loss"] = total_loss
        return losses

    # Disparity Fns
    def get_gt_disparity_depth(self, inputs, features):
        depth = Variable(inputs["depth_gt"].clone(), requires_grad=False).to(self.device)
        depth = torch.nan_to_num(depth, self.opt.max_depth, self.opt.max_depth)
        depth = torch.clamp(depth, self.opt.min_depth, self.opt.max_depth)
        depth = depth.reshape((-1, 1, *depth.shape[2:]))
        
        outputs = {}
        outputs[('depth', 0)] = depth
        return outputs

    def generate_disparity_depth(self, inputs, features):
        imgL = inputs[("color_aug", 'l', 0)]
        imgL = imgL.reshape((-1, *imgL.shape[2:]))
        imgR = inputs[("color_aug", 'r', 0)]
        imgR = imgR.reshape((-1, *imgR.shape[2:]))

        disps, _, _ = self.models['disp_module'](imgL, imgR)
        disp = torch.abs(disps[-1])
        depth = torch.nan_to_num((self.opt.baseline * self.opt.focal_length)/disp, 
                        self.opt.max_depth, self.opt.max_depth)
        depth = torch.clamp(depth, self.opt.min_depth, self.opt.max_depth)

        outputs = {}
        outputs[('depth', 0)] = depth
        for idx in range(len(disps)):
            outputs[('disp_l', idx)] = disps[idx]
            # outputs[('feature_l', idx)] = features_l[idx]

        return outputs

    def compute_disparity_losses(self, inputs, outputs):
        if self.opt.DISPARITY.with_spn and self.epoch >= self.opt.DISPARITY.start_epoch_for_spn:
            num_out = self.opt.DISPARITY.nblocks + 2
        else:
            num_out = self.opt.DISPARITY.nblocks + 1

        depth_L = inputs["depth_gt"].float().cuda()
        depth_L = depth_L.reshape((-1, depth_L.shape[-2], depth_L.shape[-1]))

        # Using -1 for inf,-inf to filter it using mask.
        disp_L = torch.nan_to_num((self.opt.baseline * self.opt.focal_length)/depth_L, -1, -1) 

        mask = disp_L > 0
        mask.detach_()

        all_losses = []
        for idx in range(num_out):
            pred = outputs[('disp_l', idx)]
            pred = pred.reshape((-1, pred.shape[-2], pred.shape[-1]))
            loss = self.opt.DISPARITY.loss_weights[idx] * \
                F.smooth_l1_loss(pred[mask], disp_L[mask], size_average=True)
            all_losses.append(loss)

        losses = {}
        losses["disparity_loss"] = sum(all_losses)
        return losses

    # RGBD Fns
    def compute_rgbd_losses(self, inputs, outputs):
        losses = {}
        losses["rgbd_loss"] = Variable(torch.tensor(0).float(), requires_grad=False).to(self.device)
        return losses

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
    def generate_dense_pred(self, inputs, outputs):
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

    def dense_reprojection_loss(self, pred, target):
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

    def compute_dense_reprojection_losses(self, inputs, outputs):
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
                reprojection_losses.append(self.dense_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for idx in range(self.num_input_frames-1):
                    pred = inputs[("color", 'l', 0)][:,idx, ...]
                    identity_reprojection_losses.append(
                        self.dense_reprojection_loss(pred, target))

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
            losses["reprojection_loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["reprojection_loss"] = total_loss
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

    def sparse_reprojection_loss(self, pred, target):
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

    def compute_sparse_reprojection_losses(self, inputs, outputs):
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
                sparse_reprojection_losses.append(self.sparse_reprojection_loss(dso_pred, dso_target))

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

            losses["reprojection_loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["reprojection_loss"] = total_loss
        return losses

    # BEV Fns
    def generate_gt_bev(self, inputs, outputs):
        gt_bevs = []
        for idx in range(self.num_input_frames):
            depth = outputs[("depth", 0)][:, idx, ...]
            depth = depth.reshape((self.opt.batch_size, *depth.shape[2:]))

            semantics = inputs["semantics_gt"][:, idx, ...]
            semantics = semantics.reshape((self.opt.batch_size, *semantics.shape[2:]))

            cam_points = self.backproject_depth[0](depth, inputs[("inv_K", 0)])
            bev = self.homography_warp.project(cam_points, semantics)

            bev = bev.reshape((self.opt.batch_size, 1, *bev.shape[1:]))
            gt_bevs.append(bev)

        gt_bevs = torch.cat(gt_bevs, dim=1)
        return gt_bevs

    def compute_bev_losses(self, inputs, outputs):
        pred = outputs[("bev", 0)]
        pred = pred.reshape((-1, *pred.shape[2:]))

        if "bev_gt" not in inputs:
            inputs["bev_gt"] = self.generate_gt_bev(inputs, outputs)

        target = inputs["bev_gt"]
        target = target.reshape((-1, *target.shape[2:]))
        
        bev_loss =  self.bev_ce_loss(pred, target).mean()

        output = {"bev_loss": bev_loss}

        return output

    def compute_discriminator_losses(self, inputs, outputs):
        loss = {}

        if self.model_optimizer_D is None:
            return loss

        fake_pred = self.models["discriminator"](outputs[("bev", 0)])
        real_pred = self.models["discriminator"](inputs["discr"].float())
 
        # Train Discriminator
        if self.epoch > self.opt.DISCRIMINATOR.discr_train_epoch:
            loss["discriminator_loss"] = self.criterion_d(fake_pred, self.fake) + \
                self.criterion_d(real_pred, self.valid)
            loss["gan_loss"] = self.opt.DISCRIMINATOR.lambda_D * \
                self.criterion_d(fake_pred, self.valid)

        return loss

    def generate_homography_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        outputs["bev_sample"] = []
        outputs["bev_rc"] = []

        for idx in range(self.num_input_frames-1):
            T = outputs[("cam_T_cam", 0)][:, idx, ...]
            pix_coords = self.homography_warp(T)

            outputs["bev_sample"].append(pix_coords)

            pred_bev = F.grid_sample(
                outputs[("bev", 0)][:, idx+1, ...],
                outputs["bev_sample"][idx],
                mode='bilinear',
                padding_mode="border")
            outputs["bev_rc"].append(pred_bev)

    def bev_homography_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target points
        """
        reprojection_loss = self.bev_ce_loss(pred, target)
        return reprojection_loss.unsqueeze(dim=1)

    def compute_homography_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        target = outputs[("bev", 0)][:,0, ...].argmax(dim=1)
        reprojection_losses = []

        for idx in range(self.num_input_frames-1):
            pred = outputs["bev_rc"][idx]
            reprojection_losses.append(self.bev_homography_loss(pred, target))

        reprojection_losses = torch.cat(reprojection_losses, 1)

        if self.opt.avg_bev_reprojection:
            reprojection_loss = reprojection_losses.mean(1, keepdim=True)
        else:
            reprojection_loss = reprojection_losses

        if reprojection_loss.shape[1] == 1:
            to_optimise = reprojection_loss
        else:
            to_optimise, _ = torch.min(reprojection_loss, dim=1)

        loss = to_optimise.mean()
        losses = {"homography_loss": loss}

        return losses

    # Metrics Fns
    def compute_depth_metrics(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        if "depth_gt" not in inputs:
            return

        depth_pred = outputs[("depth", 0)][:,0, ...].squeeze(dim=1).detach()

        depth_gt = inputs["depth_gt"][:,0, ...]
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
        if "relpose_gt" not in inputs:
            return

        gt = inputs["relpose_gt"].cpu().detach()

        pred = outputs[("cam_T_cam", 0)].cpu().detach()

        error = torch.linalg.matrix_norm(pred - gt)

        losses["pose_error"] = torch.mean(error)

        gt = gt.numpy()
        pred = pred.numpy()

        for idx, frame_id in enumerate(self.opt.frame_ids[1:]): 
            err_dict = get_pose_diff(gt[:,idx,...], pred[:,idx,...])
            for k,v in err_dict.items():
                losses["err_{}_{}".format(k, frame_id)] = v

    def compute_bev_metrics(self, inputs, outputs, losses):
        """Compute bev metrics, to allow monitoring during training
        """
        if "bev_gt" not in inputs:
            return

        mIOU, mAP = np.array([0., 0., 0]), np.array([0., 0., 0])

        for batch_idx in range(self.opt.batch_size):
            pred = torch.argmax(outputs[("bev", 0)][batch_idx, 0].detach(), \
                dim=0).cpu().numpy()
            gt = inputs["bev_gt"][batch_idx, 0].detach().cpu().numpy()
            mIOU += mean_IU(pred, gt, len(self.opt.bev_ce_weights))
            mAP += mean_precision(pred, gt, len(self.opt.bev_ce_weights))

        mIOU /= self.opt.batch_size
        mAP /= self.opt.batch_size

        losses["bev_error/mIOU"] = dict(unknown=mIOU[0], occupied=mIOU[1], free=mIOU[2])
        losses["bev_error/mAP"] =  dict(unknown=mAP[0], occupied=mAP[1], free=mAP[2])

    # Logging Fns
    def log_time(self, batch_idx, duration, losses):
        """Print a logging statement to the terminal
        """
        total_loss = losses["loss"].cpu().data
        discr_loss = losses["discriminator_loss"].cpu().data if "discriminator_loss" in losses else -1
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} |  discriminator loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, total_loss, discr_loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def parse_log_data(self, val):
        if isinstance(val, dict):
            out = {}
            for k, v in val.items():
                out[k] = self.parse_log_data(v)
            return out
        elif isinstance(val, torch.Tensor):
            return val.cpu().detach().item()
        else:
            return val

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        step_info = {}
        step_info['step'] = self.step

        writer = self.writers[mode]
        for l, v in losses.items():
            if v is None:
                continue
            elif isinstance(v, dict):
                writer.add_scalars("{}".format(l), v, self.step)
            else:
                writer.add_scalar("{}".format(l), v, self.step)
            step_info[l] = self.parse_log_data(v)

        step_logpath = os.path.join(self.log_path, 'step_logs', f'{self.step}.pkl')
        os.makedirs(os.path.dirname(step_logpath), exist_ok=True)
        with open(step_logpath, 'wb') as f:
            pickle.dump(step_info, f)


        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            '''FORMAT: key_{frame_idx}_{*scale}/{batch_idx}'''

            for idx, frame_id in enumerate(self.opt.frame_ids):

                # COLOR data
                for s in [0]:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        invnormalize_imagenet(inputs[("color", 'l', s)][j, idx].data), self.step)
                    if s == 0 and frame_id != 0 and ("color", 0) in outputs:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            invnormalize_imagenet(outputs[("color", s)][idx-1][j].data), self.step)

                # BEV data
                if "bev_gt" in inputs:
                    writer.add_image(
                        "bev_gt_{}/{}".format(frame_id, j), 
                        normalize_image(inputs['bev_gt'][j, idx].unsqueeze(0).data,
                        (0, 2)), self.step)
                
                if ("bev", 0) in outputs:
                    writer.add_image(
                        "bev_pred_{}/{}".format(frame_id, j), 
                        normalize_image(outputs[("bev", 0)][j, idx].argmax(dim=0, 
                        keepdim=True).data, (0, 2)), self.step)

                if "bev_rc" in outputs and idx > 0:
                    writer.add_image(
                        "bev_rc_{}/{}".format(frame_id, j), 
                        normalize_image(outputs['bev_rc'][idx-1][j].argmax(dim=0, 
                        keepdim=True).data, (0,2)), self.step)

                # DEPTH data
                if "depth_gt" in inputs:
                    writer.add_image(
                        "depth_{}/{}".format(frame_id, j),
                        normalize_image(inputs["depth_gt"][j, idx].unsqueeze(0).data, 
                            (self.opt.min_depth, self.opt.max_depth)), self.step)
                
                if ("depth", 0) in outputs:
                    writer.add_image(
                        "depth_{}/{}".format(frame_id, j),
                        normalize_image(outputs[("depth", 0)][j, idx], 
                            (self.opt.min_depth, self.opt.max_depth)), self.step)

                # SEMANTICS
                if "semantics_gt" in inputs:
                    floor_sem = inputs['semantics_gt'][j, frame_id] == 2  # ground floor label is 2
                    writer.add_image(
                        "semantics_{}/{}".format(frame_id, j), 
                        floor_sem.data, self.step)

            # POSE - only for the target frame.
            if "svo_map" in inputs:
                writer.add_image(
                    "svo_{}/{}".format(0, j), 
                    inputs['svo_map'][j].unsqueeze(0).data, self.step)
                writer.add_image(
                    "svo_noise_{}/{}".format(0, j), 
                    inputs['svo_map_noise'][j].unsqueeze(0).data, self.step)

    # Save Model Outputs
    def dump_raw_data(self, inputs, outputs):
        for idx in range(self.opt.batch_size):
            line = inputs['filename'][idx].split()
            outdir = os.path.join(self.log_path, "dump", str(self.epoch),
                        os.path.basename(line[0]), "0")

            # write imgL
            if (("color_aug", 'l', 0) in inputs) and ('color_aug_left' in self.opt.dump_data):
                imgL = inputs[("color_aug", 'l', 0)]
                outpath = os.path.join(outdir, 'left_rgb', '{}.png'.format(line[1]))
                os.makedirs(os.path.dirname(outpath), exist_ok=True)
                img = invnormalize_imagenet(imgL[idx, 0, ...].cpu().detach())
                img = (img.permute(1,2,0).numpy() * 255).astype(np.uint8)
                cv2.imwrite(outpath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # write imgR
            if (("color_aug", 'r', 0) in inputs)  and ('color_aug_right' in self.opt.dump_data):
                imgR = inputs[("color_aug", 'r', 0)]
                outpath = os.path.join(outdir, 'right_rgb', '{}.png'.format(line[1]))
                os.makedirs(os.path.dirname(outpath), exist_ok=True)
                img = invnormalize_imagenet(imgR[idx, 0, ...].cpu().detach())
                img = (img.permute(1,2,0).numpy() * 255).astype(np.uint8)
                cv2.imwrite(outpath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # write depth_gt
            if ("depth_gt" in inputs) and ('gt_depth' in self.opt.dump_data):
                outpath = os.path.join(outdir, 'tgt_depth', '{}.png'.format(line[1]))
                os.makedirs(os.path.dirname(outpath), exist_ok=True)
                img = inputs['depth_gt'][idx, 0].squeeze().cpu().detach().numpy()
                img = (img * 65535/10).astype(np.uint16)
                cv2.imwrite(outpath, img)

            # write depth
            if (("depth", 0) in outputs) and ('pred_depth' in self.opt.dump_data):
                pred_depth = outputs[("depth", 0)]
                outpath = os.path.join(outdir, 'pred_depth', '{}.png'.format(line[1]))
                os.makedirs(os.path.dirname(outpath), exist_ok=True)
                img = pred_depth[idx, 0, ...].squeeze().cpu().detach().numpy()
                img = (img * 65535/10).astype(np.uint16)
                cv2.imwrite(outpath, img)

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

    if cfg.script_mode == 'train':
        trainer.train()
    elif cfg.script_mode in ['val', 'predict']:
        trainer.val()
    else:
        raise NotImplementedError()
