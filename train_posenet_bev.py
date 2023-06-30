# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import pickle

import os
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
import kornia.enhance

seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
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
    os.environ['PYTHONHASHSEED'] = str(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

class Trainer:
    def __init__(self, options):
        self.opt = options
        print("Training mode: {}".format(self.opt.mode))
        assert np.all([(k in self.opt.PIPELINE.run) for k in self.opt.PIPELINE.train])

        assert self.opt.model_name is not None
        os.makedirs(self.opt.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        #self.setup_logs()

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

        if len(self.parameters_to_train) > 0:
            self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
            self.model_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.model_optimizer, mode='min', 
                factor=0.5, patience=10, threshold=1e-2, threshold_mode='rel')
            # self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            #     self.model_optimizer, self.opt.scheduler_step_size, 0.8)


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
        
        # self.bev_th_loss = nn.CrossEntropyLoss(reduction='none', \
        #     weight=torch.Tensor(self.opt.bev_ce_weights).to(self.device),
        #     ignore_index=0)

        self.bev_th_loss = nn.KLDivLoss(reduction='none', log_target=True)

        self.criterion_d = nn.BCEWithLogitsLoss()

        self.save_opts()
        self.epoch = 0
        self.step = 0

    # Setup models and datasets.
    def setup_disparity_model(self):
        if self.opt.DISPARITY.module == 'AnyNet':
            disp_module = networks.AnyNet
        elif self.opt.DISPARITY.module == 'ResnetDecoder':
            disp_module = networks.DepthResnetDecoder
        elif self.opt.DISPARITY.module == 'JointDecoder':
            disp_module = networks.JointDecoder
        elif self.opt.DISPARITY.module == 'mock':
            disp_module = networks.MockDecoder
        else:
            raise NotImplementedError()

        disp_model = disp_module(self.opt.DISPARITY, opt=self.opt)
        self.models["disp_module"] = nn.DataParallel(disp_model)
        self.models["disp_module"].cuda()

        if not self.opt.DISPARITY.freeze_weights:
            disp_params = disp_model.get_params(self.opt.DISPARITY)
            self.parameters_to_train.extend(disp_params)
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
            rgbd_params = {
                "name": "rgbd_encoder",
                "params": list(self.models["rgbd_encoder"].parameters()),
                "lr": getattr(self.opt.RGBD_ENCODER, 'lr', self.opt.learning_rate)
            }
            self.parameters_to_train.append(rgbd_params)
        else:
            for param in self.models["rgbd_encoder"].parameters():
                param.requires_grad = False

    def setup_logs(self):
        while True:
            time.sleep(1)

    def setup_bev_decoder_model(self):
        ## BEV_DECODER
        if self.opt.BEV_DECODER.module == 'mock':
            bev_decoder = networks.MockDecoder
        elif self.opt.BEV_DECODER.module == 'layout_decoder':
            bev_decoder = networks.LayoutDecoder
        elif self.opt.BEV_DECODER.module == 'resnet_layout':
            bev_decoder = networks.LayoutResnetDecoder
        elif self.opt.BEV_DECODER.module == 'occant':
            bev_decoder = networks.occant.OccupancyAnticipator
        else:
            raise NotImplementedError()

        bev_model = bev_decoder(**self.opt.BEV_DECODER, cfg=self.opt)
        self.models["bev_decoder"] = bev_model
        self.models["bev_decoder"].to(self.device)

        if not self.opt.BEV_DECODER.freeze_weights:
            bev_params = bev_model.get_params(self.opt.BEV_DECODER)
            print("#### NUMBER OF PARAMS:", len(bev_params))
            self.parameters_to_train.extend(bev_params)
        else:
            for param in self.models["bev_decoder"].parameters():
                param.requires_grad = False

    def setup_discriminator_model(self):
        self.models["discriminator"] = networks.Discriminator(n_channels=self.opt.BEV_DECODER.n_classes, \
            **self.opt.DISCRIMINATOR)
        self.models["discriminator"].to(self.device)

        if not self.opt.DISCRIMINATOR.freeze_weights:
            self.parameters_to_train_D = list(self.models["discriminator"].parameters())
            self.model_optimizer_D = optim.Adam(self.parameters_to_train_D,
                self.opt.DISCRIMINATOR.lr)
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
                    (self.opt.batch_size * self.num_input_frames,
                    *self.patch))),
            requires_grad=False).float().cuda()

        self.fake = Variable(
            torch.Tensor(
                np.zeros(
                    (self.opt.batch_size * self.num_input_frames,
                    *self.patch))),
            requires_grad=False).float().cuda()

    def setup_pose_model(self):
        pose_params = []
        if self.opt.POSE.pose_model_type == "separate_resnet":
            self.models["pose_encoder"] = networks.ResnetEncoder(
                self.opt.POSE.num_layers,
                self.opt.POSE.weights_init == "pretrained",
                num_input_images=self.num_pose_frames,
                in_channels=self.opt.POSE.in_channels)

            self.models["pose_encoder"].to(self.device)
            encoder_params = {
                'name': 'pose_encoder',
                'params': list(self.models["pose_encoder"].parameters()),
                'lr': getattr(self.opt.POSE, 'encoder_lr', self.opt.learning_rate)
            }
            pose_params.append(encoder_params)

            self.models["pose"] = networks.PoseDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2)

        elif self.opt.POSE.pose_model_type == "shared":
            self.models["pose"] = networks.PoseDecoder(
                self.models["rgbd_encoder"].num_ch_enc, self.num_pose_frames)

        elif self.opt.POSE.pose_model_type == "posecnn":
            self.models["pose"] = networks.PoseCNN(self.num_pose_frames, self.opt.POSE.in_channels)

        else:
            raise NotImplementedError(self.opt.POSE.pose_model_type)
        
        self.models["pose"].to(self.device)
        decoder_params = {
            'name': 'pose_decoder',
            'params': list(self.models["pose"].parameters()),
            'lr': getattr(self.opt.POSE, 'decoder_lr', self.opt.learning_rate)
        }
        pose_params.append(decoder_params)

        if not self.opt.POSE.freeze_weights:
            self.parameters_to_train.extend(pose_params)
        else:
            for param_group in pose_params:
                for param in param_group['params']:
                    param.requires_grad = False

    def setup_datasets(self):
        # data
        datasets_dict = {
            "habitat": datasets.HabitatDataset, 
            "gibson4": datasets.Gibson4Dataset,
            "kitti": datasets.KittiOdometry
        }
        self.dataset = datasets_dict[self.opt.dataset]

        load_keys = set()
        if 'DISPARITY' in self.opt.PIPELINE.train:
            load_keys.add('depth')
            load_keys.add('bev')
            load_keys.add('pose')
            load_keys.add('ego_map_gt')
        if 'RGBD' in self.opt.PIPELINE.train:
            load_keys.add('semantics')
        if 'BEV' in self.opt.PIPELINE.train:
            load_keys.add('bev')
            load_keys.add('ego_map_gt')
            load_keys.add('depth')
            # load_keys.add('semantics')
        if 'DISCR' in self.opt.PIPELINE.train:
            load_keys.add('discr')
        if 'POSE' in self.opt.PIPELINE.train:
            load_keys.add('pose')

        if self.opt.mode == 'debug':
            load_keys.update(['depth', 'pose', 'bev', 'discr'])

        with open(self.opt.train_split_file, 'r') as f:
            train_filenames = f.read().splitlines()
        with open(self.opt.val_split_file, 'r') as f:
            val_filenames = f.read().splitlines()

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        val_dataset = self.dataset(self.opt, filenames=val_filenames, 
            is_train=False, load_keys=load_keys)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, shuffle=True, worker_init_fn=seed_worker,
            num_workers=self.opt.val_workers, pin_memory=True, drop_last=True,
            generator=g)

        # For sparse and debug mode, we need keypoints only for training.
        if self.opt.mode == 'sparse':
            load_keys.add('keypts')

        train_dataset = self.dataset(self.opt, filenames=train_filenames, 
            is_train=True, load_keys=load_keys)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, shuffle=True, worker_init_fn=seed_worker,
            num_workers=self.opt.train_workers, pin_memory=True, drop_last=True,
            generator=g)

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
            self.extract_poses = self.get_gt_poses
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
            self.extract_poses = self.get_gt_poses
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
        # self.val()
        for self.epoch in range(self.opt.num_epochs):
            # with torch.autograd.set_detect_anomaly(True):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                val_loss = self.val()
                if self.epoch > 0:
                    self.model_lr_scheduler.step(val_loss)
                self.save_model()

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()

        metrics = defaultdict(lambda : 0)

        loss_hist = dict([(k, []) for k in self.opt.loss_hist_keys])

        # start_time = time.time()
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
                    if k in loss_hist.keys():
                        loss_hist[k].append(v.cpu().detach().item())

        for k,v in metrics.items():
            if isinstance(v, dict):
                for i in v.keys():
                    metrics[k][i] /= len(self.val_loader)
                metrics[k] = dict(metrics[k])
            else:
                metrics[k] /= len(self.val_loader)

        metrics = dict(metrics)

        self.log("val", inputs, outputs, metrics, loss_hist)
        val_loss = losses["loss"].item()

        del inputs, outputs, losses

        self.set_train()

        return val_loss

    def run_epoch(self):
        """Run a single epoch of training
        """
        if self.epoch > 0 and self.model_lr_scheduler_D is not None:
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

            if discr_backward:
                self.model_optimizer_D.zero_grad()
                losses["discriminator_loss"].backward()
                self.model_optimizer_D.step()

            for pg in self.parameters_to_train:
                nn.utils.clip_grad_norm_(pg['params'], 0.1)

            self.model_optimizer.step()

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
            outputs.update(rgbd_features)

        # BEV
        if 'BEV' in self.opt.PIPELINE.run:
            bev = self.models['bev_decoder'](inputs, outputs)
            outputs.update(bev)

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
            "homography_loss": None,
            "stc_loss": None
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
        if 'POSE' in self.opt.PIPELINE.train and 'DISPARITY' in self.opt.PIPELINE.run:
            self.generate_reprojection_pred(inputs, outputs)
            losses.update(self.compute_reprojection_losses(inputs, outputs))

        # HOMOGRAPHY LOSS
        if 'BEV' in self.opt.PIPELINE.train and 'POSE' in self.opt.PIPELINE.run:
            self.generate_homography_pred(inputs, outputs)
            losses.update(self.compute_homography_losses(inputs, outputs))

        # STC LOSS
        if 'POSE' in self.opt.PIPELINE.train:
            losses.update(self.compute_stereo_temporal_constraint_loss(inputs, outputs))

        # MERGE LOSSES
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        for key, weight in self.opt.loss_weights.items():
            if losses[key] is not None:
                total_loss += weight * losses[key]

        losses["loss"] = total_loss
        return losses

    # Disparity Fns
    def get_gt_disparity_depth(self, inputs, features):
        outputs = {}
        for side in ['l', 'r']:
            depth = Variable(inputs[("depth_gt", side)].clone(), requires_grad=False).to(self.device)
            depth = torch.nan_to_num(depth, self.opt.max_depth, self.opt.max_depth)
            depth = torch.clamp(depth, self.opt.min_depth, self.opt.max_depth)
            depth = depth.reshape((-1, 1, *depth.shape[2:]))
            
            outputs[('depth', side, 0)] = depth
        return outputs

    def generate_disparity_depth(self, inputs, features):
        outputs = {}
        outputs.update(self.models['disp_module'](inputs, features))

        return outputs

    def compute_disparity_losses(self, inputs, outputs):
        losses = {}
        losses.update(self.models['disp_module'].module.compute_loss(inputs, outputs, self.epoch))
        return losses

    # RGBD Fns
    def compute_rgbd_losses(self, inputs, outputs):
        losses = {}
        losses["rgbd_loss"] = Variable(torch.tensor(0).float(), requires_grad=False).to(self.device)
        return losses

    # Pose Fns
    def get_gt_poses(self, inputs, features):
        outputs = {}
        for side in ["l", "r"]:
            outputs[("cam_T_cam", side, 0)] = Variable(inputs[("relpose_gt", side)].clone(), 
                                                requires_grad=True).to(self.device)
        return outputs

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        tgt_side = 'l'
        for side in ['l', 'r']:
            outputs[("axisangle", side, 0)] = []
            outputs[("translation", side, 0)] = []
            outputs[("cam_T_cam", side, 0)] = []

            if self.num_pose_frames == 2:
                # In this setting, we compute the pose to each source frame via a
                # separate forward pass through the pose network.

                # select what features the pose network takes as input
                # if self.opt.POSE.pose_model_type == "separate_resnet":
                #     src_feats = inputs[("color_aug", side, 0)]
                #     tgt_feats = inputs[("color_aug", tgt_side, 0)]
                # else:
                src_feats = features[('rgbd_features', side, 0)][:,:,:self.opt.POSE.in_channels, ...]
                tgt_feats = features[('rgbd_features', tgt_side, 0)][:,:,:self.opt.POSE.in_channels, ...]

                for idx in range(1, len(self.opt.frame_ids)):
                    # To maintain ordering we always pass frames in temporal order
                    f_i = self.opt.frame_ids[idx]
                    if f_i < 0:
                        pose_inputs = [src_feats[:,idx,...], tgt_feats[:,0,...]]
                    else:
                        pose_inputs = [tgt_feats[:,0,...], src_feats[:,idx,...]]

                    if self.opt.POSE.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.POSE.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", side, 0)].append(axisangle)
                    outputs[("translation", side, 0)].append(translation)

                    # Invert the matrix if the frame id is negative
                    T = networks.layers.transformation_from_parameters(axisangle[:, 0], 
                            translation[:, 0], invert=(f_i < 0))

                    outputs[("cam_T_cam", side, 0)].append(T)

            else:
                # Here we input all frames to the pose net (and predict all poses) together
                # if self.opt.POSE.pose_model_type == "separate_resnet":
                #     pose_inputs = inputs[("color_aug", side, 0)]

                #     if self.opt.POSE.pose_model_type == "separate_resnet":
                #         pose_inputs = [self.models["pose_encoder"](pose_inputs)]

                # elif self.opt.pose_model_type in ["shared", "posecnn"]:
                #     pose_inputs = features[('rgbd_features', 0)]

                # axisangle, translation = self.models["pose"](pose_inputs)

                # outputs[("axisangle", 0)] = axisangle
                # outputs[("translation", 0)] = translation

                # for i, f_i in enumerate(self.opt.frame_ids[1:]):
                #     T = networks.layers.transformation_from_parameters(axisangle[:, i], translation[:, i])
                #     outputs[("cam_T_cam", 0)].append(T)
                raise NotImplementedError('All frames transformation to tgt frame not implemented.')

        for k,v in outputs.items():
            outputs[k] = torch.stack(v, dim=1)

        return outputs

    def compute_stereo_temporal_constraint_loss(self, inputs, outputs):
        losses = {}
        loss = 0

        baseline_matrix = torch.eye(4, requires_grad=False).to(self.device)
        baseline_matrix[0, 3] = self.opt.baseline
        baseline_matrix = baseline_matrix.unsqueeze(0).repeat(self.opt.batch_size, 1, 1)

        for idx in range(self.num_input_frames - 1):
            lposes = outputs[("cam_T_cam", 'l', 0)][:, idx, ...]
            rposes = outputs[("cam_T_cam", 'r', 0)][:, idx, ...]

            rlposes = lposes @ torch.linalg.inv(rposes)

            error = rlposes - baseline_matrix
            loss += torch.linalg.norm(error)

        losses['stc_loss'] = loss/ ((self.num_input_frames - 1) * self.opt.batch_size)
        return losses

    # Dense Fns
    def generate_dense_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            tgt_side = 'l'
            for side in ['l', 'r']:
                outputs[("sample", side, scale)] = []
                outputs[("color", side, scale)] = []

                for idx in range(self.num_input_frames-1):
                    T = outputs[("cam_T_cam", side, 0)][:, idx, ...]

                    depth = outputs[('depth', tgt_side, scale)][:,0, ...]

                    cam_points = self.backproject_depth[0](
                        depth, inputs[("inv_K", 0)])
                    pix_coords = self.project_3d[0](
                        cam_points, inputs[("K", 0)], T)

                    outputs[("sample", side, scale)].append(pix_coords)

                    pred_color = F.grid_sample(
                        inputs[("color", side, scale)][:, idx+1, ...],
                        outputs[("sample", side, scale)][idx],
                        padding_mode="border")
                    outputs[("color", side, scale)].append(pred_color)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", side, scale)] = inputs[("color", side, scale)]

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
            tgt_side = 'l'
            for side in ['l', 'r']:
                loss = 0
                reprojection_losses = []

                depth = outputs[("depth", side, scale)][:,0, ...]
                color = inputs[("color", side, scale)][:,0, ...]
                target = inputs[("color", tgt_side, 0)][:,0, ...]

                for idx in range(self.num_input_frames-1):
                    pred = outputs[("color", side, scale)][idx]
                    reprojection_losses.append(self.dense_reprojection_loss(pred, target))

                reprojection_losses = torch.cat(reprojection_losses, 1)

                if not self.opt.disable_automasking:
                    identity_reprojection_losses = []
                    for idx in range(self.num_input_frames-1):
                        pred = inputs[("color", side, 0)][:,idx+1, ...]
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
                losses["reprojection_loss_{}/{}".format(side, scale)] = loss

        total_loss /= (2 * self.num_scales)
        losses["reprojection_loss"] = total_loss
        return losses

    # Sparse Fns
    def generate_sparse_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
     
        for scale in self.opt.scales:
            tgt_side = 'l'
            for side in ['l', 'r']:
                depth = outputs[('depth', tgt_side, 0)][:, 0, ...]

                # sample depth for dso points                                                
                dso_points = inputs[('dso_points', tgt_side)]
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

                outputs[("dso_mask", side, scale)] = []
                outputs[("dso_color", side, scale)] = []

                for idx in range(self.num_input_frames):
                    if idx == 0:
                        T = torch.eye(4).view(1, 4, 4).expand(self.opt.batch_size, 4, 4).cuda()
                    else:
                        T = outputs[("cam_T_cam", side, 0)][:,idx-1,...]

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
                    outputs[("dso_mask", side, scale)].append(valid.unsqueeze(1).float())

                    # sample patch from color images
                    pred_color = F.grid_sample(
                        inputs[("color", side, 0)][:, idx, ...],
                        pix_coords,
                        padding_mode="border")
                    outputs[("dso_color", side, scale)].append(pred_color)

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
            tgt_side = 'l'
            for side in ['l', 'r']:
                loss = 0
                sparse_reprojection_losses = []

                depth = outputs[('depth', side, scale)][:, 0, ...]
                color = inputs[("color", side, scale)][:, 0, ...]
                dso_target = outputs[("dso_color", tgt_side, scale)][0]

                # dso loss
                for idx in range(1, self.num_input_frames):
                    dso_pred = outputs[("dso_color", side, scale)][idx]
                    sparse_reprojection_losses.append(self.sparse_reprojection_loss(dso_pred, dso_target))
                    
                if self.num_input_frames == 5:
                    dso_combined_1 = torch.cat((sparse_reprojection_losses[1], sparse_reprojection_losses[2]), dim=1)
                    dso_combined_2 = torch.cat((sparse_reprojection_losses[0], sparse_reprojection_losses[3]), dim=1)

                    dso_to_optimise_1, _ = torch.min(dso_combined_1, dim=1)
                    dso_to_optimise_2, _ = torch.min(dso_combined_2, dim=1)
                    dso_loss_1 = dso_to_optimise_1.mean() 
                    dso_loss_2 = dso_to_optimise_2.mean()

                    loss += dso_loss_1 + dso_loss_2

                    losses["dso_loss_1/{}".format(scale)] = dso_loss_1
                    losses["dso_loss_2/{}".format(scale)] = dso_loss_2
                else:
                    dso_combined_1 = torch.cat(sparse_reprojection_losses, dim=1)
                    dso_to_optimise_1, _ = torch.min(dso_combined_1, dim=1)
                    dso_loss_1 = dso_to_optimise_1.mean()
                    loss += dso_loss_1 

                    losses["dso_loss_1/{}".format(scale)] = dso_loss_1

                # TODO: Fix this for finetuning depth prediction
                # smooth_loss = networks.layers.get_smooth_loss(depth, color)
                # loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale) * (not self.opt.DISPARITY.freeze_weights)
                # losses["smooth_loss/{}".format(scale)] = smooth_loss

                total_loss += loss

                losses["reprojection_loss_{}/{}".format(side, scale)] = loss

        total_loss /= 2 * self.num_scales
        losses["reprojection_loss"] = total_loss
        return losses

    # BEV Fns
    def generate_gt_bev(self, inputs, outputs, side, scale):
        gt_bevs = []
        for idx in range(self.num_input_frames):
            depth = outputs[("depth", side, scale)][:, idx, ...]
            depth = depth.reshape((self.opt.batch_size, *depth.shape[2:]))

            semantics = inputs[("semantics_gt", side)][:, idx, ...]
            semantics = semantics.reshape((self.opt.batch_size, *semantics.shape[2:]))

            cam_points = self.backproject_depth[0](depth, inputs[("inv_K", 0)])
            bev = self.homography_warp.project(cam_points, semantics)

            bev = bev.reshape((self.opt.batch_size, 1, *bev.shape[1:]))
            gt_bevs.append(bev)

        gt_bevs = torch.cat(gt_bevs, dim=1)
        return gt_bevs

    def compute_bev_losses(self, inputs, outputs):
        losses = {}
        losses.update(self.models['bev_decoder'].compute_loss(inputs, outputs, self.epoch))
        return losses

    def compute_discriminator_losses(self, inputs, outputs):
        loss = {}
        side = 'l'

        if self.model_optimizer_D is None:
            return loss

        pred = outputs[("bev", side, 0)]
        pred = pred.reshape((-1, *pred.shape[2:]))
        pred_sm = F.softmax(pred, dim=1)

        target = inputs["discr"]
        target = target.reshape((-1, *target.shape[2:]))
        target = F.one_hot(target, num_classes=self.opt.BEV_DECODER.n_classes)
        target = Variable(target.permute(0,3,1,2), requires_grad=False).float().to(self.device)

        fake_pred = self.models["discriminator"](pred_sm)
        real_pred = self.models["discriminator"](target)
 
        # Train Discriminator
        if self.epoch >= self.opt.DISCRIMINATOR.discr_train_epoch:
            loss["discriminator_loss"] = self.criterion_d(fake_pred, self.fake) + \
                self.criterion_d(real_pred, self.valid)
            loss["gan_loss"] = self.criterion_d(fake_pred, self.valid)

        return loss

    def generate_homography_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        side = 'l'
        outputs[("bev_sample", side)] = []
        outputs[("bev_mask", side)] = []
        outputs[("bev_rc", side)] = []

        for idx in range(self.num_input_frames-1):
            T = outputs[("cam_T_cam", side, 0)][:, idx, ...]
            pix_coords = self.homography_warp(T)

            outputs[("bev_sample", side)].append(pix_coords)
            
            # save mask
            valid = (pix_coords[..., 0] > -1.) & (pix_coords[..., 0] < 1.) & (pix_coords[..., 1] > -1.) & (
                        pix_coords[..., 1] < 1.)
            outputs[("bev_mask", side)].append(valid.unsqueeze(1).float())

            pred_bev = F.grid_sample(
                outputs[("bev", side, 0)][:, idx+1, ...],
                outputs[("bev_sample", side)][idx],
                mode='bilinear',
                padding_mode="border")
            outputs[("bev_rc", side)].append(pred_bev)

    def bev_homography_loss(self, pred, target, mask):
        """Computes reprojection loss between a batch of predicted and target points
        """
        pred = F.logsigmoid(pred)
        target = F.logsigmoid(target)
        reprojection_loss = self.bev_th_loss(pred, target)
        reprojection_loss = reprojection_loss.unsqueeze(dim=1) * mask
        return reprojection_loss

    def compute_homography_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        if self.epoch < self.opt.BEV_DECODER.homography_train_epoch:
            return {}

        side = 'l'
        target = outputs[("bev", side, 0)][:,0, ...]
        reprojection_losses = []

        for idx in range(self.num_input_frames-1):
            pred = outputs[("bev_rc", side)][idx]
            mask = outputs[("bev_mask", side)][idx]
            reprojection_losses.append(self.bev_homography_loss(pred, target, mask))

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
        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = 0

        for side in ['l', 'r']:
            if ("depth_gt", side) not in inputs:
                return

            if ("depth", side, 0) not in outputs:
                continue

            depth_pred = outputs[("depth", side, 0)][:,0, ...].squeeze(dim=1).detach()

            depth_gt = inputs[("depth_gt", side)][:,0, ...]
            mask = (depth_gt > self.opt.min_depth) * (depth_gt < self.opt.max_depth)

            depth_gt = depth_gt[mask]
            depth_pred = depth_pred[mask]

            depth_errors = networks.layers.compute_depth_errors(depth_gt, depth_pred)

            for i, metric in enumerate(self.depth_metric_names):
                losses[metric] += np.array(depth_errors[i].cpu()) * 0.5

    def compute_pose_metrics(self, inputs, outputs, losses):
        for side in ['l', 'r']:
            if ("relpose_gt", side) not in inputs:
                continue

            gt = inputs[("relpose_gt", side)].cpu().detach()

            pred = outputs[("cam_T_cam", side, 0)].cpu().detach()

            error = torch.linalg.matrix_norm(pred - gt)

            losses["pose_error/{}".format(side)] = torch.mean(error)

            gt = gt.numpy()
            pred = pred.numpy()

            for idx, frame_id in enumerate(self.opt.frame_ids[1:]): 
                err_dict = get_pose_diff(gt[:,idx,...], pred[:,idx,...])
                for k,v in err_dict.items():
                    losses["err_{}_{}/{}".format(k, frame_id, side)] = v
                
                #log absolute pose for debugging only
                for batch_idx in range(min(4, self.opt.batch_size)):
                    pred_pose = np.expand_dims(pred[batch_idx, idx, ...], axis=0)
                    identity_pose = np.expand_dims(np.eye(4), axis=0)
                    diff_dict = get_pose_diff(identity_pose, pred_pose)
                    losses[f"pos_{frame_id}_{side}/{batch_idx}"] = diff_dict['pos']
                    losses[f"rot_{frame_id}_{side}/{batch_idx}"] = diff_dict['rot']

    def compute_bev_metrics(self, inputs, outputs, losses):
        """Compute bev metrics, to allow monitoring during training
        """
        side = 'l'
        if ("bev_gt", side) not in inputs:
            return

        mIOU_cls, mAP_cls = np.array([0., 0., 0]), np.array([0., 0., 0])
        mIOU_ch, mAP_ch = np.zeros((2, 2)), np.zeros((2, 2))

        for batch_idx in range(self.opt.batch_size):
            pred = F.sigmoid(outputs[("bev", side, 0)][:, 0])[batch_idx].detach().cpu().numpy()
            gt = inputs[("bev_gt", side)][batch_idx, 0].detach().cpu().numpy()

            pred_map = np.zeros((pred.shape[1], pred.shape[2])) # unknown
            pred_map[np.logical_and(pred[1] >= 0.5, pred[0] >= 0.5)] = 1  # known and occupied
            pred_map[np.logical_and(pred[1] >= 0.5, pred[0] < 0.5)] = 2  # known and free

            mIOU_cls += mean_IU(pred_map, gt, len(self.opt.bev_ce_weights))
            mAP_cls += mean_precision(pred_map, gt, len(self.opt.bev_ce_weights))

            pred_map = np.zeros((2, pred.shape[1], pred.shape[2])) # set values to free and unknown for the two channels
            pred_map[0, pred[0] >= 0.5] = 1  # occupied
            pred_map[1, pred[1] >= 0.5] = 1  # known

            gt_ch = np.zeros((2, pred.shape[1], pred.shape[2]))
            gt_ch[0, gt==1] = 1
            gt_ch[1, np.logical_or(gt==1, gt==2)] = 1

            # [Free, Occupied]
            mIOU_ch[0] += mean_IU(pred_map[0], gt_ch[0], 2)
            mAP_ch[0] += mean_precision(pred_map[0], gt_ch[0], 2)
            # [Unexplored, Explored]
            mIOU_ch[1] += mean_IU(pred_map[1], gt_ch[1], 2)
            mAP_ch[1] += mean_precision(pred_map[1], gt_ch[1], 2)

        mIOU_cls /= self.opt.batch_size
        mAP_cls /= self.opt.batch_size
        mIOU_ch /= self.opt.batch_size
        mAP_ch /= self.opt.batch_size

        losses["bev_error/mIOU_cls"] = dict(unknown=mIOU_cls[0], occupied=mIOU_cls[1], free=mIOU_cls[2])
        losses["bev_error/mAP_cls"] =  dict(unknown=mAP_cls[0], occupied=mAP_cls[1], free=mAP_cls[2])

        losses["bev_error/mIOU_ch"] = dict(free=mIOU_ch[0,0], occupied=mIOU_ch[0,1], unexplored=mIOU_ch[1,0], explored=mIOU_ch[1,1])
        losses["bev_error/mAP_ch"] =  dict(free=mAP_ch[0,0], occupied=mAP_ch[0,1], unexplored=mAP_ch[1,0], explored=mAP_ch[1,1])

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

    def log(self, mode, inputs, outputs, losses, loss_hist=None):
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

        if loss_hist is not None:
            for k, v in loss_hist.items():
                step_info['hist_' + k] = v
                writer.add_histogram('hist_' + k, np.array(v), self.step)

        writer.add_scalar("epoch", self.epoch, self.step)
        step_info["epoch"] = self.epoch

        if len(self.parameters_to_train) > 0:
            lr_dict = dict([(pg['name'], pg['lr']) for pg in self.model_optimizer.param_groups])
            writer.add_scalars("{}".format('learning_rate'), lr_dict, self.step)
            step_info['learning_rate'] = lr_dict

        step_logpath = os.path.join(self.log_path, mode, 'step_logs', f'{self.step}.pkl')
        os.makedirs(os.path.dirname(step_logpath), exist_ok=True)
        with open(step_logpath, 'wb') as f:
            pickle.dump(step_info, f)


        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            '''FORMAT: key_{frame_idx}_{side}/{batch_idx}'''

            for idx, frame_id in enumerate(self.opt.frame_ids):

                for side in ['l']:

                    # COLOR data
                    writer.add_image(
                        "color_gt_{}_{}/{}".format(frame_id, side, j),
                        invnormalize_imagenet(inputs[("color", side, 0)][j, idx].data), self.step)
                    if idx > 0 and ("color", side, 0) in outputs:
                        writer.add_image(
                            "color_rc_{}_{}/{}".format(frame_id, side, j),
                            invnormalize_imagenet(outputs[("color", side, 0)][idx-1][j].data), self.step)

                    # BEV data
                    if ("bev_gt", side) in inputs:
                        writer.add_image(
                            "bev_gt_{}_{}/{}".format(frame_id, side, j), 
                            normalize_image(inputs[("bev_gt", side)][j, idx].unsqueeze(0).data,
                            (0, 2)), self.step)
                
                    if ("bev", side, 0) in outputs:
                        pred = F.sigmoid(outputs[("bev", side, 0)][:, idx])[j].detach().cpu()
                        pred_map = torch.zeros((pred.shape[1], pred.shape[2]))
                        pred_map[(pred[1] >= 0.5) & (pred[0] >= 0.5)] = 1  # known and occupied
                        pred_map[(pred[1] >= 0.5) & (pred[0] < 0.5)] = 2  # known and free
                        # pred_map = pred[1]
                        writer.add_image(
                            "bev_pred_{}_{}/{}".format(frame_id, side, j), 
                            normalize_image(pred_map.unsqueeze(0).data, (0, 2)), self.step)

                    if ("bev_rc", side, 0) in outputs and idx > 0:
                        pred = F.sigmoid(outputs[("bev_rc", side, 0)][idx-1])[j].detach().cpu()
                        pred_map = torch.zeros((pred.shape[1], pred.shape[2]))
                        pred_map[(pred[1] >= 0.5) & (pred[0] >= 0.5)] = 1  # known and occupied
                        pred_map[(pred[1] >= 0.5) & (pred[0] < 0.5)] = 2  # known and free
                        writer.add_image(
                            "bev_rc_{}_{}/{}".format(frame_id, side, j), 
                            normalize_image(pred_map.unsqueeze(0).data, (0,2)), self.step)

                    # Dump ego_map_gt
                    if ("ego_map_gt", side) in inputs:
                        pred = inputs[("ego_map_gt", side)][j, idx].detach().cpu()
                        pred_map = torch.zeros((pred.shape[1], pred.shape[2]))
                        pred_map[(pred[1] >= 0.5) & (pred[0] >= 0.5)] = 1  # known and occupied
                        pred_map[(pred[1] >= 0.5) & (pred[0] < 0.5)] = 2  # known and free
                        writer.add_image(
                            "ego_map_gt_{}_{}/{}".format(frame_id, side, j), 
                            normalize_image(pred_map.unsqueeze(0).data, (0, 2)), self.step)

                    # DEPTH data
                    if ("depth_gt", side) in inputs:
                        depth_out = normalize_image(inputs[("depth_gt", side)][j, idx].unsqueeze(0).data, (self.opt.min_depth, self.opt.max_depth))
                        writer.add_image("depth_gt_{}_{}/{}".format(frame_id, side, j), depth_out, self.step)
                    
                    if ("depth", side, 0) in outputs:
                        writer.add_image(
                        "depth_pred_{}_{}/{}".format(frame_id, side, j),
                        normalize_image(outputs[("depth", side, 0)][j, idx], 
                            (self.opt.min_depth, self.opt.max_depth)), self.step)

                    # SEMANTICS
                    if ("semantics_pred", side, 0) in outputs:
                        sem_pred = outputs[('semantics_pred', side, 0)][j, frame_id].detach().cpu()
                        writer.add_image(
                            "semantics_pred_{}_{}/{}".format(frame_id, side, j), 
                            sem_pred.data, self.step)

                    if ("semantics_gt", side) in inputs:
                        sem_gt = inputs[('semantics_gt', side)][j, frame_id] # ground floor label is 2, equate to 2 to map only that
                        writer.add_image(
                            "semantics_gt_{}_{}/{}".format(frame_id, side, j), 
                            sem_gt.data, self.step)

                    if ('combined_db_mask', side, 0) in outputs:
                        loss_mask = outputs[('combined_db_mask', 'l', 0)]
                        if loss_mask.shape[0] < j * idx:
                            continue
                        mask = loss_mask[idx*j].unsqueeze(0) * 1.0

                        src_img  = invnormalize_imagenet(inputs[("color", side, 0)][j, idx])
                        mask = mask.unsqueeze(0).repeat([1,3,1,1]) * src_img # B x 3 x H x W
                        mask_overlay = kornia.enhance.add_weighted(src_img, 0.35, mask, 0.65, 0.0)
                        writer.add_image(
                            "combined_loss_mask_{}_{}/{}".format(frame_id, side, j), 
                            mask_overlay.squeeze(0).data, self.step)

            # POSE - only for the target frame.
            # if "svo_map" in inputs:
            #     writer.add_image(
            #         "svo_{}/{}".format(0, j), 
            #         inputs['svo_map'][j].unsqueeze(0).data, self.step)
            #     writer.add_image(
            #         "svo_noise_{}/{}".format(0, j), 
            #         inputs['svo_map_noise'][j].unsqueeze(0).data, self.step)

    # Save Model Outputs
    def dump_raw_data(self, inputs, outputs):
        for idx in range(self.opt.batch_size):
            scene, camera_pose, fileidx = inputs['filename'][idx].split()
            outdir = os.path.join(self.log_path, "dump", str(self.epoch), os.path.basename(scene), camera_pose)

            side = 'l'
                
            # write pred pose
            if ("cam_T_cam", side, 0) in outputs and ('pred_pose' in self.opt.dump_data):
                pred = outputs[("cam_T_cam", side, 0)][idx, 0, ...].cpu().detach().numpy()
                pose_dir = 'pose'
                outpath = os.path.join(outdir, pose_dir, '{}.npy'.format(fileidx))
                os.makedirs(os.path.dirname(outpath), exist_ok=True)
                np.save(outpath, pred)

            # write img
            if (("color_aug", side, 0) in inputs) and ('color_aug' in self.opt.dump_data):
                img = inputs[("color_aug", side, 0)]
                color_dir = 'RGB'
                outpath = os.path.join(outdir, color_dir, '{}.png'.format(fileidx))
                os.makedirs(os.path.dirname(outpath), exist_ok=True)
                img = invnormalize_imagenet(img[idx, 0, ...].cpu().detach())
                img = (img.permute(1,2,0).numpy() * 255).astype(np.uint8)
                cv2.imwrite(outpath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # write depth_gt
            if (("depth_gt", side) in inputs) and ('gt_depth' in self.opt.dump_data):
                depth_dir = 'DEPTH_GT'
                outpath = os.path.join(outdir, depth_dir, '{}.png'.format(fileidx))
                os.makedirs(os.path.dirname(outpath), exist_ok=True)
                img = inputs[("depth_gt", side)][idx, 0].squeeze().cpu().detach().numpy()
                img = (img * 65535/10).astype(np.uint16)
                cv2.imwrite(outpath, img)

            # write depth
            if (("depth", side, 0) in outputs) and ('pred_depth' in self.opt.dump_data):
                pred_depth = outputs[("depth", side, 0)]
                depth_dir = 'DEPTH'
                outpath = os.path.join(outdir, depth_dir, '{}.png'.format(fileidx))
                os.makedirs(os.path.dirname(outpath), exist_ok=True)
                img = pred_depth[idx, 0, ...].squeeze().cpu().detach().numpy()
                img = (img * 65535/10).astype(np.uint16)
                cv2.imwrite(outpath, img)

            # write bev_gt
            if (("bev_gt",side) in inputs) and ('bev_gt' in self.opt.dump_data):
                bev_dir = 'bev'
                outpath = os.path.join(outdir, bev_dir, '{}.png'.format(fileidx))
                os.makedirs(os.path.dirname(outpath), exist_ok=True)
                img = inputs[("bev_gt",side)][idx, 0].squeeze().cpu().detach().numpy()
                img = (img * 127).astype(np.uint8)
                cv2.imwrite(outpath, img)

            # write bev_pred (full occupancy)
            if (("bev", side, 0) in outputs) and ('pred_bev' in self.opt.dump_data):
                pred_bev = outputs[("bev", side, 0)]
                bev_dir = 'bev_pred'
                outpath = os.path.join(outdir, bev_dir, '{}.png'.format(fileidx))
                os.makedirs(os.path.dirname(outpath), exist_ok=True)
                pred = F.sigmoid(pred_bev[:, 0])[idx].detach().cpu()
                img = torch.zeros((pred.shape[1], pred.shape[2]))
                img[(pred[1] >= 0.5) & (pred[0] >= 0.5)] = 1  # known and occupied
                img[(pred[1] >= 0.5) & (pred[0] < 0.5)] = 2  # known and free
                img = img.cpu().detach().numpy()
                img = (img * 127).astype(np.uint8)
                cv2.imwrite(outpath, img)

                            
            # write gt_visible_occupancy (ANS target for OccAnt models)
            if (("ego_map_gt", 'l') in inputs) and ('gt_visible_occupancy' in self.opt.dump_data) and side == 'l':
                bev_dir = 'gt_visible_occupancy'
                outpath = os.path.join(outdir, bev_dir, '{}.png'.format(fileidx))
                os.makedirs(os.path.dirname(outpath), exist_ok=True)
                pred = inputs[("ego_map_gt", side)][idx, 0].squeeze().cpu().detach()
                img = torch.zeros((pred.shape[1], pred.shape[2]))
                img[(pred[1] >= 0.5) & (pred[0] >= 0.5)] = 1  # known and occupied
                img[(pred[1] >= 0.5) & (pred[0] < 0.5)] = 2  # known and free
                img = img.cpu().detach().numpy()
                img = (img * 127).astype(np.uint8)
                cv2.imwrite(outpath, img)

            
            # write bev_pred (ANS output for OccAnt models)
            if ("depth_proj_estimate" in outputs) and ('pred_visible_occupancy' in self.opt.dump_data) and side == 'l':
                pred_bev = outputs["depth_proj_estimate"]
                bev_dir = 'pred_visible_occupancy'
                outpath = os.path.join(outdir, bev_dir, '{}.png'.format(fileidx))
                os.makedirs(os.path.dirname(outpath), exist_ok=True)
                pred = F.sigmoid(pred_bev[:, 0])[idx].detach().cpu()
                img = torch.zeros((pred.shape[1], pred.shape[2]))
                img[(pred[1] >= 0.5) & (pred[0] >= 0.5)] = 1  # known and occupied
                img[(pred[1] >= 0.5) & (pred[0] < 0.5)] = 2  # known and free
                img = img.cpu().detach().numpy()
                img = (img * 127).astype(np.uint8)
                cv2.imwrite(outpath, img)

            # write grad_cam images
            if (("rgb_cam", "unknown", 0) in outputs) and ('grad_cam' in self.opt.dump_data):
                for sem_class in ["unknown", "occupied", "free"]:
                    bev_dir = 'grad_rgb'
                    os.makedirs(os.path.join(outdir, bev_dir), exist_ok=True)
                    img = outputs[("rgb_cam", sem_class, 0)][idx, 0].transpose(1,2,0)
                    outpath = os.path.join(outdir, bev_dir, '{}_{}.png'.format(sem_class, fileidx))
                    cv2.imwrite(outpath, img)

                    bev_dir = 'grad_grayscale'
                    os.makedirs(os.path.join(outdir, bev_dir), exist_ok=True)
                    img = outputs[("grayscale_cam", sem_class, 0)][idx, 0].transpose(1,2,0)
                    outpath = os.path.join(outdir, bev_dir, '{}_{}.png'.format(sem_class, fileidx))
                    img = (img.squeeze() * 255).astype(np.uint8)
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
        if self.opt.load_optimizer and os.path.isfile(optimizer_load_path):
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
