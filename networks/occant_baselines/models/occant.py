#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as tmodels
import numpy as np

from einops import rearrange

from networks.occant_baselines.models.unet import (
    FPNDecoder,
    MergeMultimodalLSTM,
    UNetDecoderLSTM,
    UNetEncoder,
    UNetDecoder,
    MiniUNetEncoder,
    LearnedRGBProjection,
    MergeMultimodal,
    MergeMultiView,
    ResNetRGBEncoder,
)

from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from utils import invnormalize_imagenet


def softmax_2d(x):
    b, h, w = x.shape
    x_out = F.softmax(rearrange(x, "b h w -> b (h w)"), dim=1)
    x_out = rearrange(x_out, "b (h w) -> b h w", h=h)
    return x_out

norm_layer = lambda x : nn.BatchNorm2d(x, affine=True, track_running_stats=False)


# ================================ Anticipation base ==================================


class BaseModel(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.config = cfg

        if cfg.GP_ANTICIPATION.OUTPUT_NORMALIZATION.channel_0 == "sigmoid":
            self.normalize_channel_0 = torch.sigmoid
        elif cfg.GP_ANTICIPATION.OUTPUT_NORMALIZATION.channel_0 == "softmax":
            self.normalize_channel_0 = softmax_2d
        elif cfg.GP_ANTICIPATION.OUTPUT_NORMALIZATION.channel_0 == "identity":
            self.normalize_channel_0 = torch.nn.Identity()

        if cfg.GP_ANTICIPATION.OUTPUT_NORMALIZATION.channel_1 == "sigmoid":
            self.normalize_channel_1 = torch.sigmoid
        elif cfg.GP_ANTICIPATION.OUTPUT_NORMALIZATION.channel_1 == "softmax":
            self.normalize_channel_1 = softmax_2d
        elif cfg.GP_ANTICIPATION.OUTPUT_NORMALIZATION.channel_1 == "identity":
            self.normalize_channel_1 = torch.nn.Identity()

        self._create_gp_models()

    def forward(self, x):
        final_outputs = {}
        gp_outputs = self._do_gp_anticipation(x)
        final_outputs.update(gp_outputs)

        return final_outputs

    def _create_gp_models(self):
        raise NotImplementedError

    def _do_gp_anticipation(self, x):
        raise NotImplementedError

    def _normalize_decoder_output(self, x_dec):
        out = []
        for ch in range(x_dec.shape[1]):
            out.append(self.normalize_channel_0(x_dec[:, ch, ...]))
        return torch.stack(out, dim=1)


# ============================= Anticipation models ===================================


class ANSRGB(BaseModel):
    """
    Predicts depth-projection from RGB only.
    """

    def _create_gp_models(self):
        gp_cfg = self.config.GP_ANTICIPATION
        resnet = tmodels.resnet18(pretrained=True)
        if self.config.GP_ANTICIPATION.freeze_ans_resnet == True:
            for p in resnet.parameters():
                p.requires_grad = False

        self.main = nn.Sequential(  # (3, 128, 128)
            # Feature extraction
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,  # (512, 4, 4)
            # FC layers equivalent
            nn.Conv2d(512, 512, 1),  # (512, 4, 4)
            norm_layer(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1),  # (512, 4, 4)
            norm_layer(512),
            nn.ReLU(),
            # Upsampling
            nn.Conv2d(512, 256, 3, padding=1),  # (256, 4, 4)
            norm_layer(256),
            nn.ReLU(),
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  # (256, 8, 8)
            nn.Conv2d(256, 128, 3, padding=1),  # (128, 8, 8)
            norm_layer(128),
            nn.ReLU(),
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  # (128, 16, 16),
            nn.Conv2d(128, 64, 3, padding=1),  # (64, 16, 16)
            norm_layer(64),
            nn.ReLU(),
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  # (64, 32, 32),
            nn.Conv2d(64, 32, 3, padding=1),  # (32, 32, 32)
            norm_layer(32),
            nn.ReLU(),
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  # (32, 64, 64),
            nn.Conv2d(32, gp_cfg.nclasses, 3, padding=1),  # (2, 64, 64)
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  # (2, 128, 128),
        )

        self.bottleneck = [self.main._modules['16']]

    def _do_gp_anticipation(self, x):
        x_dec = self.main(x["rgb"])
        x_dec = self._normalize_decoder_output(x_dec)
        outputs = {"occ_estimate": x_dec}

        return outputs


class ANSDepth(BaseModel):
    """
    Computes depth-projection from depth and camera parameters only.
    Outputs the GT projected occupancy
    """

    def _create_gp_models(self):
        pass

    def _do_gp_anticipation(self, x):
        x_dec = x["ego_map_gt"]
        outputs = {"occ_estimate": x_dec}

        return outputs


class OccAntRGB(BaseModel):
    """
    Anticipated using rgb only.
    """

    def _create_gp_models(self):
        nmodes = 2
        gp_cfg = self.config.GP_ANTICIPATION

        # Compute constants
        resnet_type = (
            gp_cfg.resnet_type if hasattr(gp_cfg, "resnet_type") else "resnet50"
        )
        infeats = 768 if resnet_type == "resnet50" else 192
        nsf = gp_cfg.unet_nsf
        unet_feat_size = nsf * 8

        # RGB encoder branch
        self.gp_rgb_encoder = ResNetRGBEncoder(resnet_type)
        self.gp_rgb_projector = LearnedRGBProjection(mtype="upsample", infeats=infeats)
        self.gp_rgb_unet = MiniUNetEncoder(infeats, unet_feat_size)

        # Depth projection estimator
        config = self.config.clone()
        self.gp_depth_proj_estimator = ANSRGB(config)

        # Depth encoder branch
        self.gp_depth_proj_encoder = UNetEncoder(gp_cfg.nclasses, nsf=nsf)

        # Merge modules
        self.gp_merge_x5 = MergeMultimodal(unet_feat_size, nmodes=nmodes)
        self.gp_merge_x4 = MergeMultimodal(unet_feat_size, nmodes=nmodes)
        self.gp_merge_x3 = MergeMultimodal(unet_feat_size // 2, nmodes=nmodes)

        self.bottleneck = [self.gp_rgb_projector] #[self.gp_merge_x5, self.gp_merge_x4, self.gp_merge_x3]

        # Decoder module
        self.gp_decoder = UNetDecoder(gp_cfg.nclasses, nsf=nsf)

        self._detach_depth_proj = gp_cfg.detach_depth_proj

        # Load pretrained model if available
        if gp_cfg.pretrained_depth_proj_model != "":
            self._load_pretrained_model(gp_cfg.pretrained_depth_proj_model)

        if gp_cfg.freeze_features:
            for p in self.gp_rgb_encoder.parameters():
                p.requires_grad = False
        if gp_cfg.freeze_depth_proj_model:
            for p in self.gp_depth_proj_estimator.parameters():
                p.requires_grad = False

    def _do_gp_anticipation(self, x):
        """
        Inputs:
            x is a dictionary containing the following keys:
                'rgb' - (bs, 3, H, W) RGB input
                'depth' - (bs, 3, H, W) Depth input - channels are repeated
                'ego_map_gt' - (bs, 2, H, W) probabilities
        """
        x_rgb = self.gp_rgb_encoder(x["rgb"])  # (bs, 768, H/8, W/8)
        x_gp = self.gp_rgb_projector(x_rgb)  # (bs, 768, H/4, W/4)

        x_rgb_enc = self.gp_rgb_unet(x_gp)  # {'x3p', 'x4p', 'x5p'}
        # Estimate projected occupancy
        x_depth_proj = self.gp_depth_proj_estimator(x)["occ_estimate"]  # (bs, nclasses, V, V)
        if self._detach_depth_proj:
            x_depth_proj_enc = self.gp_depth_proj_encoder(
                x_depth_proj.detach()
            )  # {'x1', 'x2', 'x3', 'x4', 'x5'}
        else:
            x_depth_proj_enc = self.gp_depth_proj_encoder(
                x_depth_proj
            )  # {'x1', 'x2', 'x3', 'x4', 'x5'}

        # Replace x_depth_proj_enc with merged features
        x5_inputs = [x_rgb_enc["x5p"], x_depth_proj_enc["x5"]]
        x4_inputs = [x_rgb_enc["x4p"], x_depth_proj_enc["x4"]]
        x3_inputs = [x_rgb_enc["x3p"], x_depth_proj_enc["x3"]]

        x5_enc = self.gp_merge_x5(*x5_inputs)  # (unet_feat_size  , H/16, H/16)
        x4_enc = self.gp_merge_x4(*x4_inputs)  # (unet_feat_size  , H/8 , H/8 )
        x3_enc = self.gp_merge_x3(*x3_inputs)  # (unet_feat_size/2, H/4 , H/4 )
        x_depth_proj_enc["x5"] = x5_enc
        x_depth_proj_enc["x4"] = x4_enc
        x_depth_proj_enc["x3"] = x3_enc

        x_dec = self.gp_decoder(x_depth_proj_enc)
        x_dec = self._normalize_decoder_output(x_dec)  # (bs, nclasses, H, W)

        outputs = {
            "depth_proj_estimate": x_depth_proj, 
            "occ_estimate": x_dec
        }

        enc_fts = {
            ("encoder_features", 0): x_rgb_enc["x3p"], 
            ("encoder_features", 1): x_rgb_enc["x4p"], 
            ("encoder_features", 2): x_rgb_enc["x5p"]
        }

        outputs.update(enc_fts)

        return outputs

    def _load_pretrained_model(self, path):
        depth_proj_state_dict = torch.load(
            self.config.GP_ANTICIPATION.pretrained_depth_proj_model, map_location="cpu"
        )["mapper_state_dict"]
        cleaned_state_dict = {}
        for k, v in depth_proj_state_dict.items():
            if ("mapper_copy" in k) or ("projection_unit" not in k):
                continue
            new_k = k.replace("module.", "")
            new_k = new_k.replace("mapper.projection_unit.main.main.", "")
            cleaned_state_dict[new_k] = v
        self.gp_depth_proj_estimator.load_state_dict(cleaned_state_dict)


class OccAntDepth(BaseModel):
    """
    Anticipated using depth projection only.
    """

    def _create_gp_models(self):
        gp_cfg = self.config.GP_ANTICIPATION

        # Compute constants
        nsf = gp_cfg.unet_nsf
        unet_encoder = UNetEncoder(gp_cfg.nclasses, nsf=nsf)
        unet_decoder = UNetDecoder(gp_cfg.nclasses, nsf=nsf)
        unet_feat_size = nsf * 8
        self.gp_depth_proj_encoder = unet_encoder
        self.gp_decoder = unet_decoder

    def _do_gp_anticipation(self, x):
        """
        Inputs:
            x is a dictionary containing the following keys:
                'ego_map_gt' - (bs, nclasses, H, W) input
        """
        x_enc = self.gp_depth_proj_encoder(
            x["ego_map_gt"]
        )  # dictionary with different outputs
        x_dec = self.gp_decoder(x_enc)  # (bs, nclasses, H, W)
        x_dec = self._normalize_decoder_output(x_dec)

        outputs = {"occ_estimate": x_dec}

        return outputs


class OccAntRGBD(BaseModel):
    """
    Anticipated using rgb and depth projection.
    """

    def _create_gp_models(self):
        nmodes = 2
        gp_cfg = self.config.GP_ANTICIPATION

        # Compute constants
        resnet_type = (
            gp_cfg.resnet_type if hasattr(gp_cfg, "resnet_type") else "resnet50"
        )
        infeats = 768 if resnet_type == "resnet50" else 192
        nsf = gp_cfg.unet_nsf
        unet_encoder = UNetEncoder(gp_cfg.nclasses, nsf=nsf)
        unet_decoder = UNetDecoder(gp_cfg.nclasses, nsf=nsf)
        unet_feat_size = nsf * 8

        # RGB encoder branch
        self.gp_rgb_encoder = ResNetRGBEncoder(resnet_type)
        self.gp_rgb_projector = LearnedRGBProjection(mtype="upsample", infeats=infeats)
        self.gp_rgb_unet = MiniUNetEncoder(infeats, unet_feat_size)

        # Depth encoder branch
        self.scale_ego_map_input = lambda x: F.interpolate(x, (self.config.width, self.config.height), mode='nearest')
        self.gp_depth_proj_encoder = unet_encoder

        # Merge modules
        self.gp_merge_x5 = MergeMultimodal(unet_feat_size, nmodes=nmodes)
        self.gp_merge_x4 = MergeMultimodal(unet_feat_size, nmodes=nmodes)
        self.gp_merge_x3 = MergeMultimodal(unet_feat_size // 2, nmodes=nmodes)

        self.bottleneck = [self.gp_merge_x5, self.gp_merge_x4, self.gp_merge_x3]

        # Decoder module
        self.gp_decoder = unet_decoder

        if gp_cfg.freeze_features:
            for p in self.gp_rgb_encoder.parameters():
                p.requires_grad = False

    def _do_gp_anticipation(self, x):
        """
        Inputs:
            x is a dictionary containing the following keys:
                'rgb' - (bs, 3, H, W) RGB input
                'ego_map_gt' - (bs, 2, H, W) probabilities
        """
        x_rgb = self.gp_rgb_encoder(x["rgb"])  # (bs, infeats, H/8, W/8)
        x_gp = self.gp_rgb_projector(x_rgb)  # (bs, infeats, H/4, W/4)

        x_rgb_enc = self.gp_rgb_unet(x_gp)  # {'x3p', 'x4p', 'x5p'}
        ego_map_gt = self.scale_ego_map_input(x["ego_map_gt"])
        x_depth_proj_enc = self.gp_depth_proj_encoder(
            ego_map_gt
        )  # {'x1', 'x2', 'x3', 'x4', 'x5'}

        # Replace x_depth_proj_enc with merged features
        x5_inputs = [x_rgb_enc["x5p"], x_depth_proj_enc["x5"]]
        x4_inputs = [x_rgb_enc["x4p"], x_depth_proj_enc["x4"]]
        x3_inputs = [x_rgb_enc["x3p"], x_depth_proj_enc["x3"]]

        x5_enc = self.gp_merge_x5(*x5_inputs)  # (unet_feat_size  , H/16, H/16)
        x4_enc = self.gp_merge_x4(*x4_inputs)  # (unet_feat_size  , H/8 , H/8 )
        x3_enc = self.gp_merge_x3(*x3_inputs)  # (unet_feat_size/2, H/4 , H/4 )
        x_depth_proj_enc["x5"] = x5_enc
        x_depth_proj_enc["x4"] = x4_enc
        x_depth_proj_enc["x3"] = x3_enc

        x_dec = self.gp_decoder(x_depth_proj_enc)  # (bs, nclasses, H, W)
        x_dec = self._normalize_decoder_output(x_dec)

        outputs = {"occ_estimate": x_dec}

        enc_fts = {
            ("encoder_features", 0): x_rgb_enc["x3p"], 
            ("encoder_features", 1): x_rgb_enc["x4p"], 
            ("encoder_features", 2): x_rgb_enc["x5p"]
        }
        outputs.update(enc_fts)

        return outputs

class OccAntLSTM(BaseModel):
    """
    Anticipated using rgb only.
    """

    def _create_gp_models(self):
        nmodes = 2
        gp_cfg = self.config.GP_ANTICIPATION

        self.seq_len = len(self.config.frame_ids)

        # Compute constants
        resnet_type = (
            gp_cfg.resnet_type if hasattr(gp_cfg, "resnet_type") else "resnet50"
        )
        infeats = 768 if resnet_type == "resnet50" else 192
        nsf = gp_cfg.unet_nsf
        unet_feat_size = nsf * 8

        # RGB encoder branch
        self.gp_rgb_encoder = ResNetRGBEncoder(resnet_type)
        # self.gp_merge_embeddings = ConvLSTMCell(192, 192, (3, 3), True)

        self.gp_rgb_projector = LearnedRGBProjection(mtype="upsample", infeats=infeats)
        self.gp_rgb_unet = MiniUNetEncoder(infeats, unet_feat_size)

        # Depth projection estimator
        config = self.config.clone()
        self.gp_depth_proj_estimator = ANSRGB(config)

        # Depth encoder branch
        self.gp_depth_proj_encoder = UNetEncoder(gp_cfg.nclasses, nsf=nsf)

        # Merge modules
        self.gp_merge_x5 = MergeMultimodalLSTM(unet_feat_size, nmodes=nmodes)
        self.gp_merge_x4 = MergeMultimodalLSTM(unet_feat_size, nmodes=nmodes)
        self.gp_merge_x3 = MergeMultimodalLSTM(unet_feat_size // 2, nmodes=nmodes)

        # Decoder module
        self.gp_decoder = UNetDecoderLSTM(gp_cfg.nclasses, nsf=nsf)

        self._detach_depth_proj = gp_cfg.detach_depth_proj

        # Load pretrained model if available
        if gp_cfg.pretrained_depth_proj_model != "":
            self._load_pretrained_model(gp_cfg.pretrained_depth_proj_model)

        if gp_cfg.freeze_features:
            for p in self.gp_rgb_encoder.parameters():
                p.requires_grad = False
        if gp_cfg.freeze_depth_proj_model:
            for p in self.gp_depth_proj_estimator.parameters():
                p.requires_grad = False

    def _do_gp_anticipation(self, x):
        """
        Inputs:
            x is a dictionary containing the following keys:
                'rgb' - (bs, 3, H, W) RGB input
                'depth' - (bs, 3, H, W) Depth input - channels are repeated
                'ego_map_gt' - (bs, 2, H, W) probabilities
        """
        B, T = x["rgb"].shape[0] // self.seq_len, self.seq_len
        x_rgb = self.gp_rgb_encoder(x["rgb"])  # (bs, 768, H/8, W/8)
        x_gp = self.gp_rgb_projector(x_rgb)  # (bs, 768, H/4, W/4)

        x_rgb_enc = self.gp_rgb_unet(x_gp)  # {'x3p', 'x4p', 'x5p'}

        for k, v in x_rgb_enc.items():
            x_rgb_enc[k] = v.reshape((B, T, *v.shape[1:]))

        # Estimate projected occupancy
        x_depth_proj = self.gp_depth_proj_estimator(x)["occ_estimate"]  # (bs, nclasses, V, V)
        if self._detach_depth_proj:
            x_depth_proj_enc = self.gp_depth_proj_encoder(
                x_depth_proj.detach()
            )  # {'x1', 'x2', 'x3', 'x4', 'x5'}
        else:
            x_depth_proj_enc = self.gp_depth_proj_encoder(
                x_depth_proj
            )  # {'x1', 'x2', 'x3', 'x4', 'x5'}

        for k, v in x_depth_proj_enc.items():
            x_depth_proj_enc[k] = v.reshape((B, T, *v.shape[1:]))

        # Replace x_depth_proj_enc with merged features
        x5_inputs = [x_rgb_enc["x5p"], x_depth_proj_enc["x5"]]
        x4_inputs = [x_rgb_enc["x4p"], x_depth_proj_enc["x4"]]
        x3_inputs = [x_rgb_enc["x3p"], x_depth_proj_enc["x3"]]

        x5_enc = self.gp_merge_x5(*x5_inputs)  # (unet_feat_size  , H/16, H/16)
        x4_enc = self.gp_merge_x4(*x4_inputs)  # (unet_feat_size  , H/8 , H/8 )
        x3_enc = self.gp_merge_x3(*x3_inputs)  # (unet_feat_size/2, H/4 , H/4 )
        x_depth_proj_enc["x5"] = x5_enc
        x_depth_proj_enc["x4"] = x4_enc
        x_depth_proj_enc["x3"] = x3_enc

        x_dec = self.gp_decoder(x_depth_proj_enc)
        x_dec = x_dec.reshape((B * T, *x_dec.shape[2:]))
        x_dec = self._normalize_decoder_output(x_dec)  # (bs, nclasses, H, W)

        outputs = {
            "depth_proj_estimate": x_depth_proj, 
            "occ_estimate": x_dec
        }

        enc_fts = {
            ("encoder_features", 0): x_rgb_enc["x3p"].reshape((B*T, *x_rgb_enc["x3p"].shape[2:])), 
            ("encoder_features", 1): x_rgb_enc["x4p"].reshape((B*T, *x_rgb_enc["x4p"].shape[2:])), 
            ("encoder_features", 2): x_rgb_enc["x5p"].reshape((B*T, *x_rgb_enc["x5p"].shape[2:]))
        }

        outputs.update(enc_fts)

        return outputs


class OccAntConcat(BaseModel):
    """
    Anticipated using rgb only.
    """

    def _create_gp_models(self):
        nmodes = 2
        gp_cfg = self.config.GP_ANTICIPATION

        self.seq_len = len(self.config.frame_ids)

        # Compute constants
        resnet_type = (
            gp_cfg.resnet_type if hasattr(gp_cfg, "resnet_type") else "resnet50"
        )
        infeats = 768 if resnet_type == "resnet50" else 192
        nsf = gp_cfg.unet_nsf
        unet_feat_size = nsf * 8

        # RGB encoder branch
        self.gp_rgb_encoder = ResNetRGBEncoder(resnet_type)
        self.gp_rgb_projector = LearnedRGBProjection(mtype="upsample", infeats=infeats)
        self.gp_rgb_unet = MiniUNetEncoder(infeats, unet_feat_size)

        # Depth projection estimator
        config = self.config.clone()
        self.gp_depth_proj_estimator = ANSRGB(config)

        # Depth encoder branch
        self.gp_depth_proj_encoder = UNetEncoder(gp_cfg.nclasses, nsf=nsf)

        # Merge modules
        self.gp_merge_x5 = MergeMultimodal(unet_feat_size, nmodes=nmodes)
        self.gp_merge_x4 = MergeMultimodal(unet_feat_size, nmodes=nmodes)
        self.gp_merge_x3 = MergeMultimodal(unet_feat_size // 2, nmodes=nmodes)

        self.gp_conv3d_x1 = MergeMultiView(unet_feat_size//8, nviews=self.seq_len)
        self.gp_conv3d_x2 = MergeMultiView(unet_feat_size//4, nviews=self.seq_len)
        self.gp_conv3d_x3 = MergeMultiView(unet_feat_size//2, nviews=self.seq_len)
        self.gp_conv3d_x4 = MergeMultiView(unet_feat_size, nviews=self.seq_len)
        self.gp_conv3d_x5 = MergeMultiView(unet_feat_size, nviews=self.seq_len)

        # Decoder module
        self.gp_decoder = UNetDecoder(gp_cfg.nclasses, nsf=nsf)

        self._detach_depth_proj = gp_cfg.detach_depth_proj

        # Load pretrained model if available
        if gp_cfg.pretrained_depth_proj_model != "":
            self._load_pretrained_model(gp_cfg.pretrained_depth_proj_model)

        if gp_cfg.freeze_features:
            for p in self.gp_rgb_encoder.parameters():
                p.requires_grad = False
        if gp_cfg.freeze_depth_proj_model:
            for p in self.gp_depth_proj_estimator.parameters():
                p.requires_grad = False

    def _do_gp_anticipation(self, x):
        """
        Inputs:
            x is a dictionary containing the following keys:
                'rgb' - (bs, 3, H, W) RGB input
                'depth' - (bs, 3, H, W) Depth input - channels are repeated
                'ego_map_gt' - (bs, 2, H, W) probabilities
        """
        B, T = x["rgb"].shape[0] // self.seq_len, self.seq_len
        x_rgb = self.gp_rgb_encoder(x["rgb"])  # (bs, 768, H/8, W/8)
        x_gp = self.gp_rgb_projector(x_rgb)  # (bs, 768, H/4, W/4)

        x_rgb_enc = self.gp_rgb_unet(x_gp)  # {'x3p', 'x4p', 'x5p'}

        # Estimate projected occupancy
        x_depth_proj = self.gp_depth_proj_estimator(x)["occ_estimate"]  # (bs, nclasses, V, V)
        if self._detach_depth_proj:
            x_depth_proj_enc = self.gp_depth_proj_encoder(
                x_depth_proj.detach()
            )  # {'x1', 'x2', 'x3', 'x4', 'x5'}
        else:
            x_depth_proj_enc = self.gp_depth_proj_encoder(
                x_depth_proj
            )  # {'x1', 'x2', 'x3', 'x4', 'x5'}

        # Replace x_depth_proj_enc with merged features
        x5_inputs = [x_rgb_enc["x5p"], x_depth_proj_enc["x5"]]
        x4_inputs = [x_rgb_enc["x4p"], x_depth_proj_enc["x4"]]
        x3_inputs = [x_rgb_enc["x3p"], x_depth_proj_enc["x3"]]

        x5_enc = self.gp_merge_x5(*x5_inputs)  # (unet_feat_size  , H/16, H/16)
        x4_enc = self.gp_merge_x4(*x4_inputs)  # (unet_feat_size  , H/8 , H/8 )
        x3_enc = self.gp_merge_x3(*x3_inputs)  # (unet_feat_size/2, H/4 , H/4 )
        x_depth_proj_enc["x5"] = x5_enc
        x_depth_proj_enc["x4"] = x4_enc
        x_depth_proj_enc["x3"] = x3_enc

        for k, v in x_depth_proj_enc.items():
            merge_fn = getattr(self, 'gp_conv3d_' + k)
            x_depth_proj_enc[k] = merge_fn(v.reshape((B, -1, *v.shape[2:]))).reshape((-1, *v.shape[1:]))

        x_dec = self.gp_decoder(x_depth_proj_enc)
        x_dec = self._normalize_decoder_output(x_dec)  # (bs, nclasses, H, W)

        outputs = {
            "depth_proj_estimate": x_depth_proj, 
            "occ_estimate": x_dec
        }

        enc_fts = {
            ("encoder_features", 0): x_rgb_enc["x3p"], 
            ("encoder_features", 1): x_rgb_enc["x4p"], 
            ("encoder_features", 2): x_rgb_enc["x5p"]
        }

        outputs.update(enc_fts)

        return outputs


class OccAntSemantics(BaseModel):
    """
    Anticipated using rgb only.
    """

    def _create_gp_models(self):
        nmodes = 2
        gp_cfg = self.config.GP_ANTICIPATION

        self.seq_len = len(self.config.frame_ids)

        # Compute constants
        resnet_type = (
            gp_cfg.resnet_type if hasattr(gp_cfg, "resnet_type") else "resnet50"
        )
        infeats = 768 if resnet_type == "resnet50" else 192
        nsf = gp_cfg.unet_nsf
        unet_feat_size = nsf * 8

        # RGB encoder branch
        self.gp_rgb_encoder = ResNetRGBEncoder(resnet_type)
        self.gp_rgb_projector = LearnedRGBProjection(mtype="upsample", infeats=infeats)
        self.gp_rgb_unet = MiniUNetEncoder(infeats, unet_feat_size)


        # Semantic encoder branch
        self.gp_semantic_proj_encoder = UNetEncoder(1, nsf=nsf)

        # Merge modules
        self.gp_merge_x5 = MergeMultimodal(unet_feat_size, nmodes=nmodes)
        self.gp_merge_x4 = MergeMultimodal(unet_feat_size, nmodes=nmodes)
        self.gp_merge_x3 = MergeMultimodal(unet_feat_size // 2, nmodes=nmodes)

        # Decoder module
        self.gp_decoder = UNetDecoder(gp_cfg.nclasses, nsf=nsf)

        # # Load pretrained model if available
        # if gp_cfg.pretrained_depth_proj_model != "":
        #     self._load_pretrained_model(gp_cfg.pretrained_depth_proj_model)

        if gp_cfg.freeze_features:
            for p in self.gp_rgb_encoder.parameters():
                p.requires_grad = False
        if gp_cfg.freeze_semantic_proj_model:
            for p in self.gp_semantic_proj_encoder.parameters():
                p.requires_grad = False

    def _do_gp_anticipation(self, x):
        """
        Inputs:
            x is a dictionary containing the following keys:
                'rgb' - (bs, 3, H, W) RGB input
                'depth' - (bs, 3, H, W) Depth input - channels are repeated
                'ego_map_gt' - (bs, 2, H, W) probabilities
        """
        x_rgb = self.gp_rgb_encoder(x["rgb"])  # (bs, 768, H/8, W/8)
        x_gp = self.gp_rgb_projector(x_rgb)  # (bs, 768, H/4, W/4)

        x_rgb_enc = self.gp_rgb_unet(x_gp)  # {'x3p', 'x4p', 'x5p'}

        x_semantic_proj_enc = self.gp_semantic_proj_encoder(
            x["semantics"]
        )  # {'x1', 'x2', 'x3', 'x4', 'x5'}

        # Replace x_depth_proj_enc with merged features
        x5_inputs = [x_rgb_enc["x5p"], x_semantic_proj_enc["x5"]]
        x4_inputs = [x_rgb_enc["x4p"], x_semantic_proj_enc["x4"]]
        x3_inputs = [x_rgb_enc["x3p"], x_semantic_proj_enc["x3"]]

        x5_enc = self.gp_merge_x5(*x5_inputs)  # (unet_feat_size  , H/16, H/16)
        x4_enc = self.gp_merge_x4(*x4_inputs)  # (unet_feat_size  , H/8 , H/8 )
        x3_enc = self.gp_merge_x3(*x3_inputs)  # (unet_feat_size/2, H/4 , H/4 )
        x_semantic_proj_enc["x5"] = x5_enc
        x_semantic_proj_enc["x4"] = x4_enc
        x_semantic_proj_enc["x3"] = x3_enc

        x_dec = self.gp_decoder(x_semantic_proj_enc)
        x_dec = self._normalize_decoder_output(x_dec)  # (bs, nclasses, H, W)

        outputs = {
            "occ_estimate": x_dec
        }

        enc_fts = {
            ("encoder_features", 0): x_rgb_enc["x3p"], 
            ("encoder_features", 1): x_rgb_enc["x4p"], 
            ("encoder_features", 2): x_rgb_enc["x5p"]
        }

        outputs.update(enc_fts)

        return outputs


class OccAntSingleModality(BaseModel):
    """
    Anticipated using rgb only.
    """

    def _create_gp_models(self):
        nmodes = 2
        gp_cfg = self.config.GP_ANTICIPATION

        # Compute constants
        resnet_type = (
            gp_cfg.resnet_type if hasattr(gp_cfg, "resnet_type") else "resnet50"
        )
        infeats = 768 if resnet_type == "resnet50" else 192
        nsf = gp_cfg.unet_nsf
        unet_feat_size = nsf * 8

        # RGB encoder branch
        # self.gp_rgb_encoder = ResNetRGBEncoder(resnet_type)
        # self.gp_rgb_projector = LearnedRGBProjection(mtype="upsample", infeats=infeats)
        # self.gp_rgb_unet = MiniUNetEncoder(infeats, unet_feat_size)

        # Depth projection estimator
        config = self.config.clone()
        self.gp_depth_proj_estimator = ANSRGB(config)

        # Depth encoder branch
        self.gp_depth_proj_encoder = UNetEncoder(gp_cfg.nclasses, nsf=nsf)

        # Merge modules
        # self.gp_merge_x5 = MergeMultimodal(unet_feat_size, nmodes=nmodes)
        # self.gp_merge_x4 = MergeMultimodal(unet_feat_size, nmodes=nmodes)
        # self.gp_merge_x3 = MergeMultimodal(unet_feat_size // 2, nmodes=nmodes)

        # Decoder module
        self.gp_decoder = UNetDecoder(gp_cfg.nclasses, nsf=nsf)

        self._detach_depth_proj = gp_cfg.detach_depth_proj

        # Load pretrained model if available
        if gp_cfg.pretrained_depth_proj_model != "":
            self._load_pretrained_model(gp_cfg.pretrained_depth_proj_model)

        if gp_cfg.freeze_features:
            for p in self.gp_rgb_encoder.parameters():
                p.requires_grad = False
        if gp_cfg.freeze_depth_proj_model:
            for p in self.gp_depth_proj_estimator.parameters():
                p.requires_grad = False

    def _do_gp_anticipation(self, x):
        """
        Inputs:
            x is a dictionary containing the following keys:
                'rgb' - (bs, 3, H, W) RGB input
                'depth' - (bs, 3, H, W) Depth input - channels are repeated
                'ego_map_gt' - (bs, 2, H, W) probabilities
        """
        # x_rgb = self.gp_rgb_encoder(x["rgb"])  # (bs, 768, H/8, W/8)
        # x_gp = self.gp_rgb_projector(x_rgb)  # (bs, 768, H/4, W/4)

        # x_rgb_enc = self.gp_rgb_unet(x_gp)  # {'x3p', 'x4p', 'x5p'}
        # # Estimate projected occupancy
        x_depth_proj = self.gp_depth_proj_estimator(x)["occ_estimate"]  # (bs, nclasses, V, V)
        if self._detach_depth_proj:
            x_depth_proj_enc = self.gp_depth_proj_encoder(
                x_depth_proj.detach()
            )  # {'x1', 'x2', 'x3', 'x4', 'x5'}
        else:
            x_depth_proj_enc = self.gp_depth_proj_encoder(
                x_depth_proj
            )  # {'x1', 'x2', 'x3', 'x4', 'x5'}

        # Replace x_depth_proj_enc with merged features
        # x5_inputs = [x_rgb_enc["x5p"], x_depth_proj_enc["x5"]]
        # x4_inputs = [x_rgb_enc["x4p"], x_depth_proj_enc["x4"]]
        # x3_inputs = [x_rgb_enc["x3p"], x_depth_proj_enc["x3"]]

        # x5_enc = self.gp_merge_x5(*x5_inputs)  # (unet_feat_size  , H/16, H/16)
        # x4_enc = self.gp_merge_x4(*x4_inputs)  # (unet_feat_size  , H/8 , H/8 )
        # x3_enc = self.gp_merge_x3(*x3_inputs)  # (unet_feat_size/2, H/4 , H/4 )
        # x_depth_proj_enc["x5"] = x5_enc
        # x_depth_proj_enc["x4"] = x4_enc
        # x_depth_proj_enc["x3"] = x3_enc

        x_dec = self.gp_decoder(x_depth_proj_enc)
        x_dec = self._normalize_decoder_output(x_dec)  # (bs, nclasses, H, W)

        outputs = {
            "depth_proj_estimate": x_depth_proj, 
            "occ_estimate": x_dec
        }

        return outputs

    def _load_pretrained_model(self, path):
        depth_proj_state_dict = torch.load(
            self.config.GP_ANTICIPATION.pretrained_depth_proj_model, map_location="cpu"
        )["mapper_state_dict"]
        cleaned_state_dict = {}
        for k, v in depth_proj_state_dict.items():
            if ("mapper_copy" in k) or ("projection_unit" not in k):
                continue
            new_k = k.replace("module.", "")
            new_k = new_k.replace("mapper.projection_unit.main.main.", "")
            cleaned_state_dict[new_k] = v
        self.gp_depth_proj_estimator.load_state_dict(cleaned_state_dict)



class OccAnt2Decoders(BaseModel):
    """
    Anticipated using rgb only.
    """

    def _create_gp_models(self):
        nmodes = 2
        gp_cfg = self.config.GP_ANTICIPATION

        # Compute constants
        resnet_type = (
            gp_cfg.resnet_type if hasattr(gp_cfg, "resnet_type") else "resnet50"
        )
        infeats = 768 if resnet_type == "resnet50" else 192
        nsf = gp_cfg.unet_nsf
        unet_feat_size = nsf * 8

        # RGB encoder branch
        self.gp_rgb_encoder = ResNetRGBEncoder(resnet_type)
        self.gp_rgb_projector = LearnedRGBProjection(mtype="upsample", infeats=infeats)
        self.gp_rgb_unet = MiniUNetEncoder(infeats, unet_feat_size)

        # Depth projection estimator
        config = self.config.clone()
        self.gp_depth_proj_estimator = ANSRGB(config)

        # Depth encoder branch
        self.gp_depth_proj_encoder = UNetEncoder(gp_cfg.nclasses, nsf=nsf)

        # Merge modules
        self.gp_merge_x5 = MergeMultimodal(unet_feat_size, nmodes=nmodes)
        self.gp_merge_x4 = MergeMultimodal(unet_feat_size, nmodes=nmodes)
        self.gp_merge_x3 = MergeMultimodal(unet_feat_size // 2, nmodes=nmodes)

        # Decoder module
        self.gp_decoder1 = UNetDecoder(1, nsf=nsf) # For occupied/free
        self.gp_decoder2 = UNetDecoder(1, nsf=nsf) # For explored/unexplored

        self._detach_depth_proj = gp_cfg.detach_depth_proj

        # Load pretrained model if available
        if gp_cfg.pretrained_depth_proj_model != "":
            self._load_pretrained_model(gp_cfg.pretrained_depth_proj_model)

        if gp_cfg.freeze_features:
            for p in self.gp_rgb_encoder.parameters():
                p.requires_grad = False
        if gp_cfg.freeze_depth_proj_model:
            for p in self.gp_depth_proj_estimator.parameters():
                p.requires_grad = False

    def _do_gp_anticipation(self, x):
        """
        Inputs:
            x is a dictionary containing the following keys:
                'rgb' - (bs, 3, H, W) RGB input
                'depth' - (bs, 3, H, W) Depth input - channels are repeated
                'ego_map_gt' - (bs, 2, H, W) probabilities
        """
        x_rgb = self.gp_rgb_encoder(x["rgb"])  # (bs, 768, H/8, W/8)
        x_gp = self.gp_rgb_projector(x_rgb)  # (bs, 768, H/4, W/4)

        x_rgb_enc = self.gp_rgb_unet(x_gp)  # {'x3p', 'x4p', 'x5p'}
        # Estimate projected occupancy
        x_depth_proj = self.gp_depth_proj_estimator(x)["occ_estimate"]  # (bs, nclasses, V, V)
        if self._detach_depth_proj:
            x_depth_proj_enc = self.gp_depth_proj_encoder(
                x_depth_proj.detach()
            )  # {'x1', 'x2', 'x3', 'x4', 'x5'}
        else:
            x_depth_proj_enc = self.gp_depth_proj_encoder(
                x_depth_proj
            )  # {'x1', 'x2', 'x3', 'x4', 'x5'}

        # Replace x_depth_proj_enc with merged features
        x5_inputs = [x_rgb_enc["x5p"], x_depth_proj_enc["x5"]]
        x4_inputs = [x_rgb_enc["x4p"], x_depth_proj_enc["x4"]]
        x3_inputs = [x_rgb_enc["x3p"], x_depth_proj_enc["x3"]]

        x5_enc = self.gp_merge_x5(*x5_inputs)  # (unet_feat_size  , H/16, H/16)
        x4_enc = self.gp_merge_x4(*x4_inputs)  # (unet_feat_size  , H/8 , H/8 )
        x3_enc = self.gp_merge_x3(*x3_inputs)  # (unet_feat_size/2, H/4 , H/4 )
        x_depth_proj_enc["x5"] = x5_enc
        x_depth_proj_enc["x4"] = x4_enc
        x_depth_proj_enc["x3"] = x3_enc

        x_dec1 = self.gp_decoder1(x_depth_proj_enc)
        x_dec2 = self.gp_decoder2(x_depth_proj_enc)
        x_dec = torch.cat([x_dec1, x_dec2], dim=1)
        x_dec = self._normalize_decoder_output(x_dec)  # (bs, nclasses, H, W)

        outputs = {
            "depth_proj_estimate": x_depth_proj, 
            "occ_estimate": x_dec
        }

        enc_fts = {
            ("encoder_features", 0): x_rgb_enc["x3p"], 
            ("encoder_features", 1): x_rgb_enc["x4p"], 
            ("encoder_features", 2): x_rgb_enc["x5p"]
        }

        outputs.update(enc_fts)

        return outputs

    def _load_pretrained_model(self, path):
        depth_proj_state_dict = torch.load(
            self.config.GP_ANTICIPATION.pretrained_depth_proj_model, map_location="cpu"
        )["mapper_state_dict"]
        cleaned_state_dict = {}
        for k, v in depth_proj_state_dict.items():
            if ("mapper_copy" in k) or ("projection_unit" not in k):
                continue
            new_k = k.replace("module.", "")
            new_k = new_k.replace("mapper.projection_unit.main.main.", "")
            cleaned_state_dict[new_k] = v
        self.gp_depth_proj_estimator.load_state_dict(cleaned_state_dict)


class OccAntDepthRaw(BaseModel):
    """
    Anticipated using rgb only.
    """

    def _create_gp_models(self):
        nmodes = 2
        gp_cfg = self.config.GP_ANTICIPATION

        self.seq_len = len(self.config.frame_ids)

        # Compute constants
        resnet_type = (
            gp_cfg.resnet_type if hasattr(gp_cfg, "resnet_type") else "resnet50"
        )
        infeats = 768 if resnet_type == "resnet50" else 192
        nsf = gp_cfg.unet_nsf
        unet_feat_size = nsf * 8

        # RGB encoder branch
        self.gp_rgb_encoder = ResNetRGBEncoder(resnet_type)
        self.gp_rgb_projector = LearnedRGBProjection(mtype="upsample", infeats=infeats)
        self.gp_rgb_unet = MiniUNetEncoder(infeats, unet_feat_size)


        # Semantic encoder branch
        self.gp_depth_encoder = UNetEncoder(1, nsf=nsf)

        # Merge modules
        self.gp_merge_x5 = MergeMultimodal(unet_feat_size, nmodes=nmodes)
        self.gp_merge_x4 = MergeMultimodal(unet_feat_size, nmodes=nmodes)
        self.gp_merge_x3 = MergeMultimodal(unet_feat_size // 2, nmodes=nmodes)

        # Decoder module
        self.gp_decoder = UNetDecoder(gp_cfg.nclasses, nsf=nsf)

        # # Load pretrained model if available
        # if gp_cfg.pretrained_depth_proj_model != "":
        #     self._load_pretrained_model(gp_cfg.pretrained_depth_proj_model)

        if gp_cfg.freeze_features:
            for p in self.gp_rgb_encoder.parameters():
                p.requires_grad = False
        if gp_cfg.freeze_semantic_proj_model:
            for p in self.gp_semantic_proj_encoder.parameters():
                p.requires_grad = False

    def _do_gp_anticipation(self, x):
        """
        Inputs:
            x is a dictionary containing the following keys:
                'rgb' - (bs, 3, H, W) RGB input
                'depth' - (bs, 3, H, W) Depth input - channels are repeated
                'ego_map_gt' - (bs, 2, H, W) probabilities
        """
        x_rgb = self.gp_rgb_encoder(x["rgb"])  # (bs, 768, H/8, W/8)
        x_gp = self.gp_rgb_projector(x_rgb)  # (bs, 768, H/4, W/4)

        x_rgb_enc = self.gp_rgb_unet(x_gp)  # {'x3p', 'x4p', 'x5p'}

        x_depth_enc = self.gp_depth_encoder(
            x["depth"]
        )  # {'x1', 'x2', 'x3', 'x4', 'x5'}

        # Replace x_depth_proj_enc with merged features
        x5_inputs = [x_rgb_enc["x5p"], x_depth_enc["x5"]]
        x4_inputs = [x_rgb_enc["x4p"], x_depth_enc["x4"]]
        x3_inputs = [x_rgb_enc["x3p"], x_depth_enc["x3"]]

        x5_enc = self.gp_merge_x5(*x5_inputs)  # (unet_feat_size  , H/16, H/16)
        x4_enc = self.gp_merge_x4(*x4_inputs)  # (unet_feat_size  , H/8 , H/8 )
        x3_enc = self.gp_merge_x3(*x3_inputs)  # (unet_feat_size/2, H/4 , H/4 )
        x_depth_enc["x5"] = x5_enc
        x_depth_enc["x4"] = x4_enc
        x_depth_enc["x3"] = x3_enc

        x_dec = self.gp_decoder(x_depth_enc)
        x_dec = self._normalize_decoder_output(x_dec)  # (bs, nclasses, H, W)

        outputs = {
            "occ_estimate": x_dec
        }

        enc_fts = {
            ("encoder_features", 0): x_rgb_enc["x3p"], 
            ("encoder_features", 1): x_rgb_enc["x4p"], 
            ("encoder_features", 2): x_rgb_enc["x5p"]
        }

        outputs.update(enc_fts)

        return outputs


class OccAntAuxSemantics(BaseModel):
    
    def _create_gp_models(self):
        nmodes = 2
        gp_cfg = self.config.GP_ANTICIPATION

        # Compute constants
        resnet_type = (
            gp_cfg.resnet_type if hasattr(gp_cfg, "resnet_type") else "resnet50"
        )
        infeats = 768 if resnet_type == "resnet50" else 192
        nsf = gp_cfg.unet_nsf
        unet_feat_size = nsf * 8

        # RGB encoder branch
        self.gp_rgb_encoder = ResNetRGBEncoder(resnet_type)
        self.gp_rgb_projector = LearnedRGBProjection(mtype="upsample", infeats=infeats)
        self.gp_rgb_unet = MiniUNetEncoder(infeats, unet_feat_size)

        self.gp_sem_fpn = FPNDecoder(1, nsf=nsf) # binary prediction

        # Depth projection estimator
        config = self.config.clone()
        self.gp_depth_proj_estimator = ANSRGB(config)

        # Depth encoder branch
        self.gp_depth_proj_encoder = UNetEncoder(gp_cfg.nclasses, nsf=nsf)

        # Merge modules
        self.gp_merge_x5 = MergeMultimodal(unet_feat_size, nmodes=nmodes)
        self.gp_merge_x4 = MergeMultimodal(unet_feat_size, nmodes=nmodes)
        self.gp_merge_x3 = MergeMultimodal(unet_feat_size // 2, nmodes=nmodes)

        # Decoder module
        self.gp_decoder = UNetDecoder(gp_cfg.nclasses, nsf=nsf)

        self._detach_depth_proj = gp_cfg.detach_depth_proj

        # Load pretrained model if available
        if gp_cfg.pretrained_depth_proj_model != "":
            self._load_pretrained_model(gp_cfg.pretrained_depth_proj_model)

        if gp_cfg.freeze_features:
            for p in self.gp_rgb_encoder.parameters():
                p.requires_grad = False
        if gp_cfg.freeze_depth_proj_model:
            for p in self.gp_depth_proj_estimator.parameters():
                p.requires_grad = False

    def _do_gp_anticipation(self, x):
        """
        Inputs:
            x is a dictionary containing the following keys:
                'rgb' - (bs, 3, H, W) RGB input
                'depth' - (bs, 3, H, W) Depth input - channels are repeated
                'ego_map_gt' - (bs, 2, H, W) probabilities
        """
        x_rgb = self.gp_rgb_encoder(x["rgb"])  # (bs, 768, H/8, W/8)
        x_gp = self.gp_rgb_projector(x_rgb)  # (bs, 768, H/4, W/4)

        x_rgb_enc = self.gp_rgb_unet(x_gp)  # {'x3p', 'x4p', 'x5p'}

        x_sem_out = self.gp_sem_fpn(x_rgb_enc)

        # Estimate projected occupancy
        x_depth_proj = self.gp_depth_proj_estimator(x)["occ_estimate"]  # (bs, nclasses, V, V)
        if self._detach_depth_proj:
            x_depth_proj_enc = self.gp_depth_proj_encoder(
                x_depth_proj.detach()
            )  # {'x1', 'x2', 'x3', 'x4', 'x5'}
        else:
            x_depth_proj_enc = self.gp_depth_proj_encoder(
                x_depth_proj
            )  # {'x1', 'x2', 'x3', 'x4', 'x5'}

        # Replace x_depth_proj_enc with merged features
        x5_inputs = [x_rgb_enc["x5p"], x_depth_proj_enc["x5"]]
        x4_inputs = [x_rgb_enc["x4p"], x_depth_proj_enc["x4"]]
        x3_inputs = [x_rgb_enc["x3p"], x_depth_proj_enc["x3"]]

        x5_enc = self.gp_merge_x5(*x5_inputs)  # (unet_feat_size  , H/16, H/16)
        x4_enc = self.gp_merge_x4(*x4_inputs)  # (unet_feat_size  , H/8 , H/8 )
        x3_enc = self.gp_merge_x3(*x3_inputs)  # (unet_feat_size/2, H/4 , H/4 )
        x_depth_proj_enc["x5"] = x5_enc
        x_depth_proj_enc["x4"] = x4_enc
        x_depth_proj_enc["x3"] = x3_enc

        x_dec = self.gp_decoder(x_depth_proj_enc)
        x_dec = self._normalize_decoder_output(x_dec)  # (bs, nclasses, H, W)

        outputs = {
            "depth_proj_estimate": x_depth_proj, 
            "occ_estimate": x_dec
        }

        enc_fts = {
            ("encoder_features", 0): x_rgb_enc["x3p"], 
            ("encoder_features", 1): x_rgb_enc["x4p"], 
            ("encoder_features", 2): x_rgb_enc["x5p"]
        }

        outputs.update(enc_fts)

        semantic_estimates = {
            ("semantics_pred", 'l', 0): x_sem_out["y3p"], 
            ("semantics_pred", 'l', 1): x_sem_out["y4p"], 
            ("semantics_pred", 'l', 2): x_sem_out["y5p"]
        }

        outputs.update(semantic_estimates)

        return outputs

    def _load_pretrained_model(self, path):
        depth_proj_state_dict = torch.load(
            self.config.GP_ANTICIPATION.pretrained_depth_proj_model, map_location="cpu"
        )["mapper_state_dict"]
        cleaned_state_dict = {}
        for k, v in depth_proj_state_dict.items():
            if ("mapper_copy" in k) or ("projection_unit" not in k):
                continue
            new_k = k.replace("module.", "")
            new_k = new_k.replace("mapper.projection_unit.main.main.", "")
            cleaned_state_dict[new_k] = v
        self.gp_depth_proj_estimator.load_state_dict(cleaned_state_dict)


class OccAntChandrakarInput(BaseModel):
    """
    Anticipated using rgb and depth projection.
    """

    def _create_gp_models(self):
        nmodes = 2
        gp_cfg = self.config.GP_ANTICIPATION

        self.cam_height = self.config.cam_height
        self.bev_width = self.config.bev_width
        self.map_scale = self.config.bev_res

        # Compute constants
        resnet_type = (
            gp_cfg.resnet_type if hasattr(gp_cfg, "resnet_type") else "resnet50"
        )
        infeats = 768 if resnet_type == "resnet50" else 192
        nsf = gp_cfg.unet_nsf
        unet_encoder = UNetEncoder(2, nsf=nsf)
        unet_decoder = UNetDecoder(gp_cfg.nclasses, nsf=nsf)
        unet_feat_size = nsf * 8

        # RGB encoder branch
        self.gp_rgb_encoder = ResNetRGBEncoder(resnet_type)
        self.gp_rgb_projector = LearnedRGBProjection(mtype="upsample", infeats=infeats)
        self.gp_rgb_unet = MiniUNetEncoder(infeats, unet_feat_size)

        # Depth encoder branch
        self.gp_depth_proj_encoder = unet_encoder

        # Merge modules
        self.gp_merge_x5 = MergeMultimodal(unet_feat_size, nmodes=nmodes)
        self.gp_merge_x4 = MergeMultimodal(unet_feat_size, nmodes=nmodes)
        self.gp_merge_x3 = MergeMultimodal(unet_feat_size // 2, nmodes=nmodes)

        self.bottleneck = [self.gp_merge_x5, self.gp_merge_x4, self.gp_merge_x3]

        # Decoder module
        self.gp_decoder = unet_decoder

        if gp_cfg.freeze_features:
            for p in self.gp_rgb_encoder.parameters():
                p.requires_grad = False
       

    def _floorseg_to_egomap(self, x):
        b = x["rgb"].shape[0]
        w, h = x["rgb"].shape[2:]
        img_rect = np.concatenate([np.indices((w, h)), np.ones((2, w, h))], axis=0)
        img_rect = torch.unsqueeze(torch.from_numpy(img_rect).float().to(x["rgb"].device), axis=0).repeat((b,1,1,1))

        inv_K = x["inv_K"]

        pc_proj = torch.matmul(inv_K, img_rect.reshape((b, 4, -1)))
        cos_theta = pc_proj[:, 1] / torch.linalg.norm(pc_proj, ord=2, dim=1)
        sin_theta = torch.sqrt(1 - cos_theta**2)

        floor_depth_pred = self.cam_height * sin_theta / (cos_theta + 1e-6)
        sem =  x["semantics"]

        pc = pc_proj[:, :3] * torch.unsqueeze(floor_depth_pred, dim=1)
        pc = pc.reshape((b, 3, h, w))
        pc_valid = pc * sem

        V = self.bev_width
        Vby2 = V // 2

        points = pc_valid.reshape((b, 3, -1))
        points[:, 1] *= -1
        points[:, 2] *= -1

        batch_ix = torch.cat([torch.full([1, 1, h*w], bx, device=points.device, dtype=torch.float32) for bx in range(b)])
        points = torch.cat((points, batch_ix), 1)
        points = points.transpose(1, 2)

        grid_x = ((points[..., 0] / self.map_scale) + Vby2)
        grid_y = ((points[..., 2] / self.map_scale) + V)

        valid_idx = (
            (grid_x >= 0) & (grid_x <= V - 1) & (grid_y >= 0) & (grid_y <= V - 1)
        )
        batch_indices = points[valid_idx, :][:, 3].int()
        grid_x = grid_x[valid_idx].int()
        grid_y = grid_y[valid_idx].int()

        pred_bev = torch.zeros((b*V*V), dtype=torch.float32, device=x["rgb"].device)
        bev_idx = batch_indices*V*V + grid_y*V + grid_x
        pred_bev[bev_idx.long()] = 1.0

        pred_bev = pred_bev.reshape((b, 1, V, V))
        pred_bev = pred_bev.repeat((1,2,1,1))
        return pred_bev


    def _do_gp_anticipation(self, x):
        """
        Inputs:
            x is a dictionary containing the following keys:
                'rgb' - (bs, 3, H, W) RGB input
                'ego_map_gt' - (bs, 2, H, W) probabilities
        """
        x_rgb = self.gp_rgb_encoder(x["rgb"])  # (bs, infeats, H/8, W/8)
        x_gp = self.gp_rgb_projector(x_rgb)  # (bs, infeats, H/4, W/4)

        x_rgb_enc = self.gp_rgb_unet(x_gp)  # {'x3p', 'x4p', 'x5p'}
        ego_map_gt = x["ego_map_gt"] # self._floorseg_to_egomap(x)
        x_depth_proj_enc = self.gp_depth_proj_encoder(
            ego_map_gt
        )  # {'x1', 'x2', 'x3', 'x4', 'x5'}

        # Replace x_depth_proj_enc with merged features
        x5_inputs = [x_rgb_enc["x5p"], x_depth_proj_enc["x5"]]
        x4_inputs = [x_rgb_enc["x4p"], x_depth_proj_enc["x4"]]
        x3_inputs = [x_rgb_enc["x3p"], x_depth_proj_enc["x3"]]

        x5_enc = self.gp_merge_x5(*x5_inputs)  # (unet_feat_size  , H/16, H/16)
        x4_enc = self.gp_merge_x4(*x4_inputs)  # (unet_feat_size  , H/8 , H/8 )
        x3_enc = self.gp_merge_x3(*x3_inputs)  # (unet_feat_size/2, H/4 , H/4 )
        x_depth_proj_enc["x5"] = x5_enc
        x_depth_proj_enc["x4"] = x4_enc
        x_depth_proj_enc["x3"] = x3_enc

        x_dec = self.gp_decoder(x_depth_proj_enc)  # (bs, nclasses, H, W)
        x_dec = self._normalize_decoder_output(x_dec)

        outputs = {"occ_estimate": x_dec}

        enc_fts = {
            ("encoder_features", 0): x_rgb_enc["x3p"], 
            ("encoder_features", 1): x_rgb_enc["x4p"], 
            ("encoder_features", 2): x_rgb_enc["x5p"]
        }
        outputs.update(enc_fts)

        return outputs

# ================================ Occupancy anticipator ==============================


class OccupancyAnticipator(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__()
        self.config = cfg
        model_type = cfg.GP_ANTICIPATION.type
        self._model_type = model_type
        self.input_width = cfg.width
        self.input_height = cfg.height
        self.bev_size = (self.config.bev_width, self.config.bev_height)
        cfg.defrost()
        if model_type == "ans_rgb":
            self.main = ANSRGB(cfg)
        elif model_type == "ans_depth":
            self.main = ANSDepth(cfg)
        elif model_type == "occant_rgb":
            self.main = OccAntRGB(cfg)
        elif model_type == "occant_depth":
            self.main = OccAntDepth(cfg)
        elif model_type == "occant_rgbd":
            self.main = OccAntRGBD(cfg)
        elif model_type == "occant_lstm":
            self.main = OccAntLSTM(cfg)
        elif model_type == "occant_concat":
            self.main = OccAntConcat(cfg)
        elif model_type == "occant_semantics":
            self.main = OccAntSemantics(cfg)
        elif model_type == "occant_singlemodality":
            self.main = OccAntSingleModality(cfg)
        elif model_type == "occant_2decoders":
            self.main = OccAnt2Decoders(cfg)
        elif model_type == "occant_depthraw":
            self.main = OccAntDepthRaw(cfg)
        elif model_type == "occant_auxsemantics":
            self.main = OccAntAuxSemantics(cfg)
        elif model_type == "occant_chandrakarinput":
            self.main = OccAntChandrakarInput(cfg)
        else:
            raise ValueError(f"Invalid model_type {model_type}")

        cfg.freeze()

        if cfg.use_radial_loss_mask:
            grid_indices = np.indices((self.config.bev_height, self.config.bev_width)).transpose(1,2,0)
            grid_dist = grid_indices - np.array([self.config.bev_height, self.config.bev_width//2])
            grid_normalized_dist = np.linalg.norm(grid_dist, axis=2) / np.sqrt(self.config.bev_height**2 + (self.config.bev_width//2)**2)

            self.loss_mask = torch.from_numpy(1 - grid_normalized_dist).float()
        else:
            self.loss_mask = torch.ones((self.config.bev_height, self.config.bev_width)).float()
        
        # bce_weights = torch.ones((2, self.config.bev_height, self.config.bev_width))
        # bce_weights[0, ...] *= 4

        self.bev_bce_loss = lambda x, y: (nn.BCEWithLogitsLoss(reduction='none')(x, y) * self.loss_mask.to(x.device))

    def forward(self, inputs, features):
        imgL = inputs[("color_aug", 'l', 0)]
        imgL = imgL.reshape((-1, *imgL.shape[2:]))
        
        x = {
            "rgb": imgL
        }

        if ("semantics_gt", "l") in inputs:
            semantics = inputs[("semantics_gt", "l")]
            semantics = semantics.reshape((-1, *semantics.shape[2:]))

            x["semantics"] = semantics

        if ("inv_K", 0) in inputs:
            inv_K = inputs[("inv_K", 0)]
            x["inv_K"] = inv_K

        if ("depth_gt", "l") in inputs:
            depth = inputs[("depth_gt", "l")]
            depth = depth.reshape((-1, 1, *depth.shape[2:]))

            x["depth"] = depth

        if ("ego_map_gt", "l") in inputs:
            ego_map_gt = inputs[("ego_map_gt", "l")]
            ego_map_gt = ego_map_gt.reshape((-1, *ego_map_gt.shape[2:]))

            x["ego_map_gt"] = ego_map_gt

        outputs = self.main(x)

        for k in ["occ_estimate", "depth_proj_estimate"]:
            if k not in outputs:
                continue
            outputs[k] = F.interpolate(outputs[k], self.bev_size, mode='area')

        outputs[("bev", "l", 0)] = outputs["occ_estimate"]

        # if self.config.GP_ANTICIPATION.grad_cam == True:
        #     with torch.enable_grad():
        #         model = SegmentationModelOutputWrapper(self.main)
        #         target_layers = model.model.bottleneck
        #         for sem_idx, sem_class in enumerate(["unknown", "occupied", "free"]):
        #             with GradCAM(model=model,
        #                         target_layers=target_layers,
        #                         use_cuda=torch.cuda.is_available()) as cam:
        #                 grayscale_cam = cam(input_tensor=x["rgb"], targets=[SemanticSegmentationTarget(sem_idx)])
        #                 rgb_cam = []
        #                 for idx in range(grayscale_cam.shape[0]):
        #                     rgb = torch.clamp(invnormalize_imagenet(x["rgb"][idx]), min=0, max=1).cpu().detach().numpy().transpose(1,2,0)
        #                     rgb_cam.append(show_cam_on_image(rgb, grayscale_cam[idx], use_rgb=True))
                        
        #                 outputs[("rgb_cam", sem_class, 0)] = np.array(rgb_cam).transpose(0, 3, 1, 2)
        #                 outputs[("grayscale_cam", sem_class, 0)] = np.expand_dims(grayscale_cam, axis=1)

        return outputs


    def compute_loss(self, inputs, outputs, *args, **kwargs):
        bev_losses = 0

        pred = outputs[("bev", 'l', 0)]
        pred = pred.reshape((-1, 2, *self.bev_size)) # Occupied, Explored

        target = inputs[("bev_gt", 'l')]
        target = target.reshape((-1, *self.bev_size))

        target_prob = torch.zeros_like(pred, device=pred.device, dtype=torch.float32, requires_grad=False)
        target_prob[:, 0, ...] = (target == 1) * 1.0  # Occupied
        target_prob[:, 1, ...] = (target != 0) * 1.0  # Explored

        # bev_losses +=  self.bev_bce_loss(pred, target_prob).mean()
        
        bev_occupied_loss = self.bev_bce_loss(pred[:, 0], target_prob[:, 0]).mean()
        bev_explored_loss = self.bev_bce_loss(pred[:, 1], target_prob[:, 1]).mean()
        
        bev_losses = (3 * bev_occupied_loss + bev_explored_loss) / 4

        if "depth_proj_estimate" in outputs:
            pred = outputs["depth_proj_estimate"]
            pred = pred.reshape((-1, *pred.shape[2:]))

            target = inputs[("ego_map_gt", 'l')]
            target = target.reshape((-1, *target.shape[2:]))

            bev_losses += self.bev_bce_loss(pred, target).mean()

        
        for k, v in outputs.items():
            if "semantics_pred" in k:
                pred = v
                pred = pred.reshape((-1, *pred.shape[2:]))

                target = inputs["semantics_gt", "l"]
                target = target.reshape((-1, *target.shape[2:]))

                bev_losses += self.bev_bce_loss(pred, target).mean()

        # target_logits = torch.log((target_prob + 1e-7)/(1 - target_prob + 1e-7))
        # outputs[("bev", 'l', 0)] = target_logits.reshape(outputs[("bev", 'l', 0)].shape)


        return {"bev_loss": bev_losses, "bev_occ_loss":bev_occupied_loss, "bev_exp_loss":bev_explored_loss}


    def get_params(self, opt):

        bev_decoder_dict = {
            'name': 'bev_decoder',
            'params': self.main.parameters()
        }
        if hasattr(opt, 'decoder_lr'):
            bev_decoder_dict['lr'] = opt.decoder_lr

        model_params = [bev_decoder_dict]
        return model_params


class SemanticSegmentationTarget:
    def __init__(self, category):
        self.category = category
        
    def __call__(self, model_output):
        output = F.sigmoid(model_output)
        occ_p = output[:, 0] * output[:, 1]
        free_p = (1 - output[:, 0]) * output[:, 1]
        unknown_p = 1 - output[:, 1]
        if self.category == 0:
            mask = (unknown_p >= 0.5)
            return (unknown_p * mask).sum()
        if self.category == 1:
            mask = (occ_p >= 0.5)
            return (occ_p * mask).sum()
        elif self.category == 2:
            mask = (free_p >= 0.5)
            return (free_p * mask).sum()

        raise Exception(f"Invalid category provided {self.category}")


class SegmentationModelOutputWrapper(nn.Module):
    def __init__(self, model): 
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        return [self.model({"rgb": x})["occ_estimate"]]