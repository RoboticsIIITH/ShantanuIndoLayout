# This is a modified version of the monodepth2 dataset class

# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import cv2
import random
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt

from PIL import Image  # using pillow-simd for increased speed

import torch
from torch.nn.functional import fold
import torch.utils.data as data
from torchvision import transforms

from scipy.spatial.transform import Rotation
import quaternion

from networks.occant_baselines.depthsensor import DepthProjector

import albumentations as A

imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img.load()
            return img

def cv2_loader(path):
    return pil.fromarray(cv2.imread(path, -1))

def perspective_camera_intrinsics(f_x, c_x, f_y, c_y):
    camera_intrinsics = np.zeros((3,4))
    camera_intrinsics[0][0] = f_x
    camera_intrinsics[1][1] = f_y
    camera_intrinsics[2][2] = 1
    camera_intrinsics[0][2] = c_x
    camera_intrinsics[1][2] = c_y
    
    return camera_intrinsics

def orthographic_camera_intrinsics(f_x, c_x, f_y, c_y):
    camera_intrinsics = np.eye(4)
    camera_intrinsics[0][0] = f_x
    camera_intrinsics[1][1] = f_y
    camera_intrinsics[0][3] = c_x
    camera_intrinsics[1][3] = c_y
    
    return camera_intrinsics

def img_to_rect( u, v, depth_rect, P2):
    
    cu = P2[0, 2]
    cv = P2[1, 2]
    fu = P2[0, 0]
    fv = P2[1, 1]

    x = ((u - cu) * depth_rect) / fu
    y = ((v - cv) * depth_rect) / fv
    pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
    return pts_rect

def img_to_lid(depth_map, cam_mat, label=None):

    x_range = np.arange(0, depth_map.shape[1])
    y_range = np.arange(0, depth_map.shape[0])
    x_idxs, y_idxs = np.meshgrid(x_range, y_range)
    x_idxs, y_idxs = x_idxs.reshape(-1), y_idxs.reshape(-1)

    depth = depth_map[y_idxs, x_idxs]
    pts_rect = img_to_rect(x_idxs, y_idxs, depth, cam_mat)
    
    if label is not None:
        label_intensity = label[y_idxs, x_idxs]
        filt = label_intensity == 2
        pts_rect = pts_rect[filt]

    return pts_rect

def process_topview(topview, w, h):
    # topview = topview.crop((13, 27, 115, 128)) # To crop out the bottom center 5.05x5.05 m map from 6.4x6.4m map
    topview = topview.crop((32, 64, 96, 128)) # To crop out the bottom center 3.2x3.2 m map from 6.4x6.4m map
    topview = topview.resize((w, h), pil.NEAREST)
    topview = np.array(topview)
    return topview

class Gibson4Dataset(data.Dataset):
    """Dataset class for habitat

    Args:
        opt
        filenames
        is_train
        load_keys
    """
    def __init__(self, opt, filenames, is_train, load_keys):
        super(Gibson4Dataset, self).__init__()

        self.opt = opt
        self.filenames = filenames
        self.is_train = is_train
        self.load_keys = load_keys
        self.dataset_keys = ["data_path", "ego_map_dir",
            "height", "width", "frame_ids", "scales",
            "baseline", "cam_height",
            "bev_width", "bev_height", "bev_res", "floor_path"]
        
        for k in self.dataset_keys:
            setattr(self, k, self.opt.get(k, None))

        for k in ["color_dir", "depth_dir", "bev_dir", "pose_dir", "semantics_dir"]:
            setattr(self, k, self.data_path if self.opt.get(k) is None else self.opt.get(k))

        self.interp = Image.ANTIALIAS
        self.loader = pil_loader

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**imagenet_stats),
            transforms.CenterCrop(self.height)
        ])

        self.crop_img = transforms.CenterCrop(self.height)

        full_res_shape = (1024, 1024)
        focal_length = full_res_shape[0]/ (2 * np.tan(np.deg2rad(self.opt.hfov/2)))

        self.width_ar = (full_res_shape[0] * self.height) // full_res_shape[1]
        self.focal_length = focal_length * self.width_ar / full_res_shape[0]  # Scaling based on given input shape

        print("Focal length:", self.focal_length)

        # Since we are cropping, the field of view changes, but the focal length remains the same.
        # The cropping is equal on both sides, so (cx, cy) are always at image center.
        self.K = np.array([[self.focal_length/self.width_ar, 0, 0.5, 0],
                           [0, self.focal_length/self.height, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        self.num_scales = len(self.scales)
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width_ar // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()
        self.load_pose = self.check_pose()
        # self.perspective_intrinsics = perspective_camera_intrinsics(320, 320, 320, 240)

        # self.bev_intrinsics = orthographic_camera_intrinsics(self.bev_res, \
        #     self.bev_width//2, self.bev_res, self.bev_height//2)

        if 'ego_map_gt' in self.load_keys:
            self.depth_projector = DepthProjector(self.opt)

        self.bev_transform = A.Compose([
            A.Resize(height=self.bev_height, width=self.bev_width, interpolation=cv2.INTER_NEAREST, always_apply=True),
        ])

        self.ego_map_transform = A.Compose([
            # A.Resize(height=self.height, width=self.width_ar, interpolation=cv2.INTER_NEAREST, always_apply=True), # If input is depth
            # A.CenterCrop(height=self.height, width=self.width, always_apply=True)
            A.Resize(height=self.bev_height, width=self.bev_width, interpolation=cv2.INTER_NEAREST, always_apply=True),
            # A.augmentations.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=4, min_height=16, min_width=16, p=0.5)
            # A.augmentations.geometric.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=0, \
            #     interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.8)
            # A.augmentations.GridDistortion(num_steps=7, distort_limit=0.5, 
            #                            interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, 
            #                            value=0, p=0.5)
            # A.augmentations.Affine(scale=[0.9, 1.1], translate_percent=[-0.1, 0.1], rotate=[-10,10], shear=(-10, 10), 
            #                   interpolation=cv2.INTER_NEAREST, cval=0, p=0.5)
            # A.transforms.GaussNoise(p=0.5)
            # A.transforms.MultiplicativeNoise(elementwise=True, p=0.5)
        ])

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            if "color" in k:
                n, side, i = k
                for i in range(self.num_scales):
                    inputs[(n, side, i)] = []
                    for img in inputs[(n, side, i-1)]:
                        inputs[(n, side, i)].append(self.resize[i](img))

        for k in list(inputs):
            if "color" in k:
                n, side, i = k
                inputs[(n + "_aug", side, i)] = []
                for tstep, img in enumerate(inputs[k]):
                    inputs[(n, side, i)][tstep] = self.normalize(img)
                    inputs[(n + "_aug", side, i)].append(self.normalize(color_aug(img)))

            if "depth_gt" in k:
                for tstep, img in enumerate(inputs[k]):
                    inputs[k][tstep] = self.crop_img(img)

            if "semantics_gt" in k:
                for tstep, img in enumerate(inputs[k]):
                    inputs[k][tstep] = self.crop_img(img)

        
        for key in inputs.keys():
            if "discr" in key:
                for tstep, img in enumerate(inputs[key]):
                    inputs[key][tstep] = process_topview(img, self.bev_width, self.bev_height)
                    # To bring it from 0-255 to 0-2
                    inputs[key][tstep] = torch.tensor(inputs[key][tstep] // 127, dtype=torch.int64)
                
            if "bev_gt" in key:
                for tstep, img in enumerate(inputs[key]):
                    inputs[key][tstep] = process_topview(img, self.bev_width, self.bev_height)
                    inputs[key][tstep] = torch.tensor(inputs[key][tstep] // 127, dtype=torch.int64) # To bring it from 0-255 to 0-2

                    # inputs[key][tstep] = self.bev_transform(image=img)['image']
                    # inputs[key][tstep] = torch.tensor(inputs[key][tstep], dtype=torch.int64)
            
            if "ego_map_gt" in key:
                for tstep, img in enumerate(inputs[key]):
                    inputs[key][tstep] = np.transpose(self.ego_map_transform(image=img)['image'], (2, 0, 1))
                    inputs[key][tstep] = torch.tensor(inputs[key][tstep], dtype=torch.float32)
                    # tmp = np.zeros((2, *inputs[key][tstep].shape[1:]))
                    # tmp[0, inputs[key][tstep][0] != 0] = 1
                    # tmp[1, ...] = inputs[key][tstep][0]
                    # inputs[key][tstep] = torch.tensor(tmp, dtype=torch.float32)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]
        camera_pose = line[1]
        frame_index = int(line[2])

        inputs[("color", 'l', -1)] = []
        for i in self.frame_ids:
            inputs[("color", 'l', -1)].append(self.get_color(folder, frame_index + i, camera_pose, do_flip))

        # Shape - T x BH x BW
        if 'bev' in self.load_keys:
            inputs[("bev_gt", 'l')] = [self.get_bev(folder, frame_index+i, camera_pose, do_flip) for i in self.frame_ids]

        # Project Depth to BEV based on height thresholding. (For OccAnt Models)
        if 'ego_map_gt' in self.load_keys:
            if self.ego_map_dir is not None:
                ego_map_fn = self.read_ego_map_gt
            else:
                ego_map_fn = self.get_ego_map_gt
            inputs[("ego_map_gt", 'l')] = [ego_map_fn(folder, frame_index+i, camera_pose, do_flip) for i in self.frame_ids]

        if 'discr' in self.load_keys and self.floor_path is not None:
            inputs["discr"] = [self.get_floor() for i in self.frame_ids]

        if 'pose' in self.load_keys and len(self.frame_ids) > 1:
            pose_tgt = self.get_pose(folder, frame_index, camera_pose, do_flip)
            inputs[("relpose_gt", 'l')] = []
            for i in self.frame_ids[1:]:
                pose_src = self.get_pose(folder, frame_index +i, camera_pose, do_flip)
                relpose_src_tgt = np.linalg.inv(pose_src) @ pose_tgt  # tgt to sr
                inputs[("relpose_gt", 'l')].append(torch.from_numpy(relpose_src_tgt))

        if 'semantics' in self.load_keys:
            inputs[("semantics_gt", 'l')] = []
            for i in self.frame_ids:
                semantics =  np.expand_dims(self.get_semantics(folder, frame_index + i, camera_pose, do_flip), 0)
                inputs[("semantics_gt", 'l')].append(torch.from_numpy(semantics.astype(np.float32)))

        if 'depth' in self.load_keys:
            inputs[("depth_gt", 'l')] = []
            for i in self.frame_ids:
                depth = torch.from_numpy(self.get_depth(folder, frame_index+i, camera_pose, do_flip))
                inputs[("depth_gt", 'l')].append(depth)


        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            # the focal length is dependent on camera hardware, but the center is dependent on image res (half of it to be at center).

            K[0, 0] *= self.width_ar // (2 ** scale)
            K[1, 1] *= self.height // (2 ** scale)

            # Since image is cropped from (w, h) to (h, h)
            K[0, 2] *= self.height // (2 ** scale)
            K[1, 2] *= self.height // (2 ** scale)


            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        ## TO TRAIN ANS Model
        # if self.is_train:
        #     ego_map_1ch = []
        #     for map in inputs[("ego_map_gt", 'l')]:
        #         map = map.permute(1,2,0).reshape((-1, 2)).int()
        #         ego_map_gt = torch.zeros((self.bev_height * self.bev_width)).float()
        #         ego_map_gt[(map[:, 0] == 1) & (map[:, 1] == 1)] = 1  # occupied
        #         ego_map_gt[(map[:, 0] == 0) & (map[:, 1] == 1)] = 2  # explored
        #         ego_map_gt = ego_map_gt.reshape((self.bev_height, self.bev_width))

        #         ego_map_1ch.append(ego_map_gt)
            
        #     inputs[("bev_gt", 'l')] = ego_map_1ch

        del inputs[("color", 'l', -1)]
        del inputs[("color_aug", 'l', -1)]

        for k, v in inputs.items():
            if isinstance(v, list):
                inputs[k] = torch.stack(v)

        inputs["frame"] = torch.tensor(index)
        inputs["filename"] = self.filenames[index]

        return inputs

    def get_color(self, folder, frame_index, camera_pose, do_flip):

        color_dir = os.path.join(self.color_dir, folder)
        color_path = os.path.join(color_dir, "0", camera_pose, "RGB", str(frame_index) + ".jpg")
        color = self.loader(color_path)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        camera_pose = line[1]
        frame_index = int(line[2])

        depth_img = os.path.join(
            self.depth_dir,
            scene_name,
            "0",
            camera_pose,
            "DEPTH",
            "{}.png".format(int(frame_index)))

        return os.path.isfile(depth_img)

    def check_pose(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        camera_pose = line[1]
        frame_index = int(line[2])

        pose_file = os.path.join(
            self.pose_dir,
            scene_name,
            "0",
            camera_pose,
            "pose",
            "{}.npy".format(int(frame_index)))

        return os.path.isfile(pose_file)
        
    def get_depth(self, folder, frame_index, camera_pose, do_flip):
        folder = os.path.join(self.depth_dir, folder)

        depth_path = os.path.join(folder, "0", camera_pose, "DEPTH", str(frame_index) + ".png")
        depth = self.loader(depth_path).resize((self.width_ar, self.height), pil.NEAREST)

        if do_flip:
            depth = depth.transpose(pil.FLIP_LEFT_RIGHT)

        depth = np.array(depth).astype(np.float32)/6553.5  # Nearest to maintain edge sharpness)

        return depth

    def get_semantics(self, folder, frame_index, camera_pose, do_flip, raw=False):
        sem_dir = os.path.join(self.semantics_dir, folder)
        sem_path = os.path.join(sem_dir, "0", camera_pose, "semantics", str(frame_index) + ".png")
        semantics = cv2.imread(sem_path, -1)
        semantics = cv2.resize(semantics, (self.width_ar, self.height), interpolation=cv2.INTER_NEAREST)

        if do_flip:
            semantics = np.fliplr(semantics)

        if not raw:
            semantics = np.logical_or(semantics==3, semantics==28) + 0 # 0 - occupied, 1 - free

        return semantics.copy()

    def get_pose(self, folder, frame_index, camera_pose, do_flip):
        # Refer to photometric_reconstruction notebook.
        
        cam_to_agent = np.eye(4)
        cam_to_agent[1,1] = -1  # Flip the y-axis of the point-cloud to be pointing upwards
        cam_to_agent[2,2] = -1  # Flip the z-axis of the point-cloud to follow right-handed coordinate system.

        pose_dir = os.path.join(self.pose_dir, folder)
        pose_path = os.path.join(pose_dir, "0", camera_pose, "pose", str(frame_index) + ".npy")
        agent_pose = np.load(pose_path, allow_pickle=True).item()
    
        rot = Rotation.from_quat([agent_pose['rotation'].x, agent_pose['rotation'].y, 
                                agent_pose['rotation'].z, agent_pose['rotation'].w])
        R = np.eye(4)
        R[:3, :3] = rot.as_matrix()

        T = np.eye(4)
        T[:3, 3] = agent_pose['position']
        
        M = (T @ R @ cam_to_agent).astype(np.float32)

        # The images will already be locally flipped. 
        # We need to only flip the camera's global x-coordinate.
        # Refer to registration_notebook.
        M[0,3] *= (1 - 2*do_flip)

        return M
    
    def get_ego_map_gt(self, folder, frame_index, camera_pose, do_flip):
        depth = self.get_depth(folder, frame_index, camera_pose, do_flip)[:self.height, (self.width_ar - self.height)//2 : (self.width_ar + self.height)//2]
        map = self.depth_projector.get_depth_projection(depth)

        # Convert from 2-channel(explored, occ) to 3-value 1ch (unknown, occ, free) map
        # map = map.reshape((-1, 2))
        # ego_map_gt = np.zeros((self.bev_height * self.bev_width), dtype=np.uint8)
        # ego_map_gt[map[:, 0]] = 0
        # ego_map_gt[np.logical_and(map[:, 0] == 1, map[:, 1] == 1)] = 1  # occupied
        # ego_map_gt[np.logical_and(map[:, 0] == 0, map[:, 1] == 1)] = 2  # explored
        # ego_map_gt = ego_map_gt.reshape((self.bev_height, self.bev_width, 1))

        ego_map_gt = map.astype(np.uint8)
        # ego_map_gt[..., 1] = ego_map_gt[..., 0]    # Quick hack to remove the free information from the explored channel. Occupied already doesn't have it.

        # # Hack to get free only map from ego_map
        # tmp = np.zeros((*ego_map_gt.shape[:2], 2), dtype=np.float32)
        # tmp[..., 1] = np.logical_and(ego_map_gt[..., 0] == 0, ego_map_gt[..., 1] == 1) * 1.0
        # ego_map_gt = tmp

        return ego_map_gt

    def read_ego_map_gt(self, folder, frame_index, camera_pose, do_flip):
        folder = os.path.join(self.ego_map_dir, folder)

        bev_path = os.path.join(folder, '0', camera_pose, 'pred_bev', str(frame_index) + ".png")
        bev = cv2.imread(bev_path, -1)

        if do_flip:
            bev = np.fliplr(bev)

        bev = bev // 127
        ego_map = np.zeros((*bev.shape, 2), dtype=np.float32)
        ego_map[bev == 1, 0] = 1  # Occupied 
        ego_map[np.logical_or(bev==1, bev==2), 1]= 1 # Explored

        # # Chandrakar depth
        # chandrakar_depth_path = os.path.join(folder, '0', camera_pose, 'pred_depth', str(frame_index) + ".png")
        # depth = cv2.imread(chandrakar_depth_path, -1)

        # if do_flip:
        #     depth = np.fliplr(depth)

        # # Raw continous depth with a mask
        # depth = depth/6553.5
        # ego_map = np.zeros((*depth.shape, 2), dtype=np.float32)
        # ego_map[depth!=0, 0] = 1
        # ego_map[..., 1] = depth

        # # Discretized depth 
        # num_channels = 128
        # depth = np.clip(depth * num_channels/10.0, a_min=0, a_max=num_channels-1).astype(np.uint16)
        # ego_map = np.zeros((depth.size, num_channels), dtype=np.float32)
        # ego_map[np.arange(depth.size), depth.reshape(-1)] = 1
        # ego_map = ego_map.reshape((*depth.shape, num_channels))

        # # GT Depth masked by floor segmentation (To understand if chandrakar's depth error is an issue)
        # depth = self.get_depth(folder, frame_index, camera_pose, do_flip)[:self.height, (self.width_ar - self.height)//2 : (self.width_ar + self.height)//2]
        # sem = self.get_semantics(folder, frame_index, camera_pose, do_flip)[:self.height, (self.width_ar - self.height)//2 : (self.width_ar + self.height)//2]

        # masked_depth = depth * sem
        # ego_map = np.zeros((*masked_depth.shape, 2), dtype=np.float32)
        # ego_map[masked_depth!=0, 0] = 1
        # ego_map[..., 0] = masked_depth

        return ego_map

    def get_bev(self, folder, frame_index, camera_pose, do_flip):
        folder = os.path.join(self.bev_dir, folder)

        bev_path = os.path.join(folder, camera_pose, "partial_occ", str(frame_index) + ".png")
        bev = self.loader(bev_path)

        if do_flip:
            bev = bev.transpose(pil.FLIP_LEFT_RIGHT)

        # bev = self.get_ego_map_gt(folder, frame_index, camera_pose, do_flip)
        return bev


    def get_floor(self):
        map_file = np.random.choice(os.listdir(self.floor_path))
        map_path = os.path.join(self.floor_path, map_file)
        osm = self.loader(map_path)
        return osm

if __name__ == '__main__':
    opt = dict()
    opt["data_path"] = '/scratch/jaidev/new'
    opt["height"] = 128
    opt["width"] = 128
    opt["frame_ids"] = [0]
    opt["scales"] = [0]
    opt["baseline"] = 0.2
    opt["cam_height"] = 1
    opt["focal_length"] = 64
    opt["bev_width"] = 64
    opt["bev_height"] = 64
    opt["bev_res"] = 0.05

    opt["depth_dir"] = None
    opt["bev_dir"] = '/scratch/jaidev/new_partialmaps_120'

    split_path = 'splits/gibson4/train_files.txt'
    with open(split_path, 'r') as f:
        filenames = f.read().splitlines()

    is_train = True
    load_keys = ['bev', 'depth', 'pose']

    dataset = Gibson4Dataset(opt, filenames, is_train, load_keys)
    dl = data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, \
        drop_last=False)

    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )

    for tmp in dl:
        tmp_dir = '/scratch/shantanu/tmp'
        os.makedirs(tmp_dir, exist_ok=True)
        for idx, filepath in enumerate(tmp['filename']):
            folder, camera_pose, fileidx = filepath.split()
            img_path = os.path.join(opt['data_path'], folder, '0', camera_pose, 'RGB', f'{fileidx}.jpg')
            org_img = cv2.imread(img_path, -1)
            org_img_path = os.path.join(tmp_dir, '{}_{}_{}_org.jpg'.format(folder.replace('/', '_'), camera_pose, fileidx))
            cv2.imwrite(org_img_path, org_img)
            
            color = tmp[('color_aug', 'l', 0)][idx, 0, ...]
            conv_img = (inv_normalize(color.cpu().detach()).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            conv_img_path = os.path.join(tmp_dir, '{}_{}_{}_conv.jpg'.format(folder.replace('/', '_'), camera_pose, fileidx))
            cv2.imwrite(conv_img_path, cv2.cvtColor(conv_img, cv2.COLOR_RGB2BGR))

            depth = tmp[('depth_gt', 'l')][idx, 0, ...].cpu().detach().numpy() 
            depth = (depth * 6553.5).astype(np.uint16)
            depth_path = os.path.join(tmp_dir, '{}_{}_{}_depth.png'.format(folder.replace('/', '_'), camera_pose, fileidx))
            cv2.imwrite(depth_path, depth)

            bev = tmp[('bev_gt', 'l')][idx, ...].cpu().detach().squeeze(0).numpy()
            bev = (bev * 127).astype(np.uint8)
            bev_path = os.path.join(tmp_dir, '{}_{}_{}_bev.png'.format(folder.replace('/', '_'), camera_pose, fileidx))
            cv2.imwrite(bev_path, bev)
        break
