# This is a modified version of the monodepth2 dataset class

# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

# Important URLS:
# https://github.com/alexkreimer/odometry/tree/master/devkit

# Important archives:
# data_odometry_poses.zip, data_odometry_velodyne.zip, kitti_odom_calib.zip, data_odometry_color.zip

from __future__ import absolute_import, division, print_function

import os
import PIL
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

import skimage.transform
from .kitti_utils import generate_depth_map

from .extract_svo_point import PixelSelector

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
    topview = topview.resize((w, h), pil.NEAREST)
    topview = np.array(topview)
    return topview

odom_to_depth_map = {
    '00': 'train/2011_10_03_drive_0027_sync/proj_depth/groundtruth',
    '01': 'train/2011_10_03_drive_0042_sync/proj_depth/groundtruth',
    '02': 'train/2011_10_03_drive_0034_sync/proj_depth/groundtruth',
    '05': 'train/2011_09_30_drive_0018_sync/proj_depth/groundtruth',
    '06': 'train/2011_09_30_drive_0020_sync/proj_depth/groundtruth',
    '07': 'train/2011_09_30_drive_0027_sync/proj_depth/groundtruth',
    '08': 'train/2011_09_30_drive_0028_sync/proj_depth/groundtruth',
    '09': 'train/2011_09_30_drive_0033_sync/proj_depth/groundtruth',
    '10': 'train/2011_09_30_drive_0034_sync/proj_depth/groundtruth'
 }


class KittiOdometry(data.Dataset):
    """Dataset class for habitat

    Args:
        opt
        filenames
        is_train
        load_keys
    """
    def __init__(self, opt, filenames, is_train, load_keys):
        super(KittiOdometry, self).__init__()

        self.opt = opt
        self.filenames = filenames
        self.is_train = is_train
        self.load_keys = load_keys
        self.dataset_keys = ["data_path",
            "height", "width", "frame_ids", "scales",
            "bev_width", "bev_height", "bev_res", "floor_path",
            "color_dir", "depth_dir", "bev_dir", "pose_dir", "semantics_dir"]
        
        for k in self.dataset_keys:
            setattr(self, k, self.opt.get(k, None))

        if self.color_dir is None:
            self.color_dir = os.path.join(self.data_path, "dataset", "sequences")
        if self.depth_dir is None:
            self.depth_dir = os.path.join(self.data_path, "dataset", "sequences")
        if self.pose_dir is None:
            self.pose_dir = os.path.join(self.data_path, "dataset", "poses")

        self.interp = Image.ANTIALIAS
        self.loader = pil_loader
        self.pixelselector = PixelSelector()

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**imagenet_stats)
        ])

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)

        self.T_c2 = np.array([
            [ 1.        ,  0.        ,  0.        , -0.06103058],
            [ 0.        ,  1.        ,  0.        ,  0.00143972],
            [ 0.        ,  0.        ,  1.        , -0.00620322],
            [ 0.        ,  0.        ,  0.        ,  1.        ]
        ])

        self.T_c3 = np.array([
            [ 1.        ,  0.        ,  0.        ,  0.47441838],
            [ 0.        ,  1.        ,  0.        , -0.00187031],
            [ 0.        ,  0.        ,  1.        , -0.0033185 ],
            [ 0.        ,  0.        ,  0.        ,  1.        ]
        ])

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        self.num_scales = len(self.scales)
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()
        self.load_pose = self.check_pose()
        self.perspective_intrinsics = perspective_camera_intrinsics(720, 620, 720, 187)

        self.bev_intrinsics = orthographic_camera_intrinsics(self.bev_res, \
            self.bev_width//2, self.bev_res, self.bev_height//2)

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
        
        for key in inputs.keys():
            if "discr" in key:
                inputs[key] = process_topview(inputs[key], self.bev_width, self.bev_height)
            if "bev_gt" in key:
                for tstep, img in enumerate(inputs[key]):
                    inputs[key][tstep] = process_topview(img, self.bev_width, self.bev_height)

                    # To bring it from 0-255 to 0-2
                    inputs[key][tstep] = torch.tensor(inputs[key][tstep] // 127, 
                        dtype=torch.int64)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = False #self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        frame_index = int(line[1])

        for side in ['l', 'r']:
            inputs[("color", side, -1)] = []
            for i in self.frame_ids:
                inputs[("color", side, -1)].append(self.get_color(folder, frame_index + i, side, do_flip))

        # Shape - T x BH x BW
        if 'bev' in self.load_keys:
            side = 'l'
            inputs[("bev_gt", side)] = [self.get_bev(folder, frame_index+i, side, do_flip) for i in self.frame_ids]

        if 'discr' in self.load_keys and self.floor_path is not None:
            inputs["discr"] = self.get_floor()

        if 'pose' in self.load_keys:
            pose_tgt = self.get_pose(folder, frame_index, 'l', do_flip)
            for side in ['l', 'r']:
                inputs[("relpose_gt", side)] = []
                for i in self.frame_ids[1:]:
                    pose_src = self.get_pose(folder, frame_index +i, side, do_flip)
                    relpose_src_tgt = np.linalg.inv(pose_src) @ pose_tgt  # tgt to sr
                    inputs[("relpose_gt", side)].append(torch.from_numpy(relpose_src_tgt))

        if 'semantics' in self.load_keys:
            side = 'l'
            inputs[("semantics_gt", side)] = []
            for i in self.frame_ids:
                semantics =  np.expand_dims(self.get_semantics(folder, frame_index, side, do_flip), 0)
                inputs[("semantics_gt", side)].append(torch.from_numpy(semantics.astype(np.float32)))

        if 'depth' in self.load_keys:
            for side in ['l', 'r']:
                inputs[("depth_gt", side)] = []
                for i in self.frame_ids:
                    inputs[("depth_gt", side)].append(torch.from_numpy(
                        self.generate_depth(folder, frame_index+i, side, do_flip)))

        if 'keypts' in self.load_keys:
            side = 'l'
            svo_map_resized = np.zeros((self.height, self.width))
            img = np.array(inputs[("color", side, -1)][0])
            key_points = self.pixelselector.extract_points(img)
            key_points = key_points.astype(int)
            key_points[:, 0] = key_points[:, 0] * self.height // 480
            key_points[:, 1] = key_points[:, 1] * self.width // 640

            # noise 1000 points
            noise_num = 3000 - key_points.shape[0]
            noise_points = np.zeros((noise_num, 2), dtype=np.int32)
            noise_points[:, 0] = np.random.randint(self.height, size=noise_num)
            noise_points[:, 1] = np.random.randint(self.width, size=noise_num)

            svo_map_resized[key_points[:, 0], key_points[:, 1]] = 1

            inputs[('svo_map', side)] = torch.from_numpy(svo_map_resized.copy())

            svo_map_resized[noise_points[:, 0], noise_points[:, 1]] = 1
            inputs[('svo_map_noise', side)] = torch.from_numpy(
                svo_map_resized,
            ).float()

            keypoints = np.concatenate((key_points, noise_points), axis=0)
            inputs[('dso_points', side)] = torch.from_numpy(
                keypoints,
            ).float()


        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for side in ['l', 'r']:
            del inputs[("color", side, -1)]
            del inputs[("color_aug", side, -1)]

        for k, v in inputs.items():
            if isinstance(v, list):
                inputs[k] = torch.stack(v)

        inputs["frame"] = torch.tensor(index)
        inputs["filename"] = self.filenames[index]

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):

        if side == 'r':
            side = '3'
        else:
            side = '2'

        color_dir = os.path.join(self.color_dir, folder)
        color_path = os.path.join(color_dir, f'image_{side}', '{:06d}.png'.format(frame_index))
        color = self.loader(color_path).resize(self.full_res_shape, pil.NEAREST)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def check_depth_old(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        depth_img = os.path.join(
            self.depth_dir,
            odom_to_depth_map[scene_name],
            "image_02",
            "{:010d}.png".format(int(frame_index)))

        return os.path.isfile(depth_img)
   
    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.depth_dir,
            scene_name,
            "velodyne/{:06d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def load_all_poses(self):
        self.poses = dict()
        for scene_filename in os.listdir(self.pose_dir):
            scene = os.path.splitext(scene_filename)[0]
            pose_file = os.path.join(self.pose_dir, scene_filename)
            poses = []
            with open(pose_file, 'r') as f:
                lines = f.readlines()

                for line in lines:
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                    T_w_cam0 = T_w_cam0.reshape(3, 4)
                    T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    poses.append(T_w_cam0)
            
            self.poses[scene] = np.array(poses)

    def check_pose(self):
        self.load_all_poses()
        return len(getattr(self, "poses", {})) > 0
   
    def get_depth(self, folder, frame_index, side, do_flip):
        if side == 'r':
            side = '3'
        else:
            side = '2'

        scene = odom_to_depth_map[folder]
        folder = os.path.join(self.depth_dir, scene)

        depth_path = os.path.join(folder, f"image_0{side}", "{:010d}.png".format(frame_index))
        depth = self.loader(depth_path).resize((self.width, self.height), pil.NEAREST)
        depth = np.array(depth, dtype=np.float32)/256

        if do_flip:
            depth = np.fliplr(depth)

        return depth

    def generate_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.depth_dir, folder)

        velo_filename = os.path.join(
            self.depth_dir,
            folder,
            "velodyne/{:06d}.bin".format(int(frame_index)))

        if side == 'r':
            side = 3
        else:
            side = 2

        depth_gt = generate_depth_map(calib_path, velo_filename, side)
        depth_gt = skimage.transform.resize(
            depth_gt, (self.height, self.width), order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        depth_gt = depth_gt.astype(np.float32)

        return depth_gt

    def get_semantics(self, folder, frame_index, side, do_flip):
        semantics = np.zeros((self.height, self.width), dtype=np.int64)
        return semantics

    def get_pose(self, folder, frame_index, side, do_flip):
        # Refer to photometric_reconstruction notebook.
        
        if side == 'r':
            cam_to_car = self.T_c3
        else:
            cam_to_car = self.T_c2

        car_to_world = self.poses[folder][int(frame_index)]
        
        M = (car_to_world @ cam_to_car).astype(np.float32)

        # The images will already be locally flipped. 
        # We need to only flip the camera's global x-coordinate.
        # Refer to registration_notebook.
        M[0,3] *= (1 - 2*do_flip)

        return M

    def get_bev(self, folder, frame_index, side, do_flip):
        bev = np.zeros((self.bev_height, self.bev_width), dtype=np.int64)
        bev = pil.fromarray(bev, mode='L')
        return bev

    def generate_bev_old(self, semantics, depth):

        resz = 3 / 120.0
        resx = 3 / 120.0
        xmin = -1.5  
        xmax = 1.5
        zmin = 0.1
        zmax = 3
        rows = 120
        cols = 120

        pts = img_to_lid(depth, self.bev_intrinsics, semantics)
        occupancy_map = np.zeros((rows, cols))
        flag1 = pts[:, 0] > xmin
        flag2 = pts[:, 0] < xmax
        flag3 = pts[:, 2] > zmin
        flag4 = pts[:, 2] < zmax
        flag = flag1 * flag2 * flag3 * flag4
        curr_frame_pts = pts[flag, :].T

        plane_topview = np.zeros((2, np.shape(curr_frame_pts)[1]))
        plane_topview[0, :] = (xmax + curr_frame_pts[0, :]) / resx
        plane_topview[1, :] = (zmax - curr_frame_pts[2, :]) / resz
        plane_topview = np.array(plane_topview, dtype=np.int32)
        occupancy_map[plane_topview[1, :], plane_topview[0, :]] = 1

        return occupancy_map

    def generate_bev(self, folder, frame_index, side, do_flip):
        depth = self.get_depth(folder, frame_index, side, do_flip)
        semantics = self.get_semantics(folder, frame_index, side, do_flip)

        h, w = depth.shape
        pc = img_to_lid(depth, self.perspective_intrinsics)
        X = np.concatenate([pc, np.ones((h*w, 1))], axis=1)
        
        bw, bh = self.bev_width, self.bev_height
        res = self.bev_res
        
        Kb = np.eye(4)
        Kb[0, 0] = 1/res
        Kb[1, 1] = 1/res
        Kb[0, 3] = bw/2
        Kb[1, 3] = bh/2

        Tb = np.eye(4)
        Tb[:3, 3] = [0, 0, -(res*bh + 0.1)/2]  # to move to center of grid of range 0.1 to bh*res along z.

        Rb = np.eye(4)
        Rb[:3, :3] = Rotation.from_rotvec([np.pi/2, 0, 0]).as_matrix()

        Xb = (Kb @ Rb @ Tb @ X.T).T

        valid_idx = ((Xb[:, 1] >= 0) & (Xb[:, 1] < bh) & 
                    (Xb[:, 0] >= 0) & (Xb[:, 0] < bw) &
                    (Xb[:, 2] >= 0))

        u = Xb[:, 0][valid_idx].astype(np.uint16)
        v = Xb[:, 1][valid_idx].astype(np.uint16)
        
        grid_sem = (semantics.reshape(-1)[valid_idx] == 2) + 1 # 0-empty, 1-occ, 2-free

        grid_idx = v * bw + u

        grid = np.zeros((bh, bw), dtype=np.int64).reshape(-1)
        grid[grid_idx] = grid_sem

        grid = grid.reshape((bh, bw))
        
        return grid

    def get_floor(self):
        map_file = np.random.choice(os.listdir(self.floor_path))
        map_path = os.path.join(self.floor_path, map_file)
        osm = self.loader(map_path)
        return osm

if __name__ == '__main__':
    opt = dict()
    opt["data_path"] = '/scratch/shantanu/kitti'
    opt["height"] = 192 
    opt["width"] = 640
    opt["frame_ids"] = [0, -1, 1]
    opt["scales"] = [0]
    opt["bev_width"] = 120
    opt["bev_height"] = 120
    opt["bev_res"] = 0.025
    
    opt["depth_dir"] = None
    opt["bev_dir"] = None

    split_path = 'splits/kitti/kitti_00.txt'
    with open(split_path, 'r') as f:
        filenames = f.read().splitlines()

    is_train = True
    load_keys = ['bev', 'depth', 'keypts', 'semantics', 'pose']

    dataset = KittiOdometry(opt, filenames, is_train, load_keys)
    dl = data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, \
        drop_last=False)

    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )

    for tmp in dl:
        tmp_dir = '/tmp/indoor_layout_estimation/kitti_dataset'
        os.makedirs(tmp_dir, exist_ok=True)
        for idx, filepath in enumerate(tmp['filename']):
            folder, fileidx = filepath.split()
            img_path = os.path.join(opt['data_path'], 'dataset', 'sequences', folder, 'image_2', '{:06d}.png'.format(int(fileidx)))
            org_img = cv2.imread(img_path, 1)
            org_img_path = os.path.join(tmp_dir, '{}_{}_org.jpg'.format(folder, fileidx))
            cv2.imwrite(org_img_path, org_img)
            
            color = tmp[('color_aug', 'l', 0)][idx, 0, ...]
            conv_img = (inv_normalize(color.cpu().detach()).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            conv_img_path = os.path.join(tmp_dir, '{}_{}_conv.jpg'.format(folder, fileidx))
            cv2.imwrite(conv_img_path, cv2.cvtColor(conv_img, cv2.COLOR_RGB2BGR))
        break
