import numpy as np
import cv2

class DepthProjector:
    r"""Estimates the top-down occupancy based on current depth-map.
    Args:
        config: contains the MAP_SCALE, MAP_SIZE, HEIGHT_THRESH fields to
                decide grid-size, extents of the projection, and the thresholds
                for determining obstacles and explored space.
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config= config
        # self.upscale_map = 1

        self.dilate_maps = self.config.get('dilate_maps', True)
        self.offset_baseline = self.config.get('offset_baseline', False)
        self.offset_camera_plane = self.config.get('offset_camera_plane', False)

        # Map statistics
        self.map_size = self.config.bev_width
        self.map_scale = 3.2 / self.config.bev_width
        # self.map_scale = 5.05 / self.config.bev_width
        assert np.isclose(self.map_scale, self.config.bev_res, atol=1e-5)
        
        self.max_forward_range = self.map_size * self.map_scale + (self.config.min_depth * self.offset_camera_plane)

        # Agent height for pointcloud tranforms
        self.camera_height = self.config.cam_height

        # Compute intrinsic matrix
        depth_H = self.config.height
        depth_W = self.config.width
        hfov = float(self.config.hfov) * np.pi / 180
        vfov = 2 * np.arctan((depth_H / depth_W) * np.tan(hfov / 2.0))
        self.intrinsic_matrix = np.array(
            [
                [1 / np.tan(hfov / 2.0), 0.0, 0.0, 0.0],
                [0.0, 1 / np.tan(vfov / 2.0), 0.0, 0.0],
                [0.0, 0.0, 1, 0],
                [0.0, 0.0, 0, 1],
            ]
        )
        self.inverse_intrinsic_matrix = np.linalg.inv(self.intrinsic_matrix)

        # Height thresholds for obstacles
        self.height_thresh = self.config.obstacle_height_thresholds

        # Depth processing
        self.min_depth = float(self.config.min_depth)
        self.max_depth = float(self.config.max_depth)

        # Pre-compute a grid of locations for depth projection
        W = self.config.width
        H = self.config.height
        self.proj_xs, self.proj_ys = np.meshgrid(
            np.linspace(-1, 1, W), np.linspace(1, -1, H)
        )


    def convert_to_pointcloud(self, depth_float):
        """
        Inputs:
            depth_float = (H, W) numpy array
        Returns:
            xyz_camera = (N, 3) numpy array for (X, Y, Z) in egocentric world coordinates
        """

        # =========== Convert to camera coordinates ============
        # W = depth.shape[1]
        xs = np.copy(self.proj_xs).reshape(-1)
        ys = np.copy(self.proj_ys).reshape(-1)
        depth_float = depth_float.reshape(-1)
        # Filter out invalid depths
        valid_depths = (depth_float >= self.min_depth) & (
            depth_float <= self.max_forward_range
        )
        xs = xs[valid_depths]
        ys = ys[valid_depths]
        depth_float = depth_float[valid_depths]
        # Unproject
        # negate depth as the camera looks along -Z
        xys = np.vstack(
            (
                xs * depth_float,
                ys * depth_float,
                -depth_float,
                np.ones(depth_float.shape),
            )
        )
        inv_K = self.inverse_intrinsic_matrix
        xyz_camera = np.matmul(inv_K, xys).T  # XYZ in the camera coordinate system
        xyz_camera = xyz_camera[:, :3] / xyz_camera[:, 3][:, np.newaxis]

        return xyz_camera

    def safe_assign(self, im_map, x_idx, y_idx, value):
        try:
            im_map[x_idx, y_idx] = value
        except IndexError:
            valid_idx1 = np.logical_and(x_idx >= 0, x_idx < im_map.shape[0])
            valid_idx2 = np.logical_and(y_idx >= 0, y_idx < im_map.shape[1])
            valid_idx = np.logical_and(valid_idx1, valid_idx2)
            im_map[x_idx[valid_idx], y_idx[valid_idx]] = value

    def get_depth_projection(self, depth):
        """
        Project pixels visible in depth-map to ground-plane
        """

        # if self.config.DEPTH_SENSOR.NORMALIZE_DEPTH:
        #     depth = sim_depth * (self.max_depth - self.min_depth) + self.min_depth
        # else:
        #     depth = sim_depth

        XYZ_ego = self.convert_to_pointcloud(depth)

        # Adding agent's height to the pointcloud
        if self.offset_baseline:
            XYZ_ego[:, 0] -= self.config.baseline / 2
        XYZ_ego[:, 1] += self.camera_height
        if self.offset_camera_plane:
            XYZ_ego[:, 2] += self.min_depth

        # Convert to grid coordinate system
        V = self.map_size
        Vby2 = V // 2

        points = XYZ_ego

        grid_x = (points[:, 0] / self.map_scale) + Vby2
        grid_y = (points[:, 2] / self.map_scale) + V

        # Filter out invalid points
        valid_idx = (
            (grid_x >= 0) & (grid_x <= V - 1) & (grid_y >= 0) & (grid_y <= V - 1)
        )
        points = points[valid_idx, :]
        grid_x = grid_x[valid_idx].astype(int)
        grid_y = grid_y[valid_idx].astype(int)

        # Create empty maps for the two channels
        obstacle_mat = np.zeros((self.map_size, self.map_size), np.uint8)
        explore_mat = np.zeros((self.map_size, self.map_size), np.uint8)

        # Compute obstacle locations
        high_filter_idx = points[:, 1] < self.height_thresh[1]
        low_filter_idx = points[:, 1] > self.height_thresh[0]
        obstacle_idx = np.logical_and(low_filter_idx, high_filter_idx)

        self.safe_assign(obstacle_mat, grid_y[obstacle_idx], grid_x[obstacle_idx], 1)
        # kernel = np.ones((3, 3), np.uint8)
        # obstacle_mat = cv2.dilate(obstacle_mat, kernel, iterations=1)

        # Compute explored locations
        explored_idx = high_filter_idx
        self.safe_assign(explore_mat, grid_y[explored_idx], grid_x[explored_idx], 1)
        # kernel = np.ones((3, 3), np.uint8)
        # obstacle_mat = cv2.dilate(obstacle_mat, kernel, iterations=1)

        if self.dilate_maps:
            kernel = np.ones((3, 3), np.uint8)
            obstacle_mat = cv2.dilate(obstacle_mat, kernel, iterations=2)
            # Smoothen the maps
            kernel = np.ones((3, 3), np.uint8)

            obstacle_mat = cv2.morphologyEx(obstacle_mat, cv2.MORPH_CLOSE, kernel)
            explore_mat = cv2.morphologyEx(explore_mat, cv2.MORPH_CLOSE, kernel)

        # Ensure all expanded regions in obstacle_mat are accounted for in explored_mat
        explore_mat = np.logical_or(explore_mat, obstacle_mat)

        return np.stack([obstacle_mat, explore_mat], axis=2)