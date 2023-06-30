from yacs.config import CfgNode as CN

_C = CN()

# Log configuration
_C.log_dir = '/tmp/indoor_layout_estimation/'
_C.log_frequency = 250
_C.model_name = None
_C.save_frequency = 1
_C.script_mode = 'train'  # train, eval, predict
_C.dump_data = ['pred_depth'] # Add gt_depth, pred_depth, color_aug_left,  color_aug_right
_C.loss_hist_keys = ['bev_loss', 'bev_occ_loss', 'bev_exp_loss']

# Dataset configuration
_C.dataset = 'habitat'
_C.data_path = ''
_C.width = 128
_C.height = 128
_C.baseline = 0.2
_C.cam_height = 1.0
_C.focal_length = 85.333  # 170.667/2
_C.hfov = 73.74  # For the cropped version with width as 128
_C.min_depth = 0.1
_C.max_depth = 10.0
_C.train_split_file = ''
_C.val_split_file = ''
_C.frame_ids = [0, -1, 1]
_C.scales = [0]
_C.bev_width = 128 # 64
_C.bev_height = 128 # 64
_C.bev_res = 0.025 # 0.05  # 5cm to 1 pixel
_C.floor_path = None
_C.obstacle_height_thresholds = [0.2, 1.5]

# Additional datakeys
_C.color_dir = None
_C.depth_dir = None
_C.bev_dir = None
_C.pose_dir = None
_C.semantics_dir = None
_C.ego_map_dir = None

# Model configuration
_C.load_weights_folder = None
_C.models_to_load = []
_C.load_optimizer = False

# Training hyperparameters
_C.no_cuda = False
_C.batch_size = 4
_C.num_epochs = 100
_C.train_workers = 4
_C.val_workers = 4
_C.learning_rate = 1e-4
_C.scheduler_step_size = 15
_C.mode = "dense"  # dense, sparse or debug

# Loss configuration
_C.no_ssim = False
_C.disable_automasking = True
_C.avg_reprojection = False
_C.disparity_smoothness = 1e-3
_C.avg_bev_reprojection = False
# Generic because used in homography loss for pose too
_C.bev_ce_weights = [1.0, 2.0, 8.0]  # BEV Classes 0 - Unknown, 1 - Occupied, 2 - Free
_C.use_radial_loss_mask = False

_C.loss_weights = CN(new_allowed=False)
_C.loss_weights.disparity_loss = 0.0
_C.loss_weights.rgbd_loss = 0.0
_C.loss_weights.bev_loss = 0.0
_C.loss_weights.gan_loss = 0.0
_C.loss_weights.reprojection_loss = 0.0
_C.loss_weights.homography_loss = 0.0
_C.loss_weights.stc_loss = 0.0

# Pipeline configuration
_C.PIPELINE = CN(new_allowed=False)
_C.PIPELINE.train = ['DISPARITY', 'RGBD', 'BEV', 'DISCR', 'POSE']  # Remove elements when training a smaller pipeline
_C.PIPELINE.run = ['DISPARITY', 'RGBD', 'BEV', 'DISCR', 'POSE']  # Remove elements for inference with a smaller pipeline

# Module specific configuration
_C.DISPARITY = CN(new_allowed=True)
_C.DISPARITY.freeze_weights = False

_C.RGBD_ENCODER = CN(new_allowed=True)
_C.RGBD_ENCODER.freeze_weights = False

_C.BEV_DECODER = CN(new_allowed=True)
_C.BEV_DECODER.n_classes = 3
_C.BEV_DECODER.homography_train_epoch = 0
_C.BEV_DECODER.bce_loss = 1.0
_C.BEV_DECODER.ssim_loss = 0.0
_C.BEV_DECODER.smoothness_loss = 0.0
_C.BEV_DECODER.freeze_weights = False

# Discriminator Setting
_C.DISCRIMINATOR = CN()
_C.DISCRIMINATOR.lr = 1e-4
_C.DISCRIMINATOR.scheduler_step_size = 15
_C.DISCRIMINATOR.discr_train_epoch = 5
_C.DISCRIMINATOR.freeze_weights = False

_C.POSE = CN(new_allowed=True)
_C.POSE.pose_model_input = 'pairs'
_C.POSE.in_channels = 4
_C.POSE.pose_model_type = 'posecnn'
_C.POSE.freeze_weights = False


# Distributed Training configuration
_C.DD = CN(new_allowed=False)
_C.DD.world_size = 1
_C.DD.dist_url = 'tcp://224.66.41.62:23456'
_C.DD.dist_backend = 'gloo'

# Debug configuration
_C.DEBUG = CN(new_allowed=False)
_C.DEBUG.extract_depth = 'get_gt_disparity_depth'
_C.DEBUG.extract_poses = 'get_gt_poses'
_C.DEBUG.generate_reprojection_pred = 'generate_dense_pred'
_C.DEBUG.compute_reprojection_losses = 'compute_dense_reprojection_losses'
_C.DEBUG.dump_raw_data = False
_C.DEBUG.log_folder = "logs"
_C.DEBUG.wandb = True

# =========== OccAnt GP_ANTICIPATION specific options ============
_C.GP_ANTICIPATION = CN()
# Type
_C.GP_ANTICIPATION.type = "occant_rgb"
# Model capacity factor for custom UNet
_C.GP_ANTICIPATION.unet_nsf = 16
# Freeze image features
_C.GP_ANTICIPATION.freeze_features = False
_C.GP_ANTICIPATION.freeze_ans_resnet = False
_C.GP_ANTICIPATION.nclasses = 2
_C.GP_ANTICIPATION.resnet_type = "resnet18"
# OccAnt RGB specific hyperparameters
_C.GP_ANTICIPATION.detach_depth_proj = False
_C.GP_ANTICIPATION.pretrained_depth_proj_model = ""
_C.GP_ANTICIPATION.freeze_depth_proj_model = False
_C.GP_ANTICIPATION.freeze_semantic_proj_model = False
# Normalization options for anticipation output
_C.GP_ANTICIPATION.OUTPUT_NORMALIZATION = CN()
_C.GP_ANTICIPATION.OUTPUT_NORMALIZATION.channel_0 = (
    "identity"
)
_C.GP_ANTICIPATION.OUTPUT_NORMALIZATION.channel_1 = (
    "identity"
)
# Wall occupancy option
_C.GP_ANTICIPATION.wall_fov = 120.0

# Grad CAM option
_C.GP_ANTICIPATION.grad_cam = False



def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
