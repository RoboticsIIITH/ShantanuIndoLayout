#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import cv2
import math
import os

import habitat_sim

from habitat.utils.visualizations import maps
from PIL import Image

import numpy as np
import random

import magnum as mn

# In[2]:


WIDTH=640
HEIGHT=480
HFOV=90
AGENT_HEIGHT=1
RESOLUTION=0.05
COORDINATE_MIN=0
COORDINATE_MAX=100
NUM_SAMPLES=500


# In[3]:


def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensors = {
        "left_rgb": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": [-0.25, settings["sensor_height"], 0.0],
        },
        "left_depth": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": [-0.25, settings["sensor_height"], 0.0],
        },
        "left_semantic": {
            "sensor_type": habitat_sim.SensorType.SEMANTIC,
            "resolution": [settings["height"], settings["width"]],
            "position": [-0.25, settings["sensor_height"], 0.0],
        },
        "right_rgb": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.25, settings["sensor_height"], 0.0],
        },
        "right_depth": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.25, settings["sensor_height"], 0.0],
        },
        "right_semantic": {
            "sensor_type": habitat_sim.SensorType.SEMANTIC,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.25, settings["sensor_height"], 0.0],
        },
        "center_rgb":{
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
        "center_depth": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
    }

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        sensor_spec = habitat_sim.CameraSensorSpec()
        sensor_spec.uuid = sensor_uuid
        sensor_spec.sensor_type = sensor_params["sensor_type"]
        sensor_spec.resolution = sensor_params["resolution"]
        sensor_spec.position = sensor_params["position"]
        # sensor_spec.parameters["hfov"] = settings["fov"]

        sensor_specs.append(sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.1)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=15.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=15.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def make_sim(BASE_DIR, scene="17DRP5sb8fy"):

    scene = f"{BASE_DIR}/{scene}.glb"

    rgb_sensor = True  # @param {type:"boolean"}
    depth_sensor = True  # @param {type:"boolean"}
    semantic_sensor = True  # @param {type:"boolean"}

    sim_settings = {
        "width": WIDTH,  # Spatial resolution of the observations
        "height": HEIGHT,
        "scene": scene,  # Scene path
        "default_agent": 0,
        "sensor_height": AGENT_HEIGHT,  # Height of sensors in meters
        "rgb": rgb_sensor,  # RGB sensor
        "depth": depth_sensor,  # Depth sensor
        "semantic": semantic_sensor,  # Semantic sensor
        "seed": 1,  # used in the random navigation
        "enable_physics": False,  # kinematics only
        "fov": str(HFOV),
    }

    cfg = make_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)

    seed = 0
    random.seed(seed)
    sim.seed(seed)
    np.random.seed(seed)
   
    return sim


# In[4]:


from habitat.utils.geometry_utils import (
    angle_between_quaternions,
    quaternion_from_two_vectors,
    quaternion_rotate_vector,
    quaternion_from_coeff,
    quaternion_to_list,
)


def quaternion_xyzw_to_wxyz(v: np.array):
    return np.quaternion(v[3], *v[0:3])


def quaternion_wxyz_to_xyzw(v: np.array):
    return np.quaternion(*v[1:4], v[0])


def quaternion_to_coeff(quat: np.quaternion) -> np.array:
    r"""Converts a quaternions to coeffs in [x, y, z, w] format
    """
    coeffs = np.zeros((4,))
    coeffs[3] = quat.real
    coeffs[0:3] = quat.imag
    return coeffs


def compute_heading_from_quaternion(r):
    """
    r - rotation quaternion
    Computes clockwise rotation about Y.
    """
    # quaternion - np.quaternion unit quaternion
    # Real world rotation
    direction_vector = np.array([0, 0, -1])  # Forward vector
    heading_vector = quaternion_rotate_vector(r.inverse(), direction_vector)

    phi = -np.arctan2(heading_vector[0], -heading_vector[2]).item()
    return phi


def compute_quaternion_from_heading(theta):
    """
    Setup: -Z axis is forward, X axis is rightward, Y axis is upward.
    theta - heading angle in radians --- measured clockwise from -Z to X.
    Compute quaternion that represents the corresponding clockwise rotation about Y axis.
    """
    # Real part
    q0 = math.cos(-theta / 2)
    # Imaginary part
    q = (0, math.sin(-theta / 2), 0)

    return np.quaternion(q0, *q)


def compute_egocentric_delta(p1, r1, p2, r2):
    """
    p1, p2 - (x, y, z) position
    r1, r2 - np.quaternions
    Compute egocentric change from (p1, r1) to (p2, r2) in
    the coordinates of (p1, r1)
    Setup: -Z axis is forward, X axis is rightward, Y axis is upward.
    """
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    theta_1 = compute_heading_from_quaternion(r1)
    theta_2 = compute_heading_from_quaternion(r2)

    D_rho = math.sqrt((x1 - x2) ** 2 + (z1 - z2) ** 2)
    D_phi = (
        math.atan2(x2 - x1, -z2 + z1) - theta_1
    )  # counter-clockwise rotation about Y from -Z to X
    D_theta = theta_2 - theta_1

    return (D_rho, D_phi, D_theta)


def compute_updated_pose(p, r, delta_xz, delta_y):
    """
    Setup: -Z axis is forward, X axis is rightward, Y axis is upward.
    p - (x, y, z) position
    r - np.quaternion
    delta_xz - (D_rho, D_phi, D_theta) in egocentric coordinates
    delta_y - scalar change in height
    Compute new position after a motion of delta from (p, r)
    """
    x, y, z = p
    theta = compute_heading_from_quaternion(
        r
    )  # counter-clockwise rotation about Y from -Z to X
    D_rho, D_phi, D_theta = delta_xz

    xp = x + D_rho * math.sin(theta + D_phi)
    yp = y + delta_y
    zp = z - D_rho * math.cos(theta + D_phi)
    pp = np.array([xp, yp, zp])

    thetap = theta + D_theta
    rp = compute_quaternion_from_heading(thetap)

    return pp, rp


# In[5]:


def make_global_map(sim):

    top_down_map = maps.get_topdown_map(
        sim.pathfinder, 0, meters_per_pixel=0.05
    )

    # plt.imshow(top_down_map)
    # plt.show()
    recolor_map = np.array(
        [1, 0, 0]
    )

    new_top_down_map = maps.get_topdown_map(
        sim.pathfinder, 0, meters_per_pixel=0.1,
    )
    new_top_down_map = recolor_map[new_top_down_map]
    return np.uint8(new_top_down_map), top_down_map


def get_mesh_occupancy(sim, agent_state):

    map_size = 64

    _, top_down_map = make_global_map(sim)

    # print("Map:", top_down_map.shape)

    agent_position = agent_state.position
    agent_rotation = agent_state.rotation

    a_y, a_x = maps.to_grid(
        agent_position[2],
        agent_position[0],
        (top_down_map.shape[0], top_down_map.shape[1]),
        pathfinder=sim.pathfinder
    )

    top_down_map_pad = np.pad(top_down_map, (300, 300), mode="constant", constant_values=0)
    a_x += 300
    a_y += 300

    # Crop region centered around the agent
    mrange = int(map_size * 1.5)
    ego_map = top_down_map_pad[
            (a_y - mrange) : (a_y + mrange), (a_x - mrange) : (a_x + mrange)
        ]

    if ego_map.shape[0] == 0 or ego_map.shape[1] == 0:
        print("EMPTY")
        ego_map = np.zeros((2 * mrange + 1, 2 * mrange + 1), dtype=np.uint8)

    # Rotate to get egocentric map
    # Negative since the value returned is clockwise rotation about Y,
    # but we need anti-clockwise rotation
    agent_heading = compute_heading_from_quaternion(agent_rotation)
    agent_heading = math.degrees(agent_heading)

    half_size = ego_map.shape[0] // 2
    center = (half_size, half_size)
    M = cv2.getRotationMatrix2D(center, agent_heading, scale=1.0)

    # print(center, agent_heading)

    ego_map = (
        cv2.warpAffine(
            ego_map * 255,
            M,
            (ego_map.shape[1], ego_map.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(1,),
        ).astype(np.float32)
        / 255.0
    )

    # plt.imshow(ego_map)
    # plt.show()

    mrange = int(map_size)
    ego_map = ego_map[
        (half_size - mrange) : (half_size + mrange),
        (half_size - mrange) : (half_size + mrange),
    ]

    # plt.imshow(ego_map)
    # plt.show()

    ego_map[ego_map > 0.5] = 1.0
    ego_map[ego_map <= 0.5] = 0.0

    # # This map is currently 0 if occupied and 1 if unoccupied. Flip it.
    # ego_map = 1.0 - ego_map

    # # Flip the x axis because to_grid() flips the conventions
    # ego_map = np.flip(ego_map, axis=1)

    # Get forward region infront of the agent
    half_size = ego_map.shape[0] // 2
    quarter_size = ego_map.shape[0] // 4
    center = (half_size, half_size)

    ego_map = ego_map[0:half_size, quarter_size : (quarter_size + half_size)]

    return ego_map


# In[ ]:


BASE_DIR = '/scratch/shantanu/gibson'
scene = 'Lindenwood'

scene_collected_data = os.path.join('/scratch/shantanu/HabitatGibson/data', scene)
frame_index = 2590 # 2535 # 


pose_path = os.path.join(scene_collected_data, "0", "pose", str(frame_index) + ".npy")
tgt_pose = np.load(pose_path, allow_pickle=True).item()

sim = make_sim(BASE_DIR, scene)
color_map, scene_map = make_global_map(sim)

agent = sim.agents[0]
state = agent.get_state()
state.position = tgt_pose['position']
state.rotation = tgt_pose['rotation']
agent.set_state(state)

obs = sim.get_sensor_observations()

# In[ ]:
agent_state = state
top_down_map = scene_map

agent_position = agent._sensors['center_rgb'].object.absolute_translation #agent_state.position
agent_rotation = agent_state.rotation

a_y, a_x = maps.to_grid(
    agent_position[2],
    agent_position[0],
    (top_down_map.shape[0], top_down_map.shape[1]),
    pathfinder=sim.pathfinder
)

top_down_map_pad = np.pad(top_down_map, (300, 300), mode="constant", constant_values=0)
a_x += 300
a_y += 300

map_size = 64
mrange = int(map_size * 1.5)
ego_map = top_down_map_pad[
        (a_y - mrange) : (a_y + mrange), (a_x - mrange) : (a_x + mrange)
    ]

agent_heading = compute_heading_from_quaternion(agent_rotation)
agent_heading = math.degrees(agent_heading)

half_size = ego_map.shape[0] // 2
center = (half_size, half_size)
M = cv2.getRotationMatrix2D(center, agent_heading, scale=1.0)

ego_map = (
    cv2.warpAffine(
        ego_map * 255,
        M,
        (ego_map.shape[1], ego_map.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(1,),
    ).astype(np.float32)
    / 255.0
)

mrange = int(map_size)
ego_map = ego_map[
    (half_size - mrange) : (half_size + mrange),
    (half_size - mrange) : (half_size + mrange),
]

ego_map[ego_map > 0.5] = 1.0
ego_map[ego_map <= 0.5] = 0.0

half_size = ego_map.shape[0] // 2
quarter_size = ego_map.shape[0] // 4
center = (half_size, half_size)
ego_map = ego_map[0:half_size, quarter_size : (quarter_size + half_size)]

# In[ ]:

cv2.imwrite('/scratch/shantanu/img.png', cv2.cvtColor(obs['center_rgb'][:,:,:3], cv2.COLOR_BGR2RGB))
cv2.imwrite('/scratch/shantanu/depth.png', (obs['center_depth'] * 6553.5).astype(np.uint16))
cv2.imwrite('/scratch/shantanu/bev.png', (ego_map + 1)*127)
print()
