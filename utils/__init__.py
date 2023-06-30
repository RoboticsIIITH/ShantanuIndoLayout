from numpy import degrees
from .layout_utils import *
from .exceptions import *
import torch
from torchvision import transforms
from scipy.spatial.transform import Rotation



def normalize_image(x, range=None):
    """Rescale image pixels to span range [0, 1]
    """
    if range is not None and isinstance(range, (list, tuple)) and len(range) == 2:
        mi, ma = range
    else:
        ma = float(x.max().cpu().data)
        mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    x = torch.clamp(x, mi, ma)
    return (x - mi) / d

def invnormalize_imagenet(x):
    inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
            std=[1/0.229, 1/0.224, 1/0.255]
        )
    return inv_normalize(x)




def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def get_pose_diff(pose1, pose2):
    """Get pose difference between pose1 and pose2
    """
    r1 = pose1[:, :3, :3]
    r2 = pose2[:, :3, :3]
    r21 = r2.transpose(0, 2, 1) @ r1

    err_r = np.array([0, 0, 0], dtype=np.float32)
    for idx in range(r21.shape[0]):
        err_r += np.abs(Rotation.from_matrix(r21[idx, ...]).as_rotvec(degrees=True))
    err_r = err_r/r21.shape[0]

    p1 = pose1[:, :3, 3]
    p2 = pose2[:, :3, 3]
    p21 = np.mean(np.abs(p2 - p1), axis=0)

    rot = dict(rx=err_r[0], ry=err_r[1], rz=err_r[2])
    pos = dict(px=p21[0], py=p21[1], pz=p21[2])

    output = dict(pos=pos, rot=rot)

    return output