
import numpy as np


import torch
import torch.nn as nn
from torch.nn import functional as F

from pytorch3d import transforms
import torch_scatter


from scipy.spatial.transform import Rotation


def voxel_filter(coord, color, d, device = 'cpu'):
    mins = coord.min(axis=-2)
    maxs = coord.max(axis=-2)

    vox_idx = ((coord - mins) // d).astype(int)
    shape = vox_idx.max(axis=-2) + 1
    raveled_idx = torch.tensor(np.ravel_multi_index(vox_idx.T, shape), device=device)

    n_pts_per_vox = torch_scatter.scatter(torch.ones_like(raveled_idx, device=device), raveled_idx, dim_size=shape[0]*shape[1]*shape[2])
    nonzero_vox = n_pts_per_vox.nonzero()
    n_pts_per_vox = n_pts_per_vox[nonzero_vox].squeeze(-1)

    color_vox = torch_scatter.scatter(torch.tensor(color, dtype=torch.float32, device=device), raveled_idx.unsqueeze(-1), dim=-2, dim_size=shape[0]*shape[1]*shape[2])[nonzero_vox].squeeze(-2)
    color_vox /= n_pts_per_vox.unsqueeze(-1)
    color_vox = color_vox.cpu().numpy()

    # Type 1: Center of voxel
    # coord_vox = np.stack(np.unravel_index(nonzero_vox.cpu().numpy().reshape(-1), shape), axis=-1)
    # coord_vox = coord_vox * d + mins + (d/2)

    # Type 2: Avg coord
    coord_vox = torch_scatter.scatter(torch.tensor(coord, dtype=torch.float32, device=device), raveled_idx.unsqueeze(-1), dim=-2, dim_size=shape[0]*shape[1]*shape[2])[nonzero_vox].squeeze(-2)
    coord_vox /= n_pts_per_vox.unsqueeze(-1)
    coord_vox = coord_vox.cpu().numpy()

    return coord_vox, color_vox



def print_cuda_usage(device_idx):
    t = torch.cuda.get_device_properties(device_idx).total_memory
    r = torch.cuda.memory_reserved(device_idx)
    a = torch.cuda.memory_allocated(device_idx)
    f = r-a  # free inside reserved
    print(f"Reserved: {r/1024/1024} Mib || Allocated: {a/1024/1024} Mib || Free: {(r-a)/1024/1024} Mib")


class DataTransform():
    def __init__(self, ranges, scale, R=None):
        if type(ranges) == tuple: # (xlim, ylim, zlim)
            ranges = np.array(ranges)

        center = ranges.sum(axis=-1) / 2
        length = ranges[:,1] - ranges[:,0]

        self.R = R
        self.s = scale
        self.t = -center

        self.ranges = ranges
        self.ranges_transformed = self.transform(ranges.T).T

    def transform(self, x):
        assert x.shape[-1] == 3
        x = (x + self.t) / self.s
        if self.R is not None:
            x = self.R @ x
        return x

    def inv_transform(self, x):
        assert x.shape[-1] == 3
        if self.R is not None:
            x = self.R.T @ x
        x = x * self.s
        return x - self.t

    def transform_T(self, X, R):
        X = (X + self.t) / self.s
        if self.R is not None:
            raise NotImplementedError
        return (X, R)

    def inv_transform_T(self, X, R):
        if self.R is not None:
            raise NotImplementedError
        X = (X * self.s) - self.t
        return (X, R)




def preprocess(sample, characteristic_length, pick_and_place = False):
    coord, color = sample['coord'], sample['color']
    xlim, ylim, zlim = sample['range']
    if 'grasp' in sample.keys():
        X_sdg, R_sdg = sample['grasp']
    data_transform = DataTransform((xlim, ylim, zlim), scale=characteristic_length)

    coord = data_transform.transform(coord)
    if 'grasp' in sample.keys():
        X_sdg, R_sdg = data_transform.transform_T(X_sdg, R_sdg)
    ranges = data_transform.ranges_transformed.copy()

    xlim, ylim, zlim = xlim / characteristic_length, ylim / characteristic_length, zlim / characteristic_length
    if 'grasp' in sample.keys():
        grasp = (X_sdg, R_sdg)

    color_mean = np.array([[0.5, 0.5, 0.5]])
    color_std = np.array([[0.5, 0.5, 0.5]])

    color = (color - color_mean) / color_std
    sample_out = {}
    sample_out['coord'] = coord
    sample_out['color'] = color
    sample_out['ranges'] = ranges
    if 'grasp' in sample.keys():
        sample_out['grasp'] = grasp
    sample_out['data_transform'] = data_transform


    if pick_and_place:
        coord, color = sample['coord_pick'], sample['color_pick']
        xlim, ylim, zlim = sample['range_pick']
        X_sdg, R_sdg = sample['pick_pose']
        data_transform = DataTransform((xlim, ylim, zlim), scale=characteristic_length)

        coord = data_transform.transform(coord)
        X_sdg, R_sdg = data_transform.transform_T(X_sdg, R_sdg)
        ranges = data_transform.ranges_transformed.copy()

        xlim, ylim, zlim = xlim / characteristic_length, ylim / characteristic_length, zlim / characteristic_length
        grasp = (X_sdg, R_sdg)

        color_mean = np.array([[0.5, 0.5, 0.5]])
        color_std = np.array([[0.5, 0.5, 0.5]])

        color = (color - color_mean) / color_std

        sample_out['coord_Q'] = coord
        sample_out['color_Q'] = color
        sample_out['ranges_Q'] = ranges
        sample_out['grasp_Q'] = grasp
        sample_out['data_transform_Q'] = data_transform



        coord, color = sample['coord_place'], sample['color_place']
        xlim, ylim, zlim = sample['range_place']
        if 'place' in sample.keys():
            X_sdg, R_sdg = sample['place']
        data_transform = DataTransform((xlim, ylim, zlim), scale=characteristic_length)

        coord = data_transform.transform(coord)
        if 'place' in sample.keys():
            X_sdg, R_sdg = data_transform.transform_T(X_sdg, R_sdg)
        ranges = data_transform.ranges_transformed.copy()

        xlim, ylim, zlim = xlim / characteristic_length, ylim / characteristic_length, zlim / characteristic_length
        if 'place' in sample.keys():
            grasp = (X_sdg, R_sdg)

        color_mean = np.array([[0.5, 0.5, 0.5]])
        color_std = np.array([[0.5, 0.5, 0.5]])

        color = (color - color_mean) / color_std

        sample_out['coord_K'] = coord
        sample_out['color_K'] = color
        sample_out['ranges_K'] = ranges
        if 'place' in sample.keys():
            sample_out['grasp_K'] = grasp
        sample_out['data_transform_K'] = data_transform

    return sample_out

def get_frame_from_xy(x_axis, y_axis):
    x_axis = F.normalize(x_axis, dim = -1)
    y_axis = F.normalize(y_axis, dim = -1)
    inner_product = (x_axis * y_axis).sum(dim=-1, keepdim = True)
    y_axis = y_axis - (inner_product * x_axis)
    y_axis = F.normalize(y_axis, dim = -1)
    z_axis = torch.cross(x_axis, y_axis, dim=-1)

    R = torch.stack([x_axis, y_axis, z_axis],dim=-1)
    return R

def quat_from_matrix(R):
    eps = 1e-12
    
    qw = (1 + R[:,0,0] + R[:,1,1] + R[:,2,2]).sqrt() /2
    qx = (R[:,2,1] - R[:,1,2])/( 4 *qw + eps)
    qy = (R[:,0,2] - R[:,2,0])/( 4 *qw + eps)
    qz = (R[:,1,0] - R[:,0,1])/( 4 *qw + eps)

    return torch.stack([qw,qx,qy,qz], dim = -1)

def rotation_measure_quat(R1, R2):
    q1 = quat_from_matrix(R1)
    q2 = quat_from_matrix(R2)
    inner_prod = ((q1*q2)**2).sum(dim = -1)

    #return 1-inner_prod # equals (1/2)*(1-cos_theta)
    return 0.5 - inner_prod # equals (-1/2)*cos_theta

def rotation_measure_geodesic(R1,R2):
    diff = torch.einsum('...ij,...ik->...jk', R1, R2)
    trace = torch.einsum('...ii',diff)
    cos = (trace - 1)/2
    # For numerical stability:
    eps = 1e-6
    #cos = (1-2*eps)*cos + eps
    cos = (1-eps)*cos

    theta = torch.acos(cos)
    return theta**2

rotation_measure = rotation_measure_geodesic




class OrthoTransform():
    def __init__(self, W, ranges):
        self.H = self.W = W
        self.ranges = ranges
        assert ranges.shape == (2,2)
        self.center = np.array([[self.ranges[0,0]+self.ranges[0,1], self.ranges[1,0]+self.ranges[1,1]]])/2 # (1,2)
        self.min = self.ranges.min(axis=-1).reshape(1,-1) # (1,2)
        X = ranges[0,1] - ranges[0,0]
        Y = ranges[1,1] - ranges[1,0]
        assert X == Y > 0
        self.pix_size = X / self.H

    def coord2pix(self, coord): # coord should be (X,Y) not (X,Y,Z)
        if len(coord.shape) == 1:
            coord.reshape(1,-1)
        assert len(coord.shape) == 2 and coord.shape[-1] == 2 # coord.shape == (N,2)

        pix_W = (coord[:,0] - self.min[:,0]) // self.pix_size
        pix_H = self.H - ((coord[:,1] - self.min[:,1]) // self.pix_size) - 1

        return np.stack((pix_H, pix_W), axis=-1).astype(int) # (N,2)

    def pix2coord(self, pix):
        coord = pix[...,::-1] * np.array([1., -1.]) + np.array([0., self.H-1])
        coord = coord * self.pix_size + self.min
        return coord

    def orthographic(self, coord, color):
        ortho_coord, ortho_color = voxel_filter(coord, color, d=self.pix_size)
        ortho_color = ortho_color[np.argsort(ortho_coord[:,-1])]
        ortho_coord = ortho_coord[np.argsort(ortho_coord[:,-1])]
        ortho_inrange_idx = ((ortho_coord[:,0] > self.ranges[0][0]) * (ortho_coord[:,0] < self.ranges[0][1]) * (ortho_coord[:,1] > self.ranges[1][0]) * (ortho_coord[:,1] < self.ranges[1][1])).nonzero()
        ortho_coord, ortho_color, ortho_depth = ortho_coord[ortho_inrange_idx][:,:2], ortho_color[ortho_inrange_idx], ortho_coord[ortho_inrange_idx][:,2:]
        pix_coord = self.coord2pix(ortho_coord)

        # ortho_img = np.zeros([self.H,self.W,4])
        # for pix, color, depth in zip(pix_coord, ortho_color, ortho_depth):
        #     ortho_img[pix[0], pix[1],:3] = color
        #     ortho_img[pix[0], pix[1],3] = depth

        ortho_img = np.zeros([self.H+2,self.W+2,4])
        for pix, color, depth in zip(pix_coord, ortho_color, ortho_depth):
            ortho_img[pix[0]:pix[0]+3, pix[1]:pix[1]+3, :3] = color
            ortho_img[pix[0]:pix[0]+3, pix[1]:pix[1]+3, 3] = depth

        ortho_img = ortho_img[1:-1, 1:-1, :]
        return ortho_img # RGBD (H,W,4)

    def pose2pix_yaw_zrp(self, pose, grasp = 'side'):
        X_sg, R_sg = pose
        grasp_pix = self.coord2pix(X_sg[:2].reshape(1,-1)).reshape(-1) # H,W
        height = X_sg[2]

        if grasp == 'side':
            R_sf = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]]).T
            R_fg = R_sf.T @ R_sg
            rot = Rotation.from_matrix(R_fg)
            yaw , pitch, roll = rot.as_euler('XYZ', degrees = True)
        elif grasp == 'top':
            R_sf = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]]).T
            R_fg = R_sf.T @ R_sg
            rot = Rotation.from_matrix(R_fg)
            roll, pitch, yaw = rot.as_euler('XYZ', degrees = True)
            yaw = -yaw

        return [grasp_pix, yaw, height, roll, pitch]

    def pix_yaw_zrp2pose(self, grasp_pix, yaw, height, roll, pitch, grasp = 'side'):
        if grasp == 'side':
            ypr = np.array([yaw, pitch, roll])
            R_fg = Rotation.from_euler('XYZ', ypr, degrees=True).as_matrix()
            R_sf = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]]).T
            R_sg = R_sf @ R_fg
        elif grasp == 'top':
            rpy = np.array([roll, pitch, -yaw])
            R_fg = Rotation.from_euler('XYZ', rpy, degrees=True).as_matrix()
            R_sf = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]]).T
            R_sg = R_sf @ R_fg

        coord_grasp = self.pix2coord(grasp_pix.reshape(1,2)).reshape(2)
        X_sg = np.array([coord_grasp[0], coord_grasp[1], height])
        return (X_sg, R_sg)


def check_irreps_sorted(irreps):
    max_deg = 0
    for irrep in irreps:
        deg = int(irrep.ir[0])
        if deg < max_deg:
            return False
        else:
            max_deg = deg
    return True



def voxelize_sample(sample, coord_jitter = False, color_jitter = False, pick = True, place = True):
    sample = sample.copy()

    # if coord_jitter:
    #     if pick:
    #         sample['coord'] = sample['coord'] + np.random.randn(*(sample['coord'].shape)) * sample['d'] * coord_jitter
    #     if place:
    #         sample['coord_pick'] = sample['coord_pick'] + np.random.randn(*(sample['coord_pick'].shape)) * sample['d_pick'] * coord_jitter
    #         sample['coord_place'] = sample['coord_place'] + np.random.randn(*(sample['coord_place'].shape)) * sample['d_place'] * coord_jitter

    if pick:
        sample['coord'], sample['color'] = voxel_filter(sample['coord'], sample['color'], d=sample['d'])
    if place:
        sample['coord_pick'], sample['color_pick'] = voxel_filter(sample['coord_pick'], sample['color_pick'], d=sample['d_pick'])
        sample['coord_place'], sample['color_place'] = voxel_filter(sample['coord_place'], sample['color_place'], d=sample['d_place'])

    if coord_jitter:
        if pick:
            sample['coord'] = sample['coord'] + np.random.randn(*(sample['coord'].shape)) * sample['d'] * coord_jitter
        if place:
            sample['coord_pick'] = sample['coord_pick'] + np.random.randn(*(sample['coord_pick'].shape)) * sample['d_pick'] * coord_jitter
            sample['coord_place'] = sample['coord_place'] + np.random.randn(*(sample['coord_place'].shape)) * sample['d_place'] * coord_jitter

    if color_jitter:
        if pick:
            sample['color'] = sample['color'] + np.random.randn(*(sample['color'].shape)) * color_jitter
        if place:
            sample['color_pick'] = sample['color_pick'] + np.random.randn(*(sample['color_pick'].shape)) * color_jitter
            sample['color_place'] = sample['color_place'] + np.random.randn(*(sample['color_place'].shape)) * color_jitter

        if pick:
            sample['color'] = np.where(sample['color'] > 1., 1., sample['color'])
            sample['color'] = np.where(sample['color'] < 0., 0., sample['color'])
        if place:
            sample['color_pick'] = np.where(sample['color_pick'] > 1., 1., sample['color_pick'])
            sample['color_pick'] = np.where(sample['color_pick'] < 0., 0., sample['color_pick'])
            sample['color_place'] = np.where(sample['color_place'] > 1., 1., sample['color_place'])
            sample['color_place'] = np.where(sample['color_place'] < 0., 0., sample['color_place'])


    return sample


from scipy.stats import binomtest
def binomial_test(success, n, confidence = 0.95):
    result = binomtest(k=success, n=max(n,1))
    mid = result.proportion_estimate
    low = result.proportion_ci(confidence_level=confidence, method='exact').low
    high = result.proportion_ci(confidence_level=confidence, method='exact').high

    result_str = f"{100*success/max(n,1):.1f}% ({success} / {n});   ({100*confidence:.0f}% CI: {low*100:.1f}%~{high*100:.1f}%)"

    return mid, low, high, result_str


@torch.jit.script
def normalize_quaternion(q):
    return transforms.standardize_quaternion(q/torch.norm(q, dim=-1, keepdim=True))