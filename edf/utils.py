import time
import datetime
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F

from torch_cluster import radius_graph, radius
from torch_scatter import scatter, scatter_logsumexp, scatter_log_softmax
from pytorch3d import transforms
from xitorch.interpolate import Interp1D

import e3nn.nn
from e3nn import o3
from e3nn.math import soft_one_hot_linspace, soft_unit_step

import edf
from edf.pybullet_env.utils import get_image, axiscreator, img_data_to_pointcloud
from edf.visual_utils import plot_color_and_depth, scatter_plot, scatter_plot_ax, visualize_samples, visualize_sample_cluster
from edf.pybullet_env.env import MugTask




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