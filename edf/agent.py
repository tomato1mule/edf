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

from edf.utils import preprocess
from edf.models import SE3Transformer, SE3TransformerLight, EnergyModel
from edf.mcmc import MH, LangevinMH
from edf.dist import GaussianDistSE3

class PickAgent(nn.Module):
    def __init__(self, characteristic_length, max_radius = 2.5):
        super().__init__()
        self.max_radius = max_radius
        self.characteristic_length = characteristic_length
        self.N_query = 1
        self.query_radius = 0.  
        self.field_cutoff = max_radius * 1.         
        self.std_theta_metro = 45 / 180 * np.pi
        self.std_X_metro = 5.
        self.dt = 0.1 
        self.ranges_cropped = torch.tensor([[-21., 21.],
                                            [-21., 21.],
                                            [-30., -5.]])
        self.irreps_out = o3.Irreps("5x0e + 10x1e + 4x2e + 2x3e + 2x4e")
        self.irreps_descriptor = self.irreps_out
        self.sh_lmax_descriptor = 4
        self.number_of_basis_descriptor = 10
        self.irrep_normalization = 'norm'
        self.register_buffer('dummy', torch.tensor([0.]))

        # Models
        self.se3T = SE3TransformerLight(max_neighbor_radius = max_radius, irreps_out=self.irreps_out)
        self.energy_model = EnergyModel(N_query = self.N_query, query_radius = self.query_radius, field_cutoff = self.field_cutoff,
                                        irreps_input = self.irreps_out, irreps_descriptor = self.irreps_descriptor, sh_lmax = self.sh_lmax_descriptor, number_of_basis = self.number_of_basis_descriptor, ranges = self.ranges_cropped)
        self.metropolis = MH(ranges_X = self.ranges_cropped, std_theta = self.std_theta_metro, std_X = self.std_X_metro)
        self.langevin = LangevinMH(ranges_X = self.ranges_cropped, dt = self.dt, std_theta = 1., std_X = 1.)


        # Learning
        self.lr_se3T = 5e-3
        self.lr_energy = 20e-3
        self.optimizer_se3T = torch.optim.Adam(list(self.se3T.parameters()), lr=self.lr_se3T, betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True)
        self.optimizer_energy = torch.optim.Adam(list(self.energy_model.parameters()), lr=self.lr_energy, betas=(0.9, 0.98), eps=1e-09, weight_decay=0, amsgrad=True)
        self.perturb_dist = GaussianDistSE3(std_theta = 10 / 180 * np.pi, std_X = 2.5 * 0.2)

    def initialize(self):
        self.metropolis.get_inv_cdf()
        self.langevin.get_inv_cdf()

    def sample_pose(self, inputs, temperature = 1., learning = False, ):
        self.optimizer_se3T.zero_grad()
        self.optimizer_energy.zero_grad()
        outputs = self.se3T(inputs)
        log_P = lambda T: -self.energy_model(outputs, T, temperature = temperature, learning = False)
        

    def train_once(self, sample, N_T, mh_iter, langevin_iter, temperature = 1.):
        device = self.dummy.device

        color_unprocessed = sample['color']
        sample = preprocess(sample, self.characteristic_length)

        coord, color, ranges = sample['coord'], sample['color'], sample['ranges']
        X_sdg, R_sdg = sample['grasp'][0], sample['grasp'][1]
        data_transform = sample['data_transform']

        ##### Prepare input
        target_pos = torch.tensor(X_sdg, dtype=torch.float32).unsqueeze(0).to(device) # (1, 3)
        target_orn = torch.tensor(R_sdg, dtype=torch.float32).unsqueeze(0).to(device) # (1, 3, 3)
        feature = torch.tensor(color, dtype=torch.float32)
        pos = torch.tensor(coord, dtype=torch.float32)
        in_range_cropped_idx = (((pos[:] > self.ranges_cropped[:,0]) * (pos[:] < self.ranges_cropped[:,1])).sum(dim=-1) == 3).nonzero().squeeze(-1)
        pos = pos[in_range_cropped_idx].to(device)
        feature = feature[in_range_cropped_idx].to(device)

        inputs = {'feature': feature, 'pos': pos, 'edge': None, 'max_neighbor_radius': self.max_radius}
        target_T = torch.cat([transforms.matrix_to_quaternion(target_orn), target_pos], dim=-1)
        target_T = self.perturb_dist.propose(target_T)
        target_R, target_X = transforms.quaternion_to_matrix(target_T[...,:4]), target_T[...,4:]




        self.optimizer_se3T.zero_grad()
        self.optimizer_energy.zero_grad()
        outputs = self.se3T(inputs)
        log_P = lambda T: -self.energy_model(outputs, T, temperature = temperature, learning = False)