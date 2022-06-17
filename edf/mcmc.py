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
from edf.layers import ClusteringLayer, EdgeSHLayer, SE3TransformerLayer
from edf.dist import UniformDistSE3, GaussianDistSE3, NewGaussianDistSE3





class MH(nn.Module):
    def __init__(self, ranges_X, std_X, std_theta):
        super().__init__()
        uniform_dist_T = UniformDistSE3(ranges_X=ranges_X)
        gaussian_dist_T = GaussianDistSE3(std_theta = std_theta, std_X= std_X)
        
        self.init_dist = uniform_dist_T
        self.proposal_dist = gaussian_dist_T

        self.register_buffer('dummy', torch.tensor([0]))

    def initialize(self, N_samples):
        T_init = self.init_dist.sample(N=N_samples)
        return T_init
        
    def propose(self, T):
        return self.proposal_dist.propose(T)

    def log_factor_diff(self, T_new, T_old): # log[Q(Told | Tnew) / Q(Tnew|Told)] 
        #return self.proposal_dist.log_factor(T_old, T_new) - self.proposal_dist.log_factor(T_new, T_old)
        return self.proposal_dist.log_factor_diff(T_new, T_old)

    def log_acceptance_ratio(self, T_new, T_old, log_P):
        assert T_new.shape == T_old.shape

        log_A = self.log_factor_diff(T_new, T_old)
        #log_P(T_new) - log_P(T_old) # Parallelized to gain ~2x speedup
        T_cat = torch.cat((T_new, T_old), dim=0)
        logP = log_P(T_cat)
        log_A = log_A + logP[:len(logP)//2] - logP[len(logP)//2:]

        return log_A

    def acceptance_ratio(self, T_new, T_old, log_P):
        log_A = self.log_acceptance_ratio(T_new, T_old, log_P)

        return torch.exp(torch.min(torch.tensor([0.]).to(log_A.device), log_A))

    def run_once(self, T, log_P, return_mask = False):
        T_new = self.propose(T)
        A = self.acceptance_ratio(T_new, T, log_P)

        accept_mask = (A > torch.rand(len(A)).to(A.device)).unsqueeze(-1)
        T_new = torch.where(accept_mask, T_new, T)

        if return_mask is True:
            return T_new, accept_mask
        return T_new

    def forward(self, log_P, max_iter, N_transforms = None, T_seed = None, pbar = False):
        if N_transforms is None and T_seed is None:
            raise ValueError('N_transforms must be specified if T_seed is not given')

        if T_seed is None:
            T = self.initialize(N_samples = N_transforms)
        else:
            T = T_seed
            N_transforms = T_seed.shape[0]

        Ts = [T]
        As = [torch.ones(N_transforms, dtype=torch.bool, device=T.device)]
        
        iterator = range(max_iter-1)
        if pbar == True:
            iterator = tqdm(iterator)
        for i in iterator:
            T,A = self.run_once(T, log_P, return_mask = True)
            Ts.append(T)
            As.append(A.squeeze(-1))
        
        Ts = torch.stack(Ts, dim=0) # (max_iter, Nt, 4+3)
        As = torch.stack(As, dim=0) # (max_iter, Nt)

        return {'Ts':Ts, 'As':As}

    def get_inv_cdf(self):
        self.proposal_dist.dist_R.get_inv_cdf()






class LangevinMH(nn.Module):
    def __init__(self, ranges_X, dt, std_theta = 1., std_X = 1.,):
        super().__init__()
        self.dt = dt
        uniform_dist_T = UniformDistSE3(ranges_X=ranges_X)
        gaussian_dist_T = NewGaussianDistSE3(std_theta = np.sqrt(2*self.dt) * std_theta, std_X = np.sqrt(2*self.dt) * std_X)

        self.init_dist = uniform_dist_T
        self.proposal_dist = gaussian_dist_T

        self.register_buffer('dummy', torch.tensor([0]))

    def initialize(self, N_samples):
        T_init = self.init_dist.sample(N=N_samples)
        return T_init
        
    def propose(self, T, grad):
        return self.proposal_dist.propose(T, lie_offset = grad * self.dt)

    def log_factor_diff(self, T_new, T_old, grad_new, grad_old): # log[Q(Told | Tnew) / Q(Tnew|Told)] 
        return self.proposal_dist.log_factor_diff(T_new, T_old, lie_offset_2 = grad_new * self.dt, lie_offset_1 = grad_old * self.dt)

    def get_logP_with_grad(self, T, log_P):
        assert T.dim() == 2 # (Nt, 7) # Is it necessary?? => Necessary because of logP.sum() gradient

        q = T[...,:4]
        X = T[...,4:]
        lie_R = torch.zeros_like(X)
        lie_X = torch.zeros_like(X)
        lie_R.requires_grad = True
        lie_X.requires_grad = True

        qt = transforms.axis_angle_to_quaternion(lie_R)
        q_grad = transforms.quaternion_multiply(q,qt)
        X_grad = X + lie_X
        T_grad = torch.cat([q_grad, X_grad], dim=-1)
        
        logP = log_P(T_grad)
        logP.sum().backward()

        grad = torch.cat([lie_X.grad, lie_R.grad], dim=-1) # approximation is best around dt = 0.01   (0.1 too big, 0.01 too small)
        return logP.detach(), grad.detach()

    def run_once(self, T, log_P, optim, return_mask = False):
        optim.zero_grad()
        logP_old, grad_old = self.get_logP_with_grad(T, log_P)

        optim.zero_grad()
        T_new = self.propose(T, grad_old)
        logP_new, grad_new = self.get_logP_with_grad(T_new, log_P)

        log_A = self.log_factor_diff(T_new, T, grad_new = grad_new, grad_old = grad_old) + logP_new - logP_old
        A = torch.exp(torch.min(torch.tensor([0.], device = log_A.device), log_A))
        accept_mask = (A > torch.rand(len(A), device=A.device)).unsqueeze(-1)

        T_new = torch.where(accept_mask, T_new, T)

        if return_mask is True:
            return T_new, accept_mask
        return T_new

    def forward(self, log_P, max_iter, optim, N_transforms = None, T_seed = None, pbar = False, optimize = False, dt_optim = None):
        if N_transforms is None and T_seed is None:
            raise ValueError('N_transforms must be specified if T_seed is not given')

        if T_seed is None:
            T = self.initialize(N_samples = N_transforms)
        else:
            T = T_seed
            N_transforms = T_seed.shape[0]

        Ts = [T]
        As = [torch.ones(N_transforms, dtype=torch.bool, device=T.device)]
        
        iterator = range(max_iter-1)
        if pbar == True:
            iterator = tqdm(iterator)
        for i in iterator:
            if optimize is False:
                T,A = self.run_once(T, log_P, optim = optim, return_mask = True)
                Ts.append(T)
                As.append(A.squeeze(-1))
            elif optimize is True:
                T = self.optimize_once(T, log_P, optim = optim, dt=dt_optim)
                Ts.append(T)

        if optimize is False:
            Ts = torch.stack(Ts, dim=0) # (max_iter, Nt, 4+3)
            As = torch.stack(As, dim=0) # (max_iter, Nt)
            return {'Ts':Ts, 'As':As}
        else:
            Ts = torch.stack(Ts, dim=0) # (max_iter, Nt, 4+3)
            return {'Ts':Ts}

    def optimize_once(self, T, log_P, optim, dt = None):
        if dt is None:
            dt = self.dt
        optim.zero_grad()
        logP, grad = self.get_logP_with_grad(T, log_P)

        lie = grad * dt
        T_new = self.proposal_dist.apply_lie(T, lie)
        with torch.no_grad():
            logP_new = log_P(T_new)
        log_A = logP_new - logP
        A = torch.exp(torch.min(torch.tensor([0.], device = log_A.device), log_A))
        accept_mask = (A > torch.rand(len(A), device=A.device)).unsqueeze(-1)
        T_new = torch.where(accept_mask, T_new, T)
        
        return T_new

    def get_inv_cdf(self):
        self.proposal_dist.dist_R.get_inv_cdf()