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
from edf.utils import normalize_quaternion




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
        with torch.no_grad():
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
    def __init__(self, ranges_X, dt, std_theta = 1., std_X = 1.):
        super().__init__()
        self.register_buffer('dt', torch.tensor(dt))
        self.register_buffer('std_theta', torch.tensor(std_theta))
        self.register_buffer('std_X', torch.tensor(std_X))
        self.register_buffer('ranges_X', ranges_X)
        uniform_dist_T = UniformDistSE3(ranges_X=ranges_X)
        self.init_dist = uniform_dist_T

        so3_basis = torch.tensor([[[0., 0., 0.],
                                   [0., 0.,-1.],
                                   [0., 1., 0.]],

                                   [[0., 0., 1.],
                                   [0., 0., 0.],
                                   [-1.,0., 0.]],

                                   [[0.,-1., 0.],
                                   [1., 0., 0.],
                                   [0., 0., 0.]]])
        self.register_buffer('so3_basis', so3_basis)
        self.register_buffer('q_indices', torch.tensor([[1,2,3], [0,3,2], [3,0,1], [2,1,0]], dtype=torch.long))
        self.register_buffer('q_factor', torch.tensor([[-0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, 0.5]]))

    def initialize(self, N_samples):
        raise NotImplementedError
        T_init = self.init_dist.sample(N=N_samples)
        return T_init

    def in_range(self, T):
        X = T[...,-3:]
        in_range = (self.ranges_X[:,1] >= X) * (X >= self.ranges_X[:,0])
        return in_range.all(dim=-1)

    def get_logP_with_lie_grad(self, T, log_P):
        assert T.dim() == 2 # (Nt, 7) # Is it necessary?? => Necessary because of logP.sum() gradient
        assert T.requires_grad is False

        # ##### Analytic #####
        # T = T.detach()
        # q, X = T[...,:4], T[...,4:]
        # q = transforms.standardize_quaternion(q/torch.norm(q, dim=-1, keepdim=True))
        # q, X = q.requires_grad_(), X.requires_grad_()
        # logP = log_P(torch.cat([q, X], dim=-1))
        # logP.sum().backward(inputs = [q, X])
        # grad_q, grad_X = q.grad.detach(), X.grad.detach()
        # lie_derivative = q.detach()[...,self.q_indices] * self.q_factor
        # lie_R_grad = torch.einsum('...ia,...i->...a',lie_derivative, grad_q)
        # lie_X_grad = transforms.quaternion_apply(transforms.quaternion_invert(q),grad_X)
        # grad = torch.cat([lie_X_grad.detach(), lie_R_grad.detach()], dim=-1)

        # ##### Autograd exp map #####
        # T = T.detach()
        # q, X = T[...,:4], T[...,4:]
        # q = transforms.standardize_quaternion(q/torch.norm(q, dim=-1, keepdim=True))
        # lie_R, lie_X = torch.zeros_like(X, requires_grad=True), torch.zeros_like(X, requires_grad=True)
        # dq = transforms.axis_angle_to_quaternion(lie_R)
        # q_test = transforms.quaternion_multiply(q,dq)
        # lie_R_mat = (self.so3_basis * lie_R[..., None, None]).sum(dim=-3) # (Nt, 3, 3)
        # right_jacobian = torch.eye(3, dtype=q.dtype, device=q.device) + lie_R_mat/2 + (lie_R_mat@lie_R_mat)/3
        # dX = torch.einsum('...ij,...j', right_jacobian, lie_X) # (Nt, 3)
        # X_test = X + transforms.quaternion_apply(q, dX)
        # logP = log_P(torch.cat([q_test, X_test], dim=-1))
        # logP.sum().backward(inputs = [lie_R, lie_X])
        # grad = torch.cat([lie_X.grad.detach(), lie_R.grad.detach()], dim=-1)

        # ##### Autograd add (very bad) #####
        # T = T.detach()
        # q, X = T[...,:4], T[...,4:]
        # transforms.standardize_quaternion(q/torch.norm(q, dim=-1, keepdim=True))
        # R = transforms.quaternion_to_matrix(q)
        # lie_R, lie_X = torch.zeros_like(X, requires_grad=True), torch.zeros_like(X, requires_grad=True)
        # lie_R_mat = (self.so3_basis * lie_R[..., None, None]).sum(dim=-3)
        # q_test, X_test = transforms.matrix_to_quaternion(R+lie_R_mat), X+lie_X
        # logP = log_P(torch.cat([q_test/q_test.norm(dim=-1, keepdim=True), X_test], dim=-1))
        # logP.sum().backward(inputs = [lie_R, lie_X])
        # grad = torch.cat([lie_X.grad.detach(), lie_R.grad.detach()], dim=-1)

        # ##### Lie from quaternion grad #####
        T = T.detach().requires_grad_(True)
        logP = log_P(T)
        logP.sum().backward(inputs=T)
        grad = T.grad
        L = T.detach()[...,self.q_indices] * self.q_factor
        grad = torch.cat([transforms.quaternion_apply(transforms.quaternion_invert(T[...,:4].detach()), grad[...,4:]), torch.einsum('...ia,...i', L, grad[...,:4])], dim=-1)
        
        # max_grad = 100.
        # grad_norm = grad.norm(dim=-1, keepdim=True)
        # grad_clip = grad_norm > max_grad
        # grad = torch.where(grad_clip, grad / grad_norm * max_grad, grad)

        # grad_X, grad_R = lie_X.grad.detach(), lie_R.grad.detach()
        # max_grad_X = 10.
        # grad_norm_X = grad_X.norm(dim=-1, keepdim=True)
        # grad_clip_X = grad_norm_X > max_grad_X
        # grad_clipped_X = torch.where(grad_clip_X, grad_X / grad_norm_X * max_grad_X, grad_X)
        # max_grad_R = 3.141592
        # grad_norm_R = grad_R.norm(dim=-1, keepdim=True)
        # grad_clip_R = grad_norm_R > max_grad_R
        # grad_clipped_R = torch.where(grad_clip_R, grad_R / grad_norm_R * max_grad_R, grad_R)
        # grad = torch.cat([grad_clipped_X, grad_clipped_R], dim=-1) 

        return logP.detach(), grad

    def get_logP_with_grad(self, T, log_P):
        assert T.dim() == 2 # (Nt, 7) # Is it necessary?? => Necessary because of logP.sum() gradient
        assert T.requires_grad is False

        # ##### Lie from quaternion grad #####
        T = T.detach().requires_grad_(True)
        logP = log_P(T)
        logP.sum().backward(inputs=T)

        return logP.detach(), T.grad.detach()

    def propose(self, T, grad, reject = False, var_dt = True, log_P = None):
        if reject:
            assert log_P is not None
        T, grad = T.detach(), grad.detach() # (Nt, 7), (Nt, 7)

        nat_grad = grad.clone() # (Nt, 7)
        L = T[...,self.q_indices] * self.q_factor # (Nt, 4, 3)
        if var_dt is True or reject is True:
            lie_grad = torch.empty_like(grad[...,:6]) # (Nt, 6) # (v, \omg)
            lie_grad[..., :3] = transforms.quaternion_apply(transforms.quaternion_invert(T[...,:4]), point = grad[...,4:]) # (Nt, 3), Translation part
            lie_grad[..., 3:] = torch.einsum('...ia,...i->...a', L, grad[...,:4]) # (Nt, 3), Rotation part
        Ginv = (torch.eye(4, dtype=T.dtype, device=T.device) - torch.einsum('...i,...j->...ij', T[...,:4], T[...,:4]))/4  # (Nt, 4, 4)
        nat_grad[..., :4] = torch.einsum('...ij,...j->...i', Ginv, grad[...,:4]) # (Nt, 7)
        if var_dt is True:
            dt = torch.min(3 / (lie_grad.abs().sum(dim=-1) + 1e-5), torch.tensor(1., device=grad.device, dtype=grad.dtype)).unsqueeze(-1) # (Nt,1)
            dt = dt*self.dt                                # (Nt,1)
        else:
            dt = self.dt
        std_R = torch.sqrt(2*dt) * self.std_theta      # (Nt,1) or (1,)
        std_X = torch.sqrt(2*dt) * self.std_X          # (Nt,1) or (1,)

        
        noise_R_lie = torch.randn_like(T[...,:3]) # (Nt, 3)
        noise_X_lie = torch.randn_like(noise_R_lie) # (Nt, 3)
        if reject:
            log_prob_R = -0.5 * (noise_R_lie.square().sum(dim=-1)) # (Nt,)
            log_prob_X = -0.5 * (noise_X_lie.square().sum(dim=-1)) # (Nt,)
        noise_q = torch.einsum('...ij,...j', L, noise_R_lie) # (Nt, 4)
        # noise_X = transforms.quaternion_apply(quaternion = T[...,:4], point = noise_X_lie) # (Nt, 3) # Unnecessary due to rotation invariance of normal distribution
        noise_X = noise_X_lie  # (Nt, 3)

        dT = (nat_grad*dt) + torch.cat([noise_q*std_R, noise_X*std_X], dim=-1) # (Nt, 7)
        T_prop = T + dT # (Nt, 7)
        T_prop[...,:4] = normalize_quaternion(T_prop[...,:4])

        if reject:
            logP_prop, grad_prop = self.get_logP_with_grad(T_prop, log_P)
            raise NotImplementedError
            return T_prop, logP_prop.detach(), grad_prop, wtf
        else:
            return T_prop

    def run_once(self, T, log_P, return_mask = False, reject=False): # if reject=False, this becomes SGLD, otherwise MALA
        logP_old, grad_old = self.get_logP_with_grad(T, log_P)
        T_new = self.propose(T, grad_old, reject=reject)

        if reject:
            raise NotImplementedError
            logP_new, grad_new = self.get_logP_with_grad(T_new, log_P)
            with torch.no_grad():
                log_A = self.log_factor_diff(T_new, T, grad_new = grad_new, grad_old = grad_old) + logP_new - logP_old
                A = torch.exp(torch.min(torch.tensor([0.], device = log_A.device), log_A))
                accept_mask = (A > torch.rand(len(A), device=A.device)).unsqueeze(-1)
                T_new = torch.where(accept_mask, T_new, T)
        else:
            #accept_mask = torch.ones(len(T_new), 1, dtype=torch.bool, device=T_new.device)
            accept_mask = self.in_range(T_new).unsqueeze(-1)
            T_new = torch.where(accept_mask, T_new, T)

        if return_mask is True:
            return T_new, accept_mask
        return T_new

    def forward(self, log_P, max_iter, N_transforms = None, T_seed = None, pbar = False, reject = False):
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
            T,A = self.run_once(T, log_P, return_mask = True, reject=reject)
            Ts.append(T)
            As.append(A.squeeze(-1))

        Ts = torch.stack(Ts, dim=0) # (max_iter, Nt, 4+3)
        As = torch.stack(As, dim=0) # (max_iter, Nt)
        return {'Ts':Ts, 'As':As}

    def get_inv_cdf(self):
        # self.proposal_dist.dist_R.get_inv_cdf()
        pass


class PoseOptimizer(nn.Module):
    def __init__(self, device = 'cpu'):
        super().__init__()
        self.device = device
        self.register_buffer('q_indices', torch.tensor([[1,2,3], [0,3,2], [3,0,1], [2,1,0]], dtype=torch.long))
        self.register_buffer('q_factor', torch.tensor([[-0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, 0.5]]))

    def to(self, device):
        self.device = device
        return super().to(device)

    def forward(self, T, log_P, step, lr=0.001, sort=False, pbar=False):
        device = T.device

        T = T.clone().detach().to(self.device).requires_grad_(True)
        #optim = torch.optim.SGD([T], lr=lr)
        optim = torch.optim.Adam([T], lr=lr)

        iterator = range(step-1)
        if pbar == True:
            iterator = tqdm(iterator)

        Ts = [T.clone().detach()]
        #As = [torch.ones(len(T), dtype=torch.bool, device=device)]
        Es = []
        for _ in iterator:
            optim.zero_grad()
            E = -log_P(T)
            E.sum().backward(inputs=T)
            ##### Natural Gradient #####
            # L = T.detach()[...,self.q_indices] * self.q_factor
            # Ginv = torch.einsum('...ia,...ja', L, L)
            Ginv = (torch.eye(4, dtype=T.dtype, device=T.device) - torch.einsum('...i,...j->...ij', T.detach()[...,:4], T.detach()[...,:4]))/4
            nat_grad = torch.einsum('...ij,...j', Ginv, T.grad[...,:4])
            T.grad[..., :4] = nat_grad
            ############################
            optim.step()
            T.detach()[...,:4] = transforms.standardize_quaternion(T.detach()[...,:4] / torch.norm(T.detach()[...,:4], dim=-1, keepdim=True))
            Es.append(E.clone().detach())
            Ts.append(T.clone().detach())
        with torch.no_grad():
            E = -log_P(T.detach())
            Es.append(E.clone().detach())

        Ts = torch.stack(Ts, dim=-3).to(device) # (max_iter, Nt, 4+3)
        As = torch.ones(Ts.shape[-3], Ts.shape[-2], dtype=torch.bool).to(device) # (max_iter, Nt)
        Es = torch.stack(Es, dim=0).to(device) # (max_iter, Nt)

        if sort:
            sorted = Es.argsort(dim=-2, descending=True)
            Ts = Ts[sorted, torch.arange(sorted.shape[-1]).repeat(sorted.shape[-2],1)] # (max_iter, Nt, 4+3)
            As = As[sorted, torch.arange(sorted.shape[-1]).repeat(sorted.shape[-2],1)] # (max_iter, Nt)
            Es = Es[sorted, torch.arange(sorted.shape[-1]).repeat(sorted.shape[-2],1)] # (max_iter, Nt)

        return {'Ts':Ts, 'As':As, 'Es':Es}
































































class LangevinMHDeprecated(nn.Module):
    def __init__(self, ranges_X, dt, std_theta = 1., std_X = 1.):
        super().__init__()
        self.dt = dt
        self.std_theta = std_theta
        self.std_X = std_X
        self.register_buffer('std', torch.tensor(np.sqrt(2*self.dt) * std_theta))
        self.register_buffer('ranges_X', ranges_X)
        uniform_dist_T = UniformDistSE3(ranges_X=ranges_X)
        gaussian_dist_T = NewGaussianDistSE3(std_theta = np.sqrt(2*self.dt) * std_theta, std_X = np.sqrt(2*self.dt) * std_X)

        self.init_dist = uniform_dist_T
        self.proposal_dist = gaussian_dist_T

        self.register_buffer('dummy', torch.tensor([0]))
        so3_basis = torch.tensor([[[0., 0., 0.],
                                   [0., 0.,-1.],
                                   [0., 1., 0.]],

                                   [[0., 0., 1.],
                                   [0., 0., 0.],
                                   [-1.,0., 0.]],

                                   [[0.,-1., 0.],
                                   [1., 0., 0.],
                                   [0., 0., 0.]]])
        self.register_buffer('so3_basis', so3_basis)
        self.register_buffer('q_indices', torch.tensor([[1,2,3], [0,3,2], [3,0,1], [2,1,0]], dtype=torch.long))
        self.register_buffer('q_factor', torch.tensor([[-0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, 0.5]]))


    def initialize(self, N_samples):
        T_init = self.init_dist.sample(N=N_samples)
        return T_init
        
    def propose(self, T, grad, reject = False):
        if reject:
            raise NotImplementedError
            #return self.proposal_dist.propose(T, lie_offset = grad * self.dt)
        else:
            q_old, X_old = T[...,:4], T[...,4:]
            #lie_new = (grad * self.dt) + (self.std * torch.randn(*(grad.shape), device=grad.device, dtype=grad.dtype)) 

            ### Variable timestep
            dt = torch.min(1 / (grad.abs().sum(dim=-1)+1e-5), torch.tensor(1., device=grad.device, dtype=grad.dtype)).unsqueeze(-1)
            dt = dt*self.dt
            std = torch.sqrt(2*dt) * self.std_theta
            lie_new = (grad * dt) + (std * torch.randn(*(grad.shape), device=grad.device, dtype=grad.dtype)) 

            T_new = transforms.se3_exp_map(lie_new).transpose(-1,-2)
            q_new, X_new = transforms.matrix_to_quaternion(T_new[...,:3,:3]),  T_new[..., :3, -1]
            q_new, X_new = transforms.quaternion_multiply(q_old, q_new), transforms.quaternion_apply(q_old, X_new) + X_old
            return torch.cat([q_new, X_new], dim=-1)


    def log_factor_diff(self, T_new, T_old, grad_new, grad_old): # log[Q(Told | Tnew) / Q(Tnew|Told)] 
        return self.proposal_dist.log_factor_diff(T_new, T_old, lie_offset_2 = grad_new * self.dt, lie_offset_1 = grad_old * self.dt)

    def get_logP_with_grad(self, T, log_P):
        assert T.dim() == 2 # (Nt, 7) # Is it necessary?? => Necessary because of logP.sum() gradient
        assert T.requires_grad is False

        # ##### Analytic #####
        # T = T.detach()
        # q, X = T[...,:4], T[...,4:]
        # q = transforms.standardize_quaternion(q/torch.norm(q, dim=-1, keepdim=True))
        # q, X = q.requires_grad_(), X.requires_grad_()
        # logP = log_P(torch.cat([q, X], dim=-1))
        # logP.sum().backward(inputs = [q, X])
        # grad_q, grad_X = q.grad.detach(), X.grad.detach()
        # lie_derivative = q.detach()[...,self.q_indices] * self.q_factor
        # lie_R_grad = torch.einsum('...ia,...i->...a',lie_derivative, grad_q)
        # lie_X_grad = transforms.quaternion_apply(transforms.quaternion_invert(q),grad_X)
        # grad = torch.cat([lie_X_grad.detach(), lie_R_grad.detach()], dim=-1)

        # ##### Autograd exp map #####
        # T = T.detach()
        # q, X = T[...,:4], T[...,4:]
        # q = transforms.standardize_quaternion(q/torch.norm(q, dim=-1, keepdim=True))
        # lie_R, lie_X = torch.zeros_like(X, requires_grad=True), torch.zeros_like(X, requires_grad=True)
        # dq = transforms.axis_angle_to_quaternion(lie_R)
        # q_test = transforms.quaternion_multiply(q,dq)
        # lie_R_mat = (self.so3_basis * lie_R[..., None, None]).sum(dim=-3) # (Nt, 3, 3)
        # right_jacobian = torch.eye(3, dtype=q.dtype, device=q.device) + lie_R_mat/2 + (lie_R_mat@lie_R_mat)/3
        # dX = torch.einsum('...ij,...j', right_jacobian, lie_X) # (Nt, 3)
        # X_test = X + transforms.quaternion_apply(q, dX)
        # logP = log_P(torch.cat([q_test, X_test], dim=-1))
        # logP.sum().backward(inputs = [lie_R, lie_X])
        # grad = torch.cat([lie_X.grad.detach(), lie_R.grad.detach()], dim=-1)

        # ##### Autograd add (very bad) #####
        # T = T.detach()
        # q, X = T[...,:4], T[...,4:]
        # transforms.standardize_quaternion(q/torch.norm(q, dim=-1, keepdim=True))
        # R = transforms.quaternion_to_matrix(q)
        # lie_R, lie_X = torch.zeros_like(X, requires_grad=True), torch.zeros_like(X, requires_grad=True)
        # lie_R_mat = (self.so3_basis * lie_R[..., None, None]).sum(dim=-3)
        # q_test, X_test = transforms.matrix_to_quaternion(R+lie_R_mat), X+lie_X
        # logP = log_P(torch.cat([q_test/q_test.norm(dim=-1, keepdim=True), X_test], dim=-1))
        # logP.sum().backward(inputs = [lie_R, lie_X])
        # grad = torch.cat([lie_X.grad.detach(), lie_R.grad.detach()], dim=-1)

        # ##### Lie from quaternion grad #####
        T = T.detach().requires_grad_(True)
        logP = log_P(T)
        logP.sum().backward(inputs=T)
        grad = T.grad
        L = T.detach()[...,self.q_indices] * self.q_factor
        grad = torch.cat([transforms.quaternion_apply(transforms.quaternion_invert(T[...,:4].detach()), grad[...,4:]), torch.einsum('...ia,...i', L, grad[...,:4])], dim=-1)


        
        # max_grad = 100.
        # grad_norm = grad.norm(dim=-1, keepdim=True)
        # grad_clip = grad_norm > max_grad
        # grad = torch.where(grad_clip, grad / grad_norm * max_grad, grad)

        # grad_X, grad_R = lie_X.grad.detach(), lie_R.grad.detach()
        # max_grad_X = 10.
        # grad_norm_X = grad_X.norm(dim=-1, keepdim=True)
        # grad_clip_X = grad_norm_X > max_grad_X
        # grad_clipped_X = torch.where(grad_clip_X, grad_X / grad_norm_X * max_grad_X, grad_X)
        # max_grad_R = 3.141592
        # grad_norm_R = grad_R.norm(dim=-1, keepdim=True)
        # grad_clip_R = grad_norm_R > max_grad_R
        # grad_clipped_R = torch.where(grad_clip_R, grad_R / grad_norm_R * max_grad_R, grad_R)
        # grad = torch.cat([grad_clipped_X, grad_clipped_R], dim=-1) 

        return logP.detach(), grad

    def in_range(self, T):
        X = T[...,-3:]
        in_range = (self.ranges_X[:,1] >= X) * (X >= self.ranges_X[:,0])
        return in_range.all(dim=-1)

    def run_once(self, T, log_P, return_mask = False, reject=False): # if reject=False, this becomes SGLD, otherwise MALA
        logP_old, grad_old = self.get_logP_with_grad(T, log_P)
        with torch.no_grad():
            T_new = self.propose(T, grad_old, reject=reject)
        #print(-logP_old.detach()[:8])  

        if reject:
            logP_new, grad_new = self.get_logP_with_grad(T_new, log_P)
            with torch.no_grad():
                log_A = self.log_factor_diff(T_new, T, grad_new = grad_new, grad_old = grad_old) + logP_new - logP_old
                A = torch.exp(torch.min(torch.tensor([0.], device = log_A.device), log_A))
                accept_mask = (A > torch.rand(len(A), device=A.device)).unsqueeze(-1)
                T_new = torch.where(accept_mask, T_new, T)
        else:
            #accept_mask = torch.ones(len(T_new), 1, dtype=torch.bool, device=T_new.device)
            accept_mask = self.in_range(T_new).unsqueeze(-1)
            T_new = torch.where(accept_mask, T_new, T)

        if return_mask is True:
            return T_new, accept_mask
        return T_new

    def forward(self, log_P, max_iter, N_transforms = None, T_seed = None, pbar = False, reject = False):
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
            T,A = self.run_once(T, log_P, return_mask = True, reject=reject)
            Ts.append(T)
            As.append(A.squeeze(-1))

        Ts = torch.stack(Ts, dim=0) # (max_iter, Nt, 4+3)
        As = torch.stack(As, dim=0) # (max_iter, Nt)
        return {'Ts':Ts, 'As':As}

    def get_inv_cdf(self):
        self.proposal_dist.dist_R.get_inv_cdf()
