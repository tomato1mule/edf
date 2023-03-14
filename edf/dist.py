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




class QuaternionUniformDist(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('dummy', torch.tensor([0.]), persistent=False)
        self.is_symmetric = True

    def sample(self, N=1):
        q = transforms.random_quaternions(N, device = self.dummy.device)
        return q

    def propose(self, q):
        N = len(q.view(-1,4))
        q_new = self.sample(N=N)
        q_new = transforms.quaternion_multiply(q.view(-1,4), q_new)
        return q_new.view(*(q.shape))

    def log_factor(self, q2, q1): # log Q(q2 | q1)
        assert q2.shape == q2.shape
        return torch.zeros(q1.shape[:-1], device = q2.device)

    def log_factor_diff(self, q2, q1): # log[ Q(q1 | q2) / Q(q2 | q1) ]
        assert q2.shape == q2.shape
        return torch.zeros(q1.shape[:-1], device = q2.device)








class IgSO3Dist(nn.Module):
    def __init__(self, std):
        super().__init__()
        self.register_buffer('dummy', torch.tensor([0.]), persistent=False)
        self.is_symmetric = True

        self.eps = 0.5 * (std**2)
        self.inverse_cdf = None

    def get_inv_cdf(self):
        X = torch.linspace(0, np.pi, 300, device = self.dummy.device)
        Y = self.isotropic_gaussian_so3(X) * self.haar_measure(X)

        cdf = torch.cumsum(Y, dim=-1)
        cdf = cdf / cdf.max()
        self.inverse_cdf = Interp1D(cdf, X, 'linear') # https://gist.github.com/amarvutha/c2a3ea9d42d238551c694480019a6ce1
       

    def isotropic_gaussian_so3(self, omg, eps = None, lmax = None):
        if eps is None:
            eps = self.eps

        if eps <= 1.:
            return self.isotropic_gaussian_so3_small(omg, eps)

        if lmax is None:
            lmax = max(int( 3. / np.sqrt(eps)) , 2)

        small_number = 1e-9
        sum = 0.
        for l in range(lmax + 1):
            sum = sum +         (2*l+1)    *    np.exp(-l*(l+1) * eps)    *    (  torch.sin((l+0.5)*omg) + (l+0.5)*small_number  )    /    (  torch.sin(omg/2) + 0.5*small_number  )

        return sum

    def isotropic_gaussian_so3_small(self, omg, eps = None):
        if eps is None:
            eps = self.eps

        small_number = 1e-9
        small_num = small_number / 2 
        small_dnm = (1-np.exp(-1. * np.pi**2 / eps)*(2  - 4 * (np.pi**2) / eps   )) * small_number

        return 0.5 * np.sqrt(torch.pi) * (eps ** -1.5) * torch.exp((eps - (omg**2 / eps))/4) / (torch.sin(omg/2) + small_num)            \
            * ( small_dnm + omg - ((omg - 2*torch.pi)*torch.exp(torch.pi * (omg - torch.pi) / eps) + (omg + 2*torch.pi)*torch.exp( -torch.pi * (omg+torch.pi) / eps) ))            

    def haar_measure(self, omg):
        return (1-torch.cos(omg)) / torch.pi

    def sample(self, N=1):
        if self.eps <= 0.05:
            return transforms.axis_angle_to_quaternion(torch.randn(N,3, device = self.dummy.device) * np.sqrt(2*self.eps))

        if self.inverse_cdf is None:
            raise Exception('You should call self.get_inv_cdf() at least once')
        angle = self.inverse_cdf(torch.rand(N, device = self.dummy.device)).unsqueeze(-1)
        axis = F.normalize(torch.randn(N,3, device = self.dummy.device), dim=-1)

        return transforms.axis_angle_to_quaternion(axis * angle)

    def propose(self, q):
        N = len(q.view(-1,4))

        q_new = self.sample(N=N)
        q_new = transforms.quaternion_multiply(q.view(-1,4), q_new)

        return q_new.view(*(q.shape))

    def log_factor(self, q2, q1 = None): # log Q(q2 | q1)
        if q1 is None:
            q_diff = q2
        else:
            assert q2.shape == q1.shape
            q_diff = transforms.quaternion_multiply(transforms.quaternion_invert(q1), q2)

        angle = transforms.quaternion_to_axis_angle(q_diff).norm(dim=-1)
        if self.eps <= 0.05:
            return -0.25 * angle.square() / self.eps # gaussian
        else:
            return torch.log(self.isotropic_gaussian_so3(angle))

    def log_factor_diff(self, q2, q1): # log[ Q(q1 | q2) / Q(q2 | q1) ]
        assert q2.shape == q1.shape
        return torch.zeros(q2.shape[:-1], device = q2.device)











class GaussianDistR3(nn.Module):
    def __init__(self, std):
        super().__init__()
        self.std = std
        self.register_buffer('dummy', torch.tensor([0]), persistent=False)
        self.is_symmetric = True
        self.is_isotropic = True

    def sample(self, N=1):
        return self.std * torch.randn(N,3, device=self.dummy.device)

    def propose(self, X):
        X_new = X.reshape(-1,3)
        N = len(X_new)
        return (X_new + self.sample(N)).reshape(*(X.shape))

    def log_gaussian_factor(self, X):
        return -0.5/(self.std**2) * (X**2).sum(-1)

    def log_factor(self, X2, X1 = None): # log Q(x2 | x1)
        if X1 is None:
            X_diff = X2
        else:
            assert X1.shape == X2.shape
            X_diff = X2 - X1

        return self.log_gaussian_factor(X_diff)

    def log_factor_diff(self, X2, X1): # log[ Q(x1 | x2) / Q(x2 | x1) ]
        assert X2.shape == X1.shape
        return torch.zeros(X2.shape[:-1], device=X2.device)











class UniformDistR3(nn.Module):
    def __init__(self, ranges):
        super().__init__()
        self.register_buffer('ranges', ranges, persistent=False) # (3,2)
        self.is_symmetric = True
        self.is_isotropic = False

    def sample(self, N=1):
        return torch.rand(N,3, device=self.ranges.device) * (self.ranges[:,1] - self.ranges[:,0]) + self.ranges[:,0]

    def propose(self, X):
        N = len(X)
        X_new = self.sample(N)
        return X_new.reshape(*(X.shape))

    def log_factor(self, X2, X1): #Q(X2|X1)
        assert X2.shape == X1.shape
        in_range = (self.ranges[:,1] >= X2) * (X2 >= self.ranges[:,0])
        
        return (~in_range).any(dim=-1) * -30

    def log_factor_diff(self, X2, X1): # log[ Q(x1 | x2) / Q(x2 | x1) ]
        assert X2.shape == X1.shape
        return torch.zeros(X2.shape[:-1], device=X2.device)











class DistSE3(nn.Module):
    def __init__(self, decoupled = False):
        super().__init__()
        self.dist_X = None
        self.dist_R = None
        self.decoupled = decoupled

    def sample(self, N=1):
        q = self.dist_R.sample(N=N)
        x = self.dist_X.sample(N=N)
        x = transforms.quaternion_apply(q, x)    # TODO: this does not work well with UniformDist so needs to be fixed
        return torch.cat([q, x], dim=-1)

    def propose(self, T):                                               
        q_old, X_old = T[...,:4], T[...,4:]                             
        q_new = self.dist_R.propose(q_old)                              
        X_prop = self.dist_X.propose(torch.zeros_like(X_old))           
        if self.decoupled:
            X_new = X_old + X_prop
        else:
            X_new = transforms.quaternion_apply(q_old, X_prop) + X_old
        
        return torch.cat([q_new, X_new], dim=-1)

    def log_factor(self, T2, T1):                                                                                   
        q2, X2 = T2[...,:4], T2[...,4:]                      
        q1, X1 = T1[...,:4], T1[...,4:]                                       
        
        log_factor_q = self.dist_R.log_factor(q2, q1)
        if self.decoupled:
            X_prop = X2-X1
        else:
            X_prop = transforms.quaternion_apply(transforms.quaternion_invert(q1), X2-X1) # X2 and X1 are represented in space frame, but DeltaX is sampled from gaussian in (old) body frame(=q1) so X2-X1 should be transported to q1 body frame
        log_factor_X = self.dist_X.log_factor(X_prop, torch.zeros_like(X_prop))

        return log_factor_q + log_factor_X

    def log_factor_diff(self, T2, T1): # log Q(T1 | T2) / logQ(T2 | T1)                   # (Note that numerator is T1 | T2, not T2 | T1 since we're doing MCMC)
        q2, X2 = T2[...,:4], T2[...,4:]
        q1, X1 = T1[...,:4], T1[...,4:]
        
        log_factor_q_diff = self.dist_R.log_factor_diff(q2, q1)

        if self.dist_X.is_isotropic is True or self.decoupled:
            log_factor_X_diff = self.dist_X.log_factor_diff(X2, X1)
        else:
            X_prop_21 = transforms.quaternion_apply(transforms.quaternion_invert(q1), X2-X1)
            X_prop_12 = transforms.quaternion_apply(transforms.quaternion_invert(q2), X1-X2)
            log_factor_X_21 = self.dist_X.log_factor(X_prop_21, torch.zeros_like(X_prop_21))
            log_factor_X_12 = self.dist_X.log_factor(X_prop_12, torch.zeros_like(X_prop_12))
            log_factor_X_diff = log_factor_X_12 - log_factor_X_21

        return log_factor_q_diff + log_factor_X_diff







class GaussianDistSE3(DistSE3):
    def __init__(self, std_theta, std_X, decoupled = False):
        super().__init__(decoupled=decoupled)
        self.dist_R = IgSO3Dist(std=std_theta)
        self.dist_X = GaussianDistR3(std=std_X)









class UniformDistSE3(DistSE3):
    def __init__(self, ranges_X, decoupled = False):
        super().__init__(decoupled=decoupled)
        self.dist_R = QuaternionUniformDist()
        self.dist_X = UniformDistR3(ranges = ranges_X)










class MixedDistSE3(DistSE3):
    def __init__(self, std_X, decoupled = False):
        super().__init__(decoupled=decoupled)
        self.dist_R = QuaternionUniformDist()
        self.dist_X = GaussianDistR3(std=std_X)








class NewGaussianDistSE3(nn.Module):
    def __init__(self, std_theta, std_X, decoupled = False):
        super().__init__()
        if decoupled:
            raise NotImplementedError
        self.dist_X = GaussianDistR3(std=std_X)
        self.dist_R = IgSO3Dist(std=std_theta)

    def apply_lie(self, T, lie): # lie algebra order : X,R     (following pytorch3d convention)
        q_old, X_old = T[...,:4], T[...,4:]
        lie_X, lie_R = lie[...,:3], lie[...,3:]

        q_prop = transforms.axis_angle_to_quaternion(lie_R)
        X_prop = transforms.se3_exp_map(lie)[...,-1,:3]

        q_new = transforms.quaternion_multiply(q_old, q_prop)
        X_new = transforms.quaternion_apply(q_old, X_prop) + X_old

        return torch.cat([q_new, X_new], dim=-1)

    def sample_lie(self, N=1):
        lie_R = transforms.quaternion_to_axis_angle(self.dist_R.sample(N=N))
        lie_X = self.dist_X.sample(N=N)
        
        return torch.cat([lie_X, lie_R], dim=-1)

    def propose(self, T, lie_offset = None):
        if lie_offset is None:
            lie_offset = torch.zeros(*(T.shape[:-1]),6, device = T.device)

        assert T.shape[:-1] == lie_offset.shape[:-1]

        N = len(T.view(-1,7))
        lie = (self.sample_lie(N=N) + lie_offset).reshape(*(T.shape[:-1]),6)
        
        return self.apply_lie(T, lie)

    def get_diff(self, T2, T1): # T_diff = T1^-1 @ T2
        assert T1.shape == T2.shape

        q2, X2 = T2[...,:4], T2[...,4:]
        q1, X1 = T1[...,:4], T1[...,4:]

        q1_inv = transforms.quaternion_invert(q1)
        q_diff = transforms.quaternion_multiply(q1_inv, q2)
        X_diff = transforms.quaternion_apply(q1_inv, (X2-X1))

        return torch.cat([q_diff, X_diff], dim=-1)

    def logarithm(self, T):
        q, X = T[...,:4], T[...,4:]
        T_mat = torch.cat([transforms.quaternion_to_matrix(q), X.unsqueeze(-1)], dim=-1)
        T_mat = torch.cat([T_mat, torch.tensor([[0., 0., 0., 1.]] ,device = T_mat.device).repeat(*(T_mat.shape[:-2]),1, 1)] , dim=-2)
        lie = transforms.se3_log_map(T_mat.transpose(-1,-2))
        return lie

    def log_factor(self, T2, T1, lie_offset = None):   # logQ(T2 | T1)
        if lie_offset is None:
            lie_offset = torch.zeros(*(T2.shape[:-1]),6, device = T2.device)

        assert T2.shape[:-1] == lie_offset.shape[:-1]

        T_diff = self.get_diff(T2, T1)
        lie_diff = self.logarithm(T_diff)
        lie_diff = lie_diff - lie_offset
        q_diff = transforms.axis_angle_to_quaternion(lie_diff[..., 3:])
        X_diff = lie_diff[..., :3]


        log_factor_q = self.dist_R.log_factor(q_diff)
        log_factor_X = self.dist_X.log_factor(X_diff)

        return log_factor_q + log_factor_X

    def log_factor_diff(self, T2, T1, lie_offset_2 = None, lie_offset_1 = None): # log Q(T1 | T2; grad(T2)) / logQ(T2 | T1; grad(T1))                   # (Note that numerator is T1 | T2, not T2 | T1 since we're doing MCMC)

        return self.log_factor(T1, T2, lie_offset=lie_offset_2) - self.log_factor(T2, T1, lie_offset=lie_offset_1)