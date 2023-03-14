import warnings
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


class EquivLayerNormJIT(nn.Module):
    def __init__(self, irreps, eps = 1e-5, centerize_vectors = False, affine = True):
        super().__init__()
        self.eps = eps
        self.irreps = irreps
        self.feature_dim = self.irreps.dim
        self.N_group = len(self.irreps)
        self.centerize_vectors = centerize_vectors
        self.affine = affine
        if torch.are_deterministic_algorithms_enabled():
            self.deterministic = True
        else:
            self.deterministic = False

        scatter_index_elementwise = torch.zeros(self.feature_dim, dtype=torch.int64)
        scatter_index_irrepwise = torch.zeros(self.feature_dim, dtype=torch.int64)
        counts_elementwise = []
        counts_irrepwise = []
        scalar_mask_elementwise = []

        idx = 0
        last_scatter_idx_elementwise = -1
        for i, irrep in enumerate(self.irreps):
            count, l, p = irrep[0], irrep[1].l, irrep[1].p
            #assert count > 2
            next_idx = count * (2*l+1) + idx
            scatter_index_elementwise[idx:next_idx] = last_scatter_idx_elementwise + torch.arange(1,2+2*l).repeat(count)
            scatter_index_irrepwise[idx:next_idx] = i
            if l == 0 and p == 1:
                #self.register_parameter(f'offset_{i}', torch.nn.Parameter(torch.zeros(1, requires_grad=True)))
                scalar_mask_elementwise += [True] * (count*(2*l+1))
            else:
                #self.register_buffer(f'offset_{i}', torch.zeros(count * (2*l+1)))
                scalar_mask_elementwise += [False] * (count*(2*l+1))
            last_scatter_idx_elementwise += 2*l+1
            idx = next_idx
            counts_elementwise += [count] * (2*l+1)
            counts_irrepwise.append(count)

        self.register_parameter(f'offset', torch.nn.Parameter(torch.zeros(1, self.feature_dim, requires_grad=True)))

        counts_elementwise = torch.tensor(counts_elementwise, dtype=torch.int64)
        counts_irrepwise = torch.tensor(counts_irrepwise, dtype=torch.int64)
        counts_irrepwise_unbiased = torch.max(torch.tensor(1, dtype=torch.int64), counts_irrepwise - 1)
        scalar_mask_elementwise = torch.tensor(scalar_mask_elementwise, dtype=torch.bool)

        self.max_scatter_index_elementwise = scatter_index_elementwise.max().item()
        self.feature_dim_unique = self.max_scatter_index_elementwise + 1

        self.register_buffer('scatter_index_elementwise', scatter_index_elementwise.unsqueeze(-2), persistent=False)
        self.register_buffer('scatter_index_irrepwise', scatter_index_irrepwise.unsqueeze(-2), persistent=False)
        self.register_buffer('counts_elementwise', counts_elementwise.unsqueeze(-2), persistent=False)
        self.register_buffer('counts_irrepwise', counts_irrepwise.unsqueeze(-2), persistent=False)
        self.register_buffer('counts_irrepwise_unbiased', counts_irrepwise_unbiased.unsqueeze(-2), persistent=False)
        self.register_buffer('scalar_mask_elementwise', scalar_mask_elementwise.unsqueeze(-2), persistent=False)

        #scale_logit = torch.randn(len(self.irreps))/3
        #scale_logit.requires_grad = True
        #self.scale_logit = torch.nn.Parameter(scale_logit)
        self.scale = torch.nn.Parameter(torch.ones(1, len(self.irreps), requires_grad=True))     ################################################################ TODO: register parameter #####################################

    # def get_offset(self):
    #     offset = []
    #     for i, irrep in enumerate(self.irreps):
    #         mul, l, p = irrep[0], irrep[1][0], irrep[1][1]
    #         if l == 0 and p == 1:
    #             offset.append(getattr(self, f'offset_{i}').repeat(mul))
    #         else:
    #             offset.append(getattr(self, f'offset_{i}'))
    #     return torch.cat(offset, dim=-1)

    def get_offset(self):
        return self.offset * self.scalar_mask_elementwise # (1, feature_dim)

    def expand_elementwise(self, inputs: torch.Tensor): # (batch, feature_dim_unique) -> # (batch, feature_dim)
        assert inputs.dim() == 2
        assert inputs.shape[-1] == self.max_scatter_index_elementwise + 1
        if self.deterministic:
            outputs = torch.gather(inputs.cpu(), dim=-1, index = self.scatter_index_elementwise.repeat(len(inputs),1).cpu()).to(inputs.device) 
        else:
            outputs = torch.gather(inputs, dim=-1, index = self.scatter_index_elementwise.repeat(len(inputs),1)) 

        return outputs # (batch, feature_dim)

    def expand_irrepwise(self, inputs: torch.Tensor): # (batch, N_group) -> # (batch, feature_dim_unique)
        assert inputs.dim() == 2
        assert inputs.shape[-1] == len(self.irreps)
        if self.deterministic:
            outputs = torch.gather(inputs.cpu(), dim=-1, index = self.scatter_index_irrepwise.repeat(len(inputs),1).cpu()).to(inputs.device)
        else:
            outputs = torch.gather(inputs, dim=-1, index = self.scatter_index_irrepwise.repeat(len(inputs),1))

        return outputs # (batch, feature_dim_unique)

    def mean_elementwise(self, inputs: torch.Tensor):
        assert inputs.dim() == 2
        assert inputs.shape[-1] == self.feature_dim
        if self.deterministic:
            outputs = scatter(src=inputs.cpu(), index = self.scatter_index_elementwise.cpu(), dim=-1, dim_size=self.feature_dim_unique).to(inputs.device) / self.counts_elementwise # (batch, feature_dim) -> (batch, feature_dim_unique)
        else:
            outputs = scatter(src=inputs, index = self.scatter_index_elementwise, dim=-1, dim_size=self.feature_dim_unique) / self.counts_elementwise # (batch, feature_dim) -> (batch, feature_dim_unique)

        return outputs # (batch, feature_dim_unique)

    def norm2_irrepwise(self, inputs: torch.Tensor):
        assert inputs.dim() == 2
        assert inputs.shape[-1] == self.feature_dim
        if self.deterministic:
            outputs = scatter(src=(inputs**2).cpu(), index = self.scatter_index_irrepwise.cpu(), dim=-1, dim_size=self.N_group).to(inputs.device) / self.counts_irrepwise_unbiased # (batch, feature_dim_unique) -> (batch, N_group)
        else:
            outputs = scatter(src=inputs**2, index = self.scatter_index_irrepwise, dim=-1, dim_size=self.N_group) / self.counts_irrepwise_unbiased # (batch, feature_dim_unique) -> (batch, N_group)

        return outputs # (batch, N_group)

    def zero_center(self, inputs: torch.Tensor):
        assert inputs.dim() == 2
        assert inputs.shape[-1] == self.feature_dim
        mean = self.expand_elementwise(self.mean_elementwise(inputs)) # (batch, feature_dim)
        if self.centerize_vectors is False:
            mean = mean * self.scalar_mask_elementwise # (batch, feature_dim)
        outputs = inputs - mean 
        return outputs # (batch, feature_dim)

    def normalize(self, inputs: torch.Tensor):
        assert inputs.shape[-1] == self.feature_dim
        outputs = self.zero_center(inputs) # (batch, feature_dim)
        outputs = outputs / self.expand_irrepwise((self.norm2_irrepwise(outputs)+self.eps).sqrt()) # (batch, N_group)
        return outputs

    def forward(self, feature: torch.Tensor):
        assert feature.dim() == 2
        if self.affine:
            outputs = self.expand_irrepwise(self.scale) * self.normalize(feature) + self.get_offset()
            #outputs = self.expand_irrepwise(F.softplus(self.scale_logit)) * self.normalize(feature) + self.get_offset()
        else:
            outputs = self.normalize(feature)
        
        return outputs # (batch, feature_dim)


class EquivLayerNorm(nn.Module):
    def __init__(self, irreps, eps = 1e-5, centerize_vectors = False, affine = True):
        super().__init__()
        self.layernorm_jit = torch.jit.script(EquivLayerNormJIT(irreps=irreps, eps=eps, centerize_vectors=centerize_vectors, affine=affine))

    def forward(self, inputs):
        if type(inputs) == dict:
            feature = inputs['feature']
        else:
            feature = inputs

        if feature.dim() > 2:
            outputs = self.layernorm_jit(feature.reshape(-1, *(feature.shape[-2:]))).reshape(*(feature.shape))
        elif feature.dim() == 1:
            outputs = self.layernorm_jit(feature.unsqueeze(0)).squeeze(0)
        else:
            outputs = self.layernorm_jit(feature)

        if type(inputs) == dict:
            outputs = {'feature': outputs}
            for k,v in inputs.items():
                if k not in outputs.keys():
                    outputs[k] = v
        
        return outputs


class ClusteringLayer(nn.Module):
    def __init__(self, max_neighbor_radius, self_connection = False, max_num_neighbors = None):
        super().__init__()
        self.max_radius = max_neighbor_radius
        self.max_num_neighbors = max_num_neighbors
        self.self_connection = self_connection

    def forward(self, inputs):
        feature = inputs['feature']
        pos = inputs['pos']
        
        assert feature.shape[-2] == pos.shape[-2]

        if self.max_num_neighbors is None:
            num_nodes = pos.shape[-2]
            max_num_neighbors = num_nodes -1
        else:
            max_num_neighbors = self.max_num_neighbors
        if self.self_connection is True:
            max_num_neighbors += 1

        edge_src, edge_dst = radius_graph(pos, self.max_radius * 0.999, max_num_neighbors = max_num_neighbors, loop = self.self_connection)
        edge = (edge_src, edge_dst)


        outputs = {'feature': feature, 'pos': pos, 
                   'max_neighbor_radius': self.max_radius, 'edge': edge}
        return outputs


class EdgeSHLayer(nn.Module):
    def __init__(self, sh_lmax, number_of_basis, irrep_normalization = 'norm'):
        super().__init__()
        self.number_of_basis = number_of_basis
        self.irrep_normalization = irrep_normalization
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=sh_lmax, p=1)

    def forward(self, inputs):
        feature = inputs['feature']
        pos = inputs['pos']
        max_radius = inputs['max_neighbor_radius']
        edge = inputs['edge']
        edge_src, edge_dst = edge

        assert feature.shape[-2] == pos.shape[-2]

        edge_vec = pos[edge_dst] - pos[edge_src]
        edge_length = edge_vec.norm(dim=-1)
        edge_length_embedded = soft_one_hot_linspace(
            edge_length,
            start=0.0,
            end=max_radius,
            number=self.number_of_basis,
            basis='smooth_finite',
            cutoff=True
        )
        edge_length_embedded = edge_length_embedded.mul(self.number_of_basis**0.5)
        edge_sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize = True, normalization=self.irrep_normalization)

        outputs = {'feature': feature,  
                  'pos': pos, 'max_neighbor_radius': max_radius,
                  'edge': edge, 'edge_length_embedded': edge_length_embedded, #'edge_length': edge_length,
                  'edge_sh': edge_sh, 'irreps_sh': self.irreps_sh}

        return outputs


class LinearLayerJIT(nn.Module):
    def __init__(self, irreps_list, biases = False, path_normalization = 'element'):
        super().__init__()
        assert len(irreps_list) >= 2
        self.irreps_list = irreps_list
        if path_normalization == 'none':
            path_normalization = 'element'

        layers = []
        for i in range(len(self.irreps_list)-2):
            layers.append(
                o3.Linear(self.irreps_list[i], self.irreps_list[i+1], biases = biases, path_normalization = path_normalization)
            )
            nonlinearity = e3nn.nn.NormActivation(self.irreps_list[i+1], F.gelu)
            nonlinearity.biases = 0
            layers.append(nonlinearity)
        layers.append(
            o3.Linear(self.irreps_list[-2], self.irreps_list[-1], biases = biases, path_normalization = path_normalization)
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, feature):

        outputs = self.layers(feature)

        return outputs

class LinearLayer(nn.Module):
    def __init__(self, irreps_list, biases = False, path_normalization = 'element'):
        super().__init__()
        self.linear = torch.jit.script(LinearLayerJIT(irreps_list=irreps_list, biases=biases, path_normalization=path_normalization))

    def forward(self, inputs):
        if type(inputs) == dict:
            if 'feature' in inputs.keys():
                feature = inputs['feature']
            elif 'field' in inputs.keys():
                feature = inputs['field']
            else:
                raise ValueError
        else:
            feature = inputs
        
        outputs = self.linear(feature)

        if type(inputs) == dict:
            if 'feature' in inputs.keys():
                outputs = {'feature': outputs}
                for k,v in inputs.items():
                    if k not in outputs.keys():
                        outputs[k] = v
            elif 'field' in inputs.keys():
                outputs = {'field': outputs}
                for k,v in inputs.items():
                    if k not in outputs.keys():
                        outputs[k] = v
            else:
                raise ValueError


        return outputs

class LinearLayerDeprecated(nn.Module):
    def __init__(self, irreps_list, biases = False, path_normalization = 'element'):
        super().__init__()
        assert len(irreps_list) >= 2
        self.irreps_list = irreps_list
        if path_normalization == 'none':
            path_normalization = 'element'

        layers = []
        for i in range(len(self.irreps_list)-2):
            layers.append(
                o3.Linear(self.irreps_list[i], self.irreps_list[i+1], biases = biases, path_normalization = path_normalization)
            )
            layers.append(
                e3nn.nn.NormActivation(self.irreps_list[i+1], F.gelu)
            )
        layers.append(
            o3.Linear(self.irreps_list[-2], self.irreps_list[-1], biases = biases, path_normalization = path_normalization)
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        if type(inputs) == dict:
            if 'feature' in inputs.keys():
                feature = inputs['feature']
            elif 'field' in inputs.keys():
                feature = inputs['field']
            else:
                raise ValueError
        else:
            feature = inputs
        
        outputs = self.layers(feature)

        if type(inputs) == dict:
            if 'feature' in inputs.keys():
                outputs = {'feature': outputs}
                for k,v in inputs.items():
                    if k not in outputs.keys():
                        outputs[k] = v
            elif 'field' in inputs.keys():
                outputs = {'field': outputs}
                for k,v in inputs.items():
                    if k not in outputs.keys():
                        outputs[k] = v
            else:
                raise ValueError


        return outputs

class TensorProductLayer(nn.Module):
    def __init__(self, irreps_input, irreps_output, sh_lmax, number_of_basis, irrep_normalization = 'norm', path_normalization = 'element'):
        super().__init__()
        self.irrep_normalization = irrep_normalization
        self.path_normalization = path_normalization
        self.number_of_basis = number_of_basis

        self.irreps_in = irreps_input
        self.irreps_out = irreps_output
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=sh_lmax, p=1)

        self.tp = o3.FullyConnectedTensorProduct(self.irreps_in, self.irreps_sh, self.irreps_out, shared_weights=False, internal_weights=False, irrep_normalization = self.irrep_normalization, path_normalization = self.path_normalization)
        self.fc = e3nn.nn.FullyConnectedNet([self.number_of_basis, 16, self.tp.weight_numel], act=torch.nn.functional.silu)

    def forward(self, inputs):
        feature = inputs['feature']
        edge_src, edge_dst = inputs['edge']
        edge_length_embedded = inputs['edge_length_embedded']
        edge_sh = inputs['edge_sh']

        assert feature.shape[-1] == self.irreps_in.dim
        assert self.irreps_sh == inputs['irreps_sh']

        outputs = {'feature': self.tp(feature[edge_dst], edge_sh, self.fc(edge_length_embedded))}
        for k,v in inputs.items():
            if k not in outputs.keys():
                outputs[k] = v

        return outputs

class EquivariantSelfAttention(nn.Module):
    def __init__(self, irreps_input, irreps_query, irreps_key, irreps_output, sh_lmax, number_of_basis, layernorm = False, irrep_normalization = 'norm', path_normalization = 'element'):
        super().__init__()
        self.irrep_normalization = irrep_normalization
        self.path_normalization = path_normalization
        self.number_of_basis = number_of_basis

        self.irreps_in = irreps_input
        self.irreps_out = irreps_output
        self.irreps_key = irreps_key
        self.irreps_query = irreps_query
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=sh_lmax, p=1)

        #self.query = EmbeddingLayer(self.irreps_in, self.irreps_query, biases = False, path_normalization = self.path_normalization)
        self.query = LinearLayer([self.irreps_in, self.irreps_query], biases = False, path_normalization = self.path_normalization)
        self.key = TensorProductLayer(self.irreps_in, self.irreps_key, sh_lmax = sh_lmax, number_of_basis = self.number_of_basis, irrep_normalization = self.irrep_normalization, path_normalization = self.path_normalization)
        self.value = TensorProductLayer(self.irreps_in, self.irreps_out, sh_lmax = sh_lmax, number_of_basis = self.number_of_basis, irrep_normalization = self.irrep_normalization, path_normalization = self.path_normalization)

        if layernorm:
            self.layernorm = EquivLayerNorm(self.irreps_out, eps=1e-5)
        else:
            self.layernorm = lambda x:x

    def forward(self, inputs):
        edge_src = inputs['edge'][0]

        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        num_nodes = q['feature'].shape[-2]
        num_group = len(self.irreps_key.ls)
        
        assert self.irreps_query == self.irreps_key
        logit = torch.einsum('...i,...i->...', q['feature'][edge_src], k['feature']).unsqueeze(-1)/ np.sqrt(num_group) # (EdgeNum,1)
        
        if torch.are_deterministic_algorithms_enabled():
            log_Z = scatter_logsumexp(logit.cpu(), edge_src.cpu(), dim=-2, dim_size = num_nodes).to(logit.device) # (NodeNum,1)
        else:
            log_Z = scatter_logsumexp(logit, edge_src, dim=-2, dim_size = num_nodes) # (NodeNum,1)
        log_alpha = logit - log_Z[edge_src] # (EdgeNum,1)
        alpha = torch.exp(log_alpha) # (EdgeNum,1)

        if torch.are_deterministic_algorithms_enabled():
            outputs = scatter((alpha * v['feature']).cpu(), edge_src.cpu(), dim=-2, dim_size = num_nodes).to(alpha.device)
        else:
            outputs = scatter(alpha * v['feature'], edge_src, dim=-2, dim_size = num_nodes)
        outputs = self.layernorm(outputs)

        outputs = {'feature': outputs}
        for k,v in inputs.items():
            if k not in outputs.keys():
                outputs[k] = v

        return outputs

class SE3TransformerLayer(nn.Module):
    def __init__(self, irreps_input, irreps_query, irreps_key, irreps_output, irreps_linear, sh_lmax, number_of_basis,
                 self_interaction = True, skip_connection = True, layernorm_output = False, 
                 irrep_normalization = 'norm', path_normalization = 'element'):
        super().__init__()
        self.irrep_normalization = irrep_normalization
        self.path_normalization = path_normalization
        self.number_of_basis = number_of_basis

        self.irreps_in = o3.Irreps(irreps_input)
        self.irreps_out = o3.Irreps(irreps_output)
        self.irreps_query = o3.Irreps(irreps_query)
        self.irreps_key = o3.Irreps(irreps_key)
        self.irreps_linear = o3.Irreps(irreps_linear)

        self.attention = EquivariantSelfAttention(irreps_input = self.irreps_in, irreps_output = self.irreps_out, irreps_query = self.irreps_query, irreps_key = self.irreps_key, sh_lmax = sh_lmax, number_of_basis = number_of_basis, 
                                                  layernorm = self_interaction, irrep_normalization = irrep_normalization, path_normalization = path_normalization)              # Layernorm if self interaction is used
        if self_interaction:
            self.self_interaction = nn.Sequential(LinearLayer([self.irreps_in, self.irreps_out], biases = False, path_normalization = self.path_normalization),
                                                  EquivLayerNorm(self.irreps_out, eps=1e-5))
        else:
            self.self_interaction = lambda x:0.


        self.linear = LinearLayer([self.irreps_out, self.irreps_linear, self.irreps_out], biases = True, path_normalization = self.path_normalization)
        if skip_connection:
            self.skip = lambda x:x
        else:
            self.skip = lambda x:0.        

        if layernorm_output is True:
            self.layernorm_output = EquivLayerNorm(self.irreps_out, eps=1e-5)
        else:
            self.layernorm_output = lambda x:x

    def forward(self, inputs):
        outputs = self.attention(inputs)['feature']
        outputs = (outputs + self.self_interaction(inputs['feature'])) / np.sqrt(2)
        skip = self.skip(outputs)
        outputs = self.linear(outputs)
        outputs = (outputs + skip) / np.sqrt(2)
        outputs = self.layernorm_output(outputs)
            
        outputs = {'feature': outputs}
        for k,v in inputs.items():
            if k not in outputs.keys():
                outputs[k] = v

        return outputs


@torch.jit.script
def soft_step(x, n: int = 5):
    return (x>0) * ((x<1)*((n+1)*x.pow(n)-n*x.pow(n+1)) + (x>=1))

@torch.jit.script
def soft_cutoff(x, thr:float = 0.4, n:int = 5, offset:float = 0.):
    x = (x-thr) / (1-thr)
    x = x*(1+offset)
    return 1-soft_step(x, n=n)

@torch.jit.script
def gaussian_kernels(x, n_basis: int, cutoff_start: bool = False, cutoff_end: bool = False):
    assert x.dim() == 1 
    dx = 1/(n_basis-1)
    mean = torch.linspace(0+int(cutoff_start)*dx, 1-int(cutoff_end)*dx, n_basis, device=x.device, dtype=x.dtype)
    val = (x.unsqueeze(-1) - mean) /  dx
    val = val.pow(2).neg().exp().div(1.12)
    if cutoff_start is True:
        val = val * soft_cutoff(1-x, thr = 0.8, n=4, offset=0.1).unsqueeze(-1)
    if cutoff_end is True:
        val = val * soft_cutoff(x, thr = 0.8, n=4).unsqueeze(-1)
    return val


class QuerySHLayerJIT(nn.Module):
    def __init__(self, sh_lmax, number_of_basis, field_cutoff, irrep_normalization = 'norm'):
        super().__init__()
        self.number_of_basis = number_of_basis
        self.irrep_normalization = irrep_normalization
        self.cutoff = field_cutoff

        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=sh_lmax, p=1)
        self.sh = o3.SphericalHarmonics(self.irreps_sh, normalize=True, normalization=self.irrep_normalization)

    # @torch.jit.ignore
    # def soft_one_hot_linspace_(self, edge_length):
    #     edge_length_embedded = soft_one_hot_linspace(
    #         edge_length,
    #         start=0.0,
    #         end=self.cutoff,
    #         number=self.number_of_basis,
    #         basis='smooth_finite',
    #         cutoff=True
    #     ) # (N_edge, nBasis)
    #     return edge_length_embedded

    def soft_one_hot_linspace_(self, edge_length):
        edge_length_embedded = gaussian_kernels(edge_length/self.cutoff, n_basis=self.number_of_basis, cutoff_start=True, cutoff_end=False)
        return edge_length_embedded

    def soft_one_hot_linspace(self, edge_length):
        assert edge_length.dim()==1 #(N_edges)
        
    def forward(self, pos, query_pos, feature): # x: (nx,3), y:(Nt,ny,3)
        x = pos
        y = query_pos
        assert len(y.shape) == 3
        
        Nt, Ny = y.shape[0:2]
        y = y.reshape(-1,3)

        edges = radius(x, y, r = self.cutoff, max_num_neighbors=x.shape[-2], num_workers = 1) # src: Yidx (query), dst: Xidx (input poincloud)
        edge_src, edge_dst = edges[0], edges[1]
        #print(len(edge_src))

        edge_vec = y[edge_src] - x[edge_dst] # (N_edge,3)
        # edge_length = edge_vec.norm(dim=-1, p=2) # (N_edge,)
        edge_length = (edge_vec.square().sum(dim=-1) + 1e-8).sqrt() # (N_edge,)
        edge_length_embedded = self.soft_one_hot_linspace_(edge_length)
        edge_length_embedded = edge_length_embedded.mul(self.number_of_basis**0.5) # (N_edge, nBasis)
        edge_length_normalized = edge_length / self.cutoff

        edge_sh = self.sh(edge_vec)

        # outputs = {'N_transforms': Nt, 'N_query': Ny, 'irreps_sh': self.irreps_sh, 
        #            'edge': (edge_src, edge_dst), 'edge_sh':edge_sh, 'edge_length_embedded':edge_length_embedded}
        # return outputs
        return Nt, Ny, feature, edge_src, edge_dst, edge_length_embedded, edge_sh, edge_length_normalized

class QuerySHLayer(nn.Module):
    def __init__(self, sh_lmax, number_of_basis, field_cutoff, irrep_normalization = 'norm'):
        super().__init__()
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=sh_lmax, p=1)
        self.qsh_jit = torch.jit.script(QuerySHLayerJIT(sh_lmax=sh_lmax, number_of_basis=number_of_basis, field_cutoff=field_cutoff, irrep_normalization=irrep_normalization))

    def forward(self, inputs):
        assert type(inputs) == dict
        Nt, Ny, feature, edge_src, edge_dst, edge_length_embedded, edge_sh, edge_length_normalized = self.qsh_jit.forward(pos = inputs['pos'], query_pos = inputs['query_pos'], feature = inputs['feature'])
        outputs = {'N_transforms': Nt, 'N_query': Ny, 'irreps_sh': self.irreps_sh, 'feature': feature,
                   'edge': (edge_src, edge_dst), 'edge_sh':edge_sh, 'edge_length_embedded':edge_length_embedded, 'edge_length_normalized': edge_length_normalized}
        return outputs

class QuerySHLayerDeprecated(nn.Module):
    def __init__(self, sh_lmax, number_of_basis, field_cutoff, irrep_normalization = 'norm'):
        super().__init__()
        self.number_of_basis = number_of_basis
        self.irrep_normalization = irrep_normalization
        self.cutoff = field_cutoff

        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=sh_lmax, p=1)

    def forward(self, inputs): # x: (nx,3), y:(Nt,ny,3)
        assert type(inputs) == dict
    
        x = inputs['pos']
        y = inputs['query_pos']
        assert len(y.shape) == 3
        
        Nt, Ny = y.shape[0:2]
        y = y.reshape(-1,3)

        edge_src, edge_dst = radius(x, y, r = self.cutoff * 1.2, max_num_neighbors=x.shape[-2], num_workers = 1) # src: Yidx (query), dst: Xidx (input poincloud)
        #print(len(edge_src))

        edge_vec = y[edge_src] - x[edge_dst] # (N_edge,3)

        edge_length = edge_vec.norm(dim=-1) # (N_edge,)
        edge_length_embedded = soft_one_hot_linspace(
            edge_length,
            start=0.0,
            end=self.cutoff,
            number=self.number_of_basis,
            basis='smooth_finite',
            cutoff=True
        ) # (N_edge, nBasis)
        edge_length_embedded = edge_length_embedded.mul(self.number_of_basis**0.5) # (N_edge, nBasis)
        edge_sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize = True, normalization=self.irrep_normalization) #(N_edge,irrepdim)

        outputs = {'N_transforms': Nt, 'N_query': Ny, 'irreps_sh': self.irreps_sh, 
                   'edge': (edge_src, edge_dst), 'edge_sh':edge_sh, 'edge_length_embedded':edge_length_embedded}
        if 'feature' in inputs.keys():
            outputs['feature'] = inputs['feature']

        return outputs


@torch.jit.script
def mollifier(x):
    return torch.exp(-1/(1-(2*x-1).square() + 1e-8)) *4.5 * (x>0) * (x<1) 

class TensorFieldLayerJIT(nn.Module):
    def __init__(self, irreps_input, irreps_output, sh_lmax, number_of_basis, layernorm = True, 
                 irrep_normalization = 'norm', path_normalization = 'element', tp_type='direct', vacuum=False):
        super().__init__()
        self.irrep_normalization = irrep_normalization
        self.path_normalization = path_normalization
        self.number_of_basis = number_of_basis
        self.vacuum = vacuum
        if torch.are_deterministic_algorithms_enabled():
            self.deterministic = True
        else:
            self.deterministic = False

        self.irreps_in = irreps_input
        self.irreps_out = irreps_output
        self.out_dim = self.irreps_out.dim
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=sh_lmax, p=1)
        
        if tp_type == 'direct':
            assert self.irreps_in == self.irreps_out
            instructions = []
            for i, irrep in enumerate(self.irreps_out):
                L = irrep[1].l
                for l in range(0, min(L+L, sh_lmax)+1):
                    instruction = (i,l,i,'uvu',True)
                    instructions.append(instruction)
            self.tp = o3.TensorProduct(self.irreps_out, self.irreps_sh, self.irreps_out, instructions = instructions, shared_weights=False, internal_weights=False, irrep_normalization = self.irrep_normalization, path_normalization = self.path_normalization)
        elif tp_type == 'fc':
            self.tp = o3.FullyConnectedTensorProduct(self.irreps_in, self.irreps_sh, self.irreps_out, shared_weights=False, internal_weights=False, irrep_normalization = self.irrep_normalization, path_normalization = self.path_normalization)
        else:
            raise ValueError('Unknown tp_type')

        self.fc = e3nn.nn.FullyConnectedNet([self.number_of_basis, int(1.6*self.number_of_basis), self.tp.weight_numel], act=torch.nn.functional.silu)
        self.linear_out = LinearLayerJIT([self.irreps_out, self.irreps_out], biases = True, path_normalization = self.path_normalization)
        
        if layernorm:
            if layernorm == 'no_affine':
                self.layernorm = EquivLayerNormJIT(self.irreps_out, eps=1e-5, affine=False)
            else:
                self.layernorm = EquivLayerNormJIT(self.irreps_out, eps=1e-5)
        else:
            self.layernorm = None

    def get_vacuum_feature(self):
        return torch.cat((self.vacuum_scalar_feature, self.vacuum_vector_feature), dim=-1)

    def forward(self, Nt: int, Nq: int, feature, edge_src, edge_dst, edge_length_embedded, edge_sh, edge_length=None, vacuum_feature=None):
        # Nt, Nq = inputs['N_transforms'], inputs['N_query']
        # feature = inputs['feature'] # (Nx, dimIrrep)
        # edge_src, edge_dst = inputs['edge'] # src: query, dst: pointcloud
        # edge_length_embedded = inputs['edge_length_embedded'] # (Nedge, NumBasis)
        # edge_sh = inputs['edge_sh'] # (Nedge, dimIrrep)
        # assert self.irreps_sh == inputs['irreps_sh']

        n_neighbor = torch.ones(len(edge_src)).to(feature.device) # (Nedge,)
        if self.deterministic:
            n_neighbor = scatter(n_neighbor.cpu(), edge_src.cpu(), dim=-1, dim_size = Nt * Nq).to(n_neighbor.device).reshape(Nt,Nq) # (Nt,Nq,)
        else:
            n_neighbor = scatter(n_neighbor, edge_src, dim=-1, dim_size = Nt * Nq).reshape(Nt,Nq) # (Nt,Nq,)

        # if self.use_mollifier:
        #     assert edge_length is not None
        #     edge_length_cutoff = mollifier(edge_length) # (Nedge,)
        # else:
        #     edge_length_cutoff = torch.ones_like(edge_length) # (Nedge,)

        outputs = self.tp(feature[edge_dst], edge_sh, self.fc(edge_length_embedded)) # (Nedge, dimIrrep_out)

        #outputs = edge_length_cutoff.unsqueeze(-1) * outputs # (Nedge, dimIrrep_out)
        if self.vacuum is True:
            assert vacuum_feature is not None 
            assert edge_length is not None
            assert vacuum_feature.dim() == 2 # (1, feature_dim)
            # assert (edge_length > torch.tensor(1., device=edge_length.device, dtype=edge_length.dtype)).any().item() is False
            cutoff = soft_cutoff(edge_length).unsqueeze(-1) # (Nedge,1)
            outputs = (cutoff * outputs) + ((1-cutoff) * vacuum_feature) # (N_edge, feature_dim)

        if self.deterministic:
            outputs = scatter(outputs.cpu(), edge_src.cpu(), dim=-2, dim_size = Nt * Nq).to(outputs.device) # (Nt*Nq, feature_dim)
        else:
            outputs = scatter(outputs, edge_src, dim=-2, dim_size = Nt * Nq) # (Nt*Nq, feature_dim)

        if self.vacuum is True:
            vacuum_feature = (n_neighbor.reshape(Nt * Nq) == 0).unsqueeze(-1) * vacuum_feature # (Nt*Nq, feature_dim)
            outputs = outputs + vacuum_feature # (Nt*Nq, feature_dim)
        
        if self.layernorm is not None:
            outputs = self.layernorm(outputs) # (Nt*Nq, feature_dim)
        outputs = self.linear_out(outputs).reshape(Nt,Nq,self.out_dim) # (Nt,Nq,dimIrrep_out)

        #outputs = {'field': outputs, 'n_neighbor': n_neighbor} #(Nt, Nq, d), (Nt, Nq)
        #return outputs
        return outputs, n_neighbor # (Nt, Nq, d), (Nt, Nq)

class TensorFieldLayer(nn.Module):
    def __init__(self, irreps_input, irreps_output, sh_lmax, number_of_basis, layernorm = True, irrep_normalization = 'norm', path_normalization = 'element', tp_type='direct'):
        super().__init__()
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=sh_lmax, p=1)
        self.tf_jit = torch.jit.script(
            TensorFieldLayerJIT(irreps_input=irreps_input, irreps_output=irreps_output, 
                                sh_lmax=sh_lmax, number_of_basis=number_of_basis,
                                layernorm=layernorm, irrep_normalization=irrep_normalization, path_normalization=path_normalization, tp_type=tp_type)
                                )

    def forward(self, inputs):
        Nt, Nq = inputs['N_transforms'], inputs['N_query']
        feature = inputs['feature'] # (Nx, dimIrrep)
        edge_src, edge_dst = inputs['edge'] # src: query, dst: pointcloud
        try:
            edge_length = inputs['edge_length_normalized']
        except KeyError:
            edge_length = None
        edge_length_embedded = inputs['edge_length_embedded'] # (Nedge, NumBasis)
        edge_sh = inputs['edge_sh'] # (Nedge, dimIrrep)
        assert self.irreps_sh == inputs['irreps_sh']

        outputs, n_neighbor = self.tf_jit.forward(Nt=Nt, Nq=Nq, feature=feature, edge_src=edge_src, edge_dst=edge_dst, edge_length=edge_length, edge_length_embedded=edge_length_embedded, edge_sh=edge_sh)
        
        outputs = {'field': outputs, 'n_neighbor': n_neighbor} #(Nt, Nq, d), (Nt, Nq)
        return outputs




class TensorFieldLayerDeprecated(nn.Module):
    def __init__(self, irreps_input, irreps_output, sh_lmax, number_of_basis, layernorm = True, irrep_normalization = 'norm', path_normalization = 'element', tp_type='direct'):
        super().__init__()
        self.irrep_normalization = irrep_normalization
        self.path_normalization = path_normalization
        self.number_of_basis = number_of_basis

        self.irreps_in = irreps_input
        self.irreps_out = irreps_output
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=sh_lmax, p=1)
        
        self.linear_in = lambda x:x
        
        if tp_type == 'direct':
            assert self.irreps_in == self.irreps_out
            instructions = []
            for i, irrep in enumerate(self.irreps_out):
                L = irrep[1].l
                for l in range(0, min(L+L, sh_lmax)+1):
                    instruction = (i,l,i,'uvu',True)
                    instructions.append(instruction)
            self.tp = o3.TensorProduct(self.irreps_out, self.irreps_sh, self.irreps_out, instructions = instructions, shared_weights=False, internal_weights=False, irrep_normalization = self.irrep_normalization, path_normalization = self.path_normalization)
        elif tp_type == 'fc':
            self.tp = o3.FullyConnectedTensorProduct(self.irreps_in, self.irreps_sh, self.irreps_out, shared_weights=False, internal_weights=False, irrep_normalization = self.irrep_normalization, path_normalization = self.path_normalization)
        else:
            raise ValueError('Unknown tp_type')

        self.fc = e3nn.nn.FullyConnectedNet([self.number_of_basis, 16, self.tp.weight_numel], act=torch.nn.functional.silu)
        self.linear_out = LinearLayer([self.irreps_out, self.irreps_out], biases = True, path_normalization = self.path_normalization)
        
        if layernorm:
            if layernorm == 'no_affine':
                self.layernorm = EquivLayerNorm(self.irreps_out, eps=1e-5, affine=False)
            else:
                self.layernorm = EquivLayerNorm(self.irreps_out, eps=1e-5)
        else:
            self.layernorm = lambda x:x

    def get_vacuum_feature(self):
        return torch.cat((self.vacuum_scalar_feature, self.vacuum_vector_feature), dim=-1)

    def forward(self, inputs):
        Nt, Nq = inputs['N_transforms'], inputs['N_query']
        feature = inputs['feature'] # (Nx, dimIrrep)
        edge_src, edge_dst = inputs['edge'] # src: query, dst: pointcloud
        edge_length_embedded = inputs['edge_length_embedded'] # (Nedge, NumBasis)
        edge_sh = inputs['edge_sh'] # (Nedge, dimIrrep)
        assert self.irreps_sh == inputs['irreps_sh']

        n_neighbor = torch.ones(len(edge_src)).to(feature.device) #(Nedge,)
        n_neighbor = scatter(n_neighbor, edge_src, dim=-1, dim_size = Nt * Nq).reshape(Nt,Nq) #(Nt,Nq,)

        feature = self.linear_in(feature)
        outputs = self.tp(feature[edge_dst], edge_sh, self.fc(edge_length_embedded)) # (Nedge, dimIrrep_out)
        outputs = scatter(outputs, edge_src, dim=-2, dim_size = Nt * Nq)
        outputs = self.layernorm(outputs)
        outputs = self.linear_out(outputs).reshape(Nt,Nq,self.irreps_out.dim) # (Nt,Nq,dimIrrep_out)

        outputs = {'field': outputs, 'n_neighbor': n_neighbor} #(Nt, Nq, d), (Nt, Nq)
        return outputs




class IrrepwiseDotProduct(nn.Module):
    def __init__(self, irreps):
        super().__init__()
        irreps_out = o3.Irreps(''.join([f"{mul}x0e+" for mul,ir in irreps])[:-1])
        self.tp = o3.TensorProduct(irreps, irreps, irreps_out, [
                                    (i, i, i, 'uuu', False)
                                    for i, (mul, ir) in enumerate(irreps)
                                ], irrep_normalization='component', path_normalization='element')
        self.register_buffer('inv_normalizer', torch.sqrt(torch.tensor(irreps.ls, dtype=torch.float32)*2 + 1), persistent=False)

    def forward(self, x, y):
        return self.tp(x, y) * self.inv_normalizer

