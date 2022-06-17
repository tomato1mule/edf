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


class EquivLayerNorm(nn.Module):
    def __init__(self, irreps, eps = 1e-8, centerize_vectors = False):
        super().__init__()
        self.eps = eps
        self.irreps = irreps
        self.centerize_vectors = centerize_vectors

        scatter_index_elementwise = torch.zeros(self.irreps.dim, dtype=torch.int64)
        scatter_index_irrepwise = torch.zeros(self.irreps.dim, dtype=torch.int64)
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
                self.register_parameter(f'offset_{i}', torch.nn.Parameter(torch.zeros(1, requires_grad=True)))
                scalar_mask_elementwise += [True] * (count*(2*l+1))
            else:
                self.register_buffer(f'offset_{i}', torch.zeros(count * (2*l+1)))
                scalar_mask_elementwise += [False] * (count*(2*l+1))
            last_scatter_idx_elementwise += 2*l+1
            idx = next_idx
            counts_elementwise += [count] * (2*l+1)
            counts_irrepwise.append(count)

        counts_elementwise = torch.tensor(counts_elementwise, dtype=torch.int64)
        counts_irrepwise = torch.tensor(counts_irrepwise, dtype=torch.int64)
        counts_irrepwise_unbiased = torch.max(torch.tensor(1, dtype=torch.int64), counts_irrepwise - 1)
        scalar_mask_elementwise = torch.tensor(scalar_mask_elementwise, dtype=torch.bool)

        self.max_scatter_index_elementwise = scatter_index_elementwise.max().item()

        self.register_buffer('scatter_index_elementwise', scatter_index_elementwise)
        self.register_buffer('scatter_index_irrepwise', scatter_index_irrepwise)
        self.register_buffer('counts_elementwise', counts_elementwise)
        self.register_buffer('counts_irrepwise', counts_irrepwise)
        self.register_buffer('counts_irrepwise_unbiased', counts_irrepwise_unbiased)
        self.register_buffer('scalar_mask_elementwise', scalar_mask_elementwise)

        #scale_logit = torch.randn(len(self.irreps))/3
        #scale_logit.requires_grad = True
        #self.scale_logit = torch.nn.Parameter(scale_logit)
        self.scale = torch.nn.Parameter(torch.ones(len(self.irreps), requires_grad=True))

    def get_offset(self):
        offset = []
        for i, irrep in enumerate(self.irreps):
            count, l, p = irrep[0], irrep[1].l, irrep[1].p
            if l == 0 and p == 1:
                offset.append(getattr(self, f'offset_{i}').repeat(count))
            else:
                offset.append(getattr(self, f'offset_{i}'))
        return torch.cat(offset, dim=-1)

    def expand_elementwise(self, inputs):
        assert inputs.shape[-1] == self.max_scatter_index_elementwise + 1
        outputs = torch.gather(inputs, dim=-1, index = self.scatter_index_elementwise.repeat(*(inputs.shape[:-1]),1))

        return outputs

    def expand_irrepwise(self, inputs):
        assert inputs.shape[-1] == len(self.irreps)
        outputs = torch.gather(inputs, dim=-1, index = self.scatter_index_irrepwise.repeat(*(inputs.shape[:-1]),1))

        return outputs

    def mean_elementwise(self, inputs):
        assert inputs.shape[-1] == self.irreps.dim
        outputs = scatter(src=inputs, index = self.scatter_index_elementwise, dim=-1) / self.counts_elementwise

        return outputs

    def norm2_irrepwise(self, inputs):
        assert inputs.shape[-1] == self.irreps.dim
        outputs = scatter(src=inputs**2, index = self.scatter_index_irrepwise, dim=-1) / self.counts_irrepwise_unbiased

        return outputs

    def zero_center(self, inputs):
        assert inputs.shape[-1] == self.irreps.dim
        mean = self.expand_elementwise(self.mean_elementwise(inputs))
        if self.centerize_vectors is False:
            mean = mean * self.scalar_mask_elementwise
        outputs = inputs - mean
        return outputs

    def normalize(self, inputs):
        assert inputs.shape[-1] == self.irreps.dim
        outputs = self.zero_center(inputs)
        outputs = outputs / self.expand_irrepwise((self.norm2_irrepwise(outputs)+self.eps).sqrt())
        return outputs

    def forward(self, inputs):
        if type(inputs) == dict:
            feature = inputs['feature']
        else:
            feature = inputs

        outputs = self.expand_irrepwise(self.scale) * self.normalize(feature) + self.get_offset()
        #outputs = self.expand_irrepwise(F.softplus(self.scale_logit)) * self.normalize(feature) + self.get_offset()

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


class LinearLayer(nn.Module):
    def __init__(self, irreps, biases = False, path_normalization = 'element'):
        super().__init__()
        self.irreps = [irrep for irrep in irreps]
        assert len(self.irreps) >= 2
        if path_normalization == 'none':
            path_normalization = 'element'

        layers = []
        for i in range(len(self.irreps)-2):
            layers.append(
                o3.Linear(self.irreps[i], self.irreps[i+1], biases = biases, path_normalization = path_normalization)
            )
            layers.append(
                e3nn.nn.NormActivation(self.irreps[i+1], F.gelu)
            )
        layers.append(
            o3.Linear(self.irreps[-2], self.irreps[-1], biases = biases, path_normalization = path_normalization)
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        if type(inputs) == dict:
            feature = inputs['feature']
        else:
            feature = inputs

        outputs = self.layers(feature)

        if type(inputs) == dict:
            outputs = {'feature': outputs}
            for k,v in inputs.items():
                if k not in outputs.keys():
                    outputs[k] = v

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
            self.layernorm = EquivLayerNorm(self.irreps_out, eps=1e-12)
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
        
        log_Z = scatter_logsumexp(logit, edge_src, dim=-2, dim_size = num_nodes) # (NodeNum,1)
        log_alpha = logit - log_Z[edge_src] # (EdgeNum,1)
        alpha = torch.exp(log_alpha) # (EdgeNum,1)

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
                                                  EquivLayerNorm(self.irreps_out, eps=1e-12))
        else:
            self.self_interaction = lambda x:0.


        self.linear = LinearLayer([self.irreps_out, self.irreps_linear, self.irreps_out], biases = True, path_normalization = self.path_normalization)
        if skip_connection:
            self.skip = lambda x:x
        else:
            self.skip = lambda x:0.        

        if layernorm_output is True:
            self.layernorm_output = EquivLayerNorm(self.irreps_out, eps=1e-12)
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



class QuerySHLayer(nn.Module):
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




class TensorFieldLayer(nn.Module):
    def __init__(self, irreps_input, irreps_output, sh_lmax, number_of_basis, N_query = 1, layernorm = True, irrep_normalization = 'norm', path_normalization = 'element'):
        super().__init__()
        self.irrep_normalization = irrep_normalization
        self.path_normalization = path_normalization
        self.number_of_basis = number_of_basis
        self.N_query = N_query

        self.irreps_in = irreps_input
        self.irreps_out = irreps_output
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=sh_lmax, p=1)
        
        self.linear_in = lambda x:x
        
        #"""
        assert self.irreps_in == self.irreps_out
        instructions = []
        for i, irrep in enumerate(self.irreps_out):
            L = irrep[1].l
            for l in range(0, min(L+L, sh_lmax)+1):
                instruction = (i,l,i,'uvu',True)
                instructions.append(instruction)
        self.tp = o3.TensorProduct(self.irreps_out, self.irreps_sh, self.irreps_out, instructions = instructions, shared_weights=False, internal_weights=False, irrep_normalization = self.irrep_normalization, path_normalization = self.path_normalization)
        #"""
                
        #self.tp = o3.FullyConnectedTensorProduct(self.irreps_in, self.irreps_sh, self.irreps_out, shared_weights=False, internal_weights=False, irrep_normalization = self.irrep_normalization, path_normalization = self.path_normalization)

        self.fc = e3nn.nn.FullyConnectedNet([self.number_of_basis, 16, self.tp.weight_numel], act=torch.nn.functional.silu)
        self.linear_out = LinearLayer([self.irreps_out, self.irreps_out], biases = True, path_normalization = self.path_normalization)
        
        if layernorm:
            self.layernorm = EquivLayerNorm(self.irreps_out, eps=1e-12)
        else:
            self.layernorm = lambda x:x

        self.scalar_idx = list(range(0, self.irreps_out[0][0]))
        self.vector_idx = list(range(self.irreps_out[0][0], self.irreps_out.dim))
        vacuum_scalar_feature = torch.randn((self.N_query, len(self.scalar_idx)))
        vacuum_scalar_feature.requires_grad = True
        self.register_parameter('vacuum_scalar_feature', torch.nn.Parameter(vacuum_scalar_feature))
        self.register_buffer('vacuum_vector_feature', torch.zeros((self.N_query, len(self.vector_idx))))

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

        #is_void = (n_neighbor == 0).unsqueeze(-1) #(Nt, Nq, 1)
        #outputs = ~is_void * outputs + is_void * (self.get_vacuum_feature() + 100.)

        outputs = {'field': outputs, 'n_neighbor': n_neighbor} #(Nt, Nq, d), (Nt, Nq)
        return outputs