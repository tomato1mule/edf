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
from edf.layers import ClusteringLayer, EdgeSHLayer, SE3TransformerLayer, QuerySHLayer, TensorFieldLayer

class SE3TransformerLight(nn.Module):
    def __init__(self, max_neighbor_radius, irreps_out):
        super().__init__()
        sh_lmax = 3
        number_of_basis = 10
        max_neighbor_radius = max_neighbor_radius

        irreps_in = "3x0e"
        irreps_emb = "5x0e + 5x1e + 5x2e"
        irreps_query = "5x0e + 5x1e + 5x2e"
        irreps_key = "5x0e + 5x1e + 5x2e"
        irreps_linear = "10x0e + 10x1e + 10x2e"
        self.irreps_out = irreps_out

        path_normalization = 'element'
        irrep_normalization = 'norm'
        self.model = nn.Sequential( ClusteringLayer(max_neighbor_radius = max_neighbor_radius, 
                                                    self_connection = False),

                                    EdgeSHLayer(sh_lmax = sh_lmax, number_of_basis = number_of_basis, irrep_normalization = irrep_normalization),

                                    SE3TransformerLayer(irreps_input = irreps_in, irreps_query = irreps_query, irreps_key = irreps_key, irreps_output = irreps_emb, 
                                                        irreps_linear = irreps_linear, sh_lmax = sh_lmax, number_of_basis = number_of_basis,
                                                        self_interaction = True, skip_connection = False, layernorm_output = True, 
                                                        irrep_normalization = irrep_normalization, path_normalization = 'none'),

                                    SE3TransformerLayer(irreps_input = irreps_emb, irreps_query = irreps_query, irreps_key = irreps_key, irreps_output = irreps_emb, 
                                                        irreps_linear = irreps_linear, sh_lmax = sh_lmax, number_of_basis = number_of_basis,
                                                        self_interaction = True, skip_connection = True, layernorm_output = True, 
                                                        irrep_normalization = irrep_normalization, path_normalization = path_normalization),

                                    SE3TransformerLayer(irreps_input = irreps_emb, irreps_query = irreps_query, irreps_key = irreps_key, irreps_output = irreps_emb, 
                                                        irreps_linear = irreps_linear, sh_lmax = sh_lmax, number_of_basis = number_of_basis,
                                                        self_interaction = True, skip_connection = True, layernorm_output = True, 
                                                        irrep_normalization = irrep_normalization, path_normalization = path_normalization),

                                    SE3TransformerLayer(irreps_input = irreps_emb, irreps_query = irreps_query, irreps_key = irreps_key, irreps_output = irreps_out,
                                                        irreps_linear = irreps_linear, sh_lmax = sh_lmax, number_of_basis = number_of_basis,
                                                        self_interaction = True, skip_connection = True, layernorm_output = True, 
                                                        irrep_normalization = irrep_normalization, path_normalization = path_normalization),

                                    #LinearLayer([irreps_out, irreps_out], biases = True, path_normalization = path_normalization)
                                  )

    def forward(self, inputs):
        outputs = self.model(inputs)

        return outputs

    def save_tp(self, path):
        assert path[-1] == '/'
        import gzip
        import pickle
        if not os.path.exists(path):
            os.makedirs(path)
        for i in range(2,6):
            with gzip.open(path +f"key_{i}.gzip", 'wb') as f:
                pickle.dump(self.model[i].attention.key.tp, f)
            with gzip.open(path +f"value_{i}.gzip", 'wb') as f:
                pickle.dump(self.model[i].attention.value.tp, f)
            with gzip.open(path +f"linear_{i}.gzip", 'wb') as f:
                pickle.dump(self.model[i].linear.layers[1].scalar_multiplier, f)

    def load_tp(self, path):
        assert path[-1] == '/'
        import gzip
        import pickle
        for i in range(2,6):
            with gzip.open(path +f"key_{i}.gzip", 'rb') as f:
                self.model[i].attention.key.tp = pickle.load(f)
            with gzip.open(path +f"value_{i}.gzip", 'rb') as f:
                self.model[i].attention.value.tp = pickle.load(f)
            with gzip.open(path +f"linear_{i}.gzip", 'rb') as f:
                self.model[i].linear.layers[1].scalar_multiplier = pickle.load(f)
        

class SE3Transformer(nn.Module):
    def __init__(self, max_neighbor_radius, irreps_out):
        super().__init__()
        sh_lmax = 3
        number_of_basis = 10
        max_neighbor_radius = max_neighbor_radius
        C = 8

        irreps_in = "3x0e"
        irreps_emb = f"{C}x0e + {C}x1e + {C//2}x2e + {C//2}x3e"
        irreps_query = f"{C//2}x0e + {C//2}x1e + {C//2}x2e"
        irreps_key = f"{C//2}x0e + {C//2}x1e + {C//2}x2e"
        irreps_linear = f"{C*2}x0e + {C*2}x1e + {C}x2e + {C}x3e"
        self.irreps_out = irreps_out

        path_normalization = 'element'
        irrep_normalization = 'norm'
        self.model = nn.Sequential( ClusteringLayer(max_neighbor_radius = max_neighbor_radius, 
                                                    self_connection = False),

                                    EdgeSHLayer(sh_lmax = sh_lmax, number_of_basis = number_of_basis, irrep_normalization = irrep_normalization),

                                    SE3TransformerLayer(irreps_input = irreps_in, irreps_query = irreps_query, irreps_key = irreps_key, irreps_output = irreps_emb, 
                                                        irreps_linear = irreps_linear, sh_lmax = sh_lmax, number_of_basis = number_of_basis,
                                                        self_interaction = True, skip_connection = False, layernorm_output = True, 
                                                        irrep_normalization = irrep_normalization, path_normalization = 'none'),

                                    SE3TransformerLayer(irreps_input = irreps_emb, irreps_query = irreps_query, irreps_key = irreps_key, irreps_output = irreps_emb, 
                                                        irreps_linear = irreps_linear, sh_lmax = sh_lmax, number_of_basis = number_of_basis,
                                                        self_interaction = True, skip_connection = True, layernorm_output = True, 
                                                        irrep_normalization = irrep_normalization, path_normalization = path_normalization),

                                    SE3TransformerLayer(irreps_input = irreps_emb, irreps_query = irreps_query, irreps_key = irreps_key, irreps_output = irreps_emb, 
                                                        irreps_linear = irreps_linear, sh_lmax = sh_lmax, number_of_basis = number_of_basis,
                                                        self_interaction = True, skip_connection = True, layernorm_output = True, 
                                                        irrep_normalization = irrep_normalization, path_normalization = path_normalization),

                                    SE3TransformerLayer(irreps_input = irreps_emb, irreps_query = irreps_query, irreps_key = irreps_key, irreps_output = irreps_emb, 
                                                        irreps_linear = irreps_linear, sh_lmax = sh_lmax, number_of_basis = number_of_basis,
                                                        self_interaction = True, skip_connection = True, layernorm_output = True, 
                                                        irrep_normalization = irrep_normalization, path_normalization = path_normalization),

                                    SE3TransformerLayer(irreps_input = irreps_emb, irreps_query = irreps_query, irreps_key = irreps_key, irreps_output = irreps_emb, 
                                                        irreps_linear = irreps_linear, sh_lmax = sh_lmax, number_of_basis = number_of_basis,
                                                        self_interaction = True, skip_connection = True, layernorm_output = True, 
                                                        irrep_normalization = irrep_normalization, path_normalization = path_normalization),

                                    SE3TransformerLayer(irreps_input = irreps_emb, irreps_query = irreps_query, irreps_key = irreps_key, irreps_output = irreps_out,
                                                        irreps_linear = irreps_linear, sh_lmax = sh_lmax, number_of_basis = number_of_basis,
                                                        self_interaction = True, skip_connection = True, layernorm_output = True, 
                                                        irrep_normalization = irrep_normalization, path_normalization = path_normalization),

                                    #LinearLayer([irreps_out, irreps_out], biases = True, path_normalization = path_normalization)
                                  )

    def forward(self, inputs):
        outputs = self.model(inputs)

        return outputs





class EnergyModel(nn.Module):
    def __init__(self, N_query, query_radius, field_cutoff,
                 irreps_input, irreps_descriptor, sh_lmax, number_of_basis, ranges,
                 irrep_normalization = 'norm', path_normalization = 'element', learnable_query = True,
                 ):
        super().__init__()
        self.learnable_query = learnable_query
        self.N_query = N_query
        self.query_radius = query_radius
        self.field_cutoff = field_cutoff
        self.register_buffer('ranges', ranges) # (3,2)

        self.irreps_input = o3.Irreps(irreps_input)
        self.irreps_descriptor = o3.Irreps(irreps_descriptor)

        query_points = self.init_query_points(N_query, query_radius)
        self.register_buffer('query_points', query_points)
        query_features = self.irreps_descriptor.randn(self.N_query, -1, requires_grad=True, normalization=irrep_normalization)
        if learnable_query:
            self.register_parameter('query_features', torch.nn.Parameter(query_features))
        self.register_parameter('query_attention', torch.nn.Parameter(torch.zeros(self.N_query, requires_grad=True))) # (N_query,)
        self.register_parameter('scalar_coeff', torch.nn.Parameter(torch.tensor([np.log(3. / (2. * self.irreps_descriptor[0][0]))], requires_grad=True, dtype=torch.float32)))
        #self.register_parameter('vector_coeff', torch.nn.Parameter(torch.tensor([np.log(3. / (0.2 * (len(self.irreps_descriptor.ls) - self.irreps_descriptor[0][0])))], requires_grad=True, dtype=torch.float32))) # Inner prod. type
        self.register_parameter('vector_coeff', torch.nn.Parameter(torch.tensor([np.log(3. / (2. * (len(self.irreps_descriptor.ls) - self.irreps_descriptor[0][0])))], requires_grad=True, dtype=torch.float32))) # MSE type

        self.tensor_field = nn.Sequential(QuerySHLayer(sh_lmax = sh_lmax, number_of_basis = number_of_basis, field_cutoff = field_cutoff, irrep_normalization = irrep_normalization),

                                          TensorFieldLayer(irreps_input = self.irreps_input, irreps_output = self.irreps_descriptor, 
                                                           sh_lmax = sh_lmax, number_of_basis = number_of_basis, N_query=1, #N_query = self.N_query, 
                                                           irrep_normalization = irrep_normalization, path_normalization = path_normalization)
                                         )

    def save_tp(self, path):
        assert path[-1] == '/'
        import gzip
        import pickle
        if not os.path.exists(path):
            os.makedirs(path)
        path = path + "tf.gzip"
        with gzip.open(path, 'wb') as f:
            pickle.dump(self.tensor_field[1].tp, f)

    def load_tp(self, path):
        assert path[-1] == '/'
        import gzip
        import pickle
        path = path + "tf.gzip"
        with gzip.open(path,'rb') as f:
            self.tensor_field[1].tp = pickle.load(f)

    def energy_function(self, descriptor, query_feature, n_neighbor = None):
        scalar_idx = list(range(0, self.irreps_descriptor[0][0]))
        vector_idx = list(range(self.irreps_descriptor[0][0], self.irreps_descriptor.dim))

        scalar_desc = descriptor[..., scalar_idx]
        scalar_query = query_feature[..., scalar_idx]

        vector_desc = descriptor[..., vector_idx]
        vector_query = query_feature[[..., vector_idx]]

        scalar_energy = (scalar_desc-scalar_query).square().sum(dim=-1) # MSE, (Nt, Nq)
        vector_energy = (vector_desc-vector_query).square().sum(dim=-1) # MSE, (Nt, Nq)
        #vector_energy = torch.einsum('...i,...i->...', vector_desc, vector_query) # Inner product, (Nt, Nq)
        attention = F.softmax(self.query_attention, dim=-1) # (Nq,)

        energy = torch.exp(self.scalar_coeff)*scalar_energy + torch.exp(self.vector_coeff)*vector_energy    #(Nt, Nq)
        energy = (energy * attention).sum(dim=-1) #(Nt,)

        if n_neighbor is not None:
            energy = torch.where(n_neighbor.sum(dim=-1) == 0, torch.tensor([500.], device=energy.device), energy) # (Nt,)
            #pass
        return energy # (Nt,)

    def init_query_points(self, N, max_radius):
        query_points = torch.randn((N+20) * 2, 3) / 2 
        idx = (query_points.norm(dim=-1) < 1.).nonzero()
        query_points = query_points[idx].squeeze(-2)[:N]
        query_points = max_radius * query_points

        assert query_points.shape[0] == N
        return query_points

    def transform_query_points(self, T, query_points = None):
        if query_points is None:
            query_points = self.query_points

        q, X = T[...,:4], T[...,4:] # (Nt,4), (Nt,3)
        query_points = transforms.quaternion_apply(q.unsqueeze(-2), query_points) # (Nt, 1, 4) x (Nq, 3) -> (Nt, Nq, 3)
        query_points = query_points + X.unsqueeze(-2) # (Nt, Nq, 3) + (Nt, 1, 3) -> (Nt, Nq, 3)
        return query_points # (Nt, Nq, 3)

    def get_field_value(self, inputs, query_points, input_feature_grad = True):
        assert query_points.dim() == 3 # (Nt, Nq, 3)
        feature = inputs['feature']
        pos = inputs['pos']
        if input_feature_grad is False:
            feature = feature.detach()
            pos = pos.detach()

        inputs = {'feature': feature, 'pos': pos, 'query_pos': query_points}
        outputs = self.tensor_field(inputs)

        return outputs # {'field':(Nt, Nq, dim_descriptor), 'n_neighbor': (Nt, Nq,)}

    def get_energy_transformed(self, inputs, query_points, query_features, temperature = 1., input_feature_grad = True):   # query points and query features should not be detached (to calculate gradient for transformation)
        assert query_points.dim() == 3 # (Nt, Nquery, 3)
        assert query_features.dim() == 3 # (Nt, Nquery, dim_Descriptor)
        assert type(inputs) == dict

        outputs = self.get_field_value(inputs, query_points, input_feature_grad = input_feature_grad) 
        descriptor, n_neighbor =  outputs['field'], outputs['n_neighbor'] # (Nt, Nquery, dim_Descriptor), (Nt, Nquery)

        energy = self.energy_function(descriptor, query_features, n_neighbor = n_neighbor) # (Nt, Nquery, dim_Descriptor), (Nt, Nquery, dim_Descriptor) -> (Nt,)
        energy = energy / temperature
        
        return energy # (Nt,)

    def get_energy(self, inputs, T, query_points, query_features, temperature = 1., input_feature_grad = True, query_feature_grad = True):
        assert type(inputs) == dict
        assert T.dim() == 2
        assert query_points.dim() == query_features.dim() == 2

        if query_feature_grad is False:
            query_features = query_features.detach()

        q, X = T[...,:4], T[...,4:] # (Nt,4), (Nt,3)
        
        query_points = self.transform_query_points(T, query_points = query_points) # (Nt, Nq, 3)

        #D = self.irreps_descriptor.D_from_matrix(transforms.quaternion_to_matrix(q)) # (Nt, irrepdim, irrepdim)
        D = self.irreps_descriptor.D_from_quaternion(q) # (Nt, irrepdim, irrepdim)
        query_features = torch.einsum('tij,nj->tni', D, query_features) # (Nt, d, d) x (nq, d) -> (Nt, nq, d)        
        energy = self.get_energy_transformed(inputs=inputs, query_points=query_points, query_features=query_features, temperature = temperature, input_feature_grad = input_feature_grad) # (Nt,)

        in_range = (self.ranges[:,1] >= X) * (X >= self.ranges[:,0])
        energy = torch.where((~in_range).any(dim=-1), torch.tensor([1000.], device=energy.device), energy) # (Nt,)
        return energy # (Nt,)

    def forward(self, inputs, T, temperature = 1., learning = True):
        assert type(inputs) == dict
        if self.learnable_query is True:
            energy = self.get_energy(inputs = inputs, T = T, query_points = self.query_points, query_features = self.query_features, temperature = temperature, input_feature_grad = learning, query_feature_grad = learning)
        else:
            query_features = inputs['query_feature']
            assert query_features.shape == (self.N_query, self.irreps_descriptor.dim)
            if learning is False:
                query_features = query_features.detach()
            energy = self.get_energy(inputs = inputs, T = T, query_points = self.query_points, query_features = query_features, temperature = temperature, input_feature_grad = learning, query_feature_grad = learning)

        return energy


class QueryTensorField(nn.Module):
    def __init__(self, N_query, field_cutoff,
                 irreps_input, irreps_output, sh_lmax, number_of_basis,
                 irrep_normalization = 'norm', path_normalization = 'element'
                 ):
        super().__init__()
        self.model = nn.Sequential(QuerySHLayer(sh_lmax = sh_lmax, number_of_basis = number_of_basis, field_cutoff = field_cutoff, irrep_normalization = irrep_normalization),
                                TensorFieldLayer(irreps_input = o3.Irreps(irreps_input), irreps_output = o3.Irreps(irreps_output), 
                                                sh_lmax = sh_lmax, number_of_basis = number_of_basis, N_query=N_query, 
                                                irrep_normalization = irrep_normalization, path_normalization = path_normalization))

    def forward(self, inputs):
        return self.model(inputs)

    def save_tp(self, path):
        assert path[-1] == '/'
        import gzip
        import pickle
        if not os.path.exists(path):
            os.makedirs(path)
        path = path + "tf.gzip"
        with gzip.open(path, 'wb') as f:
            pickle.dump(self.model[1].tp, f)

    def load_tp(self, path):
        assert path[-1] == '/'
        import gzip
        import pickle
        path = path + "tf.gzip"
        with gzip.open(path,'rb') as f:
            self.model[1].tp = pickle.load(f)