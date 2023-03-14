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
import torch.autograd.forward_ad as fwAD

from torch_cluster import radius_graph, radius
from torch_scatter import scatter, scatter_logsumexp, scatter_log_softmax, scatter_softmax, scatter_mean
from pytorch3d import transforms
from xitorch.interpolate import Interp1D

import e3nn.nn
from e3nn import o3
from e3nn.math import soft_one_hot_linspace, soft_unit_step

import edf
from edf.visual_utils import plot_color_and_depth, scatter_plot, scatter_plot_ax, visualize_samples, visualize_sample_cluster
from edf.layers import ClusteringLayer, EdgeSHLayer, SE3TransformerLayer, QuerySHLayer, QuerySHLayerJIT, TensorFieldLayer, TensorFieldLayerJIT, LinearLayer, EquivLayerNorm, EquivLayerNormJIT, LinearLayerJIT
from edf.utils import check_irreps_sorted
from edf.wigner import TransformFeatureQuaternion

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

    # def save_tp(self, path):
    #     assert path[-1] == '/'
    #     import gzip
    #     import pickle
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     for i in range(2,6):
    #         with gzip.open(path +f"key_{i}.gzip", 'wb') as f:
    #             pickle.dump(self.model[i].attention.key.tp, f)
    #         with gzip.open(path +f"value_{i}.gzip", 'wb') as f:
    #             pickle.dump(self.model[i].attention.value.tp, f)
    #         # with gzip.open(path +f"linear_{i}.gzip", 'wb') as f:
    #         #     pickle.dump(self.model[i].linear.linear.layers[1].scalar_multiplier, f)

    # def load_tp(self, path):
    #     assert path[-1] == '/'
    #     import gzip
    #     import pickle
    #     for i in range(2,6):
    #         with gzip.open(path +f"key_{i}.gzip", 'rb') as f:
    #             self.model[i].attention.key.tp = pickle.load(f)
    #         with gzip.open(path +f"value_{i}.gzip", 'rb') as f:
    #             self.model[i].attention.value.tp = pickle.load(f)
    #         # with gzip.open(path +f"linear_{i}.gzip", 'rb') as f:
    #         #     self.model[i].linear.linear.layers[1].scalar_multiplier = pickle.load(f)
        

class SE3Transformer(nn.Module):
    def __init__(self, max_neighbor_radius, irreps_out):
        super().__init__()
        sh_lmax = 3
        number_of_basis = 10
        max_neighbor_radius = max_neighbor_radius

        irreps_in = "3x0e"
        irreps_emb = "10x0e + 10x1e + 5x2e + 3x3e"
        irreps_query = "10x0e + 10x1e + 5x2e + 3x3e"
        irreps_key = "10x0e + 10x1e + 5x2e + 3x3e"
        irreps_linear = "20x0e + 20x1e + 10x2e + 6x3e"
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

    # def save_tp(self, path):
    #     assert path[-1] == '/'
    #     import gzip
    #     import pickle
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     for i in range(2,8):
    #         with gzip.open(path +f"key_{i}.gzip", 'wb') as f:
    #             pickle.dump(self.model[i].attention.key.tp, f)
    #         with gzip.open(path +f"value_{i}.gzip", 'wb') as f:
    #             pickle.dump(self.model[i].attention.value.tp, f)
    #         # with gzip.open(path +f"linear_{i}.gzip", 'wb') as f:
    #         #     pickle.dump(self.model[i].linear.linear.layers[1].scalar_multiplier, f)

    # def load_tp(self, path):
    #     assert path[-1] == '/'
    #     import gzip
    #     import pickle
    #     for i in range(2,8):
    #         with gzip.open(path +f"key_{i}.gzip", 'rb') as f:
    #             self.model[i].attention.key.tp = pickle.load(f)
    #         with gzip.open(path +f"value_{i}.gzip", 'rb') as f:
    #             self.model[i].attention.value.tp = pickle.load(f)
    #         # with gzip.open(path +f"linear_{i}.gzip", 'rb') as f:
    #         #     self.model[i].linear.linear.layers[1].scalar_multiplier = pickle.load(f)

class SquareError(nn.Module):
    def __init__(self, irreps, irrep_normalization):
        super().__init__()
        self.dot_product = o3.ElementwiseTensorProduct(irreps, irreps, ["0e"], irrep_normalization=irrep_normalization)

    def forward(self, descriptor, query_feature):
        diff = descriptor - query_feature
        err = self.dot_product(diff, diff) # Mean: 3.2470  ||  Std: 2.2999
        multiplier = 2. ## multiply to make std around 4~5
        return err * multiplier

class IrrepwiseL1Error(nn.Module):
    def __init__(self, irreps, irrep_normalization):
        super().__init__()
        self.dot_product = o3.ElementwiseTensorProduct(irreps, irreps, ["0e"], irrep_normalization=irrep_normalization)

    def forward(self, descriptor, query_feature):
        diff = descriptor - query_feature
        err = self.dot_product(diff, diff).sqrt() # Mean: 1.6600  ||  Std: 0.7149
        multiplier = 6. ## multiply to make std around 4~5
        return err * multiplier

class InnerProductError(nn.Module):
    def __init__(self, irreps, irrep_normalization):
        super().__init__()
        self.dot_product = o3.ElementwiseTensorProduct(irreps, irreps, ["0e"], irrep_normalization=irrep_normalization)

    def forward(self, descriptor, query_feature):
        err = self.dot_product(descriptor, query_feature) # Mean: -0.1325  ||  Std: 1.0026
        multiplier = 4. ## multiply to make std around 4~5
        return err * multiplier

class EnergyModel(nn.Module):
    def __init__(self, N_query, field_cutoff, irreps_input, irreps_descriptor, sh_lmax, number_of_basis, ranges,
                 irrep_normalization = 'norm', path_normalization = 'element', energy_type='mse', layernorm = True, learnable_irrep_weight = False, tp_type='direct'):
        super().__init__()
        # self.requires_graD = True

        self.N_query = N_query
        self.field_cutoff = field_cutoff
        self.register_buffer('ranges', ranges, persistent=False) # (3,2)

        self.irreps_input = o3.Irreps(irreps_input)
        self.irreps_descriptor = o3.Irreps(irreps_descriptor)
        self.feature_dim = self.irreps_descriptor.dim
        self.N_irrep_group = len(self.irreps_descriptor.ls)

        #self.query_model = SimpleQueryModel(irreps_descriptor=self.irreps_descriptor, N_query=self.N_query, query_radius=self.query_radius, irrep_normalization=irrep_normalization)
        #self.learnable_irrep_weight = False
        self.learnable_irrep_weight = learnable_irrep_weight
        if self.learnable_irrep_weight:
            self.register_parameter('irrep_weight_logit', torch.nn.Parameter(1 + torch.zeros(len(self.irreps_descriptor.ls), requires_grad=True, dtype=torch.float32))) # (ls,)
        else:
            self.register_buffer('irrep_weight_logit', torch.nn.Parameter(2 + torch.zeros(len(self.irreps_descriptor.ls), dtype=torch.float32)), persistent=False) # (ls,)


        self.energy_type = energy_type
        if self.energy_type == 'mse':
            self.len_scalar = (np.array(self.irreps_descriptor.ls)==0).sum()
            self.vacuum_feature_multiplier = 1. * np.sqrt(len(self.irreps_descriptor.ls)/self.len_scalar)
            self.register_parameter('vacuum_feature', torch.nn.Parameter(torch.randn(1, self.len_scalar, requires_grad=True, dtype=torch.float32))) # (N_query, len_scalar,)
            self.irrepwise_energy_func = SquareError(irreps=self.irreps_descriptor, irrep_normalization=irrep_normalization)
        elif self.energy_type == 'inner_product':
            self.len_scalar = (np.array(self.irreps_descriptor.ls)==0).sum()
            self.vacuum_feature_multiplier = 1. * np.sqrt(len(self.irreps_descriptor.ls)/self.len_scalar)
            self.register_parameter('vacuum_feature', torch.nn.Parameter(torch.randn(1, self.len_scalar, requires_grad=True, dtype=torch.float32))) # (N_query, len_scalar,)
            self.irrepwise_energy_func = InnerProductError(irreps=self.irreps_descriptor, irrep_normalization=irrep_normalization)
        elif self.energy_type == 'l1':
            self.len_scalar = (np.array(self.irreps_descriptor.ls)==0).sum()
            self.vacuum_feature_multiplier = 0.7 * np.sqrt(len(self.irreps_descriptor.ls)/self.len_scalar)
            self.register_parameter('vacuum_feature', torch.nn.Parameter(torch.randn(1, self.len_scalar, requires_grad=True, dtype=torch.float32))) # (N_query, len_scalar,)
            self.irrepwise_energy_func = IrrepwiseL1Error(irreps=self.irreps_descriptor, irrep_normalization=irrep_normalization)
        else:
            raise ValueError('Wrong energy type')
        self.vacuum_layernorm = EquivLayerNormJIT(irreps=o3.Irreps(f"{self.len_scalar}x0e"), eps=1e-5, centerize_vectors=False, affine=False)

        # self.tensor_field = nn.Sequential(QuerySHLayer(sh_lmax = sh_lmax, number_of_basis = number_of_basis, field_cutoff = field_cutoff, irrep_normalization = irrep_normalization),

        #                                   TensorFieldLayer(irreps_input = self.irreps_input, irreps_output = self.irreps_descriptor, 
        #                                                    sh_lmax = sh_lmax, number_of_basis = number_of_basis, layernorm=layernorm,
        #                                                    irrep_normalization = irrep_normalization, path_normalization = path_normalization)
        #                                  )
        self.qsh = QuerySHLayerJIT(sh_lmax = sh_lmax, number_of_basis = number_of_basis, field_cutoff = field_cutoff, irrep_normalization = irrep_normalization)
        self.tfl = TensorFieldLayerJIT(irreps_input = self.irreps_input, irreps_output = self.irreps_descriptor, sh_lmax = sh_lmax, number_of_basis = number_of_basis, layernorm=layernorm, 
                                       irrep_normalization = irrep_normalization, path_normalization = path_normalization,
                                       tp_type=tp_type, vacuum=True)
        #self.linear_tfl = LinearLayerJIT([self.irreps_descriptor, self.irreps_descriptor], biases = True, path_normalization = "element")


        #Js = [_Jd[ir.l].to(dtype = torch.float32) for mul, ir in self.irreps_descriptor]
        # for mul, ir in self.irreps_descriptor:
        #     self.register_buffer(f'J_{ir.l}', _Jd[ir.l].to(dtype = torch.float32))
        q_example = torch.randn(6,4)
        self.query_feature_transform = torch.jit.trace(TransformFeatureQuaternion(self.irreps_descriptor), 
                                                        example_inputs=(self.irreps_descriptor.randn(5,-1), transforms.standardize_quaternion(q_example/q_example.norm(dim=-1, keepdim=True))))
    
    # def get_Js(self):
    #     return [self.__getattr__(f'J_{ir.l}') for mul, ir in self.irreps_descriptor]
        
    # def requires_grad_(self, requires_grad = True):
    #     self.requires_graD = requires_grad
    #     return super().requires_grad_(requires_grad = requires_grad)

    # @torch.jit.ignore
    # def save_tp(self, path):
    #     assert path[-1] == '/'
    #     import gzip
    #     import pickle
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     # with gzip.open(path + "tf.gzip", 'wb') as f:
    #     #     pickle.dump(self.tensor_field[1].tp, f)
    #     # with gzip.open(path + "dp.gzip", 'wb') as f:
    #     #     pickle.dump(self.dot_product, f)

    # @torch.jit.ignore
    # def load_tp(self, path):
    #     assert path[-1] == '/'
    #     import gzip
    #     import pickle
    #     # with gzip.open(path + "tf.gzip",'rb') as f:
    #     #     self.tensor_field[1].tp = pickle.load(f)
    #     # with gzip.open(path + "dp.gzip",'rb') as f:
    #     #     self.dot_product = pickle.load(f)

    def get_irrep_weight(self):
        return F.softplus(self.irrep_weight_logit) / (0.693147180559945309417*self.N_irrep_group) # 0.693147180559945309417 = log(2)

    def cat_vacuum_feature(self, vacuum_feature):
        assert vacuum_feature.dim() == 2
        device = vacuum_feature.device
        vector_dim = self.feature_dim - vacuum_feature.shape[-1]
        #vector_feature = torch.zeros(*(vacuum_feature.shape[:-1]), vector_dim, device=device, dtype=torch.float32)
        vector_feature = torch.zeros(vacuum_feature.shape[0], vector_dim, device=device, dtype=torch.float32)
        return torch.cat([vacuum_feature, vector_feature], dim=-1)

    def get_vacuum_feature(self):
        #vacuum_feature = self.vacuum_layernorm(self.vacuum_feature) # (1, self.len_scalar) -> (1, self.feature_dim)
        #vacuum_feature = self.vacuum_feature
        vacuum_feature = torch.zeros_like(self.vacuum_feature)
        return self.cat_vacuum_feature(vacuum_feature * self.vacuum_feature_multiplier) # (1, self.feature_dim)

    
    def energy_function(self, descriptor, query_feature, query_attention):   # descriptor, query_feature: (Nt, Nq, feature_dim)
        irrepwise_energy = self.irrepwise_energy_func(descriptor=descriptor, query_feature=query_feature) # (Nt, Nq, ls)

        ### Debug
        #print(irrepwise_energy)
        #print(irrepwise_energy.mean())
        #print(irrepwise_energy.std())
        energy = (irrepwise_energy * self.get_irrep_weight()).sum(dim=-1) #(Nt, Nq)
        energy = (energy * query_attention).sum(dim=-1) #(Nt,)

        return energy # (Nt,)

    def transform_query_points(self, T, query_points):
        q, X = T[...,:4], T[...,4:] # (Nt,4), (Nt,3)
        query_points = transforms.quaternion_apply(q.unsqueeze(-2), query_points) # (Nt, 1, 4) x (Nq, 3) -> (Nt, Nq, 3)
        query_points = query_points + X.unsqueeze(-2) # (Nt, Nq, 3) + (Nt, 1, 3) -> (Nt, Nq, 3)
        return query_points # (Nt, Nq, 3)

    def get_field_value(self, feature, pos, query_points):
        assert query_points.dim() == 3 # (Nt, Nq, 3)

        #inputs = {'feature': feature, 'pos': pos, 'query_pos': query_points}
        #outputs = self.tensor_field(inputs)
        Nt, Ny, feature, edge_src, edge_dst, edge_length_embedded, edge_sh, edge_length_normalized = self.qsh(pos=pos, query_pos=query_points, feature=feature)
        field, n_neighbor = self.tfl(Nt=Nt, Nq=Ny, feature=feature, edge_src=edge_src, edge_dst=edge_dst, edge_length_embedded=edge_length_embedded, edge_sh=edge_sh, edge_length=edge_length_normalized, vacuum_feature = self.get_vacuum_feature())
        
        #return outputs # {'field':(Nt, Nq, dim_descriptor), 'n_neighbor': (Nt, Nq,)}
        return field, n_neighbor # (Nt, Nq, dim_descriptor), (Nt, Nq,)
    
    def get_energy_transformed(self, feature, pos, query_points, query_feature, query_attention, temperature: float):   # query points and query features should not be detached (to calculate gradient for transformation)
        assert query_points.dim() == 3 # (Nt, Nquery, 3)
        assert query_feature.dim() == 3 # (Nt, Nquery, dim_Descriptor)
        assert query_attention.dim() == 1 # (N_query,)

        # outputs = self.get_field_value(feature = feature, pos = pos, query_points = query_points) 
        # descriptor, n_neighbor =  outputs['field'], outputs['n_neighbor'] # (Nt, Nquery, dim_Descriptor), (Nt, Nquery)
        descriptor, n_neighbor = self.get_field_value(feature = feature, pos = pos, query_points = query_points)  # (Nt, Nquery, dim_Descriptor), (Nt, Nquery)

        # if n_neighbor is not None:
        #     vacuum_feature = (n_neighbor == 0).unsqueeze(-1) * self.get_vacuum_feature() # (Nt, Nq, feature_dim==ls)
        # descriptor = descriptor + vacuum_feature
        #descriptor = self.linear_tfl(descriptor)

        energy = self.energy_function(descriptor=descriptor, query_feature=query_feature, query_attention=query_attention) # (Nt,)
        energy = energy / temperature
        
        return energy, descriptor # (Nt,), (Nt, Nquery, dim_Descriptor)

    # def get_wigner_D(self, q):
    #     # D = self.irreps_descriptor.D_from_quaternion(q/q.norm(dim=-1, keepdim=True))
    #     D = self.irreps_descriptor.D_from_angles(*quat_to_angle_fast(q))
    #     return D          # (Nt, feature_len, feature_len)

    def transform_feature(self, q, feature):
        # D = self.get_wigner_D(q)
        # feature_transformed = torch.einsum('tij,qj->tqi', D, feature) # (Nt, f, f) x (Nq, f) -> (Nt, Nq, f)
        #feature_transformed = transform_feature_quat(self.irreps_descriptor, feature, q, self.get_Js())
        feature_transformed = self.query_feature_transform(feature, q)

        return feature_transformed

    def get_energy(self, T: torch.Tensor, feature: torch.Tensor, pos: torch.Tensor, query_points: torch.Tensor, query_feature: torch.Tensor, query_attention: torch.Tensor, temperature: float):
        assert feature.dim() == 2 and pos.dim() == 2   # feature: (N_points, feature_len), pos: (N_points, 3)
        assert T.dim() == 2 # (Nt, 7=4+3)
        assert query_points.dim() == query_feature.dim() == 2  # (N_q, 3), (N_q, feature_len)
        #assert query_feature.shape == vacuum_feature.shape == (self.N_query, self.irreps_descriptor.dim)

        q, X = T[...,:4], T[...,4:] # (Nt,4), (Nt,3)
        q = transforms.standardize_quaternion(q / torch.norm(q, dim=-1, keepdim=True))
        query_points_transformed = self.transform_query_points(T, query_points = query_points) # (Nt, Nq, 3)
        query_feature_transformed = self.transform_feature(q, query_feature) # (Nt, f, f) x (Nq, f) -> (Nt, Nq, f)   
        energy, descriptor = self.get_energy_transformed(feature = feature, pos = pos, 
                                             query_points = query_points_transformed, 
                                             query_feature = query_feature_transformed, 
                                             query_attention = query_attention,
                                             temperature = temperature) # (Nt,), (Nt, Nquery, dim_Descriptor)

        # Penalize out-of-range configurations with high energy.
        in_range = (self.ranges[:,1] >= X) * (X >= self.ranges[:,0])
        energy = torch.where((~in_range).any(dim=-1), torch.tensor([100000.], device=energy.device), energy) # (Nt,)
        return energy, descriptor # (Nt,), (Nt, Nquery, dim_Descriptor)

    def forward(self, T: torch.Tensor, feature: torch.Tensor, pos: torch.Tensor, query_points: torch.Tensor, query_feature: torch.Tensor, query_attention: torch.Tensor, temperature: float = 1.):
        return self.get_energy(T=T, feature=feature, pos=pos, query_points=query_points, query_feature=query_feature, query_attention=query_attention, temperature=temperature)

    @torch.jit.ignore
    def fast_parameters(self):
        if self.learnable_irrep_weight:
            return [self.vacuum_feature, self.irrep_weight_logit]
        else:
            return [self.vacuum_feature,]

    @torch.jit.ignore
    def slow_parameters(self):
        return [p for p in self.parameters() if p not in set(self.fast_parameters())]




class QueryTensorField(nn.Module):
    def __init__(self, field_cutoff,
                 irreps_input, irreps_output, sh_lmax, number_of_basis,
                 irrep_normalization = 'norm', path_normalization = 'element'
                 ):
        super().__init__()
        self.model = nn.Sequential(QuerySHLayer(sh_lmax = sh_lmax, number_of_basis = number_of_basis, field_cutoff = field_cutoff, irrep_normalization = irrep_normalization),
                                TensorFieldLayer(irreps_input = o3.Irreps(irreps_input), irreps_output = o3.Irreps(irreps_output), 
                                                sh_lmax = sh_lmax, number_of_basis = number_of_basis,
                                                irrep_normalization = irrep_normalization, path_normalization = path_normalization))

    def forward(self, inputs):
        return self.model(inputs)

    # def save_tp(self, path):
    #     assert path[-1] == '/'
    #     import gzip
    #     import pickle
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     # path = path + "tf.gzip"
    #     # with gzip.open(path, 'wb') as f:
    #     #     pickle.dump(self.model[1].tp, f)

    # def load_tp(self, path):
    #     assert path[-1] == '/'
    #     import gzip
    #     import pickle
    #     # path = path + "tf.gzip"
    #     # with gzip.open(path,'rb') as f:
    #     #     self.model[1].tp = pickle.load(f)


class QueryModel(nn.Module):
    def __init__(self, irreps_descriptor, N_query, irrep_normalization = 'norm'):
        super().__init__()
        self.irreps_descriptor = irreps_descriptor
        self.N_query = N_query                         # Number of query points
        self.irrep_normalization = irrep_normalization # Assumed normalization type for query feature

    def get_query(self, inputs = None, temperature = None, requires_grad = True):
        raise NotImplementedError

    def forward(self, inputs = None, temperature = None, requires_grad = True, **kwargs):
        return self.get_query(inputs = inputs, temperature = temperature, requires_grad=requires_grad, **kwargs)

    # def save_tp(self, path):
    #     pass

    # def load_tp(self, path):
    #     pass

    def fast_parameters(self):
        raise NotImplementedError

    def slow_parameters(self):
        raise NotImplementedError


class SimpleQueryModel(QueryModel):
    def __init__(self, irreps_descriptor, N_query, query_radius, irrep_normalization = 'norm', layernorm = False, max_N_query = None, query_center = torch.tensor([0., 0., 0.])):
        super().__init__(irreps_descriptor=irreps_descriptor, N_query=N_query, irrep_normalization=irrep_normalization)
        self.query_radius = query_radius   # maximum radius for query points' positions.

        self.register_buffer('query_points', self.init_query_points(self.N_query, self.query_radius, center=query_center)) # (N_query, 3)
        self.register_parameter('query_feature', torch.nn.Parameter(self.irreps_descriptor.randn(self.N_query, -1, requires_grad=True, normalization=irrep_normalization))) # (N_query, feature_len)
        self.register_parameter('query_attention', torch.nn.Parameter(torch.zeros(self.N_query, requires_grad=True))) # (N_query,)
        if layernorm is True:
            self.layernorm = EquivLayerNorm(irreps=self.irreps_descriptor, eps=1e-5, centerize_vectors=False, affine=True)
        elif layernorm == 'no_affine':
            self.layernorm = EquivLayerNorm(irreps=self.irreps_descriptor, eps=1e-5, centerize_vectors=False, affine=False)
        else:
            self.layernorm = nn.Identity()
            self.register_parameter('dummy_param', torch.nn.Parameter(torch.randn(1, requires_grad=True)))
        self.max_N_query = max_N_query

    def init_query_points(self, N, max_radius, center):
        assert center.ndim ==1 and center.shape[-1] == 3
        ### Generates N points within a ball whose radius is max_radius
        query_points = torch.randn((N+20) * 2, 3) / 2 
        idx = (query_points.norm(dim=-1) < 1.).nonzero()
        query_points = query_points[idx].squeeze(-2)[:N]
        query_points = max_radius * query_points + center

        assert query_points.shape[0] == N
        return query_points

    def get_query(self, inputs = None, temperature = None, requires_grad = True):
        if requires_grad is True:
            query_feature = self.query_feature                         # (N_query, feature_len)
            query_feature = self.layernorm(query_feature)
            query_points = self.query_points                           # (N_query, 3)
            if self.max_N_query is not None:
                if self.max_N_query != len(self.query_attention):
                    indices = self.query_attention.argsort(descending=True)[:self.max_N_query] # (max_N_query,)
                    query_feature = query_feature[indices]  # (max_N_query, feature_len)
                    query_points = query_points[indices] # (max_N_query, 3)
                    query_attention = F.softmax(self.query_attention[indices], dim=-1)  # (max_N_query,)
                else:
                    query_attention = F.softmax(self.query_attention, dim=-1)  # (N_query,)
            else:
                query_attention = F.softmax(self.query_attention, dim=-1)  # (N_query,)

            return {'query_feature': query_feature, 'query_points': query_points, 'query_attention': query_attention}
        else:
            with torch.no_grad():
                query_feature = self.query_feature                         # (N_query, feature_len)
                query_feature = self.layernorm(query_feature)
                query_points = self.query_points                           # (N_query, 3)
                if self.max_N_query is not None:
                    if self.max_N_query != len(self.query_attention):
                        indices = self.query_attention.argsort(descending=True)[:self.max_N_query] # (max_N_query,)
                        query_feature = query_feature[indices]  # (max_N_query, feature_len)
                        query_points = query_points[indices] # (max_N_query, 3)
                        query_attention = F.softmax(self.query_attention[indices], dim=-1)  # (max_N_query,)
                    else:
                        query_attention = F.softmax(self.query_attention, dim=-1)  # (N_query,)
                else:
                    query_attention = F.softmax(self.query_attention, dim=-1)  # (N_query,)

                return {'query_feature': query_feature, 'query_points': query_points, 'query_attention': query_attention}

    def fast_parameters(self):
        return [self.query_feature, self.query_attention]

    def slow_parameters(self):
        return [p for p in self.parameters() if p not in set(self.fast_parameters())]


class EquivWeightQueryModel(QueryModel):
    def __init__(self, irreps_descriptor, N_query, max_radius, field_cutoff, sh_lmax, number_of_basis, query_radius, irrep_normalization='norm', max_N_query = None, layernorm = "no_affine"):
        super().__init__(irreps_descriptor=irreps_descriptor, N_query=N_query, irrep_normalization=irrep_normalization)
        self.max_radius = max_radius
        self.field_cutoff = field_cutoff
        self.sh_lmax = sh_lmax
        self.number_of_basis = number_of_basis
        self.irreps_se3T = self.irreps_descriptor
        self.N_query = N_query
        self.max_N_query = max_N_query
        self.query_radius = query_radius
        self.se3T = SE3Transformer(max_neighbor_radius = self.max_radius, irreps_out = self.irreps_se3T)
        self.qsh = QuerySHLayerJIT(sh_lmax = sh_lmax, number_of_basis = number_of_basis, field_cutoff = field_cutoff, irrep_normalization = irrep_normalization)

        self.weight_field = TensorFieldLayerJIT(irreps_input = self.irreps_se3T, irreps_output = o3.Irreps("10x0e"), 
                                                sh_lmax = sh_lmax, number_of_basis = number_of_basis, layernorm=True, 
                                                irrep_normalization = irrep_normalization, path_normalization = "element",
                                                tp_type='fc', vacuum=True)
        self.weight_linear = LinearLayerJIT([o3.Irreps('10x0e'), o3.Irreps('1x0e')], biases = True, path_normalization = "element")

        self.feature_field = TensorFieldLayerJIT(irreps_input = self.irreps_se3T, irreps_output = self.irreps_descriptor, 
                                                sh_lmax = sh_lmax, number_of_basis = number_of_basis, layernorm=layernorm, 
                                                irrep_normalization = irrep_normalization, path_normalization = "element",
                                                tp_type='fc', vacuum=True)
        self.feature_linear = torch.nn.Identity()

    def get_cluster_idx(self, weight, pos, edge, N_query, max_left_points):
        assert weight.dim() == 1
        device = weight.device

        src, dst = edge
        src, dst = src.detach().cpu().numpy(), dst.detach().cpu().numpy()
        weight = weight.detach().cpu().numpy().copy()

        cluster_gather_idx = []
        cluster_point_idx = []
        idx_set = set(range(len(weight)))
        n_cluster = 0
        for iter in range(N_query):
            if len(idx_set) <= max_left_points:
                break
            idx = weight.argmax()
            neighbors = src[dst==idx]
            if len(neighbors) == 0:
                weight[idx] = -10000
                idx_set.remove(idx)
                continue # if no neighbor then pass
            else:
                neighbors = np.append(neighbors, idx)
                weight[neighbors] = -10000
                for i in neighbors:
                    try:
                        idx_set.remove(i)
                        cluster_gather_idx.append(n_cluster)
                        cluster_point_idx.append(i)
                    except KeyError:
                        pass
                n_cluster += 1
        
        cluster_gather_idx = torch.tensor(cluster_gather_idx, dtype=torch.int64, device=device) # (N_points_with_cluster)
        cluster_point_idx = torch.tensor(cluster_point_idx, dtype=torch.int64, device=device) # (N_points_with_cluster)

        return cluster_gather_idx, cluster_point_idx

    def get_init_query_pos(self, pos, edge, weight_logit):
        assert weight_logit.dim() == 1
        cluster_gather_idx, cluster_point_idx = self.get_cluster_idx(weight=weight_logit, pos=pos, edge=edge, 
                                                                     N_query = self.N_query, #if self.max_N_query is None else min(self.N_query, self.max_N_query),
                                                                     max_left_points=int(len(weight_logit)*0.02))
        query_pos = pos[cluster_point_idx]
        if torch.are_deterministic_algorithms_enabled():
            intra_cluster_query_weight = scatter_softmax(weight_logit[cluster_point_idx].cpu(), cluster_gather_idx.cpu(), dim=-1).to(weight_logit.device) # (N_points_with_cluster,3), (N_points_with_cluster,)
            query_pos = scatter((query_pos * intra_cluster_query_weight.unsqueeze(-1)).cpu(), cluster_gather_idx.cpu(), dim=-2, dim_size = cluster_gather_idx.max()+1).to(query_pos.device) # (N_query, 3)
            #query_weight = scatter(F.softmax(weight_logit, dim=-1)[cluster_point_idx], cluster_gather_idx, dim=-1, dim_size = cluster_gather_idx.max()+1) #(N_query, )
        else:
            intra_cluster_query_weight = scatter_softmax(weight_logit[cluster_point_idx], cluster_gather_idx, dim=-1) # (N_points_with_cluster,3), (N_points_with_cluster,)
            query_pos = scatter(query_pos * intra_cluster_query_weight.unsqueeze(-1), cluster_gather_idx, dim=-2, dim_size = cluster_gather_idx.max()+1) # (N_query, 3)
            #query_weight = scatter(F.softmax(weight_logit, dim=-1)[cluster_point_idx], cluster_gather_idx, dim=-1, dim_size = cluster_gather_idx.max()+1) #(N_query, )

        return query_pos #, query_weight

    def rbf(self, x1, x2, h):
        return torch.exp(-(x1-x2).square().sum(dim=-1)/h)

    def rbf_grad_x1(self, x1, x2, h):
        return -2/h * (x1-x2) * self.rbf(x1,x2,h).unsqueeze(-1)

    def stein_vgd_deprecated(self, x, log_P, iters, lr, pbar = False):
        if iters > 0:
            iterator_ = range(iters)
            if pbar:
                iterator_ = tqdm(iterator_)
            x = torch.nn.Parameter(x.detach().clone().requires_grad_(True))
            #optim = torch.optim.Adam(lr=lr, params=[x])
            optim = torch.optim.SGD(lr=lr, params=[x])

            x1 = x.detach()
            graph = radius_graph(x1, r = torch.inf)
            med = (x1[graph[1]] - x1[graph[0]]).norm(dim=-1).median(dim=-1).values
            h = med.square() / np.log(max(len(x1), 1))

            for i in iterator_:
                optim.zero_grad()
                x1 = x.detach()
                rkhs = self.rbf(x1.unsqueeze(1), x1.unsqueeze(0), h) # (Nq, Nq)
                rkhs_grad = self.rbf_grad_x1(x1.detach().unsqueeze(1), x1.unsqueeze(0), h) # (Nq, Nq, 3)
                logP = log_P(x) # (Nq,)
                logP.sum().backward(inputs=x)
                grad = x.grad.detach()
                phi = ((grad.unsqueeze(1) * rkhs.unsqueeze(-1)) + rkhs_grad).mean(dim=0)
                x.grad.detach()[:] = -phi
                optim.step()

        return x.detach().clone()

    def stein_vgd(self, x, log_P, iters, lr, pbar = False):
        requires_grad = x.requires_grad
        if iters > 0:
            iterator_ = range(iters)
            if pbar:
                iterator_ = tqdm(iterator_)

            x1 = x.detach()
            graph = radius_graph(x1, r = torch.inf)
            try:
                if torch.are_deterministic_algorithms_enabled():
                    med = (x1[graph[1]] - x1[graph[0]]).norm(dim=-1).cpu().median(dim=-1).values.to(x1.device)
                else:
                    med = (x1[graph[1]] - x1[graph[0]]).norm(dim=-1).median(dim=-1).values
            except:
                raise NotImplementedError
                print(x1)
                print(graph[1])
                print((x1[graph[1]] - x1[graph[0]]).norm(dim=-1))
                raise ValueError
            h = med.square() / np.log(max(len(x1), 1))

            for i in iterator_:
                rkhs = self.rbf(x.unsqueeze(1), x.unsqueeze(0), h) # (Nq, Nq)
                rkhs_grad = self.rbf_grad_x1(x.unsqueeze(1), x.unsqueeze(0), h) # (Nq, Nq, 3)
                if not requires_grad:
                    x_ = x.detach().requires_grad_(True)
                    grad = torch.autograd.grad(log_P(x_).sum(dim=-1), x_, create_graph = False)[0] # (Nq, 3)
                else:
                    grad = torch.autograd.grad(log_P(x).sum(dim=-1), x, create_graph = True)[0] # (Nq, 3)
                phi = ((grad.unsqueeze(1) * rkhs.unsqueeze(-1)) + rkhs_grad).mean(dim=0)
                x = x + lr*phi

        return x


    def get_weight(self, feature, pos, query_points, temperature):
        Nt, Ny, feature, edge_src, edge_dst, edge_length_embedded, edge_sh, edge_length_normalized = self.qsh(pos=pos, query_pos=query_points, feature=feature)
        field, n_neighbor = self.weight_field(Nt=Nt, Nq=Ny, feature=feature, edge_src=edge_src, edge_dst=edge_dst, edge_length_embedded=edge_length_embedded, edge_sh=edge_sh, edge_length=edge_length_normalized, vacuum_feature = torch.zeros(1,1, device=feature.device))
        weight = self.weight_linear(field) / temperature # (Nt=1, Nq, f=1)
        #weight = weight - 100000.*(n_neighbor == 0).unsqueeze(-1) # (Nt=1, Nq, f=1)
        return weight

    def get_feature(self, feature, pos ,query_points):
        Nt, Ny, feature, edge_src, edge_dst, edge_length_embedded, edge_sh, edge_length_normalized = self.qsh(pos=pos, query_pos=query_points, feature=feature)
        field, n_neighbor = self.feature_field(Nt=Nt, Nq=Ny, feature=feature, edge_src=edge_src, edge_dst=edge_dst, edge_length_embedded=edge_length_embedded, edge_sh=edge_sh, edge_length=edge_length_normalized, vacuum_feature = torch.zeros(1,1, device=feature.device))
        feature = self.feature_linear(field) # (Nt=1, Nq, f)

        return feature

    def get_query(self, inputs, temperature = None, requires_grad = True, stein_iter = 100, stein_lr = 1e-1):      
        if temperature is None:
            temperature = 1.
        if requires_grad is True:
            outputs = self.se3T(inputs)
            feature_se3T, pos = outputs['feature'], outputs['pos']
        else:
            with torch.no_grad():
                outputs = self.se3T(inputs)
                feature_se3T, pos = outputs['feature'].detach(), outputs['pos'].detach()
        num_nodes = pos.shape[-2]
        max_num_neighbors = num_nodes -1
        edge_src, edge_dst = radius_graph(pos.detach(), self.query_radius, max_num_neighbors = max_num_neighbors, loop = False)
        edge = (edge_src, edge_dst)

        if requires_grad is True:
            pos_weight_logit = self.get_weight(feature=feature_se3T, pos=pos, query_points = pos.unsqueeze(0), temperature = temperature).squeeze(0).squeeze(-1)
            query_points_init = self.get_init_query_pos(pos = pos, edge = edge, weight_logit = pos_weight_logit) # (N_query, 3)
        else:
            with torch.no_grad():
                pos_weight_logit = self.get_weight(feature=feature_se3T.detach(), pos=pos.detach(), query_points = pos.detach().unsqueeze(0), temperature = temperature).squeeze(0).squeeze(-1)
                query_points_init = self.get_init_query_pos(pos = pos.detach(), edge = edge, weight_logit = pos_weight_logit.detach()) # (N_query, 3)

        if stein_iter > 0:
            if requires_grad is True:
                log_P = lambda x: self.get_weight(feature=feature_se3T, pos=pos, query_points = x.unsqueeze(0), temperature = temperature).squeeze(0).squeeze(-1) # (N_query)
                #log_P = lambda x: self.get_weight(feature=feature_se3T, pos=pos, query_points = x.unsqueeze(0), temperature = temperature).squeeze(0).squeeze(-1) - x.norm(dim=-1)*4  # Debug
                query_points = self.stein_vgd(x=query_points_init, log_P = log_P, iters=stein_iter, lr = stein_lr) # (N_query, 3)
            else:
                with torch.no_grad():
                    log_P = lambda x: self.get_weight(feature=feature_se3T, pos=pos, query_points = x.unsqueeze(0), temperature = temperature).squeeze(0).squeeze(-1) # (N_query)
                    #log_P = lambda x: self.get_weight(feature=feature_se3T, pos=pos, query_points = x.unsqueeze(0), temperature = temperature).squeeze(0).squeeze(-1) - x.norm(dim=-1)*4  # Debug
                query_points = self.stein_vgd(x=query_points_init.detach(), log_P = log_P, iters=stein_iter, lr = stein_lr) # (N_query, 3)
        else:
            query_points = query_points_init

        if requires_grad is True:
            query_feature = self.get_feature(feature=feature_se3T, pos=pos, query_points = query_points.unsqueeze(0)).squeeze(0) # (N_query, f)
            query_attention = self.get_weight(feature=feature_se3T, pos=pos, query_points = query_points.unsqueeze(0), temperature = temperature).squeeze(0) # (N_query, 1)
        else:
            with torch.no_grad():
                query_feature = self.get_feature(feature=feature_se3T, pos=pos, query_points = query_points.unsqueeze(0)).squeeze(0) # (N_query, f)
                query_attention = self.get_weight(feature=feature_se3T, pos=pos, query_points = query_points.unsqueeze(0), temperature = temperature).squeeze(0) # (N_query, 1)
        # if self.max_N_query is not None:
        #     query_feature = query_feature[query_attention.argsort(dim=-2, descending=True).squeeze(-1)][:self.max_N_query]
        #     query_points = query_points[query_attention.argsort(dim=-2, descending=True).squeeze(-1)][:self.max_N_query]
        #     query_attention = query_attention[query_attention.argsort(dim=-2, descending=True).squeeze(-1)][:self.max_N_query]

        query_attention = F.softmax(query_attention.squeeze(-1), dim=-1) # (N_query,)
        assert query_attention.dim() == 1
        sorted_idx = query_attention.argsort(descending=True)
        if self.max_N_query is not None:
            sorted_idx = sorted_idx[:self.max_N_query]
        query_attention = query_attention[sorted_idx]
        query_feature = query_feature[sorted_idx]
        query_points = query_points[sorted_idx]
        if self.max_N_query is not None:
            query_attention = query_attention / query_attention.sum() # renormalize
        
        
        return {'query_feature': query_feature, 'query_points': query_points, 'query_attention': query_attention}

    def get_query_deprecated(self, inputs, temperature = None, requires_grad = True, stein_iter = 100, stein_lr = 1e-1):      
        if temperature is None:
            temperature = 1.
        if requires_grad is True:
            outputs = self.se3T(inputs)
            feature_se3T, pos = outputs['feature'], outputs['pos']
        else:
            with torch.no_grad():
                outputs = self.se3T(inputs)
                feature_se3T, pos = outputs['feature'].detach(), outputs['pos'].detach()
        num_nodes = pos.shape[-2]
        max_num_neighbors = num_nodes -1
        edge_src, edge_dst = radius_graph(pos.detach(), self.query_radius, max_num_neighbors = max_num_neighbors, loop = False)
        edge = (edge_src, edge_dst)

        if stein_iter > 0 and requires_grad is True:
            pos_weight_logit = self.get_weight(feature=feature_se3T, pos=pos, query_points = pos.unsqueeze(0), temperature = temperature).squeeze(0).squeeze(-1)
            query_points_init = self.get_init_query_pos(pos = pos, edge = edge, weight_logit = pos_weight_logit) # (N_query, 3)
        else:
            with torch.no_grad():
                pos_weight_logit = self.get_weight(feature=feature_se3T.detach(), pos=pos.detach(), query_points = pos.detach().unsqueeze(0), temperature = temperature).squeeze(0).squeeze(-1)
                query_points_init = self.get_init_query_pos(pos = pos.detach(), edge = edge, weight_logit = pos_weight_logit.detach()) # (N_query, 3)

        if stein_iter > 0:
            self.weight_field.requires_grad_(False), self.weight_linear.requires_grad_(False)
            self.weight_field.zero_grad(set_to_none=True), self.weight_linear.zero_grad(set_to_none=True)
            #log_P = lambda x: self.get_weight(feature=feature_se3T.detach(), pos=pos.detach(), query_points = x.unsqueeze(0), temperature = temperature).squeeze(0).squeeze(-1) # (N_query)
            log_P = lambda x: self.get_weight(feature=feature_se3T.detach(), pos=pos.detach(), query_points = x.unsqueeze(0), temperature = temperature).squeeze(0).squeeze(-1) - x.norm(dim=-1)*4  # Debug
            query_points = self.stein_vgd(x=query_points_init, log_P = log_P, iters=stein_iter, lr = stein_lr) # (N_query, 3)
            self.weight_field.requires_grad_(True), self.weight_linear.requires_grad_(True)
        else:
            query_points = query_points_init

        if requires_grad is True:
            query_feature = self.get_feature(feature=feature_se3T, pos=pos, query_points = query_points.unsqueeze(0)).squeeze(0) # (N_query, f)
            query_attention = self.get_weight(feature=feature_se3T, pos=pos, query_points = query_points.unsqueeze(0), temperature = temperature).squeeze(0) # (N_query, 1)
        else:
            with torch.no_grad():
                query_feature = self.get_feature(feature=feature_se3T, pos=pos, query_points = query_points.unsqueeze(0)).squeeze(0) # (N_query, f)
                query_attention = self.get_weight(feature=feature_se3T, pos=pos, query_points = query_points.unsqueeze(0), temperature = temperature).squeeze(0) # (N_query, 1)
        # if self.max_N_query is not None:
        #     query_feature = query_feature[query_attention.argsort(dim=-2, descending=True).squeeze(-1)][:self.max_N_query]
        #     query_points = query_points[query_attention.argsort(dim=-2, descending=True).squeeze(-1)][:self.max_N_query]
        #     query_attention = query_attention[query_attention.argsort(dim=-2, descending=True).squeeze(-1)][:self.max_N_query]

        query_attention = F.softmax(query_attention.squeeze(-1), dim=-1) # (N_query,)
        assert query_attention.dim() == 1
        sorted_idx = query_attention.argsort(descending=True)
        if self.max_N_query is not None:
            sorted_idx = sorted_idx[:self.max_N_query]
        query_attention = query_attention[sorted_idx]
        query_feature = query_feature[sorted_idx]
        query_points = query_points[sorted_idx]
        if self.max_N_query is not None:
            query_attention = query_attention / query_attention.sum() # renormalize
        
        
        return {'query_feature': query_feature, 'query_points': query_points, 'query_attention': query_attention}

    # def save_tp(self, path):
    #     assert path[-1] == '/'
    #     self.se3T.save_tp(path + "se3T/")
    #     import gzip
    #     import pickle
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     # with gzip.open(path + "weight_field_tp.gzip", 'wb') as f:
    #     #     pickle.dump(self.weight_field[1].tp, f)
    #     # with gzip.open(path + "feature_field_tp.gzip", 'wb') as f:
    #     #     pickle.dump(self.feature_field[1].tp, f)

    # def load_tp(self, path):
    #     assert path[-1] == '/'
    #     self.se3T.load_tp(path + "se3T/")
    #     import gzip
    #     import pickle
    #     # with gzip.open(path + "weight_field_tp.gzip",'rb') as f:
    #     #     self.weight_field[1].tp = pickle.load(f)
    #     # with gzip.open(path + "feature_field_tp.gzip",'rb') as f:
    #     #     self.feature_field[1].tp = pickle.load(f)

    def slow_parameters(self):
        return [p for p in self.parameters() if p not in set(self.fast_parameters())]

    def fast_parameters(self):
        return list(self.feature_field.parameters()) + list(self.feature_linear.parameters())






























































































































































































































class EquivWeightQueryModelDeprecated(QueryModel):
    def __init__(self, irreps_descriptor, N_query, max_radius, field_cutoff, sh_lmax, number_of_basis, query_radius, irrep_normalization='norm', max_N_query = None, layernorm = "no_affine"):
        super().__init__(irreps_descriptor=irreps_descriptor, N_query=N_query, irrep_normalization=irrep_normalization)
        self.max_radius = max_radius
        self.field_cutoff = field_cutoff
        self.sh_lmax = sh_lmax
        self.number_of_basis = number_of_basis
        self.irreps_se3T = self.irreps_descriptor
        self.N_query = N_query
        self.max_N_query = max_N_query
        self.query_radius = query_radius
        self.se3T = SE3Transformer(max_neighbor_radius = self.max_radius, irreps_out = self.irreps_se3T)

        self.weight_field = nn.Sequential(QuerySHLayer(sh_lmax = sh_lmax, number_of_basis = number_of_basis, field_cutoff = field_cutoff, irrep_normalization = irrep_normalization),
                                          TensorFieldLayer(irreps_input = self.irreps_se3T, irreps_output = o3.Irreps("10x0e"), 
                                                           sh_lmax = sh_lmax, number_of_basis = number_of_basis, layernorm=True,
                                                           irrep_normalization = irrep_normalization, path_normalization = "element",
                                                           tp_type='fc'),
                                          LinearLayer([o3.Irreps('10x0e'), o3.Irreps('1x0e')], biases = True, path_normalization = "element"))

        self.feature_field = nn.Sequential(QuerySHLayer(sh_lmax = sh_lmax, number_of_basis = number_of_basis, field_cutoff = field_cutoff, irrep_normalization = irrep_normalization),
                                           TensorFieldLayer(irreps_input = self.irreps_se3T, irreps_output = self.irreps_descriptor, 
                                                            sh_lmax = sh_lmax, number_of_basis = number_of_basis, layernorm = layernorm,
                                                            irrep_normalization = irrep_normalization, path_normalization = "element",
                                                            tp_type='fc'),
                                            #LinearLayer([self.irreps_descriptor, self.irreps_descriptor], biases = True, path_normalization = "element")
                                           )

    def get_cluster_idx(self, weight, pos, edge, N_query, max_left_points):
        assert weight.dim() == 1
        device = weight.device

        src, dst = edge
        src, dst = src.detach().cpu().numpy(), dst.detach().cpu().numpy()
        weight = weight.detach().cpu().numpy().copy()

        cluster_gather_idx = []
        cluster_point_idx = []
        idx_set = set(range(len(weight)))
        n_cluster = 0
        for iter in range(N_query):
            if len(idx_set) <= max_left_points:
                break
            idx = weight.argmax()
            neighbors = src[dst==idx]
            if len(neighbors) == 0:
                weight[idx] = -10000
                idx_set.remove(idx)
                continue # if no neighbor then pass
            else:
                neighbors = np.append(neighbors, idx)
                weight[neighbors] = -10000
                for i in neighbors:
                    try:
                        idx_set.remove(i)
                        cluster_gather_idx.append(n_cluster)
                        cluster_point_idx.append(i)
                    except KeyError:
                        pass
                n_cluster += 1
        
        cluster_gather_idx = torch.tensor(cluster_gather_idx, dtype=torch.int64, device=device) # (N_points_with_cluster)
        cluster_point_idx = torch.tensor(cluster_point_idx, dtype=torch.int64, device=device) # (N_points_with_cluster)

        return cluster_gather_idx, cluster_point_idx

    def get_init_query_pos(self, pos, edge, weight_logit):
        assert weight_logit.dim() == 1
        cluster_gather_idx, cluster_point_idx = self.get_cluster_idx(weight=weight_logit, pos=pos, edge=edge, 
                                                                     N_query = self.N_query if self.max_N_query is None else min(self.N_query, self.max_N_query),
                                                                     max_left_points=int(len(weight_logit)*0.02))
        query_pos = pos[cluster_point_idx]
        intra_cluster_query_weight = scatter_softmax(weight_logit[cluster_point_idx], cluster_gather_idx, dim=-1) # (N_points_with_cluster,3), (N_points_with_cluster,)
        query_pos = scatter(query_pos * intra_cluster_query_weight.unsqueeze(-1), cluster_gather_idx, dim=-2, dim_size = cluster_gather_idx.max()+1) # (N_query, 3)
        #query_weight = scatter(F.softmax(weight_logit, dim=-1)[cluster_point_idx], cluster_gather_idx, dim=-1, dim_size = cluster_gather_idx.max()+1) #(N_query, )

        return query_pos #, query_weight

    def get_query(self, inputs, temperature = None):      
        if temperature is None:
            temperature = 1.
        outputs = self.se3T(inputs)
        # feature_se3T, pos, edge = outputs['feature'], outputs['pos'], outputs['edge']
        feature_se3T, pos = outputs['feature'], outputs['pos']
        num_nodes = pos.shape[-2]
        max_num_neighbors = num_nodes -1
        edge_src, edge_dst = radius_graph(pos, self.query_radius * 0.999, max_num_neighbors = max_num_neighbors, loop = False)
        edge = (edge_src, edge_dst)

        pos_weight_logit = self.weight_field({'feature': feature_se3T, 'pos': pos, 'query_pos': pos.unsqueeze(0)})['field'].squeeze(0).squeeze(-1)
        query_points = self.get_init_query_pos(pos = pos, edge = edge, weight_logit = pos_weight_logit)

        inputs = {'feature': feature_se3T, 'pos': pos, 'query_pos': query_points.unsqueeze(0)}
        query_feature = self.feature_field(inputs)['field'].squeeze(0)
        query_attention = self.weight_field(inputs)['field'].squeeze(0)

        # if self.max_N_query is not None:
        #     query_feature = query_feature[query_attention.argsort(dim=-2, descending=True).squeeze(-1)][:self.max_N_query]
        #     query_points = query_points[query_attention.argsort(dim=-2, descending=True).squeeze(-1)][:self.max_N_query]
        #     query_attention = query_attention[query_attention.argsort(dim=-2, descending=True).squeeze(-1)][:self.max_N_query]

        query_attention = F.softmax(query_attention/temperature, dim=-2).squeeze(-1)

        return {'query_feature': query_feature, 'query_points': query_points, 'query_attention': query_attention}

    # def save_tp(self, path):
    #     assert path[-1] == '/'
    #     self.se3T.save_tp(path + "se3T/")
    #     import gzip
    #     import pickle
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     # with gzip.open(path + "weight_field_tp.gzip", 'wb') as f:
    #     #     pickle.dump(self.weight_field[1].tp, f)
    #     # with gzip.open(path + "feature_field_tp.gzip", 'wb') as f:
    #     #     pickle.dump(self.feature_field[1].tp, f)

    # def load_tp(self, path):
    #     assert path[-1] == '/'
    #     self.se3T.load_tp(path + "se3T/")
    #     import gzip
    #     import pickle
    #     # with gzip.open(path + "weight_field_tp.gzip",'rb') as f:
    #     #     self.weight_field[1].tp = pickle.load(f)
    #     # with gzip.open(path + "feature_field_tp.gzip",'rb') as f:
    #     #     self.feature_field[1].tp = pickle.load(f)

    def slow_parameters(self):
        return [p for p in self.parameters() if p not in set(self.fast_parameters())]

    def fast_parameters(self):
        return list(self.feature_field.parameters())




class EquivMultiHeadQueryModel(QueryModel):
    def __init__(self, irreps_descriptor, N_query, max_radius, field_cutoff, sh_lmax, number_of_basis, irrep_normalization='norm'):
        super().__init__(irreps_descriptor=irreps_descriptor, N_query=N_query, irrep_normalization=irrep_normalization)
        self.max_radius = max_radius
        self.field_cutoff = field_cutoff
        self.sh_lmax = sh_lmax
        self.number_of_basis = number_of_basis
        self.irreps_se3T = self.irreps_descriptor
        self.se3T = SE3Transformer(max_neighbor_radius = self.max_radius, irreps_out = self.irreps_se3T)
        self.se3T_feature = SE3Transformer(max_neighbor_radius = self.max_radius, irreps_out = self.irreps_se3T)
        self.attention_field = nn.Sequential(QuerySHLayer(sh_lmax = sh_lmax, number_of_basis = number_of_basis, field_cutoff = field_cutoff, irrep_normalization = irrep_normalization),
                                             TensorFieldLayer(irreps_input = self.irreps_se3T, irreps_output = o3.Irreps("10x0e"), 
                                                              sh_lmax = sh_lmax, number_of_basis = number_of_basis, layernorm=True,
                                                              irrep_normalization = irrep_normalization, path_normalization = "element",
                                                              tp_type='fc'),
                                          LinearLayer([o3.Irreps('10x0e'), o3.Irreps('1x0e')], biases = True, path_normalization = "element"))
        self.pos_weight_field = nn.Sequential(QuerySHLayer(sh_lmax = sh_lmax, number_of_basis = number_of_basis, field_cutoff = field_cutoff, irrep_normalization = irrep_normalization),
                                              TensorFieldLayer(irreps_input = self.irreps_se3T, irreps_output = o3.Irreps(f"{2*self.N_query}x0e"), 
                                                               sh_lmax = sh_lmax, number_of_basis = number_of_basis, layernorm=True,
                                                               irrep_normalization = irrep_normalization, path_normalization = "element",
                                                               tp_type='fc'),
                                              LinearLayer([o3.Irreps(f"{2*self.N_query}x0e"), o3.Irreps(f'{self.N_query}x0e')], biases = True, path_normalization = "element"))
        self.feature_field = nn.Sequential(QuerySHLayer(sh_lmax = sh_lmax, number_of_basis = number_of_basis, field_cutoff = field_cutoff, irrep_normalization = irrep_normalization),
                                           TensorFieldLayer(irreps_input = self.irreps_se3T, irreps_output = self.irreps_descriptor, 
                                                            sh_lmax = sh_lmax, number_of_basis = number_of_basis, layernorm='no_affine',
                                                            irrep_normalization = irrep_normalization, path_normalization = "element",
                                                            tp_type='fc'),
                                           #LinearLayer([self.irreps_descriptor, self.irreps_descriptor], biases = True, path_normalization = "element")
                                           )

        self.pos_disp_field = nn.Sequential(QuerySHLayer(sh_lmax = sh_lmax, number_of_basis = number_of_basis, field_cutoff = field_cutoff, irrep_normalization = irrep_normalization),
                                            TensorFieldLayer(irreps_input = self.irreps_se3T, irreps_output = o3.Irreps(f"{self.N_query*2}x1e"), 
                                                                sh_lmax = sh_lmax, number_of_basis = number_of_basis, layernorm=True,
                                                                irrep_normalization = irrep_normalization, path_normalization = "element",
                                                                tp_type='fc'),
                                            LinearLayer([o3.Irreps(f"{self.N_query*2}x1e"), o3.Irreps(f"{self.N_query}x1e")], biases = True, path_normalization = "element"))

    def get_init_query_pos(self, pos, weight_logit, temperature = 0.03): # pos: (N_points, 3),   weight_logit: (N_points, N_query)
        #weight = F.softmax(weight_logit, dim=-2) # (N_points, N_query)
        weight = F.softmax(weight_logit, dim=-1) + torch.ones_like(weight_logit)*1e-4 # (N_points, N_query)
        weight = weight ** (1/temperature)
        weight = weight / weight.sum(dim=-2) # (N_points, N_query)
        query_pos = torch.einsum('iq,ix->qx', weight, pos) # (N_query, 3)

        return query_pos # (N_query, 3)

    def get_query(self, inputs):      
        outputs = self.se3T(inputs)
        feature_se3T, pos = outputs['feature'], outputs['pos']
        pos_weight_logit = self.pos_weight_field({'feature': feature_se3T, 'pos': pos, 'query_pos': pos.unsqueeze(-3)})['field'].squeeze(-3) # (N_points, N_query)
        query_points = self.get_init_query_pos(pos = pos, weight_logit = pos_weight_logit) # (N_query, 3)

        dt = 1.
        for iter in range(7):
            disp = self.pos_disp_field({'feature': feature_se3T, 'pos': pos, 'query_pos': query_points.unsqueeze(-3)})['field'].squeeze(-3).reshape(self.N_query, self.N_query, 3) # (N_query, N_query, 3)
            disp = torch.diagonal(input = disp, dim1=-3, dim2=-2).T # (N_query, 3)
            #print(disp.norm(dim=-1))
            query_points = query_points + dt*disp
            
        outputs = self.se3T_feature(inputs)
        feature_se3T, pos = outputs['feature'], outputs['pos']

        inputs_feature = {'feature': feature_se3T, 'pos': pos, 'query_pos': query_points.unsqueeze(-3)}
        query_feature = self.feature_field(inputs_feature)['field'].squeeze(-3) # (N_q, N_feat)
        query_attention = self.attention_field(inputs_feature)['field'].squeeze(-3).squeeze(-1) # (N_q)
        query_attention_temperature = 1.
        query_attention = F.softmax(query_attention/query_attention_temperature, dim=-1) # (N_q)

        return {'query_feature': query_feature, 'query_points': query_points, 'query_attention': query_attention}

    # def save_tp(self, path):
    #     assert path[-1] == '/'
    #     self.se3T.save_tp(path + "se3T/")

    # def load_tp(self, path):
    #     assert path[-1] == '/'
    #     self.se3T.load_tp(path + "se3T/")

class EquivWeightQueryModelOld(QueryModel):
    def __init__(self, irreps_descriptor, N_query, max_radius, field_cutoff, sh_lmax, number_of_basis, irrep_normalization='norm', ):
        super().__init__(irreps_descriptor=irreps_descriptor, N_query=N_query, irrep_normalization=irrep_normalization)
        self.max_radius = max_radius
        self.field_cutoff = field_cutoff
        self.sh_lmax = sh_lmax
        self.number_of_basis = number_of_basis
        #self.irreps_se3T = o3.Irreps('1x0e+' + self.irreps_descriptor.__str__()) #(self.irreps_descriptor*2).sort()[0].simplify()
        #self.irreps_se3T = (self.irreps_descriptor*2).sort()[0].simplify()
        self.irreps_se3T = self.irreps_descriptor
        self.se3T = SE3Transformer(max_neighbor_radius = self.max_radius, irreps_out = self.irreps_se3T)
        #self.linear_layer = LinearLayer([self.irreps_mid, self.irreps_out], biases = True, path_normalization = "element")

        self.weight_field = nn.Sequential(QuerySHLayer(sh_lmax = sh_lmax, number_of_basis = number_of_basis, field_cutoff = field_cutoff, irrep_normalization = irrep_normalization),
                                          TensorFieldLayer(irreps_input = self.irreps_se3T, irreps_output = o3.Irreps("10x0e"), 
                                                           sh_lmax = sh_lmax, number_of_basis = number_of_basis, layernorm=True,
                                                           irrep_normalization = irrep_normalization, path_normalization = "element",
                                                           tp_type='fc'),
                                          LinearLayer([o3.Irreps('10x0e'), o3.Irreps('1x0e')], biases = True, path_normalization = "element"))
        # self.pos_weight_field = nn.Sequential(QuerySHLayer(sh_lmax = sh_lmax, number_of_basis = number_of_basis, field_cutoff = field_cutoff, irrep_normalization = irrep_normalization),
        #                                   TensorFieldLayer(irreps_input = self.irreps_se3T, irreps_output = o3.Irreps("1x0e"), 
        #                                                    sh_lmax = sh_lmax, number_of_basis = number_of_basis, layernorm=False,
        #                                                    irrep_normalization = irrep_normalization, path_normalization = "element",
        #                                                    tp_type='fc'))
        #self.pos_weight_field = LinearLayer([self.irreps_se3T, o3.Irreps('1x0e')], biases = True, path_normalization = "element")
        self.feature_field = nn.Sequential(QuerySHLayer(sh_lmax = sh_lmax, number_of_basis = number_of_basis, field_cutoff = field_cutoff, irrep_normalization = irrep_normalization),
                                           TensorFieldLayer(irreps_input = self.irreps_se3T, irreps_output = self.irreps_descriptor, 
                                                            sh_lmax = sh_lmax, number_of_basis = number_of_basis, layernorm='no_affine',
                                                            irrep_normalization = irrep_normalization, path_normalization = "element",
                                                            tp_type='fc'),
                                           #LinearLayer([self.irreps_descriptor, self.irreps_descriptor], biases = True, path_normalization = "element")
                                           )

    def get_cluster_idx(self, weight, pos, edge, max_N_query, max_left_points = 50):
        assert weight.dim() == 1
        device = weight.device

        src, dst = edge
        src, dst = src.detach().cpu().numpy(), dst.detach().cpu().numpy()
        weight = weight.detach().cpu().numpy().copy()

        cluster_gather_idx = []
        cluster_point_idx = []
        idx_set = set(range(len(weight)))
        n_cluster = 0
        for iter in range(max_N_query):
            if len(idx_set) <= max_left_points:
                break
            idx = weight.argmax()
            neighbors = src[dst==idx]
            if len(neighbors) == 0:
                weight[idx] = -10000
                idx_set.remove(idx)
                continue # if no neighbor then pass
            else:
                neighbors = np.append(neighbors, idx)
                weight[neighbors] = -10000
                for i in neighbors:
                    try:
                        idx_set.remove(i)
                        cluster_gather_idx.append(n_cluster)
                        cluster_point_idx.append(i)
                    except KeyError:
                        pass
                n_cluster += 1
        
        cluster_gather_idx = torch.tensor(cluster_gather_idx, dtype=torch.int64, device=device) # (N_points_with_cluster)
        cluster_point_idx = torch.tensor(cluster_point_idx, dtype=torch.int64, device=device) # (N_points_with_cluster)

        return cluster_gather_idx, cluster_point_idx

    def get_init_query_pos(self, pos, edge, weight_logit):
        assert weight_logit.dim() == 1
        cluster_gather_idx, cluster_point_idx = self.get_cluster_idx(weight=weight_logit, pos=pos, edge=edge, 
                                                max_N_query=self.N_query, max_left_points=int(len(weight_logit)*0.02))
        query_pos = pos[cluster_point_idx]
        intra_cluster_query_weight = scatter_softmax(weight_logit[cluster_point_idx], cluster_gather_idx, dim=-1) # (N_points_with_cluster,3), (N_points_with_cluster,)
        query_pos = scatter(query_pos * intra_cluster_query_weight.unsqueeze(-1), cluster_gather_idx, dim=-2, dim_size = cluster_gather_idx.max()+1) # (N_query, 3)
        #query_weight = scatter(F.softmax(weight_logit, dim=-1)[cluster_point_idx], cluster_gather_idx, dim=-1, dim_size = cluster_gather_idx.max()+1) #(N_query, )

        return query_pos #, query_weight

    def get_query(self, inputs):      
        outputs = self.se3T(inputs)
        feature_se3T, pos, edge = outputs['feature'], outputs['pos'], outputs['edge']
        #weight_logit, feature_se3T = feature_se3T[..., 0], feature_se3T[..., 1:]
        #pos_weight_logit = self.pos_weight_field(feature_se3T).squeeze(0).squeeze(-1)
        pos_weight_logit = self.weight_field({'feature': feature_se3T, 'pos': pos, 'query_pos': pos.unsqueeze(0)})['field'].squeeze(0).squeeze(-1)
        query_points = self.get_init_query_pos(pos = pos, edge = edge, weight_logit = pos_weight_logit)

        inputs = {'feature': feature_se3T, 'pos': pos, 'query_pos': query_points.unsqueeze(0)}
        query_feature = self.feature_field(inputs)['field'].squeeze(0)
        query_attention = self.weight_field(inputs)['field'].squeeze(0)
        #print(f"Debug||query attention: {query_attention.std()}")
        #print(query_attention)
        #print(query_attention / np.sqrt(len(self.irreps_se3T.ls)))
        #query_attention_temperature = np.sqrt(self.irreps_descriptor[0].dim)
        query_attention_temperature = 1.
        query_attention = F.softmax(query_attention/query_attention_temperature, dim=-2).squeeze(-1)
        #query_attention = query_weight


        return {'query_feature': query_feature, 'query_points': query_points, 'query_attention': query_attention}

    # def save_tp(self, path):
    #     assert path[-1] == '/'
    #     self.se3T.save_tp(path + "se3T/")

    # def load_tp(self, path):
    #     assert path[-1] == '/'
    #     self.se3T.load_tp(path + "se3T/")


class EquivMultiHeadQueryModelOld(QueryModel):
    def __init__(self, irreps_descriptor, N_query, max_radius, field_cutoff, sh_lmax, number_of_basis, irrep_normalization='norm'):
        super().__init__(irreps_descriptor=irreps_descriptor, N_query=N_query, irrep_normalization=irrep_normalization)
        self.max_radius = max_radius
        self.field_cutoff = field_cutoff
        self.sh_lmax = sh_lmax
        self.number_of_basis = number_of_basis
        self.irreps_se3T = self.irreps_descriptor
        self.se3T = SE3Transformer(max_neighbor_radius = self.max_radius, irreps_out = self.irreps_se3T)
        self.se3T_feature = SE3Transformer(max_neighbor_radius = self.max_radius, irreps_out = self.irreps_se3T)
        self.attention_field = nn.Sequential(QuerySHLayer(sh_lmax = sh_lmax, number_of_basis = number_of_basis, field_cutoff = field_cutoff, irrep_normalization = irrep_normalization),
                                             TensorFieldLayer(irreps_input = self.irreps_se3T, irreps_output = o3.Irreps("10x0e"), 
                                                              sh_lmax = sh_lmax, number_of_basis = number_of_basis,
                                                              irrep_normalization = irrep_normalization, path_normalization = "element",
                                                              tp_type='fc'),
                                          LinearLayer([o3.Irreps('10x0e'), o3.Irreps('1x0e')], biases = True, path_normalization = "element"))
        self.pos_weight_field = nn.Sequential(QuerySHLayer(sh_lmax = sh_lmax, number_of_basis = number_of_basis, field_cutoff = field_cutoff, irrep_normalization = irrep_normalization),
                                              TensorFieldLayer(irreps_input = self.irreps_se3T, irreps_output = o3.Irreps(f"{2*self.N_query}x0e"), 
                                                               sh_lmax = sh_lmax, number_of_basis = number_of_basis,
                                                               irrep_normalization = irrep_normalization, path_normalization = "element",
                                                               tp_type='fc'),
                                              LinearLayer([o3.Irreps(f"{2*self.N_query}x0e"), o3.Irreps(f'{self.N_query}x0e')], biases = True, path_normalization = "element"))
        self.feature_field = nn.Sequential(QuerySHLayer(sh_lmax = sh_lmax, number_of_basis = number_of_basis, field_cutoff = field_cutoff, irrep_normalization = irrep_normalization),
                                           TensorFieldLayer(irreps_input = self.irreps_se3T, irreps_output = self.irreps_descriptor, 
                                                            sh_lmax = sh_lmax, number_of_basis = number_of_basis,
                                                            irrep_normalization = irrep_normalization, path_normalization = "element",
                                                            tp_type='fc'),
                                           LinearLayer([self.irreps_descriptor, self.irreps_descriptor], biases = True, path_normalization = "element"))


        irreps_emb = "10x0e + 10x1e + 5x2e + 3x3e"
        irreps_query = "10x0e + 10x1e + 5x2e + 3x3e"
        irreps_key = "10x0e + 10x1e + 5x2e + 3x3e"
        irreps_linear = "20x0e + 20x1e + 10x2e + 6x3e"
        self.pos_disp_feature_field = nn.Sequential(QuerySHLayer(sh_lmax = sh_lmax, number_of_basis = number_of_basis, field_cutoff = field_cutoff, irrep_normalization = irrep_normalization),
                                                    TensorFieldLayer(irreps_input = self.irreps_se3T, irreps_output = o3.Irreps(irreps_emb), 
                                                                        sh_lmax = sh_lmax, number_of_basis = number_of_basis,
                                                                        irrep_normalization = irrep_normalization, path_normalization = "element",
                                                                        tp_type='fc'),
                                                    LinearLayer([o3.Irreps(irreps_emb), o3.Irreps(irreps_emb)], biases = True, path_normalization = "element"))

        self.pos_disp_field = nn.Sequential(ClusteringLayer(max_neighbor_radius = self.max_radius, 
                                                            self_connection = False),

                                            EdgeSHLayer(sh_lmax = sh_lmax, number_of_basis = number_of_basis, irrep_normalization = irrep_normalization),

                                            SE3TransformerLayer(irreps_input = irreps_emb, irreps_query = irreps_query, irreps_key = irreps_key, irreps_output = irreps_emb, 
                                                                irreps_linear = irreps_linear, sh_lmax = sh_lmax, number_of_basis = number_of_basis,
                                                                self_interaction = True, skip_connection = True, layernorm_output = True, 
                                                                irrep_normalization = irrep_normalization, path_normalization = 'element'),

                                            SE3TransformerLayer(irreps_input = irreps_emb, irreps_query = irreps_query, irreps_key = irreps_key, irreps_output = "1x0e + 10x1e",
                                                                irreps_linear = irreps_linear, sh_lmax = sh_lmax, number_of_basis = number_of_basis,
                                                                self_interaction = True, skip_connection = True, layernorm_output = True, 
                                                                irrep_normalization = irrep_normalization, path_normalization = 'element'),

                                            LinearLayer([o3.Irreps("1x0e + 10x1e"), o3.Irreps(f"1x1e")], biases = True, path_normalization = 'element')
                                            )

    def get_init_query_pos(self, pos, weight_logit, temperature = 0.03): # pos: (N_points, 3),   weight_logit: (N_points, N_query)
        #weight = F.softmax(weight_logit, dim=-2) # (N_points, N_query)
        weight = F.softmax(weight_logit, dim=-1) + torch.ones_like(weight_logit)*1e-4 # (N_points, N_query)
        weight = weight ** (1/temperature)
        weight = weight / weight.sum(dim=-2) # (N_points, N_query)
        query_pos = torch.einsum('iq,ix->qx', weight, pos) # (N_query, 3)

        return query_pos # (N_query, 3)

    def get_query(self, inputs):      
        outputs = self.se3T(inputs)
        feature_se3T, pos = outputs['feature'], outputs['pos']
        pos_weight_logit = self.pos_weight_field({'feature': feature_se3T, 'pos': pos, 'query_pos': pos.unsqueeze(-3)})['field'].squeeze(-3) # (N_points, N_query)
        query_points = self.get_init_query_pos(pos = pos, weight_logit = pos_weight_logit) # (N_query, 3)

        # dt = 0.5
        # for iter in range(20):
        #     pos_disp_feature = self.pos_disp_feature_field({'feature': feature_se3T, 'pos': pos, 'query_pos': query_points.unsqueeze(-3)})['field'].squeeze(-3) # (N_query, N_feat_disp)
        #     inputs_point = {'feature': pos_disp_feature, 'pos': query_points, 'edge': None, 'max_neighbor_radius': 100000.}
        #     disp = self.pos_disp_field(inputs_point)['feature'] # (N_query, 3)
        #     #print(disp.norm(dim=-1))
        #     query_points = query_points + dt*disp
            
        outputs = self.se3T_feature(inputs)
        feature_se3T, pos = outputs['feature'], outputs['pos']

        inputs_feature = {'feature': feature_se3T, 'pos': pos, 'query_pos': query_points.unsqueeze(-3)}
        query_feature = self.feature_field(inputs_feature)['field'].squeeze(-3) # (N_q, N_feat)
        query_attention = self.attention_field(inputs_feature)['field'].squeeze(-3).squeeze(-1) # (N_q)
        query_attention_temperature = 1.
        query_attention = F.softmax(query_attention/query_attention_temperature, dim=-1) # (N_q)

        return {'query_feature': query_feature, 'query_points': query_points, 'query_attention': query_attention}

    # def save_tp(self, path):
    #     assert path[-1] == '/'
    #     self.se3T.save_tp(path + "se3T/")

    # def load_tp(self, path):
    #     assert path[-1] == '/'
    #     self.se3T.load_tp(path + "se3T/")