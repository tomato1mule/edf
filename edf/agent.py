from typing import List, Tuple, Optional, Union, Any
import os
import time
import datetime
import yaml
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch_cluster import radius_graph, radius
from torch_scatter import scatter, scatter_logsumexp, scatter_log_softmax, scatter_softmax, scatter_mean
from pytorch3d import transforms
from e3nn import o3

from edf.models import SE3Transformer, EnergyModel, QueryTensorField, SimpleQueryModel, EquivWeightQueryModel, EquivMultiHeadQueryModel
from edf.mcmc import MH, LangevinMH, PoseOptimizer
from edf.dist import GaussianDistSE3
from edf.layers import IrrepwiseDotProduct
from edf.visual_utils import scatter_plot, scatter_plot_ax

class PickAgent(nn.Module):
    def __init__(self, config_dir, device = 'cpu', lr_se3T = None, lr_energy_fast = None, lr_energy_slow = None, lr_query_fast = None, lr_query_slow = None, std_theta_perturb = None, std_X_perturb = None, max_N_query = None, langevin_dt = 1e-3):
        super().__init__()
        self.right_equiv = False
        try:
            self.edf_layernorm
        except AttributeError:
            self.edf_layernorm = True
        self.param_synced = False

        self.device = device
        self.lr_se3T = lr_se3T
        self.lr_energy_fast = lr_energy_fast
        self.lr_energy_slow = lr_energy_slow
        self.lr_query_fast = lr_query_fast
        self.lr_query_slow = lr_query_slow
        self.std_theta_perturb = std_theta_perturb
        self.std_X_perturb = std_X_perturb
        self.max_N_query = max_N_query
        self.langevin_dt = langevin_dt

        self.load_config(config_dir)
        self.init_models()

        # Load tensor product pickles for reproducibility
        # E3NN's tensor product modules are not deterministic for unknown reason, even when they do not have any parameters that require random initialization.
        # DEPRECATED
        # if tp_pickle_path is not None:
        #     self.load_tp_path(tp_pickle_path)

        # Choose device for the models
        self.device = device
        self.model_to(self.device)

        # Choose device for the sampling
        self.sampling_device = 'cpu'
        self.sampling_to(self.sampling_device)

        # Pre-calculate Inverse CDF for sampling
        self.metropolis.get_inv_cdf()
        self.langevin.get_inv_cdf()

        # Set Optimizers
        if lr_se3T is not None:
            self.init_optimizers()

        # Initialize perturb distribution for data augmentation
        if self.std_theta_perturb is not None:
            self.perturb_dist = GaussianDistSE3(std_theta = self.std_theta_perturb, std_X = self.std_X_perturb).to(device)

    def init_models(self):
        self.se3T = SE3Transformer(max_neighbor_radius = self.max_radius, irreps_out=self.irreps_descriptor)

        self.energy_model = EnergyModel(N_query = self.N_query, field_cutoff = self.field_cutoff,
                                irreps_input = self.irreps_descriptor, irreps_descriptor = self.irreps_descriptor, 
                                sh_lmax = self.sh_lmax_descriptor, number_of_basis = self.number_of_basis_descriptor, 
                                ranges = self.ranges, layernorm=self.edf_layernorm, tp_type=self.tp_type).requires_grad_(False)

        self.energy_model_train = EnergyModel(N_query = self.N_query, field_cutoff = self.field_cutoff,
                                        irreps_input = self.irreps_descriptor, irreps_descriptor = self.irreps_descriptor, 
                                        sh_lmax = self.sh_lmax_descriptor, number_of_basis = self.number_of_basis_descriptor, 
                                        ranges = self.ranges, layernorm=self.edf_layernorm, tp_type=self.tp_type)


        self.synchronize_params()
        self.energy_model = torch.jit.script(self.energy_model)
        self.energy_model_train = torch.jit.script(self.energy_model_train)
        
        self.metropolis = MH(ranges_X = self.ranges, std_theta = self.std_theta, std_X = self.std_X)
        #self.langevin = LangevinMH(ranges_X = self.ranges, dt = 0.1, std_theta = 1., std_X = 1.)
        self.pose_optim = PoseOptimizer()
        self.irrepwise_dot = IrrepwiseDotProduct(self.irreps_descriptor)

        self._init_models_optional()

    def synchronize_params(self):
        self.energy_model.load_state_dict(self.energy_model_train.state_dict())
        self.param_synced = True

    def _init_models_optional(self):
        self.langevin = LangevinMH(ranges_X = self.ranges, dt = self.langevin_dt, std_theta = 1., std_X = 1.)
        self.query_model = SimpleQueryModel(irreps_descriptor=self.irreps_descriptor, N_query=self.N_query, 
                                            query_radius = self.query_radius, irrep_normalization=self.irrep_normalization, layernorm=False)

    def init_optimizers(self):
        self.optimizer_se3T = torch.optim.Adam(list(self.se3T.parameters()), lr=self.lr_se3T, betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True)
        self.optimizer_energy_fast = torch.optim.Adam(self.energy_model_train.fast_parameters(), lr=self.lr_energy_fast, betas=(0.9, 0.98), eps=1e-09, weight_decay=0, amsgrad=True)
        self.optimizer_energy_slow = torch.optim.Adam(self.energy_model_train.slow_parameters(), lr=self.lr_energy_slow, betas=(0.9, 0.98), eps=1e-09, weight_decay=0, amsgrad=True)
        self.optimizer_query_fast = torch.optim.Adam(self.query_model.fast_parameters(), lr=self.lr_query_fast, betas=(0.9, 0.98), eps=1e-09, weight_decay=0, amsgrad=True)
        self.optimizer_query_slow = torch.optim.Adam(self.query_model.slow_parameters(), lr=self.lr_query_slow, betas=(0.9, 0.98), eps=1e-09, weight_decay=0, amsgrad=True)

    def rescale_lr(self, factor):
        for g in self.optimizer_se3T.param_groups:
            g['lr'] = self.lr_se3T * factor
        for g in self.optimizer_energy_fast.param_groups:
            g['lr'] = self.lr_energy_fast * factor
        for g in self.optimizer_energy_slow.param_groups:
            g['lr'] = self.lr_energy_slow * factor
        for g in self.optimizer_query_fast.param_groups:
            g['lr'] = self.lr_query_fast * factor
        for g in self.optimizer_query_slow.param_groups:
            g['lr'] = self.lr_query_slow * factor

    def step(self):
        self.optimizer_se3T.step()
        self.optimizer_energy_fast.step()
        self.optimizer_energy_slow.step()
        self.optimizer_query_fast.step()
        self.optimizer_query_slow.step()
        self.param_synced = False # Unnecessary but for safety
        self.synchronize_params()

    def load_config(self, config_dir):
        with open(config_dir) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        # if self.max_N_query is not None:
        #     self.N_query = min(config['N_query'], self.max_N_query)
        # else:
        #     self.N_query = config['N_query']
        self.N_query = config['N_query']
        # if self.max_N_query is None:
        #     self.max_N_query = self.N_query
        self.max_radius = config['max_radius']
        self.field_cutoff = config['field_cutoff']
        self.std_theta = config['std_theta_degree'] / 180 * np.pi
        self.std_X = config['std_X']                      
        self.ranges = torch.tensor(config['ranges'])
        self.irreps_descriptor = o3.Irreps(config['irreps_descriptor'])
        self.sh_lmax_descriptor = config['sh_lmax_descriptor']
        self.number_of_basis_descriptor = config['number_of_basis_descriptor']
        self.irrep_normalization = config['irrep_normalization']       
        self.learnable_query = config['learnable_query']
        self.mh_iter = config['mh_iter']
        self.langevin_iter = config['langevin_iter']
        self.tp_type = config['tp_type']
        self._load_config_optional(config)

    def _load_config_optional(self, config):
        self.query_radius = config['query_radius']

    # def load_tp_path(self, tp_pickle_path):
    #     try:
    #         self.se3T.load_tp(tp_pickle_path + "se3T/")
    #         self.energy_model.load_tp(tp_pickle_path + "energy_model/")
    #         self.query_model.load_tp(tp_pickle_path + "query_model/")
    #     except FileNotFoundError:
    #         # Debug
    #         self.save_tp_path(tp_pickle_path)

    # def save_tp_path(self, tp_pickle_path):
    #     self.se3T.save_tp(tp_pickle_path + "se3T/")
    #     self.energy_model.save_tp(tp_pickle_path + "energy_model/")
    #     self.query_model.save_tp(tp_pickle_path + "query_model/")

    def model_to(self, device):
        self.se3T = self.se3T.to(device)
        self.energy_model_train = self.energy_model_train.to(device)
        self.query_model = self.query_model.to(device)
        self.ranges = self.ranges.to(device)
        self.irrepwise_dot.to(device)

    def sampling_to(self, device):
        self.energy_model = self.energy_model.to(device)
        self.metropolis = self.metropolis.to(device)
        self.langevin = self.langevin.to(device)
        self.pose_optim = self.pose_optim.to(device)


    def get_descriptor(self, inputs, requires_grad = True):
        if requires_grad:
            se3T_outputs = self.se3T(inputs)
            return se3T_outputs['feature'], se3T_outputs['pos']
        else:
            with torch.no_grad():
                se3T_outputs = self.se3T(inputs)
            return se3T_outputs['feature'].detach(), se3T_outputs['pos'].detach()

    def get_query(self, inputs_Q = None, requires_grad = True, temperature = None, **kwargs):
        if requires_grad is True:
            query_outputs = self.query_model(inputs_Q, temperature = temperature, requires_grad = True, **kwargs)
            return query_outputs['query_points'], query_outputs['query_feature'], query_outputs['query_attention']
        else:
            query_outputs = self.query_model(inputs_Q, temperature = temperature, requires_grad = False, **kwargs)
            return query_outputs['query_points'].detach(), query_outputs['query_feature'].detach(), query_outputs['query_attention'].detach()

    def get_log_P(self, feature, pos, query_points, query_feature, query_attention, temperature):
        log_P = lambda T: -self.energy_model(T=T, feature=feature, pos=pos, query_points=query_points, query_feature=query_feature, query_attention=query_attention, temperature=temperature)[0]
        return log_P

    def mcmc(self, T_seed, feature, pos, query_points, query_feature, query_attention, mh_iter, langevin_iter, 
             temperature = 1., pbar = True, traj_len = 1, optim_iter = 0, optim_lr = 1e-3, resample = False, output_device = None):
        input_device = T_seed.device
        if output_device is None:
            output_device = input_device
        #assert self.energy_model.requires_graD is False
        log_P = self.get_log_P(feature=feature.detach().to(self.sampling_device), 
                               pos=pos.detach().to(self.sampling_device), 
                               query_points=query_points.detach().to(self.sampling_device), 
                               query_feature=query_feature.detach().to(self.sampling_device), 
                               query_attention=query_attention.detach().to(self.sampling_device), 
                               temperature=temperature)

        if mh_iter == 0:
            assert langevin_iter > 0
            samples = self.langevin.forward(log_P, max_iter = langevin_iter, T_seed = T_seed.to(self.sampling_device), pbar=pbar)
            Ts = samples['Ts']
            As = samples['As']
        else:
            samples = self.metropolis.forward(log_P, max_iter = mh_iter, T_seed = T_seed.to(self.sampling_device), pbar=pbar)
            Ts = samples['Ts']
            As = samples['As']
            if resample is True:
                with torch.no_grad():
                    P = log_P(Ts[-1]) # (n_transforms)
                    P = P/ (P.std(dim=-1)*1.)
                    P = P.softmax(dim=-1)
                    resampled = Ts[-1][P.multinomial(len(P), replacement=True)]
                    Ts = torch.cat([Ts, resampled.unsqueeze(0)], dim=0)
                    As = torch.cat([As, torch.ones_like(As[-1]).unsqueeze(0)], dim=0)

            if langevin_iter > 0:
                samples = self.langevin.forward(log_P, max_iter = langevin_iter, T_seed = Ts[-1], pbar=pbar)
                Ts = torch.cat([Ts, samples['Ts']], dim=0)
                As = torch.cat([As, samples['As']], dim=0)

        if optim_iter > 0:
            samples_optim = self.pose_optim.forward(T=Ts[-1], log_P=log_P, step=optim_iter, lr=optim_lr, sort=True, pbar = pbar)
            Ts = torch.cat([Ts, samples_optim['Ts']], dim=0)
            As = torch.cat([As, samples_optim['As']], dim=0)
            #print(samples_optim['Es'][:,samples_optim['Es'][-1].argmin()]) #debug

        logs = {}
        logs['N_rejected'] = (~As).sum(dim=0).to(output_device)
        logs['N_rejected_langevin'] = (~samples['As']).sum(dim=0).to(output_device)
        

        Ts = Ts[-traj_len:].reshape(-1,7)
        with torch.no_grad():
            E = -log_P(Ts)
            logs['E'] = E.to(output_device)
        
        return Ts.to(output_device), logs

    def surrogate_query(self, query_points, query_attention, input_points, target_T, std = 0.5, cutoff_thr = 0.99, min_logit = 1e-3):
        assert std > 0
        if target_T.requires_grad is True or query_points.requires_grad is True or input_points.requires_grad is True:
            raise NotImplementedError
        assert target_T.dim() == 2 and target_T.shape[0] == 1 and target_T.shape[-1] == 7                # (1, 7)
        assert query_points.dim() == 2 and query_points.shape[-1] == 3                                   # (Nq, 3)
        assert input_points.dim() == 2 and input_points.shape[-1] == 3                                   # (Np, 3)
        assert query_attention.dim() == 1 and query_attention.shape[-1] == query_points.shape[-2]        # (Nq)
        query_points = self.energy_model.transform_query_points(T=target_T, query_points=query_points)   # (1, Nq, 3)
        
        Nq, Np = query_points.shape[-2], input_points.shape[-2]
        query_points = query_points.reshape(-1,3) # (Nq, 3)

        edges = radius(input_points.detach(), query_points.detach(), r = self.energy_model.qsh.cutoff * cutoff_thr, max_num_neighbors=Np, num_workers = 1) # src: Yidx (query), dst: Xidx (input poincloud)
        edge_src, edge_dst = edges[0], edges[1]
        n_neighbor = torch.ones(len(edge_src), device = query_points.device) # (Nedge,)
        n_neighbor = scatter(n_neighbor, edge_src, dim=-1, dim_size = Nq).reshape(Nq) # (Nq,)
        has_neighbor = n_neighbor > 0
        #print(has_neighbor.detach().cpu())
        
        original_logit = torch.log(query_attention) # (Nq,)
        # min_logit = original_logit.detach()[has_neighbor.nonzero().squeeze(-1)].min() # Scalar
        # min_logit = min_logit - 5*std
        log_Z = torch.logsumexp(original_logit.detach(), dim=-1) # Scalar
        min_logit = log_Z + np.log(min_logit) # such that all attention are at least (almost) higher than minimum prob

        # replace_target = ~has_neighbor # (Nq,)
        not_too_low = original_logit > min_logit     # (Nq,)
        replace_target = ~(has_neighbor * not_too_low)  # (Nq,)

        noise = torch.randn_like(original_logit) # (Nq, )
        surrogate_logit = torch.where(replace_target, min_logit, original_logit + (std*noise)) # (Nq, )
        #print(original_logit.detach().cpu())
        #print(surrogate_logit.detach().cpu())

        kld = (surrogate_logit.detach()[(replace_target).nonzero().squeeze(-1)] - original_logit[(replace_target).nonzero().squeeze(-1)]).div(std).square() # (Nq_vaccum, )
        kld = 0.5 * kld.sum(dim = -1, keepdim=True) # (1, )
        assert kld.shape == torch.Size([1])

        surrogate_attention = torch.softmax(surrogate_logit, dim=-1) # (Nq, )
        assert surrogate_attention.shape == query_attention.shape

        return surrogate_attention, kld


    def sample(self, inputs, inputs_Q, T_seed, mh_iter, langevin_iter, 
                temperature = 1., pbar = True, learning = False, traj_len = 1, optim_iter = 0, optim_lr = 1e-3, query_temperature = None, surrogate_query = False, target_T = None, resample = False, output_device = None):
        if surrogate_query is True:
            assert target_T is not None

        t_edf_begin = time.time()
        feature, pos = self.get_descriptor(inputs=inputs, requires_grad=learning) # (Nt, Nquery, feature_len), (Nt, Nquery, 3)
        query_points, query_feature, query_attention = self.get_query(inputs_Q=inputs_Q, requires_grad=learning, temperature = query_temperature) # (N_query, 3), (N_query, feature_len), (N_query,)
        if surrogate_query is True:
            query_attention, kld = self.surrogate_query(query_points.detach(), query_attention, pos.detach(), target_T = target_T)
        t_edf_end = time.time()

        t_mcmc_begin = time.time()
        Ts, logs = self.mcmc(T_seed=T_seed, feature=feature, pos=pos, 
                             query_points=query_points, query_feature=query_feature, query_attention=query_attention,
                             mh_iter = mh_iter, langevin_iter = langevin_iter, temperature = temperature, pbar = pbar, 
                             traj_len=traj_len, optim_iter=optim_iter, optim_lr=optim_lr, resample=resample, output_device=output_device)
        t_mcmc_end = time.time()
        logs['edf_time'] = t_edf_end - t_edf_begin
        logs['mcmc_time'] = t_mcmc_end - t_mcmc_begin

        if learning:
            edf_outputs = {'feature': feature, 'pos': pos, 'query_points': query_points, 'query_feature': query_feature, 'query_attention': query_attention}
        else:
            edf_outputs = {'feature': feature.detach(), 'pos': pos.detach(), 'query_points': query_points.detach(), 'query_feature': query_feature.detach(), 'query_attention': query_attention.detach()}
        if surrogate_query is True:
            return Ts, edf_outputs, logs, kld
        else:
            return Ts, edf_outputs, logs

    def greedy_policy(self, Ts, E):
        best_T_idx = E.detach().argmin().item()
        best_T = Ts[best_T_idx]
        return best_T

    def sorted_policy(self, Ts, E):
        #print(E.sort().values[:20]) # debug
        return Ts[E.argsort()]

    def softmax_policy(self, Ts, E, temperature = 1., replacement = True):
        prob = torch.softmax(-E / temperature, dim=-1)
        idx = torch.multinomial(prob, num_samples=len(prob), replacement=replacement)
        return Ts[idx]

    def train_once(self, inputs, target_T, N_transforms, mh_iter, langevin_iter, edf_norm_std = False,
                   temperature = 1., pbar = True, verbose = True, visual_info = None, inputs_Q = None, CD_ratio = 0., query_temperature = None, surrogate_query = False):
        assert self.param_synced is True

        t1 = time.time()
        target_T = self.perturb_dist.propose(target_T)
        

        N_transforms_CD = round(N_transforms*CD_ratio)
        CD_ratio = N_transforms_CD / N_transforms

        if N_transforms_CD == 0:
            #T_seed = self.metropolis.initialize(N_transforms).to(self.device)
            X_seed = torch.rand(N_transforms, 3, device=target_T.device) * (self.ranges[:,1] - self.ranges[:,0]) + self.ranges[:,0]
            q_seed = torch.randn(N_transforms, 4, device=target_T.device)
            T_seed = torch.cat((q_seed/q_seed.norm(dim=-1, keepdim=True), X_seed), dim=-1)
        elif N_transforms_CD == N_transforms:
            T_seed = target_T.repeat(N_transforms, 1)
        elif N_transforms_CD > 0 and N_transforms_CD < N_transforms:
            N_transforms_random = N_transforms - N_transforms_CD
            T_seed_CD = target_T.repeat(N_transforms_CD, 1)
            X_seed_random = torch.rand(N_transforms_random, 3, device=target_T.device) * (self.ranges[:,1] - self.ranges[:,0]) + self.ranges[:,0]
            q_seed_random = torch.randn(N_transforms_random, 4, device=target_T.device)
            T_seed_random = torch.cat((q_seed_random/q_seed_random.norm(dim=-1, keepdim=True), X_seed_random), dim=-1)
            T_seed = torch.cat([T_seed_CD, T_seed_random], dim=0)
        else:
            raise ValueError





        self.zero_grad()
        if surrogate_query is True:
            Ts, edf_outputs, logs, kld = self.sample(inputs = inputs, inputs_Q = inputs_Q, 
                                                     T_seed = T_seed, mh_iter = mh_iter, langevin_iter = langevin_iter,
                                                     temperature = temperature, pbar = pbar, learning = True, query_temperature = query_temperature, surrogate_query=surrogate_query, target_T=target_T)
        else:
            Ts, edf_outputs, logs = self.sample(inputs = inputs, inputs_Q = inputs_Q, 
                                                T_seed = T_seed, mh_iter = mh_iter, langevin_iter = langevin_iter,
                                                temperature = temperature, pbar = pbar, learning = True, query_temperature = query_temperature, surrogate_query=surrogate_query)

        
        #assert self.energy_model.requires_graD is False
        #self.energy_model.requires_grad_(True)
        E, descriptor = self.energy_model_train(T=torch.cat([target_T , Ts], dim=0), 
                                                feature = edf_outputs['feature'], 
                                                pos = edf_outputs['pos'],
                                                query_points = edf_outputs['query_points'], 
                                                query_feature = edf_outputs['query_feature'], 
                                                query_attention = edf_outputs['query_attention'],
                                                temperature = temperature) # (Nt,), (Nt, Nquery, dim_Descriptor)

        reg_edf = self.irrepwise_dot(descriptor, descriptor) # (Nt, Nquery, len(irrep.ls))
        if True: #self.right_equiv:
            reg_query = self.irrepwise_dot(edf_outputs['query_feature'], edf_outputs['query_feature']) # (Nquery, len(irrep.ls))

        E_pos = E[...,0]
        E_neg = E[...,1:]
        Loss_nll = E_pos - E_neg.mean(dim=-1)
        if edf_norm_std is not False:
            Loss_reg_edf = (reg_edf.sum(dim=-1).sum(dim=-1) / (2.*(edf_norm_std**2))) #(Nt,)
            Loss_reg_edf_pos, Loss_reg_edf_neg = Loss_reg_edf[...,0], Loss_reg_edf[...,1:].mean()
            Loss = Loss_nll + Loss_reg_edf_pos + Loss_reg_edf_neg
            Loss_reg_query = (reg_query.sum(dim=-1).sum(dim=-1) / (2.*(edf_norm_std**2)))
            Loss = Loss + Loss_reg_query
        else:
            Loss = Loss_nll
        if surrogate_query is True:
            Loss = Loss + kld
        Loss.backward()
        self.step()
        #self.energy_model.requires_grad_(False)
        best_neg_T = self.greedy_policy(Ts = Ts, E = E[..., 1:])
        t2 = time.time()





        if visual_info is not None:
            try:
                figsize = visual_info['figsize']
            except KeyError:
                figsize = (8*(1+self.right_equiv), 8)
            fig, axes = plt.subplots(1, 1+self.right_equiv, figsize=figsize, subplot_kw={'projection':'3d'})
            if self.right_equiv:
                visual_info['ax'] = axes[0]
                visual_info['ax_query'] = axes[1]
            else:
                visual_info['ax'] = axes

            self.visualize(visual_info = visual_info, 
                           edf_outputs = {k: v.detach().cpu() for k, v in edf_outputs.items()},
                           T = best_neg_T, target_T = target_T, Ts = Ts)
            if self.right_equiv:
                self.visualize_query(visual_info=visual_info, edf_outputs={k: v.detach().cpu() for k, v in edf_outputs.items()})

            try:
                show = visual_info['show']
            except KeyError:
                show = True
            try:
                save_path = visual_info['save_path']
                assert 'file_name' in visual_info.keys()
                file_name = visual_info['file_name']
            except KeyError:
                save_path = None
                file_name = None
            if save_path is not None:
                if os.path.exists(save_path) is False:
                    os.makedirs(save_path)
                fig.savefig(save_path + file_name)
            if show:
                plt.show()
            fig.clear()
            plt.close('all')


        if verbose:
            print(f"Target Energy: {E_pos.detach().cpu().item()}")
            print(f"Sample Energy: {E_neg.detach().cpu()}")
            print(f"CD ratio: {CD_ratio} || Query temperature: {query_temperature if query_temperature is not None else 1.}")
            print(f"Loss_NLL: {Loss_nll.item()} || E_pos: {E_pos.detach().cpu()} || E_neg_min: {E_neg.min(dim=-1)[0].detach().cpu()}")
            print(f"Irrep Weight: {self.energy_model.get_irrep_weight().detach().cpu()}")
            print(f"Mean Feature Norm: {reg_edf.detach().mean(dim=-2).mean(dim=-2).sqrt().cpu().mean()}")
            print(f"Mean Query Feature Norm: {reg_query.detach().mean(dim=-2).sqrt().cpu().mean()}")
            print(f"Query Attention: {edf_outputs['query_attention'].detach().cpu()}")
            print(f"MCMC Time: {logs['mcmc_time']:.4f} || Net time: {t2-t1}")
            print(f"Mean Reject ratio: {(logs['N_rejected'].type(torch.float32)/mh_iter).mean()}", flush=True)
            print(f"Mean Langevin Reject ratio: {(logs['N_rejected_langevin'].type(torch.float32)/langevin_iter).mean()}", flush=True)
        else:
            print(f"MCMC Time: {logs['mcmc_time']:.4f} || Total time: {t2-t1} || Mean Reject ratio: {(logs['N_rejected'].type(torch.float32)/mh_iter).mean()} || Loss: {Loss.item()}", flush=True)

    def forward(self, inputs: dict, T_seed: Union[torch.Tensor, int], mh_iter: int = None, langevin_iter: int = None, 
                temperature: float = 1., pbar: bool = True, inputs_Q: Union[None, dict] = None, 
                policy: str = 'sorted', policy_temperature: float = None, 
                traj_len: int = 1, optim_iter: int = 0, optim_lr: float = 1e-3, query_temperature: Union[float, None] = None, resample=True):
        assert self.param_synced is True

        if mh_iter is None:
            mh_iter = self.mh_iter
        if langevin_iter is None:
            langevin_iter = self.langevin_iter
        if type(T_seed) == int:
            N_transforms = T_seed
            X_seed = torch.rand(N_transforms, 3, device=self.device) * (self.ranges[:,1] - self.ranges[:,0]) + self.ranges[:,0]
            q_seed = torch.randn(N_transforms, 4, device=self.device)
            T_seed = torch.cat((q_seed/q_seed.norm(dim=-1, keepdim=True), X_seed), dim=-1)

        #assert self.energy_model.requires_graD is False
        Ts, edf_outputs, logs = self.sample(inputs = inputs, inputs_Q = inputs_Q, 
                                            T_seed = T_seed, mh_iter = mh_iter, langevin_iter = langevin_iter,
                                            temperature = temperature, pbar = pbar, learning = False, 
                                            traj_len=traj_len, optim_iter=optim_iter, optim_lr=optim_lr, query_temperature=query_temperature, resample=resample, output_device=self.sampling_device)

        # with torch.no_grad():
        #     E, descriptor = self.energy_model_train(T=Ts, 
        #                                             feature = edf_outputs['feature'], 
        #                                             pos = edf_outputs['pos'],
        #                                             query_points = edf_outputs['query_points'], 
        #                                             query_feature = edf_outputs['query_feature'], 
        #                                             query_attention = edf_outputs['query_attention'],
        #                                             temperature = temperature)                        # shape == (len(T_seed),)
        E = logs['E']

        if policy == 'greedy':
            best_T = self.greedy_policy(Ts = Ts, E = E)
            logs = {}
            return best_T.unsqueeze(-2), edf_outputs, logs
        elif policy == 'sorted':
            logs = {}
            return self.sorted_policy(Ts = Ts, E = E), edf_outputs, logs
        elif policy == 'softmax':
            logs = {}
            return self.softmax_policy(Ts = Ts, E = E, temperature=policy_temperature, replacement=True), edf_outputs, logs
        elif policy == 'softmax_noreplace':
            logs = {}
            return self.softmax_policy(Ts = Ts, E = E, temperature=policy_temperature, replacement=False), edf_outputs, logs
        else:
            raise ValueError('Wrong policy name')

    def visualize(self, visual_info, edf_outputs, T, target_T = None, Ts = None):
        coord = visual_info['coord']
        color = visual_info['color']
        ranges = visual_info['ranges']
        ax = visual_info['ax']


        if target_T is not None:
            target_R, target_X = transforms.quaternion_to_matrix(target_T[...,:4]), target_T[...,4:]
        R, X = transforms.quaternion_to_matrix(T[...,:4]), T[...,4:]
        

        frames = []
        # Visualize T's frame with query points
        #query_attn = (torch.softmax(self.energy_model.query_attention.detach(), dim=-1) ** 0.5).unsqueeze(-1).cpu().numpy()
        query_attn = edf_outputs['query_attention'].detach().cpu()
        query_attn = ((query_attn/query_attn.max()) ** 0.5).unsqueeze(-1).cpu().numpy()
        coord_query = self.energy_model.transform_query_points(T.detach().cpu(), edf_outputs['query_points'].detach().cpu()).numpy()
        color_query = torch.tensor([1.,0.,1.]).repeat(len(coord_query),1).cpu().numpy()
        color_query = np.concatenate([color_query,query_attn], axis=-1)
        frame_info_T = {'frame': R.cpu().numpy(),
                        'origin': X.cpu().numpy(), 
                        'alpha': 1.,
                        'pointcloud': (coord_query, color_query)
                        }
        frames.append(frame_info_T)

        # Visualize target T's frame with query points
        if target_T is not None:
            coord_query = self.energy_model.transform_query_points(target_T.detach().cpu(), edf_outputs['query_points'].detach().cpu()).numpy().squeeze(0)
            color_query = torch.tensor([0.,0.,1.]).repeat(len(coord_query),1).cpu().numpy()
            color_query = np.concatenate([color_query,query_attn], axis=-1)
            frame_info_target = {'frame': target_R.squeeze(0).cpu().numpy(),
                            'origin': target_X.squeeze(0).cpu().numpy(), 
                            'alpha': 0.1,
                            'pointcloud': None #(coord_query, color_query)
                            }
            frames.append(frame_info_target)

        if Ts is not None:
            world_origin = np.array([0., 0., -16.])
            frame_info_Ts = {'frame': np.eye(3),
                            'origin': world_origin,
                            'alpha': 0.,
                            'pointcloud': (Ts.detach().cpu().numpy()[:,4:], torch.tensor([0.,0.,1.,0.5]).repeat(len(Ts),1).numpy())
                            }
            frames.append(frame_info_Ts)

        color_alpha = np.concatenate([color, np.ones((len(color),1), dtype=int)], axis=-1)
        scatter_plot_ax(ax, coord, color_alpha, ranges, frame_infos = frames)

    def visualize_query(self, visual_info, edf_outputs):
        coord = visual_info['coord_query']
        color = visual_info['color_query']
        ranges = visual_info['ranges_query']
        ax = visual_info['ax_query']


        frames = []


        n_blob = 40
        r_blob = (ranges[0][1]-ranges[0][0])/80
        query_attn = edf_outputs['query_attention'].detach().cpu()
        query_attn = ((query_attn/query_attn.max()) / (n_blob*0.25)).unsqueeze(-1).cpu().numpy()
        coord_query = edf_outputs['query_points'].detach().cpu().numpy()
        disp = np.random.randn(n_blob,1,3)
        disp = disp / np.linalg.norm(disp, axis=-1, keepdims = True)
        coord_query = (coord_query.reshape(1,-1,3) + disp*r_blob).reshape(-1,3)
        query_attn = np.tile(query_attn.reshape(1,-1,1), reps=(n_blob,1,1)).reshape(-1,1)

        color_query = torch.tensor([0.0,0.8,0.8]).repeat(len(coord_query),1).cpu().numpy()
        color_query = np.concatenate([color_query,query_attn], axis=-1)

        frame_info_T = {'frame': np.eye(3),
                        'origin': np.zeros(3), 
                        'alpha': 0.,
                        'pointcloud': (coord_query, color_query)
                        }
        frames.append(frame_info_T)

        color_alpha = np.concatenate([color, np.ones((len(color),1), dtype=int)], axis=-1)
        scatter_plot_ax(ax, coord, color_alpha, ranges, frame_infos = frames)

    def crop_range_idx(self, pos):
        in_range_cropped_idx = (((pos > self.ranges[:,0]) * (pos < self.ranges[:,1])).sum(dim=-1) == 3).nonzero().squeeze(-1)
        return in_range_cropped_idx

    def save(self, path, filename):
        if os.path.exists(path) is False:
            os.makedirs(path)

        torch.save({'se3T_state_dict': self.se3T.state_dict(),
                    'energy_model_state_dict': self.energy_model.state_dict(),
                    'query_model_state_dict': self.query_model.state_dict()}, 
                   path + filename)

    def load(self, dir):
        loaded = torch.load(dir, map_location=self.device)
        self.se3T.load_state_dict(loaded['se3T_state_dict'])
        self.energy_model.load_state_dict(loaded['energy_model_state_dict'])
        self.param_synced = False # Unnecessary but for safety
        self.energy_model_train.load_state_dict(loaded['energy_model_state_dict'])
        self.param_synced = True # # Unnecessary but for safety
        self.query_model.load_state_dict(loaded['query_model_state_dict'])

    def debug_energy(self, inputs, T, temperature = 1., inputs_Q = None, grad_type = 'lie_grad'):
        input_device = T.device
        learning = False
        feature, pos = self.get_descriptor(inputs=inputs, requires_grad=learning) # (Nt, Nquery, feature_len), (Nt, Nquery, 3)
        query_points, query_feature, query_attention = self.get_query(inputs_Q=inputs_Q, requires_grad=learning) # (N_query, 3), (N_query, feature_len), (N_query,)
        if learning:
            edf_outputs = {'feature': feature, 'pos': pos, 'query_points': query_points, 'query_feature': query_feature, 'query_attention': query_attention}
        else:
            edf_outputs = {'feature': feature.detach(), 'pos': pos.detach(), 'query_points': query_points.detach(), 'query_feature': query_feature.detach(), 'query_attention': query_attention.detach()}
        log_P = self.get_log_P(feature=feature.detach().to(self.sampling_device), 
                               pos=pos.detach().to(self.sampling_device), 
                               query_points=query_points.detach().to(self.sampling_device), 
                               query_feature=query_feature.detach().to(self.sampling_device), 
                               query_attention=query_attention.detach().to(self.sampling_device), 
                               temperature=temperature)

        if grad_type == 'lie_grad':
            ##### Analytic #####
            T = T.detach().to(self.sampling_device)
            q, X = T[...,:4], T[...,4:]
            q = transforms.standardize_quaternion(q/torch.norm(q, dim=-1, keepdim=True))
            q, X = q.requires_grad_(), X.requires_grad_()
            logP = log_P(torch.cat([q, X], dim=-1))
            logP.sum().backward(inputs = [q, X])
            grad_q, grad_X = q.grad.detach(), X.grad.detach()
            q_indices = torch.tensor([[1,2,3], [0,3,2], [3,0,1], [2,1,0]], dtype=torch.long, device=grad_q.device)
            q_factor = torch.tensor([[-0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, 0.5]], dtype=grad_q.dtype, device=grad_q.device)
            lie_derivative = q.detach()[...,q_indices] * q_factor
            lie_R_grad = torch.einsum('...ia,...i->...a',lie_derivative, grad_q)
            lie_X_grad = transforms.quaternion_apply(transforms.quaternion_invert(q), grad_X)
            grad = torch.cat([lie_X_grad.detach(), lie_R_grad.detach()], dim=-1)

            ##### Autograd exp map #####
            # T = T.detach().to(self.sampling_device)
            # q, X = T[...,:4], T[...,4:]
            # q = q/torch.norm(q, dim=-1, keepdim=True)
            # lie_R, lie_X = torch.zeros_like(X, requires_grad=True), torch.zeros_like(X, requires_grad=True)
            # dq = transforms.axis_angle_to_quaternion(lie_R)
            # q_test = transforms.quaternion_multiply(q,dq)
            # lie_R_mat = (self.langevin.so3_basis * lie_R[..., None, None]).sum(dim=-3) # (Nt, 3, 3)
            # right_jacobian = torch.eye(3, dtype=q.dtype, device=q.device) + lie_R_mat/2 + (lie_R_mat@lie_R_mat)/3
            # dX = torch.einsum('...ij,...j', right_jacobian, lie_X) # (Nt, 3)
            # X_test = X + transforms.quaternion_apply(q, dX)
            # logP = log_P(torch.cat([q_test, X_test], dim=-1))
            # logP.sum().backward(inputs = [lie_R, lie_X])
            # grad = torch.cat([lie_X.grad.detach(), lie_R.grad.detach()], dim=-1)

            ##### Autograd add (very bad) #####
            # T = T.detach().to(self.sampling_device)
            # q, X = T[...,:4], T[...,4:]
            # R = transforms.quaternion_to_matrix(q)
            # lie_R, lie_X = torch.zeros_like(X, requires_grad=True), torch.zeros_like(X, requires_grad=True)
            # lie_R_mat = (self.langevin.so3_basis * lie_R[..., None, None]).sum(dim=-3)
            # q_test, X_test = transforms.matrix_to_quaternion(R+lie_R_mat), X+lie_X
            # logP = log_P(torch.cat([q_test/q_test.norm(dim=-1, keepdim=True), X_test], dim=-1))
            # logP.sum().backward(inputs = [lie_R, lie_X])
            # grad = torch.cat([lie_X.grad.detach(), lie_R.grad.detach()], dim=-1)

        elif grad_type == 'quaternion':
            T = T.clone().detach().to(self.sampling_device).requires_grad_(True)
            logP = log_P(T)
            logP.sum().backward(inputs=T)
            grad = T.grad

        elif grad_type == 'lie_from_quat':
            T = T.clone().detach().to(self.sampling_device).requires_grad_(True)
            logP = log_P(T)
            logP.sum().backward(inputs=T)
            grad = T.grad
            L = T.detach()[...,self.langevin.q_indices] * self.langevin.q_factor
            grad = torch.cat([transforms.quaternion_apply(transforms.quaternion_invert(T[...,:4].detach()), grad[...,4:]), torch.einsum('...ia,...i', L, grad[...,:4])], dim=-1)

        elif grad_type == 'natural':
            T = T.clone().detach().to(self.sampling_device).requires_grad_(True)
            logP = log_P(T)
            logP.sum().backward(inputs=T)
            grad = T.grad
            L = T.detach()[...,self.langevin.q_indices] * self.langevin.q_factor
            Ginv = torch.einsum('...ia,...ja', L, L)
            nat_grad = torch.einsum('...ij,...j', Ginv, grad[...,:4])
            grad = torch.cat([nat_grad, grad[...,4:]], dim=-1)

        else:
            raise ValueError

        visualizer = lambda visual_info, T : self.visualize(visual_info=visual_info, edf_outputs={k: v.detach().cpu() for k, v in edf_outputs.items()}, T=T)
        visualizer_query = None

        return logP.detach().to(input_device), grad.to(input_device), visualizer, visualizer_query


        
class PlaceAgent(PickAgent):
    def __init__(self, config_dir, device = 'cpu', lr_se3T = None, lr_energy_fast = None, lr_energy_slow = None, lr_query_fast = None, lr_query_slow = None, std_theta_perturb = None, std_X_perturb = None, max_N_query=None, langevin_dt=1e-3):
        self.edf_layernorm = True
        super().__init__(config_dir=config_dir, device=device, lr_se3T = lr_se3T, lr_energy_fast = lr_energy_fast, lr_energy_slow = lr_energy_slow, lr_query_fast = lr_query_fast, lr_query_slow = lr_query_slow, std_theta_perturb = std_theta_perturb, std_X_perturb = std_X_perturb, max_N_query=max_N_query, langevin_dt=langevin_dt)
        self.right_equiv = True

    def _load_config_optional(self, config):
        self.max_radius_Q = config['max_radius_Q']
        self.field_cutoff_Q = config['field_cutoff_Q']                   
        self.ranges_Q = torch.tensor(config['ranges_Q'])
        self.irreps_out_Q = o3.Irreps(config['irreps_out_Q'])
        self.sh_lmax_descriptor_Q = config['sh_lmax_descriptor_Q']
        self.number_of_basis_descriptor_Q = config['number_of_basis_descriptor_Q']
        self.irrep_normalization_Q = config['irrep_normalization_Q']
        self.query_radius = config['query_radius']

    def _init_models_optional(self):
        self.langevin = LangevinMH(ranges_X = self.ranges, dt = self.langevin_dt, std_theta = 1., std_X = 1.)
        #self.langevin = MH(ranges_X = self.ranges, std_theta = 0.2, std_X = 0.2)
        self.query_model = EquivWeightQueryModel(irreps_descriptor=self.irreps_out_Q, N_query = self.N_query, max_N_query = self.max_N_query,
                                                 max_radius=self.max_radius_Q, field_cutoff=self.field_cutoff_Q, sh_lmax=self.sh_lmax_descriptor_Q, 
                                                 number_of_basis=self.number_of_basis_descriptor_Q, query_radius=self.query_radius, irrep_normalization=self.irrep_normalization, layernorm=self.edf_layernorm)

    def init_optimizers(self):
        self.optimizer_se3T = torch.optim.Adam(list(self.se3T.parameters()), lr=self.lr_se3T, betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True)
        self.optimizer_energy_fast = torch.optim.Adam(self.energy_model_train.fast_parameters(), lr=self.lr_energy_fast, betas=(0.9, 0.98), eps=1e-09, weight_decay=0, amsgrad=True)
        self.optimizer_energy_slow = torch.optim.Adam(self.energy_model_train.slow_parameters(), lr=self.lr_energy_slow, betas=(0.9, 0.98), eps=1e-09, weight_decay=0, amsgrad=True)
        self.optimizer_query_fast = torch.optim.Adam(self.query_model.fast_parameters(), lr=self.lr_query_fast, betas=(0.9, 0.98), eps=1e-09, weight_decay=0, amsgrad=True)
        self.optimizer_query_slow = torch.optim.Adam(self.query_model.slow_parameters(), lr=self.lr_query_slow, betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True)

        # self.optimizer_se3T = torch.optim.SGD(list(self.se3T.parameters()), lr=self.lr_se3T, weight_decay=1e-4, momentum=0.9)
        # self.optimizer_energy_fast = torch.optim.SGD(self.energy_model_train.fast_parameters(), lr=self.lr_energy_fast, weight_decay=0, momentum=0.9)
        # self.optimizer_energy_slow = torch.optim.SGD(self.energy_model_train.slow_parameters(), lr=self.lr_energy_slow, weight_decay=0, momentum=0.9)
        # self.optimizer_query_fast = torch.optim.SGD(self.query_model.fast_parameters(), lr=self.lr_query_fast, weight_decay=1e-4, momentum=0.9)
        # self.optimizer_query_slow = torch.optim.SGD(self.query_model.slow_parameters(), lr=self.lr_query_slow, weight_decay=1e-4, momentum=0.9)

    def model_to(self, device):
        super().model_to(device)
        self.ranges_Q = self.ranges_Q.to(device)

    def crop_range_idx_Q(self, pos):
        in_range_cropped_idx_Q = (((pos > self.ranges_Q[:,0]) * (pos < self.ranges_Q[:,1])).sum(dim=-1) == 3).nonzero().squeeze(-1)
        return in_range_cropped_idx_Q



    
        


