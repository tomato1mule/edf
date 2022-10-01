import os
import argparse
import time
import datetime
import random
import numpy as np
import yaml
import gzip
import pickle


import torch
from pytorch3d import transforms

from edf.utils import preprocess, voxelize_sample
from edf.agent import PlaceAgent

def train_place(sample_dir, train_config_dir, agent_config_dir, visualize, show_plot, save_plot, save_checkpoint, checkpoint_path, plot_path, save_tp, seed = 0, deterministic = True):
    print(f"Train Place begin with seed {seed} at {str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}")
    if save_plot is False:
        plot_path = None

    ##### Load train config #####
    with open(train_config_dir) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    device = config['device']
    characteristic_length = config['characteristic_length']
    temperature = config['temperature']
    max_epochs = config['max_epochs']
    N_transform_init = config['N_transform_init']
    mh_iter_init = config['mh_iter_init']
    langevin_iter_init = config['langevin_iter_init']
    langevin_begin_epoch = config['langevin_begin_epoch']
    report_freq = config['report_freq']
    init_CD_ratio = config['init_CD_ratio']
    end_CD_ratio = config['end_CD_ratio']
    CD_end_iter = config['CD_end_iter']
    lr_se3T = config['lr_se3T']
    lr_energy_fast = config['lr_energy_fast']
    lr_energy_slow = config['lr_energy_slow']
    lr_query_fast = config['lr_query_fast']
    lr_query_slow = config['lr_query_slow']
    std_theta_perturb = config['std_theta_degree_perturb'] / 180 * np.pi
    std_X_perturb = config['std_X_perturb']
    edf_norm_std = config['edf_norm_std']
    query_anneal_end_iter = config['query_anneal_end_iter']
    query_init_temp = config['query_init_temp']
    query_end_temp = config['query_end_temp']
    elbo_end_iter = config['elbo_end_iter']
    langevin_dt = config['langevin_dt']

    #device = 'cuda'
    #mh_iter_init = 3
    #langevin_iter_init = 3


    ##### Load samples #####
    with gzip.open(sample_dir,'rb') as f:
        train_samples = pickle.load(f)

    ##### Set seeds #####
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
    torch.set_printoptions(precision=4, sci_mode=False)

    ##### Load pickled tensor product modules (parameterless) for reproduciblity #####
    # if save_tp:
    #     place_agent = PlaceAgent(config_dir=agent_config_dir, device=device, lr_se3T=lr_se3T, lr_energy_fast = lr_energy_fast, lr_energy_slow = lr_energy_slow, lr_query_fast = lr_query_fast, lr_query_slow = lr_query_slow, std_theta_perturb=std_theta_perturb, std_X_perturb=std_X_perturb)
    #     place_agent.save_tp_path(tp_pickle_dir)
    # else:
    #     place_agent = PlaceAgent(config_dir=agent_config_dir, tp_pickle_path=tp_pickle_dir, device=device, lr_se3T=lr_se3T, lr_energy_fast = lr_energy_fast, lr_energy_slow = lr_energy_slow, lr_query_fast = lr_query_fast, lr_query_slow = lr_query_slow, std_theta_perturb=std_theta_perturb, std_X_perturb=std_X_perturb)
    place_agent = PlaceAgent(config_dir=agent_config_dir, device=device, lr_se3T=lr_se3T, lr_energy_fast = lr_energy_fast, 
                             lr_energy_slow = lr_energy_slow, lr_query_fast = lr_query_fast, lr_query_slow = lr_query_slow, 
                             std_theta_perturb=std_theta_perturb, std_X_perturb=std_X_perturb, langevin_dt=langevin_dt)

    ##### Save initial parameters #####
    if save_checkpoint:
        filename = f'model_iter_{0}.pt'
        place_agent.save(checkpoint_path, filename)

    ##### Train #####
    max_iter = len(train_samples) * max_epochs
    iter = 0
    for epoch in range(1, max_epochs+1):
        train_sample_indices = list(range(len(train_samples)))
        np.random.shuffle(train_sample_indices)
        for train_sample_idx in train_sample_indices:
            iter += 1
            sample = train_samples[train_sample_idx]
            sample = voxelize_sample(sample, coord_jitter=0.1, color_jitter=0.03, pick=False, place=True)

            ##### Prepare input #####
            color_unprocessed_Q = sample['color_pick']
            color_unprocessed_K = sample['color_place']
            sample = preprocess(sample, characteristic_length, pick_and_place=True)

            coord_Q, color_Q, ranges_Q = sample['coord_Q'], sample['color_Q'], sample['ranges_Q']
            data_transform_Q = sample['data_transform_Q']

            coord_K, color_K, ranges_K = sample['coord_K'], sample['color_K'], sample['ranges_K']
            X_sdg_K, R_sdg_K = sample['grasp_K'][0], sample['grasp_K'][1]
            data_transform_K = sample['data_transform_K']

            target_pos_K = torch.tensor(X_sdg_K, dtype=torch.float32, device=device).unsqueeze(0) # (1, 3)
            target_orn_K = torch.tensor(R_sdg_K, dtype=torch.float32, device=device).unsqueeze(0) # (1, 3, 3)

            feature_Q = torch.tensor(color_Q, dtype=torch.float32, device=device)
            pos_Q = torch.tensor(coord_Q, dtype=torch.float32, device=device)
            in_range_cropped_idx_Q = place_agent.crop_range_idx_Q(pos_Q)
            pos_Q, feature_Q  = pos_Q[in_range_cropped_idx_Q], feature_Q[in_range_cropped_idx_Q]

            feature_K = torch.tensor(color_K, dtype=torch.float32, device=device)
            pos_K = torch.tensor(coord_K, dtype=torch.float32, device=device)
            in_range_cropped_idx_K = place_agent.crop_range_idx(pos_K)
            pos_K, feature_K = pos_K[in_range_cropped_idx_K], feature_K[in_range_cropped_idx_K]

            inputs_Q = {'feature': feature_Q, 'pos': pos_Q, 'edge': None, 'max_neighbor_radius': place_agent.max_radius_Q}
            inputs_K = {'feature': feature_K, 'pos': pos_K, 'edge': None, 'max_neighbor_radius': place_agent.max_radius}
            target_T_sp = torch.cat([transforms.matrix_to_quaternion(target_orn_K), target_pos_K], dim=-1) # Place pose in space frame

            ##### Train #####
            N_transforms = N_transform_init
            #mh_iter = int( mh_iter_init * np.exp(iter / max_iter) )
            mh_iter = mh_iter_init
            if epoch >= langevin_begin_epoch:
                langevin_iter = int( langevin_iter_init * (1+ iter / max_iter) )
            else:
                langevin_iter = 0
            if iter % report_freq == 0 or iter == 1 or save_checkpoint is False:
                pbar = True
                verbose = True
                if visualize:
                    visual_info = {'coord':coord_K[in_range_cropped_idx_K.detach().cpu()], 
                    'color': color_unprocessed_K[in_range_cropped_idx_K.detach().cpu()], 
                    'ranges': ranges_K,
                    'show': show_plot,
                    'save_path': plot_path,
                    'file_name': f'{iter}.png',
                    'coord_query': coord_Q[in_range_cropped_idx_Q.detach().cpu()],
                    'color_query': color_unprocessed_Q[in_range_cropped_idx_Q.detach().cpu()],
                    'ranges_query': ranges_Q,
                    }
                else:
                    visual_info = None

                
            else:
                pbar = False
                verbose = False
                visual_info = None

            if iter > CD_end_iter:
                CD_ratio = end_CD_ratio
            else:
                p_CD = 1 - (iter-1)/CD_end_iter
                CD_ratio = p_CD*init_CD_ratio + (1-p_CD)*end_CD_ratio
            if iter > query_anneal_end_iter:
                query_temperature = query_end_temp
            else:
                p_qt = 1 - (iter-1)/query_anneal_end_iter
                query_temperature = p_qt*query_init_temp + (1-p_qt)*query_end_temp

            if iter > elbo_end_iter:
                use_surrogate = False
            else:
                use_surrogate = True

            if iter == int(max_iter * 0.9):
                print("Lower lr rate")
                place_agent.rescale_lr(factor=0.2)

            if verbose:
                print(f"=========Iter {iter}=========", flush=True)
            place_agent.train_once(inputs = inputs_K, target_T = target_T_sp, N_transforms = N_transforms,
                                mh_iter = mh_iter, langevin_iter = langevin_iter, temperature = temperature, 
                                pbar = pbar, verbose = verbose, visual_info = visual_info, inputs_Q=inputs_Q, 
                                CD_ratio = CD_ratio, edf_norm_std=edf_norm_std, query_temperature=query_temperature, surrogate_query=use_surrogate)
            if iter % report_freq == 0 or iter == 1 or save_checkpoint is False:
                if save_checkpoint:
                    filename = f'model_iter_{iter}.pt'
                    place_agent.save(checkpoint_path, filename)
            if verbose:
                print("===============================", flush=True)

    print(f"Train Place finished at {str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EDF Place-Agent')

    parser.add_argument('--sample-dir', type=str, default='demo/mug_task.gzip',
                        help='Path to demonstration samples for training')
    parser.add_argument('--train-config-dir', type=str, default='config/train_config/train_place.yaml',
                        help='Path to config files for training')
    parser.add_argument('--agent-config-dir', type=str, default='config/agent_config/place_agent.yaml',
                        help='Path to config files for training')
    parser.add_argument('--tp-pickle-dir', type=str, default='reproducible_pickles/place/',
                        help='Path to pickles of E3NN tensor product modules (parameterless) for reproducibility')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize if flagged')
    parser.add_argument('--show-plot', action='store_true',
                        help='Show pyplot figure for visualization in another window.')
    parser.add_argument('--save-plot', action='store_true',
                        help='Save pyplot figure for visualization in another window.')
    parser.add_argument('--save-checkpoint', action='store_true',
                        help='Save pytorch model checkpoints')
    parser.add_argument('--checkpoint-path', type=str, default='checkpoint/train_place/',
                        help='Where to save pytorch model checkpoints')
    parser.add_argument('--plot-path', type=str, default='logs/train/place_agent/plot/',
                        help='Where to save visualization image')
    parser.add_argument('--save-tp', action='store_true',
                        help='Save pickles of E3NN tensor product modules (parameterless) for reproducibility')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for random number generators')
    args = parser.parse_args()

    sample_dir = args.sample_dir
    train_config_dir = args.train_config_dir
    agent_config_dir = args.agent_config_dir
    tp_pickle_dir = args.tp_pickle_dir
    visualize = args.visualize
    show_plot = args.show_plot
    save_plot = args.save_plot
    save_checkpoint = args.save_checkpoint
    checkpoint_path = args.checkpoint_path
    plot_path = args.plot_path
    save_tp = args.save_tp
    seed = args.seed

    train_place(sample_dir=sample_dir, train_config_dir=train_config_dir, agent_config_dir=agent_config_dir, 
                visualize=visualize, show_plot=show_plot, save_plot=save_plot, 
                save_checkpoint=save_checkpoint, checkpoint_path=checkpoint_path, plot_path=plot_path, save_tp=save_tp, seed=seed)














