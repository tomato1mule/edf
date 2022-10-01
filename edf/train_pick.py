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
from edf.agent import PickAgent

def train_pick(sample_dir, train_config_dir, agent_config_dir, visualize, show_plot, save_plot, save_checkpoint, checkpoint_path, plot_path, save_tp, seed=0, deterministic = True):
    print(f"Train Pick begin with seed {seed} at {str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}")
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
    langevin_dt = config['langevin_dt']

    # device = 'cuda'

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

    # ##### Load pickled tensor product modules (parameterless) for reproduciblity #####
    # if save_tp:
    #     pick_agent = PickAgent(config_dir=agent_config_dir, device=device, lr_se3T=lr_se3T, lr_energy_fast = lr_energy_fast, lr_energy_slow = lr_energy_slow, lr_query_fast = lr_query_fast, lr_query_slow = lr_query_slow, std_theta_perturb=std_theta_perturb, std_X_perturb=std_X_perturb)
    #     pick_agent.save_tp_path(tp_pickle_dir)
    # else:
    #     pick_agent = PickAgent(config_dir=agent_config_dir, tp_pickle_path=tp_pickle_dir, device=device, lr_se3T=lr_se3T, lr_energy_fast = lr_energy_fast, lr_energy_slow = lr_energy_slow, lr_query_fast = lr_query_fast, lr_query_slow = lr_query_slow, std_theta_perturb=std_theta_perturb, std_X_perturb=std_X_perturb)
    pick_agent = PickAgent(config_dir=agent_config_dir, device=device, lr_se3T=lr_se3T, lr_energy_fast = lr_energy_fast, 
                          lr_energy_slow = lr_energy_slow, lr_query_fast = lr_query_fast, lr_query_slow = lr_query_slow, 
                          std_theta_perturb=std_theta_perturb, std_X_perturb=std_X_perturb, langevin_dt=langevin_dt)



    ##### Save initial parameters #####
    if save_checkpoint:
        filename = f'model_iter_{0}.pt'
        pick_agent.save(checkpoint_path, filename)

    ##### Train #####
    max_iter = len(train_samples) * max_epochs
    iter = 0
    for epoch in range(1, max_epochs+1):
        train_sample_indices = list(range(len(train_samples)))
        np.random.shuffle(train_sample_indices)
        for train_sample_idx in train_sample_indices:
            iter += 1
            sample = train_samples[train_sample_idx]
            sample = voxelize_sample(sample, coord_jitter=0.1, color_jitter=0.03, pick=True, place=False)

            ##### Prepare input #####
            color_unprocessed = sample['color']
            sample = preprocess(sample, characteristic_length)

            coord, color, ranges = sample['coord'], sample['color'], sample['ranges']
            X_sdg, R_sdg = sample['grasp'][0], sample['grasp'][1]
            data_transform = sample['data_transform']

            
            target_pos = torch.tensor(X_sdg, dtype=torch.float32, device=device).unsqueeze(0) # (1, 3)
            target_orn = torch.tensor(R_sdg, dtype=torch.float32, device=device).unsqueeze(0) # (1, 3, 3)
            feature = torch.tensor(color, dtype=torch.float32, device=device)
            pos = torch.tensor(coord, dtype=torch.float32, device=device)
            in_range_cropped_idx = pick_agent.crop_range_idx(pos)
            pos, feature = pos[in_range_cropped_idx], feature[in_range_cropped_idx]

            inputs = {'feature': feature, 'pos': pos, 'edge': None, 'max_neighbor_radius': pick_agent.max_radius}
            target_T = torch.cat([transforms.matrix_to_quaternion(target_orn), target_pos], dim=-1)

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
                    visual_info = {'coord':coord[in_range_cropped_idx.cpu()], 
                                'color': color_unprocessed[in_range_cropped_idx.cpu()], 
                                'ranges': ranges,
                                'show': show_plot,
                                'save_path': plot_path,
                                'file_name': f'{iter}.png'
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

            if iter == int(max_iter * 0.9):
                print("Lower lr rate")
                pick_agent.rescale_lr(factor=0.2)

            if verbose:
                print(f"=========Iter {iter}=========", flush=True)
            pick_agent.train_once(inputs = inputs, target_T = target_T, N_transforms = N_transforms,
                                mh_iter = mh_iter, langevin_iter = langevin_iter, temperature = temperature, 
                                pbar = pbar, verbose = verbose, visual_info = visual_info, CD_ratio=CD_ratio, edf_norm_std=edf_norm_std)
            if iter % report_freq == 0 or iter == 1 or save_checkpoint is False:         
                if save_checkpoint:
                    filename = f'model_iter_{iter}.pt'
                    pick_agent.save(checkpoint_path, filename)
            if verbose:
                print("===============================", flush=True)

    print(f"Train Pick finished at {str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EDF Pick-Agent')

    parser.add_argument('--sample-dir', type=str, default='demo/mug_task.gzip',
                        help='Path to demonstration samples for training')
    parser.add_argument('--train-config-dir', type=str, default='config/train_config/train_pick.yaml',
                        help='Path to config files for training')
    parser.add_argument('--agent-config-dir', type=str, default='config/agent_config/pick_agent.yaml',
                        help='Path to config files for training')
    parser.add_argument('--tp-pickle-dir', type=str, default='reproducible_pickles/pick/',
                        help='Path to pickles of E3NN tensor product modules (parameterless) for reproducibility')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize if flagged')
    parser.add_argument('--show-plot', action='store_true',
                        help='Show pyplot figure for visualization in another window.')
    parser.add_argument('--save-plot', action='store_true',
                        help='Save pyplot figure for visualization in another window.')
    parser.add_argument('--save-checkpoint', action='store_true',
                        help='Save pytorch model checkpoints')
    parser.add_argument('--checkpoint-path', type=str, default='checkpoint/train_pick/',
                        help='Where to save pytorch model checkpoints')
    parser.add_argument('--plot-path', type=str, default='logs/train/pick_agent/plot/',
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

    train_pick(sample_dir=sample_dir, train_config_dir=train_config_dir, agent_config_dir=agent_config_dir,
               visualize=visualize, show_plot=show_plot, save_plot=save_plot, 
               save_checkpoint=save_checkpoint, checkpoint_path=checkpoint_path, plot_path=plot_path, save_tp=save_tp, seed=seed)














