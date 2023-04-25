import os

from edf.pc_utils import draw_geometry, create_o3d_points
from edf.data import PointCloud, SE3, TargetPoseDemo, DemoSequence, DemoSeqDataset, gzip_save
from edf.preprocess import Rescale, NormalizeColor, Downsample, PointJitter, ColorJitter
from edf.agent import PickAgent, PlaceAgent

import numpy as np
import yaml
import plotly as pl
import plotly.express as ple
import open3d as o3d

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose


torch.set_printoptions(precision= 3, sci_mode=False, linewidth=120)

agent_config_dir = "config/agent_config/place_agent.yaml"
train_config_dir = "config/train_config/train_place.yaml"
agent_param_dir = "checkpoint/mug_10_demo/place"

with open(train_config_dir) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
device = config['device']
unit_len = config['characteristic_length']
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


load_transforms = Compose([Rescale(rescale_factor=1/unit_len),
                          ])
trainset = DemoSeqDataset(dataset_dir="demo/test_demo", annotation_file="data.yaml", load_transforms = load_transforms, device=device)
train_dataloader = DataLoader(trainset, shuffle=True, collate_fn=lambda x:x)


scene_voxel_size = 1.7
grasp_voxel_size = 1.4
scene_points_jitter = scene_voxel_size * 0.1
grasp_points_jitter = grasp_voxel_size * 0.1
scene_color_jitter = 0.03
grasp_color_jitter = 0.03

scene_proc_fn = Compose([Downsample(voxel_size=1.7, coord_reduction="average"),
                         NormalizeColor(color_mean = torch.tensor([0.5, 0.5, 0.5]), color_std = torch.tensor([0.5, 0.5, 0.5])),
                         PointJitter(jitter_std=scene_points_jitter),
                         ColorJitter(jitter_std=scene_color_jitter)
                         ])
grasp_proc_fn = Compose([Downsample(voxel_size=1.4, coord_reduction="average"),
                         NormalizeColor(color_mean = torch.tensor([0.5, 0.5, 0.5]), color_std = torch.tensor([0.5, 0.5, 0.5])),
                         PointJitter(jitter_std=grasp_points_jitter),
                         ColorJitter(jitter_std=grasp_color_jitter)
                         ])



place_agent = PlaceAgent(config_dir=agent_config_dir, device=device, lr_se3T=lr_se3T, lr_energy_fast = lr_energy_fast, 
                            lr_energy_slow = lr_energy_slow, lr_query_fast = lr_query_fast, lr_query_slow = lr_query_slow, 
                            std_theta_perturb=std_theta_perturb, std_X_perturb=std_X_perturb, langevin_dt=langevin_dt)



# # Train



save_checkpoint = True
visualize = False



max_iter = len(trainset) * max_epochs
iter = 0
for epoch in range(1, max_epochs+1):
    for train_batch in train_dataloader:
        iter += 1
        assert len(train_batch) == 1, "Batch training is not supported yet."

        data = train_batch[0]
        demo_seq: DemoSequence = data.to(device)
        place_demo: TargetPoseDemo = demo_seq[1]
        scene_raw: PointCloud = place_demo.scene_pc
        grasp_raw: PointCloud = place_demo.grasp_pc
        target_poses: SE3 = place_demo.target_poses
        scene_proc = scene_proc_fn(scene_raw)
        grasp_proc = grasp_proc_fn(grasp_raw)
        

        ################################################# Train #########################################################
        N_transforms = N_transform_init
        mh_iter = mh_iter_init
        if epoch >= langevin_begin_epoch:
            langevin_iter = int( langevin_iter_init * (1+ iter / max_iter) )
        else:
            langevin_iter = 0
        
        if iter % report_freq == 0 or iter == 1:
            pbar = True
            verbose = True
            if visualize:
                raise NotImplementedError
            else:
                pass
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
        train_logs = place_agent.train_once(scene=scene_proc, target_T = target_poses, N_transforms = N_transforms,
                                            mh_iter = mh_iter, langevin_iter = langevin_iter, temperature = temperature, 
                                            pbar = pbar, verbose = verbose, grasp=grasp_proc, 
                                            CD_ratio = CD_ratio, edf_norm_std=edf_norm_std, query_temperature=query_temperature, surrogate_query=use_surrogate)
        
        train_logs['scene_raw'] = scene_raw.to('cpu')
        train_logs['grasp_raw'] = grasp_raw.to('cpu')
        train_logs['scene_proc'] = scene_proc.to('cpu')
        train_logs['grasp_proc'] = grasp_proc.to('cpu')

        if iter % report_freq == 0 or iter == 1:
            if save_checkpoint:
                filename = f'model_iter_{iter}.pt'
                place_agent.save(agent_param_dir, filename)

                log_filename = f'trainlog_iter_{iter}.gzip'
                gzip_save(train_logs, path=os.path.join(agent_param_dir, log_filename))
                
        if verbose:
            print("===============================", flush=True)
        
        
    


