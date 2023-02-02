import time
import datetime
import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import yaml
import gc

import torch
from pytorch3d import transforms

from edf.utils import preprocess, voxel_filter, binomial_test, voxelize_sample
from edf.agent import PickAgent, PlaceAgent
from edf.pybullet_env.env import MugTask, StickTask, BowlTask, BottleTask
from edf.visual_utils import scatter_plot_ax


def eval(eval_config_dir, task_config_dir, pick_agent_config_dir, 
         checkpoint_path_pick, place_agent_config_dir, checkpoint_path_place, 
         plot_path, use_gui, visualize_plot, save_plot, deterministic = True, place_max_distance_plan=(0.05, 1.5), task_type='mug_task'):

    def draw_result():
        pc = task.observe_pointcloud(stride = (1, 1))
        scatter_plot_ax(axes[3], pc['coord'], pc['color'], pc['ranges'])
        images = task.observe()
        for i in range(3):
            axes_img[i].imshow(images[i]['color'])

    def save_plot_func():
        if os.path.exists(plot_path + "inference/") is False:
            os.makedirs(plot_path + "inference/")
        fig.savefig(plot_path + "inference/" + f"{seed}.png")
        if os.path.exists(plot_path + "result/") is False:
            os.makedirs(plot_path + "result/")
        fig_img.savefig(plot_path + "result/" + f"{seed}.png")

    def plot():
        draw_result()
        if save_plot:
            save_plot_func()
        if visualize_plot:
            plt.show()

        fig.clear()
        fig_img.clear()
        fig.clf()
        fig_img.clf()
        for ax in axes:
            ax.cla()
        for ax in axes_img:
            ax.cla()
        plt.clf()
        plt.cla()
        plt.close(fig)
        plt.close(fig_img)
        plt.close('all')
        gc.collect()

    def report():
        confidence = 0.95
        _, _, _, pick_result = binomial_test(success=N_success_pick, n=N_tests, confidence=confidence)
        _, _, _, place_result = binomial_test(success=N_success_place, n=N_success_pick, confidence=confidence)
        _, _, _, total_result = binomial_test(success=N_success_place, n=N_tests, confidence=confidence)

        print(f"Pick Success Rate: {pick_result}    ||   Place Success Rate: {place_result}    ||   Place-and-Place Success Rate: {total_result})", flush=True)
        plot()
        print("======================================", flush=True)

    def pick(T):
        R, X = transforms.quaternion_to_matrix(T[...,:4]), T[...,4:]
        X_sdg, R_sdg = data_transform.inv_transform_T(X.detach().cpu().numpy(), R.detach().cpu().numpy())
        z_axis = R_sdg[:,-1]
        
        # R_dg_dgpre = np.eye(3)
        # R_s_dgpre = R_sdg @ R_dg_dgpre
        # X_dg_dgpre = np.array([0., 0., -0.03])
        # sX_dg_dgpre = R_sdg @ X_dg_dgpre
        # X_s_dgpre = X_sdg + sX_dg_dgpre

        # pre_pick = (X_s_dgpre, R_s_dgpre)
        pick = (X_sdg, R_sdg)

        try:
            task.pick(None, pick)
            print("Pick IK Success", flush=True)
            return True
        except StopIteration:
            #print("Pick IK Failed", flush=True)
            return False

    def place(T):
        R, X = transforms.quaternion_to_matrix(T[...,:4]), T[...,4:]
        X_sdg, R_sdg = data_transform_K.inv_transform_T(X.detach().cpu().numpy(), R.detach().cpu().numpy())
        # R_dg_dgpre = np.eye(3)
        # R_s_dgpre = R_sdg @ R_dg_dgpre
        # X_dg_dgpre = np.array([0., 0., -0.03])
        # sX_dg_dgpre = R_sdg @ X_dg_dgpre
        # X_s_dgpre = X_sdg + sX_dg_dgpre

        # pre_place = (X_s_dgpre, R_s_dgpre)
        place = (X_sdg, R_sdg)

        try:
            task.place(None, place, max_distance_plan=place_max_distance_plan)
            print("Place IK Success", flush=True)
            return True
        except StopIteration:
            #print("Place IK Failed", flush=True)
            return False



    ##### Load eval config #####
    with open(eval_config_dir) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    device = config['device']
    characteristic_length = config['characteristic_length']
    plot_figsize = config['plot_figsize']
    plot_result_figsize = config['plot_result_figsize']
    pick_only = config['pick_only']

    pick_policy = config['pick_policy']
    pick_dist_temp = config['pick_dist_temp']
    pick_policy_temp = config['pick_policy_temp']
    pick_attempt_max = config['pick_attempt_max']
    N_transform_pick = config['N_transform_pick']
    mh_iter_pick = config['mh_iter_pick']
    langevin_iter_pick = config['langevin_iter_pick']
    optim_iter_pick = config['optim_iter_pick']
    langevin_dt_pick = config['langevin_dt_pick']
    optim_lr_pick = config['optim_lr_pick']
    X_seed_mean_pick = config['X_seed_mean_pick']
    X_seed_std_pick = config['X_seed_std_pick']
    max_N_query_pick = config['max_N_query_pick']

    place_policy = config['place_policy']
    place_dist_temp = config['place_dist_temp']
    place_policy_temp = config['place_policy_temp']
    place_attempt_max = config['place_attempt_max']
    N_transform_place = config['N_transform_place']
    mh_iter_place = config['mh_iter_place']
    langevin_iter_place = config['langevin_iter_place']
    optim_iter_place = config['optim_iter_place']
    langevin_dt_place = config['langevin_dt_place']
    optim_lr_place = config['optim_lr_place']
    X_seed_mean_place = config['X_seed_mean_place']
    X_seed_std_place = config['X_seed_std_place']
    max_N_query_place = config['max_N_query_place']
    query_temp_place = config['query_temp_place']


    schedules = config['schedules']

    ##### Load train config #####
    with open(task_config_dir) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    sleep = config['sleep']
    d = config['d']
    d_pick = config['d_pick']
    d_place = config['d_place']

    model_seed = 0
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
    torch.set_printoptions(precision=4, sci_mode=False)

    ##### Load agent models #####
    # if random_init_tp:
    #     pick_agent = PickAgent(config_dir=pick_agent_config_dir, device=device, max_N_query=max_N_query_pick)
    #     place_agent = PlaceAgent(config_dir=place_agent_config_dir, device=device, max_N_query=max_N_query_place)
    # else:
    #     pick_agent = PickAgent(config_dir=pick_agent_config_dir, tp_pickle_path=pick_tp_pickle_dir, device=device, max_N_query=max_N_query_pick)
    #     place_agent = PlaceAgent(config_dir=place_agent_config_dir, tp_pickle_path=place_tp_pickle_dir, device=device, max_N_query=max_N_query_place)
    pick_agent = PickAgent(config_dir=pick_agent_config_dir, device=device, max_N_query=max_N_query_pick, langevin_dt=langevin_dt_pick).requires_grad_(False)
    place_agent = PlaceAgent(config_dir=place_agent_config_dir, device=device, max_N_query=max_N_query_place, langevin_dt=langevin_dt_place).requires_grad_(False)
    pick_agent.load(checkpoint_path_pick)
    place_agent.load(checkpoint_path_place)

    ##### Initialize task env #####
    if task_type=='mug_task':
        task = MugTask(use_gui=use_gui)
    elif task_type=='stick_task':
        task = StickTask(use_gui=use_gui)
    elif task_type=='bowl_task':
        task = BowlTask(use_gui=use_gui)
    elif task_type=='bottle_task':
        task = BottleTask(use_gui=use_gui)
    else:
        raise ValueError



    ##### Evaluate #####
    N_tests = 0
    N_success_pick = 0
    N_success_place = 0
    N_IKFAIL_pick = 0
    N_IKFAIL_place = 0
    pick_times = []
    place_times = []

    for schedule in schedules:
        mug_pose = schedule['mug_pose']
        mug_type = schedule['mug_type']
        distractor = schedule['distractor']
        use_support = schedule['use_support']
        for seed in range(schedule['init_seed'], schedule['end_seed']):
            N_tests += 1
            print(f"=================Sample {seed}==================", flush=True)
            plt.clf()
            plt.cla()
            plt.close('all')
            fig, axes = plt.subplots(1,4, figsize=plot_figsize, subplot_kw={'projection':'3d'})
            fig_img, axes_img = plt.subplots(1,3, figsize=plot_figsize)

            ##### Observe #####
            task.reset(seed = seed, mug_pose=mug_pose, mug_type=mug_type, distractor=distractor, use_support=use_support)
            pc = task.observe_pointcloud(stride = (1,1))
            sample_unprocessed = {}
            sample_unprocessed['coord'] ,sample_unprocessed['color'] = voxel_filter(pc['coord'], pc['color'], d=d)
            # vox = voxelize_sample({'coord': pc['coord'], 'color': pc['color'], 'd': d}, coord_jitter=0.1, color_jitter=0.03, pick=True, place=False)
            # sample_unprocessed['coord'] ,sample_unprocessed['color'] = vox['coord'], vox['color']
            sample_unprocessed['range'] = pc['ranges']
            sample_unprocessed['images'] = task.observe()
            sample_unprocessed['center'] = task.center
            color_unprocessed = sample_unprocessed['color']
            sample = preprocess(sample_unprocessed, characteristic_length)


            ##### Prepare Input #####
            coord, color, ranges = sample['coord'], sample['color'], sample['ranges']
            data_transform = sample['data_transform']
            feature = torch.tensor(color, dtype=torch.float32, device=device)
            pos = torch.tensor(coord, dtype=torch.float32, device=device)
            in_range_cropped_idx = pick_agent.crop_range_idx(pos)
            pos, feature = pos[in_range_cropped_idx], feature[in_range_cropped_idx]
            inputs = {'feature': feature, 'pos': pos, 'edge': None, 'max_neighbor_radius': pick_agent.max_radius}


            ##### Pick Inference #####
            t1 = time.time()
            N_transforms = N_transform_pick
            mh_iter = mh_iter_pick
            langevin_iter = langevin_iter_pick
            #T_seed_pos = torch.tensor([X_seed_std_pick])* torch.randn(N_transforms,3) + torch.tensor(X_seed_mean_pick)
            # T_seed_pos = torch.rand(N_transforms,3, device=device) * (pick_agent.ranges[:,1] - pick_agent.ranges[:,0]) + pick_agent.ranges[:,0].unsqueeze(-2)
            # T_seed = torch.cat([transforms.random_quaternions(N_transforms, device=device), T_seed_pos.to(device)] , dim=-1)
            T_seed = N_transforms
            visual_info = {'coord':coord[in_range_cropped_idx.cpu()], 
                            'color': color_unprocessed[in_range_cropped_idx.cpu()], 
                            'ranges': ranges,
                            'ax': axes[0]}
            Ts, edf_outputs, logs = pick_agent.forward(inputs=inputs, T_seed=T_seed, policy = pick_policy, mh_iter=mh_iter, langevin_iter=langevin_iter, 
                                                       temperature=pick_dist_temp, policy_temperature=pick_policy_temp, optim_iter=optim_iter_pick, optim_lr=optim_lr_pick)
            t2 = time.time()
            print(f"Pick inference time: {t2-t1}", flush=True)

            ##### Pick #####
            for T in Ts[:pick_attempt_max]:
                pick_ik_success = pick(T)
                if pick_ik_success:
                    break
            pick_agent.visualize(visual_info, edf_outputs=edf_outputs, T = T, target_T = None, Ts = Ts)

            if not pick_ik_success:
                print("Pick fail: Couldn't find IK solution", flush=True)
                N_IKFAIL_pick += 1
                report()
                continue

            if task.check_pick_success():
                print("Pick success", flush=True)
                N_success_pick += 1
            else:
                print("Pick fail: Found IK solution but failed", flush=True)
                report()
                continue
            
            if pick_only:
                report()
                continue


            ############################################# Pick Finished #######################################
            ############################################# Place Starts  #######################################



            
            ##### Observe after pick #####
            task.retract_robot(gripper_val=1., IK_time=1., back=True)
            pc_pick = task.observe_pointcloud_pick(stride = (1, 1))
            sample_unprocessed['range_pick'] = pc_pick['ranges']
            sample_unprocessed['pick_pose'] = (pc_pick['X_sg'], pc_pick['R_sg'])
            sample_unprocessed['images_pick'] = task.observe_pick()
            pc_place = task.observe_pointcloud(stride = (1, 1))
            sample_unprocessed['range_place'] = pc_place['ranges']
            sample_unprocessed['images_place'] = task.observe()

            sample_unprocessed['coord_pick'], sample_unprocessed['color_pick'] = voxel_filter(pc_pick['coord'], pc_pick['color'], d=d_pick)
            sample_unprocessed['coord_place'], sample_unprocessed['color_place'] = voxel_filter(pc_place['coord'], pc_place['color'], d=d_place)
            # vox = voxelize_sample({'coord_pick': pc_pick['coord'], 'color_pick': pc_pick['color'], 'd_pick': d_pick, 
            #                        'coord_place': pc_place['coord'], 'color_place': pc_place['color'], 'd_place': d_place,}, coord_jitter=0.1, color_jitter=0.03, pick=False, place=True)
            # sample_unprocessed['coord_pick'], sample_unprocessed['color_pick'], sample_unprocessed['coord_place'], sample_unprocessed['color_place'] = vox['coord_pick'], vox['color_pick'], vox['coord_place'], vox['color_place']

            color_unprocessed_Q = sample_unprocessed['color_pick']
            color_unprocessed_K = sample_unprocessed['color_place']
            sample = preprocess(sample_unprocessed, characteristic_length, pick_and_place=True)


            ##### Prepare input #####
            coord_Q, color_Q, ranges_Q = sample['coord_Q'], sample['color_Q'], sample['ranges_Q']
            data_transform_Q = sample['data_transform_Q']
            coord_K, color_K, ranges_K = sample['coord_K'], sample['color_K'], sample['ranges_K']
            data_transform_K = sample['data_transform_K']

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


            ##### Place Inference #####
            t1 = time.time()
            N_transforms = N_transform_place
            mh_iter = mh_iter_place
            langevin_iter = langevin_iter_place
            # T_seed_pos = torch.tensor([X_seed_std_place])* torch.randn(N_transforms,3) + torch.tensor(X_seed_mean_place)
            # T_seed_pos = torch.rand(N_transforms,3, device=device) * (place_agent.ranges[:,1] - place_agent.ranges[:,0]) + place_agent.ranges[:,0].unsqueeze(-2)
            # T_seed = torch.cat([transforms.random_quaternions(N_transforms, device=device), T_seed_pos.to(device)] , dim=-1)
            T_seed = N_transforms
            visual_info_K = {'coord':coord_K[in_range_cropped_idx_K.cpu()], 
                            'color': color_unprocessed_K[in_range_cropped_idx_K.cpu()], 
                            'ranges': ranges_K,
                            'ax': axes[1],
                            'coord_query': coord_Q[in_range_cropped_idx_Q.cpu()],
                            'color_query': color_unprocessed_Q[in_range_cropped_idx_Q.cpu()],
                            'ranges_query': ranges_Q,
                            'ax_query': axes[2]
                            }
            Ts, edf_outputs, logs = place_agent.forward(inputs=inputs_K, T_seed=T_seed, inputs_Q=inputs_Q, policy = place_policy, mh_iter=mh_iter, langevin_iter=langevin_iter, 
                                                        temperature=place_dist_temp, policy_temperature=place_policy_temp, optim_iter=optim_iter_place, optim_lr=optim_lr_place, query_temperature=query_temp_place)
            t2 = time.time()
            print(f"Place inference time: {t2-t1}", flush=True)

            ##### Place #####
            task.retract_robot(gripper_val=1., IK_time=1., back=False)
            for T in Ts[:place_attempt_max]:
                place_ik_success = place(T)
                if place_ik_success:
                    break
            place_agent.visualize(visual_info_K, edf_outputs=edf_outputs, T = T, target_T = None, Ts = Ts)
            place_agent.visualize_query(visual_info_K, edf_outputs=edf_outputs)

            if not place_ik_success:
                print("Place fail: Couldn't find IK solution", flush=True)
                N_IKFAIL_place += 1
                report()
                continue

            if task.check_place_success():
                N_success_place += 1
                print('Place Success', flush=True)
            else:
                print('Place Fail', flush=True)

            ##### Visualize final #####
            report()
    task.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate EDF agents for pick-and-place task')

    parser.add_argument('--eval-config-dir', type=str, default='config/eval_config/eval.yaml',
                        help='')
    parser.add_argument('--task-config-dir', type=str, default='config/task_config/mug_task.yaml',
                        help='')
    parser.add_argument('--pick-agent-config-dir', type=str, default='config/agent_config/pick_agent.yaml',
                        help='')
    parser.add_argument('--checkpoint-path-pick', type=str, default='checkpoint/train_pick/model_iter_500.pt',
                        help='')
    parser.add_argument('--place-agent-config-dir', type=str, default='config/agent_config/place_agent.yaml',
                        help='')
    parser.add_argument('--checkpoint-path-place', type=str, default='checkpoint/train_place/model_iter_500.pt',
                        help='')
    parser.add_argument('--plot-path', type=str, default='logs/eval/eval_name/plot/',
                        help='')
    parser.add_argument('--use-gui', action='store_true',
                        help='')
    parser.add_argument('--visualize-plot', action='store_true',
                        help='')
    parser.add_argument('--save-plot', action='store_true',
                        help='')
    parser.add_argument('--place-max-distance-plan', type=float, default=[0.05, 1.5], nargs='+',
                        help='')
    parser.add_argument('--task-type', type=str, default='mug_task',
                        help='')              
    args = parser.parse_args()

    eval_config_dir = args.eval_config_dir
    task_config_dir = args.task_config_dir
    pick_agent_config_dir = args.pick_agent_config_dir
    checkpoint_path_pick = args.checkpoint_path_pick
    place_agent_config_dir = args.place_agent_config_dir
    checkpoint_path_place = args.checkpoint_path_place
    plot_path = args.plot_path

    use_gui = args.use_gui
    visualize_plot = args.visualize_plot
    save_plot = args.save_plot
    place_max_distance_plan = args.place_max_distance_plan
    task_type = args.task_type

    if save_plot is False:
        plot_path = None

    eval(eval_config_dir=eval_config_dir, task_config_dir=task_config_dir, pick_agent_config_dir=pick_agent_config_dir,
         checkpoint_path_pick=checkpoint_path_pick, place_agent_config_dir=place_agent_config_dir, 
         checkpoint_path_place=checkpoint_path_place, plot_path=plot_path, use_gui=use_gui,
         visualize_plot=visualize_plot, save_plot=save_plot, place_max_distance_plan=place_max_distance_plan, task_type=task_type)