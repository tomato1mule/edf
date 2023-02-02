import os
import argparse
import time
import datetime
import random
import numpy as np
import yaml
import gzip
import pickle
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2

import torch
from pytorch3d import transforms

from edf.utils import preprocess, voxelize_sample, OrthoTransform, binomial_test
from edf.visual_utils import scatter_plot_ax
from edf.pybullet_env.env import MugTask, StickTask, BowlTask, BottleTask
from edf.dist import GaussianDistSE3

from baselines.equiv_tn.sixdof_non_equi_transporter import TransporterAgent
from baselines.equiv_tn.utils import perturb


def eval(schedules, plot_path='logs/baselines/TN/', use_gui=True, visualize_plot=True, save_plot=False, root_dir = 'checkpoint_tn/rim', checkpoint_iter = 1000, task_config_dir = 'config/task_config/mug_task.yaml', task_type='mug_task', pick_grasp = 'top', place_grasp = 'top'):
    seed = 0
    device = 'cuda'

    with open(task_config_dir) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    sleep = config['sleep']
    d = config['d']
    d_pick = config['d_pick']
    d_place = config['d_place']

    plot_figsize = [28,7]
    pick_attempt_max = 100
    place_attempt_max = 100
    pick_only = False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cpu':
        torch.use_deterministic_algorithms(True)
    elif device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    torch.set_printoptions(precision=4, sci_mode=False)


    def pix2pose(p, yaw, z, roll, pitch, grasp='top'):
        if yaw > np.pi:
            yaw -= 2*np.pi
        yaw = yaw * 180 / np.pi

        if roll > np.pi:
            roll -= 2*np.pi
        roll = roll * 180 / np.pi

        pitch = min(max(pitch, -np.pi), np.pi)
        pitch = pitch * 180 / np.pi

        T = ortho_transform.pix_yaw_zrp2pose(grasp_pix=p, yaw=yaw, height=z, roll=roll, pitch=pitch, grasp=grasp)
        return T

    def save_plot_func():
        if os.path.exists(plot_path + "inference/") is False:
            os.makedirs(plot_path + "inference/")
        fig.savefig(plot_path + "inference/" + f"{seed}.png")
        if os.path.exists(plot_path + "result/") is False:
            os.makedirs(plot_path + "result/")
        fig_img.savefig(plot_path + "result/" + f"{seed}.png")

    def draw_result():
        #pc = task.observe_pointcloud(stride = (1, 1))
        #scatter_plot_ax(axes[3], pc['coord'], pc['color'], pc['ranges'])
        axes[0].imshow(img_out)
        images = task.observe()
        for i in range(3):
            axes_img[i].imshow(images[i]['color'])

    def plot():
        draw_result()
        if save_plot:
            save_plot_func()
        if visualize_plot:
            plt.show()
        else:
            plt.close(fig)
            plt.close(fig_img)

    def report():
        confidence = 0.95
        _, _, _, pick_result = binomial_test(success=N_success_pick, n=N_tests, confidence=confidence)
        _, _, _, place_result = binomial_test(success=N_success_place, n=N_success_pick, confidence=confidence)
        _, _, _, total_result = binomial_test(success=N_success_place, n=N_tests, confidence=confidence)

        print(f"Pick Success Rate: {pick_result}    ||   Place Success Rate: {place_result}    ||   Place-and-Place Success Rate: {total_result})", flush=True)
        plot()
        print("======================================", flush=True)

    def pick(T):
        # R, X = transforms.quaternion_to_matrix(T[...,:4]), T[...,4:]
        # X_sdg, R_sdg = data_transform.inv_transform_T(X.detach().cpu().numpy(), R.detach().cpu().numpy())
        X_sdg, R_sdg = T
        z_axis = R_sdg[:,-1]
        
        R_dg_dgpre = np.eye(3)
        R_s_dgpre = R_sdg @ R_dg_dgpre
        X_dg_dgpre = np.array([0., 0., -0.03])
        sX_dg_dgpre = R_sdg @ X_dg_dgpre
        X_s_dgpre = X_sdg + sX_dg_dgpre

        pre_pick = (X_s_dgpre, R_s_dgpre)
        pick = (X_sdg, R_sdg)

        try:
            task.pick(pre_pick, pick)
            print("Pick IK Success", flush=True)
            return True
        except StopIteration:
            #print("Pick IK Failed", flush=True)
            return False

    def place(T):
        # R, X = transforms.quaternion_to_matrix(T[...,:4]), T[...,4:]
        # X_sdg, R_sdg = data_transform_K.inv_transform_T(X.detach().cpu().numpy(), R.detach().cpu().numpy())
        X_sdg, R_sdg = T

        R_dg_dgpre = np.eye(3)
        R_s_dgpre = R_sdg @ R_dg_dgpre
        X_dg_dgpre = np.array([0., 0., -0.03])
        sX_dg_dgpre = R_sdg @ X_dg_dgpre
        X_s_dgpre = X_sdg + sX_dg_dgpre

        pre_place = (X_s_dgpre, R_s_dgpre)
        place = (X_sdg, R_sdg)

        try:
            task.place(pre_place, place)
            print("Place IK Success", flush=True)
            return True
        except StopIteration:
            #print("Place IK Failed", flush=True)
            return False



    H = W = 160
    crop_size = 16*6
    ortho_ranges = np.array([[0.4, 0.8],[-0.2, 0.2], [0., 0.4]])
    ortho_transform = OrthoTransform(W = W, ranges = ortho_ranges[:2])
    pix_size = (ortho_ranges[0,1] - ortho_ranges[0,0]) / H

    perturb_dist = GaussianDistSE3(std_theta = 2./180*np.pi, std_X = 0.2 * 0.01)
    perturb_dist.dist_R.get_inv_cdf()


    agent = TransporterAgent(name='any', task='any', root_dir=root_dir, device=device, load=False, crop_size = crop_size, pix_size = pix_size, bounds = ortho_ranges, H=H, W=W, n_rotations=36)
    agent.load(n_iter=checkpoint_iter)


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
            fig, axes = plt.subplots(1,4, figsize=plot_figsize)
            fig_img, axes_img = plt.subplots(1,3, figsize=plot_figsize)

            ##### Observe #####
            task.reset(seed = seed, mug_pose=mug_pose, mug_type=mug_type, distractor=distractor, use_support=use_support)
            pc = task.observe_pointcloud(stride = (1,1))
            sample = {}
            sample['coord'], sample['color'] = pc['coord'], pc['color']
            sample['range'] = pc['ranges']
            sample['d'] = 0.001
            sample = voxelize_sample(sample, coord_jitter=3., color_jitter=0.03, pick=True, place=False)

            in_range_idx = ((sample['coord'][..., -1] > ortho_ranges[-1][0]) * (sample['coord'][..., -1] < ortho_ranges[-1][1]))
            coord = sample['coord'][in_range_idx]
            color = sample['color'][in_range_idx]

            img = ortho_transform.orthographic(coord, color)

            img_mean, img_std = np.array([[0.5, 0.5, 0.5, 0.25]]), np.array([[0.5, 0.5, 0.5, 0.25]])
            img = (img - img_mean) / img_std
            img = np.concatenate((img[Ellipsis, :3],
                                img[Ellipsis, 3:4],
                                img[Ellipsis, 3:4],
                                img[Ellipsis, 3:4]), axis=2).astype(np.float32)

            img_visual = img[...,:4].copy() * img_std + img_mean
            img_visual = img_visual - img_visual.min()
            img_visual = img_visual / img_visual.max()

            img_out = (img_visual.copy()[...,:3]*255).astype(np.uint8)
            
            with torch.no_grad():
                pick_conf, zrp, zrp_log_std = agent.act_pick(img) # (H,W,nRot), (H,W,nRot,3), (H,W,nRot,3)
            indices = pick_conf.reshape(-1).argsort()[-pick_attempt_max:][::-1]
            hs,ws,theta_is = np.unravel_index(indices, pick_conf.shape)

            for (h, w, theta_i) in zip(hs,ws,theta_is):
                p0 = np.array((h, w))
                p0_theta = theta_i * (2 * np.pi / pick_conf.shape[2])
                z,r,p = zrp[h,w,theta_i].detach().cpu().numpy() #+ np.random.randn(3) * zrp_log_std[h,w,theta_i].detach().cpu().exp().numpy()
                T = pix2pose(p0, p0_theta, z, r, p, grasp=pick_grasp)

                #T[0][-1] = 0.09
                pick_ik_success = pick(T)
                if pick_ik_success:
                    break
                axes[1].imshow(pick_conf[...,np.unravel_index(np.argmax(pick_conf), pick_conf.shape)[-1]])
                img_out = cv2.arrowedLine(img_out, np.array(p0)[...,::-1], (np.array(p0)[...,::-1] + np.array([np.cos(p0_theta), -np.sin(p0_theta)]) * 30).astype(int), (255,0,255), thickness = 3, tipLength=0.3)


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

            task.retract_robot(gripper_val=1., IK_time=1., back=True)


            crop_test = (img_visual.copy()[...,:3]*255).astype(np.uint8)
            crop_test = np.pad(crop_test, ((crop_size//2,crop_size//2), (crop_size//2,crop_size//2), (0, 0)))
            crop_test = crop_test[p0[0]:p0[0]+crop_size, p0[1]:p0[1]+crop_size]
            axes[2].imshow(crop_test)

            with torch.no_grad():
                place_conf, zrp_place, zrp_log_std_place = agent.act_place(img, p0_pix = p0, p0_z = z, p0_roll = r, p0_pitch = p) # (H,W,nRot), (H,W,nRot,3), (H,W,nRot,3)
            indices = place_conf.reshape(-1).argsort()[-place_attempt_max:][::-1]
            hs,ws,theta_is = np.unravel_index(indices, place_conf.shape)
            
            for (h, w, theta_i) in zip(hs,ws,theta_is):
                p1 = np.array((h, w))
                p1_theta = theta_i * (2 * np.pi / place_conf.shape[2]) + p0_theta
                p1_theta = (p1_theta + 2*np.pi) % (2*np.pi)
                z,r,p = zrp_place[h,w,theta_i].detach().cpu().numpy() #+ np.random.randn(3) * zrp_log_std_place[h,w,theta_i].detach().cpu().exp().numpy()

                T = pix2pose(p1, p1_theta, z, r, p, grasp=place_grasp)

                place_ik_success = place(T)
                if place_ik_success:
                    break
            axes[3].imshow(place_conf[...,np.unravel_index(np.argmax(place_conf), place_conf.shape)[-1]])
            img_out = cv2.arrowedLine(img_out, np.array(p1)[...,::-1], (np.array(p1)[...,::-1] + np.array([np.cos(p1_theta), -np.sin(p1_theta)]) * 30).astype(int), (0,0,255), thickness = 3, tipLength=0.3)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for evaluation of baseline models (Transporter Nets)')

    parser.add_argument('--plot-path', type=str, default='logs/baselines/TN/',
                        help='')
    parser.add_argument('--use-gui', action='store_true',
                        help='')
    parser.add_argument('--visualize-plot', action='store_true',
                        help='')
    parser.add_argument('--save-plot', action='store_true',
                        help='')
    parser.add_argument('--root-dir', type=str, default='checkpoint_tn/rim',
                        help='')
    parser.add_argument('--task-config-dir', type=str, default='config/task_config/mug_task.yaml',
                        help='')
    parser.add_argument('--checkpoint-iter', type=int, default=1000,
                        help='')
    parser.add_argument('--init-seed', type=int, default=100,
                        help='')
    parser.add_argument('--end-seed', type=int, default=200,
                        help='')
    parser.add_argument('--mug-pose', type=str, default='upright',
                        help='')
    parser.add_argument('--mug-type', type=str, default='default',
                        help='')
    parser.add_argument('--distractor', action='store_true',
                        help='')
    parser.add_argument('--use-support', action='store_true',
                        help='')
    parser.add_argument('--task-type', type=str, default='mug_task',
                        help='')
    parser.add_argument('--pick-grasp', type=str, default='top',
                        help='')
    parser.add_argument('--place-grasp', type=str, default='top',
                        help='')

    args = parser.parse_args()



    plot_path = args.plot_path
    use_gui = args.use_gui
    visualize_plot = args.visualize_plot
    save_plot = args.save_plot
    root_dir = args.root_dir
    checkpoint_iter = args.checkpoint_iter
    task_config_dir = args.task_config_dir

    init_seed = args.init_seed
    end_seed = args.end_seed
    mug_pose = args.mug_pose
    mug_type = args.mug_type
    distractor = args.distractor
    use_support = args.use_support
    task_type = args.task_type
    pick_grasp = args.pick_grasp
    place_grasp = args.place_grasp

    schedule = {'mug_pose': mug_pose, 'mug_type': mug_type, 
                'distractor': distractor, 'use_support': use_support, 
                'init_seed': init_seed, 'end_seed': end_seed}

    schedules = [schedule]

    eval(schedules=schedules, plot_path=plot_path, use_gui=use_gui, visualize_plot=visualize_plot, save_plot=save_plot, root_dir = root_dir, checkpoint_iter = checkpoint_iter, task_config_dir = task_config_dir, task_type=task_type, pick_grasp=pick_grasp, place_grasp=place_grasp)