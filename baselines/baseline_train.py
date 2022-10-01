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

from edf.utils import preprocess, voxelize_sample, OrthoTransform
from edf.dist import GaussianDistSE3

from baselines.equiv_tn.sixdof_non_equi_transporter import TransporterAgent
from baselines.equiv_tn.utils import perturb



def train(sample_dir = 'demo/mug_task_rim.gzip', root_dir = 'checkpoint_tn/rim', plot_path = 'logs/baselines/TN/train/', imshow = True, saveplot = False, lr=1e-5, max_epoch = 200):
    seed = 0
    device = 'cuda'


    if device == 'cpu':
        torch.use_deterministic_algorithms(True)
    elif device == 'cuda':
        #torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_printoptions(precision=4, sci_mode=False)

    
    with gzip.open(sample_dir,'rb') as f:
        train_samples = pickle.load(f)


    H = W = 160
    crop_size = 16*6
    ortho_ranges = np.array([[0.4, 0.8],[-0.2, 0.2], [0., 0.4]])
    ortho_transform = OrthoTransform(W = W, ranges = ortho_ranges[:2])
    pix_size = (ortho_ranges[0,1] - ortho_ranges[0,0]) / H

    perturb_dist = GaussianDistSE3(std_theta = 2./180*np.pi, std_X = 0.002)
    perturb_dist.dist_R.get_inv_cdf()


    agent = TransporterAgent(name='any', task='any', root_dir=root_dir, device=device, load=False, crop_size = crop_size, pix_size = pix_size, bounds = ortho_ranges, H=H, W=W, n_rotations=36, lr=lr)



    max_epochs = max_epoch
    iters = 0

    for epoch in range(1, max_epochs+1):
        train_sample_indices = list(range(len(train_samples)))
        np.random.shuffle(train_sample_indices)
        for train_sample_idx in train_sample_indices:
            iters += 1
            sample = train_samples[train_sample_idx].copy()
            sample['d'] = 0.001
            sample = voxelize_sample(sample, coord_jitter=3., color_jitter=0.03, pick=True, place=False)

            in_range_idx = ((sample['coord'][..., -1] > ortho_ranges[-1][0]) * (sample['coord'][..., -1] < ortho_ranges[-1][1]))
            coord = sample['coord'][in_range_idx]
            color = sample['color'][in_range_idx]

            img = ortho_transform.orthographic(coord, color)


            pick, place = sample['grasp'], sample['place']
            pick = torch.cat([transforms.matrix_to_quaternion(torch.from_numpy(pick[1])), torch.from_numpy(pick[0])], dim=-1)
            place = torch.cat([transforms.matrix_to_quaternion(torch.from_numpy(place[1])), torch.from_numpy(place[0])], dim=-1)

            pick = perturb_dist.propose(pick)
            place = perturb_dist.propose(place)

            pick = (pick[4:].numpy(), transforms.quaternion_to_matrix(pick[:4]).numpy())
            place = (place[4:].numpy(), transforms.quaternion_to_matrix(place[:4]).numpy())

            pick = ortho_transform.pose2pix_yaw_zrp(pick, grasp='top') # grasp_pix, yaw, height, roll, pitch 
            place = ortho_transform.pose2pix_yaw_zrp(place, grasp='top') # grasp_pix, yaw, height, roll, pitch 

            # img_test = (img.copy()[...,:3]*255).astype(np.uint8)
            # img_test = cv2.arrowedLine(img_test, pick[0][...,::-1], (pick[0][...,::-1] + np.array([np.cos(pick[1]/180*np.pi), -np.sin(pick[1]/180*np.pi)]) * 30).astype(int), (255,0,255), thickness = 3, tipLength=0.3)
            # img_test = cv2.arrowedLine(img_test, place[0][...,::-1], (place[0][...,::-1] + np.array([np.cos(place[1]/180*np.pi), -np.sin(place[1]/180*np.pi)]) * 30).astype(int), (0,0,255), thickness = 3, tipLength=0.3)
            # #img_test = Image.fromarray(img_test)
            # crop_size = 16*14
            # crop_test = (img.copy()[...,:3]*255).astype(np.uint8)
            # crop_test = crop_test[pick[0][0]-crop_size//2:pick[0][0]+crop_size//2, pick[0][1]-crop_size//2:pick[0][1]+crop_size//2]
            # #crop_test = Image.fromarray(crop_test)

            img_mean, img_std = np.array([[0.5, 0.5, 0.5, 0.25]]), np.array([[0.5, 0.5, 0.5, 0.25]])
            img = (img - img_mean) / img_std
            img = np.concatenate((img[Ellipsis, :3],
                                img[Ellipsis, 3:4],
                                img[Ellipsis, 3:4],
                                img[Ellipsis, 3:4]), axis=2).astype(np.float32)

            pick[1] = (pick[1] /180 *np.pi + 2*np.pi)%(2*np.pi)     # (-180~180) -> (0 ~ 2pi) Yaw
            pick[3] = (pick[3] /180 *np.pi + 2*np.pi)%(2*np.pi)     # (-180~180) -> (0 ~ 2pi) Roll
            pick[4] = (pick[4] /180 *np.pi)                         # (-90~90)   -> (-pi/2 ~ pi/2) Pitch
            place[1] = (place[1] /180 *np.pi + 2*np.pi)%(2*np.pi)   # (-180~180) -> (0 ~ 2pi) Yaw
            place[3] = (place[3] /180 *np.pi + 2*np.pi)%(2*np.pi)   # (-180~180) -> (0 ~ 2pi) Roll
            place[4] = (place[4] /180 *np.pi)                       # (-90~90)   -> (-pi/2 ~ pi/2) Pitch
            img, _, (pick[0], place[0]), (theta, trans, pivot) = perturb(img, [pick[0], place[0]], rim_offset= H//6)
            pick[1] = (pick[1] - theta + 2*np.pi) % (2*np.pi)
            place[1] = (place[1] - theta + 2*np.pi) % (2*np.pi)

            img_visual = img[...,:4].copy() * img_std + img_mean
            img_visual = img_visual - img_visual.min()
            img_visual = img_visual / img_visual.max()

            img_out = (img_visual.copy()[...,:3]*255).astype(np.uint8)
            img_gt = (img_visual.copy()[...,:3]*255).astype(np.uint8)
            crop_test = (img_visual.copy()[...,:3]*255).astype(np.uint8)
            crop_test = np.pad(crop_test, ((crop_size//2,crop_size//2), (crop_size//2,crop_size//2), (0, 0)))
            crop_test = crop_test[pick[0][0]:pick[0][0]+crop_size, pick[0][1]:pick[0][1]+crop_size]

            data = (img, pick, place)
            

            agent.train(data)
            
            if iters % 50 == 0 or iters == 1:
                agent.save()
                with torch.no_grad():
                    p0, p1, confs = agent.act(img=img, return_output=True, gt_data = data)
                pick_conf, place_conf, crop = confs
                pick_conf = pick_conf - pick_conf.min()
                pick_conf = pick_conf / pick_conf.max() * 255
                place_conf = place_conf - place_conf.min()
                place_conf = place_conf / place_conf.max() * 255
                # print(f"pick:    {p0}")
                # print(f"pick_gt:    {pick[0], pick[1]}")
                # print(f"place:    {p1}")
                # print(f"place_gt:    {place[0], place[1]}")
                # print(np.unravel_index(np.argmax(pick_conf), pick_conf.shape))
                # print(np.unravel_index(np.argmax(place_conf), place_conf.shape))

                img_out = cv2.arrowedLine(img_out, np.array(p0[0])[...,::-1], (np.array(p0[0])[...,::-1] + np.array([np.cos(p0[1]), -np.sin(p0[1])]) * 30).astype(int), (255,0,255), thickness = 3, tipLength=0.3)
                img_out = cv2.arrowedLine(img_out, np.array(p1[0])[...,::-1], (np.array(p1[0])[...,::-1] + np.array([np.cos(p1[1]), -np.sin(p1[1])]) * 30).astype(int), (0,0,255), thickness = 3, tipLength=0.3)
                img_gt = cv2.arrowedLine(img_gt, pick[0][...,::-1], (pick[0][...,::-1] + np.array([np.cos(pick[1]), -np.sin(pick[1])]) * 30).astype(int), (255,0,255), thickness = 3, tipLength=0.3)
                img_gt = cv2.arrowedLine(img_gt, place[0][...,::-1], (place[0][...,::-1] + np.array([np.cos(place[1]), -np.sin(place[1])]) * 30).astype(int), (0,0,255), thickness = 3, tipLength=0.3)
                # img_out = cv2.arrowedLine(img_out, np.array(p0[0]), (np.array(p0[0]) + np.array([np.cos(p0[1]), -np.sin(p0[1])]) * 30).astype(int), (255,0,255), thickness = 3, tipLength=0.3)
                # img_out = cv2.arrowedLine(img_out, np.array(p1[0]), (np.array(p1[0]) + np.array([np.cos(p1[1]), -np.sin(p1[1])]) * 30).astype(int), (0,0,255), thickness = 3, tipLength=0.3)
                # img_gt = cv2.arrowedLine(img_gt, pick[0], (pick[0] + np.array([np.cos(pick[1]), -np.sin(pick[1])]) * 30).astype(int), (255,0,255), thickness = 3, tipLength=0.3)
                # img_gt = cv2.arrowedLine(img_gt, place[0], (place[0] + np.array([np.cos(place[1]), -np.sin(place[1])]) * 30).astype(int), (0,0,255), thickness = 3, tipLength=0.3)
                print(f"PICK || Target z, roll, pitch: {pick[2]}, {pick[3]}, {pick[4]}", flush=True)
                print(f"PICK || z, roll, pitch: {p0[2]}, {p0[3]}, {p0[4]}", flush=True)
                print(f"PLACE || Target z, roll, pitch: {place[2]}, {place[3]}, {place[4]}", flush=True)
                print(f"PLACE || z, roll, pitch: {p1[2]}, {p1[3]}, {p1[4]}", flush=True)

                w = 7

                if imshow or saveplot:
                    fig, axes = plt.subplots(1, 5, figsize=(w*5,w))
                    axes[0].imshow(img_out)
                    axes[1].imshow(img_gt)
                    #axes[2].imshow(crop_test)
                    axes[2].imshow(crop)
                    axes[3].imshow(pick_conf[...,np.unravel_index(np.argmax(pick_conf), pick_conf.shape)[-1]])
                    axes[4].imshow(place_conf[...,np.unravel_index(np.argmax(place_conf), place_conf.shape)[-1]])

                    if saveplot:
                        if os.path.exists(plot_path) is False:
                            os.makedirs(plot_path)
                        fig.savefig(plot_path + f"{iters}.png")
                    if imshow:
                        plt.show()
                    if saveplot:
                        plt.clf()
                        plt.cla()
                        plt.close(fig)
                        plt.close('all')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for evaluation of baseline models (Transporter Nets)')

    parser.add_argument('--plot-path', type=str, default='logs_baseline/TN/rim_lowvar/train/imgs/',
                        help='')
    parser.add_argument('--root-dir', type=str, default='checkpoint_baseline/rim_lowvar',
                        help='')
    parser.add_argument('--sample-dir', type=str, default='demo/mug_task_rim_lowvar.gzip',
                        help='')
    parser.add_argument('--imshow', action='store_true',
                        help='')
    parser.add_argument('--saveplot', action='store_true',
                        help='')
    parser.add_argument('--max-epoch', type=int, default=2000,
                    help='')
    parser.add_argument('--lr', type=float, default=0.0001,
                    help='')
    args = parser.parse_args()



    sample_dir = args.sample_dir
    root_dir = args.root_dir
    plot_path = args.plot_path
    imshow = args.imshow
    saveplot = args.saveplot
    max_epoch = args.max_epoch
    lr = args.lr

    train(sample_dir = sample_dir, root_dir = root_dir, plot_path = plot_path, imshow = imshow, saveplot = saveplot, max_epoch = max_epoch, lr = lr)