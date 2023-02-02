import os
import argparse
import random
import numpy as np
import yaml

from edf.pybullet_env.utils import voxel_filter
from edf.pybullet_env.env import BottleTask
from edf.utils import OrthoTransform

# PYTHONHASHSEED=0 python3 generate_demo_bottle.py --use-gui --file-name='bottle_task_rim.gzip' --pick-type='rim' --dont-save
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Oracle policy test for mug pick and place test')

    parser.add_argument('--seeds', type=int, default=[0,1,2,3,4,5,6,7,8,9], nargs='+',
                        help='List of seeds for generating oracle demonstration samples')
    parser.add_argument('--cup-pose', type=str, default='upright',
                        help='Cup pose for generating oracle demonstration samples, either \'upright\' or \'lying\'.')
    parser.add_argument('--use-gui', action='store_true',
                        help='Use gui if flagged')
    parser.add_argument('--task-config-dir', type=str, default='config/task_config/mug_task.yaml',
                        help='Path to config file for tasks')
    parser.add_argument('--folder-name', type=str, default='demo/',
                        help='Folder name to save demo files. Default: ./demo/')
    parser.add_argument('--file-name', type=str, default='mug_task.gzip',
                        help='Demo file name')
    parser.add_argument('--pick-type', type=str, default='handle',
                        help='How to pick the mug. Either by picking the \'handle\' or the \'rim\'.')
    parser.add_argument('--low-var', action='store_true',
                        help='')
    parser.add_argument('--dont-save', action='store_true', help='')
    args = parser.parse_args()

    seeds = args.seeds
    cup_pose = args.cup_pose
    use_gui = args.use_gui
    task_config_dir = args.task_config_dir
    folder_name = args.folder_name
    file_name = args.file_name
    pick_type = args.pick_type
    low_var = args.low_var
    save = not args.dont_save

    mixed_pick_type = False
    if pick_type == 'mixed':
        mixed_pick_type = True

    with open(task_config_dir) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    d = config['d']
    d_pick = config['d_pick']
    d_place = config['d_place']

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
 
    path = folder_name + file_name
    task = BottleTask(use_gui=use_gui)

    # Generate demo samples
    samples = []
    for seed in seeds:
        if cup_pose == 'arbitrary':
            task.reset(seed = seed, mug_pose='upright' if seed%2==0 else 'lying', mug_type='train', distractor=False, use_support=False)
        else:
            task.reset(seed = seed, mug_pose=cup_pose, mug_type='train', distractor=False, use_support=False)

        if mixed_pick_type:
            if seed % 2 == 0:
                pick_type = 'handle'
            else:
                pick_type = 'rim'

        # Pick Demo
        pc = task.observe_pointcloud(stride=(1,1))

        max_iter = 100
        for iter_ in range(max_iter):
            try:
                if pick_type == 'handle':
                    pre_grasp, grasp = task.oracle_pick_handle(random_180_flip=False, force_flip = True if cup_pose == 'arbitrary' and iter_ %2 ==0 else False)
                    task.pick(pre_grasp=pre_grasp, grasp=grasp)
                elif pick_type == 'rim':
                    if low_var:
                        mod = 0
                    else:
                        mod = np.random.rand() < 0.5
                    pre_grasp, grasp = task.oracle_pick_rim(mod = mod)
                    task.pick(pre_grasp=pre_grasp, grasp=grasp)
                else:
                    raise ValueError("wrong pick_type")
                break
            except StopIteration:
                continue
        if iter_ == max_iter - 1:
            raise TimeoutError('couldn\'t find IK solution')

        sample = {}
        # sample['coord'], sample['color'] = voxel_filter(pc['coord'], pc['color'], d=d)
        sample['coord'], sample['color'] = pc['coord'], pc['color']
        sample['range'] = pc['ranges']
        sample['pre_grasp'] = pre_grasp 
        sample['grasp'] = grasp
        sample['images'] = task.observe()
        sample['center'] = task.center
        sample['d'] = d

        # Orthographic projection for baseline experiment (Transporter Networks)
        H = W = 360
        ortho_ranges = np.array([[0.4, 0.8],[-0.2, 0.2]])
        ortho_transform = OrthoTransform(W = W, ranges = ortho_ranges)
        sample['ortho_transform'] = ortho_transform
        sample['ortho_img'] = ortho_transform.orthographic(pc['coord'], pc['color'])
        sample['ortho_label_pick'] = ortho_transform.pose2pix_yaw_zrp(grasp) # grasp_pix, yaw, height, roll, pitch 

        # Place Demo
        #task.pick(pre_grasp, grasp, sleep=False)
        task.retract_robot(gripper_val=1., back=True)
        pc = task.observe_pointcloud_pick(stride=(1,1))
        # sample['coord_pick'], sample['color_pick'] = voxel_filter(pc['coord'], pc['color'], d=d_pick)
        sample['coord_pick'], sample['color_pick'] = pc['coord'], pc['color']
        sample['range_pick'] = pc['ranges']
        sample['pick_pose'] = (pc['X_sg'], pc['R_sg'])
        sample['images_pick'] = task.observe_pick()
        sample['d_pick'] = d_pick
        
        pc = task.observe_pointcloud(stride=(1,1))
        # sample['coord_place'], sample['color_place'] = voxel_filter(pc['coord'], pc['color'], d=d_place)
        sample['coord_place'], sample['color_place'] = pc['coord'], pc['color']
        sample['range_place'] = pc['ranges']
        sample['images_place'] = task.observe()
        sample['d_place'] = d_place

        if mixed_pick_type is True or pick_type == 'handle':
            max_iter = 100
            for iter_ in range(max_iter):
                if pick_type == 'handle':
                    mod = np.random.rand() < 0.5
                try:
                    pre_grasp, grasp = task.oracle_place_handle_horizontal(mod = mod, low_var=low_var)
                    task.place(pre_grasp, grasp, sleep=False, max_distance_plan=(0.07, 1.5))
                    break
                except StopIteration:
                    pass
            if iter_ == max_iter - 1:
                raise TimeoutError('couldn\'t find IK solution')

        # elif pick_type == 'handle':
        #     try:
        #         pre_grasp, grasp = task.oracle_place_handle()
        #         task.place(pre_grasp, grasp, sleep=False)
        #     except StopIteration:
        #         pre_grasp, grasp = task.oracle_place_handle(flip_x=True)
        #         task.place(pre_grasp, grasp, sleep=False)

        elif pick_type == 'rim':
            max_iter = 100
            for iter_ in range(max_iter):
                try:
                    pre_grasp, grasp = task.oracle_place_handle_horizontal(mod = mod, low_var=low_var)
                    task.place(pre_grasp, grasp, sleep=False)
                    break
                except StopIteration:
                    pass
            if iter_ == max_iter - 1:
                raise TimeoutError('couldn\'t find IK solution')

        sample['pre_place'] = pre_grasp 
        sample['place'] = grasp

        # Orthographic projection for baseline experiment (Transporter Networks)
        sample['ortho_label_place'] = ortho_transform.pose2pix_yaw_zrp(grasp) # grasp_pix, yaw, height, roll, pitch 

        samples.append(sample)
    task.close()

    # Save demo samples
    if save:
        import gzip
        import pickle

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        with gzip.open(path, 'wb') as f:
            pickle.dump(samples, f)














