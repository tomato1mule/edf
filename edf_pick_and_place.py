import time
from typing import Union, Optional, Dict, List, Tuple, Any

from ros_edf.ros_interface import EdfRosInterface
from ros_edf.pc_utils import pcd_from_numpy, draw_geometry, reconstruct_surface
from edf.data import PointCloud, SE3, TargetPoseDemo, DemoSequence, save_demos
from edf.pc_utils import check_pcd_collision, optimize_pcd_collision
from edf.env_interface import PLAN_FAIL, EXECUTION_FAIL, SUCCESS, RESET, FEASIBLE, INFEASIBLE

from edf.data import PointCloud, SE3, TargetPoseDemo, DemoSequence, DemoSeqDataset, save_demos
from edf.pc_utils import optimize_pcd_collision, draw_geometry, check_pcd_collision
from edf.preprocess import Rescale, NormalizeColor, Downsample, ApplySE3
from edf.agent import PickAgent, PlaceAgent

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

torch.set_printoptions(precision= 3, sci_mode=False, linewidth=120)





save_demo = False
n_episodes = 10
count_reset_episodes = False




device = 'cuda:0'
# device = 'cpu'
unit_len = 0.01
##### Initialize Pick Agent #####
pick_agent_config_dir = "config/agent_config/pick_agent.yaml"
pick_agent_param_dir = "checkpoint/mug_10_demo/pick/model_iter_600.pt"
max_N_query_pick = 1
langevin_dt_pick = 0.001

pick_agent = PickAgent(config_dir=pick_agent_config_dir, 
                       device = device,
                       max_N_query = max_N_query_pick, 
                       langevin_dt = langevin_dt_pick).requires_grad_(False)

pick_agent.load(pick_agent_param_dir)
pick_agent.warmup(warmup_iters=10, N_poses=100, N_points_scene=2000)



##### Initialize Place Agent #####
place_agent_config_dir = "config/agent_config/place_agent.yaml"
place_agent_param_dir = "checkpoint/mug_10_demo/place/model_iter_600.pt"
max_N_query_place = 3
langevin_dt_place = 0.001

place_agent = PlaceAgent(config_dir=place_agent_config_dir, 
                         device = device,
                         max_N_query = max_N_query_place, 
                         langevin_dt = langevin_dt_place).requires_grad_(False)

place_agent.load(place_agent_param_dir, strict=False)
place_agent.warmup(warmup_iters=10, N_poses=100, N_points_scene=1500, N_points_grasp=900)



##### Initialize Preprocessing functions #####
scene_proc_fn = Compose([Rescale(rescale_factor=1/unit_len),
                         Downsample(voxel_size=1.7, coord_reduction="average"),
                         NormalizeColor(color_mean = torch.tensor([0.5, 0.5, 0.5]), color_std = torch.tensor([0.5, 0.5, 0.5]))])
grasp_proc_fn = Compose([
                         Rescale(rescale_factor=1/unit_len),
                         Downsample(voxel_size=1.4, coord_reduction="average"),
                         NormalizeColor(color_mean = torch.tensor([0.5, 0.5, 0.5]), color_std = torch.tensor([0.5, 0.5, 0.5]))])
recover_scale = Rescale(rescale_factor=unit_len)







###### Define Primitives ######
pick_checklist = [('collision_check', {'colcheck_r': 0.003})]
place_checklist = [('collision_check', {'colcheck_r': 0.0015})]


def get_pick(scene: PointCloud, grasp: PointCloud) -> Union[str, SE3]:
    ##### Preprocess Observations #####
    scene_proc = scene_proc_fn(scene).to(device)
    grasp_proc = grasp_proc_fn(grasp).to(device)

    ##### Sample Pick Poses #####
    T_seed = 100
    pick_policy = 'sorted'
    pick_mh_iter = 1000
    pick_langevin_iter = 300
    pick_dist_temp = 1.
    pick_policy_temp = 1.
    pick_optim_iter = 100
    pick_optim_lr = 0.005

    Ts, edf_outputs, logs = pick_agent.forward(scene=scene_proc, T_seed=T_seed, policy = pick_policy, mh_iter=pick_mh_iter, langevin_iter=pick_langevin_iter, 
                                                temperature=pick_dist_temp, policy_temperature=pick_policy_temp, optim_iter=pick_optim_iter, optim_lr=pick_optim_lr)

    pick_poses = recover_scale(SE3(Ts.cpu()))
    # T_eg = SE3([ 0.707,  0.000,  0.000,  0.707,  0.000,  0.000, 0.150])
    # pick_poses = SE3.multiply(pick_poses, T_eg.inv())

    return pick_poses


def get_place(scene: PointCloud, grasp: PointCloud) -> Union[str, SE3]:
    scene_proc = scene_proc_fn(scene).to(device)
    grasp_proc = grasp_proc_fn(grasp).to(device)

    T_seed = 100
    place_policy = 'sorted'
    place_mh_iter = 1000
    place_langevin_iter = 300
    place_dist_temp = 1.
    place_policy_temp = 1.
    place_optim_iter = 100
    place_optim_lr = 0.005
    place_query_temp = 1.

    Ts, edf_outputs, logs = place_agent.forward(scene=scene_proc, T_seed=T_seed, grasp=grasp_proc, policy = place_policy, mh_iter=place_mh_iter, langevin_iter=place_langevin_iter, 
                                                temperature=place_dist_temp, policy_temperature=place_policy_temp, optim_iter=place_optim_iter, optim_lr=place_optim_lr, query_temperature=place_query_temp)

    place_poses = recover_scale(SE3(Ts.cpu()))

    return place_poses


def update_system_msg(msg: str, wait_sec: float = 0.):
    print(msg)
    if wait_sec:
        time.sleep(wait_sec)

def cleanup():
    pass












def move_robot_near_target(pose: SE3, env_interface: EdfRosInterface):
    assert len(pose) == 1

    rel_pos = torch.tensor([-0.7, 0.], device=pose.device, dtype=pose.poses.dtype)
    pos = pose.poses[0,4:6] + rel_pos
    if pos[0] > -0.6:
        pos[0] = -0.6

    env_interface.move_robot_base(pos=pos) # x,y

def check_collision(pose: SE3, 
                    scene: PointCloud, 
                    grasp: PointCloud, 
                    colcheck_r: float # Should be similar to voxel filter size
                    ) -> bool:
    assert len(pose) == 1

    col_check = check_pcd_collision(x=scene, y=grasp.transformed(pose)[0], r = colcheck_r)

    return col_check

def feasibility_check(context: Dict[str, Any], check_list: List[Tuple[str, Dict[str, Any]]]) -> Tuple[str, str]:
    available_check_types = ['collision_check']
    
    feasibility, msg = FEASIBLE, 'FEASIBLE'
    for check in check_list:
        check_type, check_kwarg = check
        assert check_type in available_check_types

        if check_type == 'collision_check':
            col_check = check_collision(pose=context['pose'], 
                                        scene=context['scene'], 
                                        grasp=context['grasp'], **check_kwarg)
            if col_check:
                return INFEASIBLE, 'COLLISION_DETECTED'
            
    return feasibility, msg

def get_pre_post_pick(scene: PointCloud, grasp: PointCloud, pick_poses: SE3) -> Tuple[SE3, SE3]:
    # _, pre_pick_poses = optimize_pcd_collision(x=scene, y=grasp, 
    #                                             cutoff_r = 0.03, dt=0.01, eps=1., iters=50,
    #                                             rel_pose=pick_poses)
    pre_pick_poses = pick_poses * SE3(torch.tensor([1., 0., 0., 0., 0., 0., -0.05], device=pick_poses.device))
    #post_pick_poses = pre_pick_poses
    post_pick_poses = SE3(pick_poses.poses + torch.tensor([0., 0., 0., 0., 0., 0., 0.1], device=pick_poses.device))

    return pre_pick_poses, post_pick_poses


def get_pre_post_place(scene: PointCloud, grasp: PointCloud, place_poses: SE3, pre_pick_pose: SE3, pick_pose: SE3) -> Tuple[SE3, SE3]:
    assert len(pick_pose) == len(pre_pick_pose) == 1

    _, pre_place_poses = optimize_pcd_collision(x=scene, y=grasp, 
                                                cutoff_r = 0.03, dt=0.01, eps=1., iters=5,
                                                rel_pose=place_poses)
    post_place_poses = place_poses * pick_pose.inv() * pre_pick_pose

    return pre_place_poses, post_place_poses


def observe(env_interface, max_try: int, attach: bool) -> bool:
    success = True
    update_system_msg("Move to Observe...")


    # Move to default pose before moving to observation pose
    for _ in range(max_try):
        move_result, _info = env_interface.move_to_named_target("init")
        if move_result == SUCCESS:
            break
        else:
            continue

    # Move to observation pose
    if move_result == SUCCESS:
        env_interface.move_robot_base(pos = torch.tensor([-1.5, 0.]))
        for _ in range(max_try):
            move_result, _info = env_interface.move_to_named_target("observe")
            if move_result == SUCCESS:
                break
            else:
                continue
    
    # Observe
    if move_result == SUCCESS:
        if attach:
            env_interface.detach()
        grasp_raw = env_interface.observe_eef(obs_type = 'pointcloud', update = True)
        if attach:
            env_interface.attach(obj = grasp_raw)
        scene_raw = env_interface.observe_scene(obs_type = 'pointcloud', update = True)


    # Come back to default pose
    if move_result == SUCCESS:
        for _ in range(max_try):
            move_result, _info = env_interface.move_to_named_target("init")
            if move_result == SUCCESS:
                break
            else:
                continue
    if move_result == SUCCESS:
        env_interface.move_robot_base(pos = torch.tensor([-0.7, 0.0]))
    
    if move_result != SUCCESS:
        update_system_msg(f"Cannot Move to Observation Pose ({move_result}). Resetting env...", wait_sec=2.0)
        success = False
        
    return success, (scene_raw, grasp_raw)









env_interface = EdfRosInterface(reference_frame = "scene")
env_interface.moveit_interface.arm_group.set_planning_time(seconds=1)
env_interface.moveit_interface.arm_group.allow_replanning(True)

demo_list = []
episode_count = 0
reset_signal = False
while True:
    if episode_count >= n_episodes:
        break
    ###### Reset Env ######
    update_system_msg('Resetting Environment...')
    env_interface.reset()
    if reset_signal and not count_reset_episodes:
        pass
    else:
        episode_count += 1
    reset_signal = False

    ###### Observe ######
    success, (scene_raw, grasp_raw) = observe(env_interface=env_interface, max_try = 10, attach=False)
    if not success:
        reset_signal = True
        continue


    ###### Sample Pick Pose ######
    pick_max_try = 100000
    for n_trial in range(pick_max_try):
        ###### Infer pick poses ######
        if n_trial == 0:
            update_system_msg('Waiting for pick poses...')
        pick_inference_result = get_pick(scene=scene_raw, grasp=grasp_raw)
        
        ###### Infer pre-pick and post-pick poses ######
        if isinstance(pick_inference_result, SE3):
            update_system_msg('Looking for feasible pick poses...')
            pick_poses: SE3 = pick_inference_result
            pre_pick_poses, post_pick_poses = get_pre_post_pick(scene=scene_raw, grasp=grasp_raw, pick_poses=pick_poses)

            ###### Check Feasiblity ######
            for idx in range(len(pick_poses)):
                pick_pose, pre_pick_pose, post_pick_pose = pick_poses[idx], pre_pick_poses[idx], post_pick_poses[idx]
                context = {'pose': pick_pose, 'scene': scene_raw, 'grasp': grasp_raw}
                feasibility, _info = feasibility_check(context=context, check_list=pick_checklist)
                if feasibility == FEASIBLE:
                    move_robot_near_target(pose=pick_pose, env_interface=env_interface)
                    pick_plan_result, pick_plans = env_interface.pick_plan(pre_pick_pose=pre_pick_pose, pick_pose=pick_pose)
                    if pick_plan_result == SUCCESS:
                        break
                    else:
                        _info = pick_plans
                        feasibility = INFEASIBLE
                        continue
                else:
                    continue
                     
            if feasibility == FEASIBLE:
                update_system_msg("Found feasible pick-pose! Executing")
                break
            else:
                if len(pick_pose) == 1:
                    update_system_msg(f"No feasible pick-pose found. Try again! (Reason: {_info})")
                else:
                    update_system_msg("No feasible pick-pose found. Try again!")
                continue
        ###### Reset Signal ######
        elif pick_inference_result == RESET:
            reset_signal = True
            break
        else:
            raise NotImplementedError(f"Unknown pick_inference_result: {pick_inference_result}")
        
    if reset_signal:
        continue
    elif n_trial == pick_max_try - 1:
        reset_signal = True
        continue
    else:
        pass

    ###### Execute Pick ######
    pick_result, _info = env_interface.pick_execute(plans=pick_plans, post_pick_pose=post_pick_pose)
    if pick_result == SUCCESS:
        update_system_msg(f"Moving to pick pose result: {pick_result}")
        pick_demo = TargetPoseDemo(target_poses=pick_poses, scene_pc=scene_raw, grasp_pc=grasp_raw)
        env_interface.detach()
        env_interface.attach_placeholder(size=0.15) # To avoid collsion with the grasped object
    else:
        update_system_msg(f"Moving to pick pose result: {pick_result}, Resetting env...", wait_sec=2.0)
        reset_signal = True
        continue

    

    ###### Observe for Place ######
    success, (scene_raw, grasp_raw) = observe(env_interface=env_interface, max_try = 10, attach=True)
    if not success:
        reset_signal = True
        continue


    ###### Sample Place Pose ######
    place_max_try = 100000
    for n_trial in range(place_max_try):
        ###### Infer place poses ######
        if n_trial == 0:
            update_system_msg('Waiting for place poses...')
        place_inference_result = get_place(scene=scene_raw, grasp=grasp_raw)
        
        ###### Infer pre-place and post-place poses ######
        if isinstance(place_inference_result, SE3):
            update_system_msg('Looking for feasible place poses...')
            place_poses: SE3 = place_inference_result
            pre_place_poses, post_place_poses = get_pre_post_place(scene=scene_raw, grasp=grasp_raw, place_poses=place_poses, pre_pick_pose=pre_pick_pose, pick_pose=pick_pose)

            ###### Check Feasiblity ######
            for idx in range(len(place_poses)):
                place_pose, pre_place_pose, post_place_pose = place_poses[idx], pre_place_poses[idx], post_place_poses[idx]
                context = {'pose': place_pose, 'scene': scene_raw, 'grasp': grasp_raw}
                feasibility, _info = feasibility_check(context=context, check_list=place_checklist)
                if feasibility == FEASIBLE:
                    move_robot_near_target(pose=place_pose, env_interface=env_interface)
                    place_plan_result, place_plans = env_interface.place_plan(pre_place_pose=pre_place_pose, place_pose=place_pose)
                    if place_plan_result == SUCCESS:
                        break
                    else:
                        _info = place_plans
                        feasibility = INFEASIBLE
                        continue
                else:
                    continue
                     
            if feasibility == FEASIBLE:
                update_system_msg("Found feasible place-pose! Executing")
                break
            else:
                if len(place_pose) == 1:
                    update_system_msg(f"No feasible place-pose found. Try again! (Reason: {_info})")
                else:
                    update_system_msg("No feasible place-pose found. Try again!")
                continue
        ###### Reset Signal ######
        elif place_inference_result == RESET:
            reset_signal = True
            break
        else:
            raise NotImplementedError(f"Unknown place_inference_result: {place_inference_result}")
        
    if reset_signal:
        reset_signal = True
        continue
    elif n_trial == place_max_try - 1:
        reset_signal = True
        continue
    else:
        pass

    ###### Execute place ######
    place_result, _info = env_interface.place_execute(plans=place_plans, post_place_pose=post_place_pose)
    if place_result == SUCCESS:
        update_system_msg(f"Moving to place pose result: {place_result}")
        place_demo = TargetPoseDemo(target_poses=place_poses, scene_pc=scene_raw, grasp_pc=grasp_raw)
        env_interface.detach()
        env_interface.release()
    else:
        update_system_msg(f"Moving to place pose move result: {place_result}, Resetting env...", wait_sec=2.0)
        reset_signal = True
        continue


    demo_seq = DemoSequence(demo_seq = [pick_demo, place_demo])
    demo_list.append(demo_seq)