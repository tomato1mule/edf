import time
import os
from itertools import islice

import numpy as np
import pybullet as p
import pybullet_data

from scipy.spatial.transform import Rotation
from pybullet_tools.utils import add_data_path, connect, dump_body, disconnect, wait_for_user, \
    get_movable_joints, get_sample_fn, set_joint_positions, get_joint_name, LockRenderer, link_from_name, get_link_pose, \
    multiply, multiply_quats, Pose, Point, interpolate_poses, HideOutput, draw_pose, set_camera_pose, load_pybullet, \
    assign_link_colors, add_line, point_from_pose, remove_handles, BLUE, INF, set_client, stable_z, create_box, set_point, reset_simulation, \
    add_fixed_constraint, remove_fixed_constraint, set_numpy_seed, set_random_seed, compute_jacobian, get_joint_positions, get_links, get_joints, get_length
from pybullet_tools.ikfast.ikfast import get_ik_joints, either_inverse_kinematics, check_ik_solver, ikfast_inverse_kinematics, get_difference_fn

from edf.pybullet_env.utils import get_image, axiscreator, img_data_to_pointcloud

def observe(cam_configs, physicsClientId = 0, **kwargs):
    outputs = []
    for n, config in enumerate(cam_configs):
        config = config.copy()
        if 'target_pos' in kwargs.keys():
            config['target_pos'] = kwargs['target_pos']
        data = get_image(config, physicsClientId = physicsClientId)
        W, H, image, depth = data['W'], data['H'], data['color'], data['depth']
        if 'R_sg' in kwargs.keys():
            data['R_sg'] = kwargs['R_sg']
        if 'X_sg' in kwargs.keys():
            data['X_sg'] = kwargs['X_sg']
        outputs.append(data)

    return outputs

class BulletTask():
    def __init__(self, use_gui = True):
        self.physicsClientId = connect(use_gui=use_gui)
        add_data_path()
        self.init_task()

    def init_task(self):
        draw_pose(Pose(), length=1.)
        p.setGravity(0,0,-10, physicsClientId = self.physicsClientId)
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId = self.physicsClientId)
        self.freq = 1/1000
        p.setTimeStep(self.freq)

    def reset(self, seed = None):
        raise NotImplementedError

    def close(self):
        disconnect()

    def observe(self):
        return observe(self.cam_configs, physicsClientId=self.physicsClientId)

    def observe_pointcloud(self, stride=(1,1)):   # sample_rate = (sr_Row, sr_Column)
        data = self.observe()
        return img_data_to_pointcloud(data = data, xlim = self.xlim, ylim = self.ylim, zlim = self.zlim, stride = stride)











































































from pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO, FRANKA_URDF
FRANKA_URDF = 'pybullet-planning/' + FRANKA_URDF


class FrankaTask(BulletTask):
    def __init__(self, use_gui = True):
        self.center = [0.6, 0., 0.15]
        camera_0_config = {'target_pos': self.center,
                   'distance': 0.5,
                   'ypr': (90, -70, 0),
                   'W': 480,
                   'H': 360,
                   'up': [0,0,1],
                   'up_axis_idx': 2,
                   'near': 0.01,
                   'far' : 100,
                   'fov' : 60 
                   }
        camera_1_config = {'target_pos': self.center,
                        'distance': 0.5,
                        'ypr': (90+135, -70, 0),
                        'W': 480,
                        'H': 360,
                        'up': [0,0,1],
                        'up_axis_idx': 2,
                        'near': 0.01,
                        'far' : 100,
                        'fov' : 60 
                        }
        camera_2_config = {'target_pos': self.center,
                        'distance': 0.5,
                        'ypr': (90+135+90, -70, 0),
                        'W': 480,
                        'H': 360,
                        'up': [0,0,1],
                        'up_axis_idx': 2,
                        'near': 0.01,
                        'far' : 100,
                        'fov' : 60 
                        }
        self.center = np.array(self.center)

        self.cam_configs = [camera_0_config, camera_1_config, camera_2_config]

        self.xlim = np.array((-0.3, 0.3)) + self.center[0]
        self.ylim = np.array((-0.3, 0.3)) + self.center[1]
        self.zlim = np.array((-0.1,0.5)) + self.center[2]

        pickcam_1_config = {'target_pos': None,
                            'distance': 0.3,
                            'ypr': (90, 40, 0),
                            'W': 480,
                            'H': 360,
                            'up': [0,0,1],
                            'up_axis_idx': 2,
                            'near': 0.01,
                            'far' : 100,
                            'fov' : 60 
                            }
        pickcam_2_config = {'target_pos': None,
                    'distance': 0.3,
                    'ypr': (90+135, 40, 0),
                    'W': 480,
                    'H': 360,
                    'up': [0,0,1],
                    'up_axis_idx': 2,
                    'near': 0.01,
                    'far' : 100,
                    'fov' : 60 
                    }
        pickcam_3_config = {'target_pos': None,
                    'distance': 0.3,
                    'ypr': (90+135+90, 40, 0),
                    'W': 480,
                    'H': 360,
                    'up': [0,0,1],
                    'up_axis_idx': 2,
                    'near': 0.01,
                    'far' : 100,
                    'fov' : 60 
                    }
        pickcam_4_config = {'target_pos': None,
                            'distance': 0.3,
                            'ypr': (90, -40, 0),
                            'W': 480,
                            'H': 360,
                            'up': [0,0,1],
                            'up_axis_idx': 2,
                            'near': 0.01,
                            'far' : 100,
                            'fov' : 60 
                            }
        pickcam_5_config = {'target_pos': None,
                    'distance': 0.3,
                    'ypr': (90+135, -40, 0),
                    'W': 480,
                    'H': 360,
                    'up': [0,0,1],
                    'up_axis_idx': 2,
                    'near': 0.01,
                    'far' : 100,
                    'fov' : 60 
                    }
        pickcam_6_config = {'target_pos': None,
                    'distance': 0.3,
                    'ypr': (90+135+90, -40, 0),
                    'W': 480,
                    'H': 360,
                    'up': [0,0,1],
                    'up_axis_idx': 2,
                    'near': 0.01,
                    'far' : 100,
                    'fov' : 60 
                    }

        self.pickcam_configs =  [pickcam_1_config, pickcam_2_config, pickcam_3_config] + [pickcam_4_config, pickcam_5_config, pickcam_6_config]
        # self.xlim_pick = np.array([-0.15, 0.15]) 
        # self.ylim_pick = np.array([-0.15, 0.15]) 
        # self.zlim_pick = np.array([-0.15, 0.15])
        self.xlim_pick = np.array([-0.20, 0.20]) 
        self.ylim_pick = np.array([-0.20, 0.20]) 
        self.zlim_pick = np.array([-0.20, 0.20])

        self.max_gripper_val = 0.025

        self.default_robot_conf = np.array([0.,    # 0
                                            -0.1873423921928338,       # 1
                                            0.0004878642201449068,     # 2
                                            -1.8468279976025963,       # 3
                                            9.125848660129776e-05,     # 4
                                            1.6594855718413641,        # 5
                                            0.7853235942872554,        # 6
                                            self.max_gripper_val,                       # 9: Finger1    0 ~ 0.04
                                            self.max_gripper_val])                      # 10: Finger2   0 ~ 0.04

        self.back_robot_conf = np.array([np.pi,    # 0
                                            -0.1873423921928338,       # 1
                                            0.0004878642201449068,     # 2
                                            -1.8468279976025963,       # 3
                                            9.125848660129776e-05,     # 4
                                            1.6594855718413641,        # 5
                                            0.7853235942872554,        # 6
                                            self.max_gripper_val,                       # 9: Finger1    0 ~ 0.04
                                            self.max_gripper_val])                      # 10: Finger2   0 ~ 0.04

        super().__init__(use_gui=use_gui)

    def spawn_robot(self):
        with HideOutput():
            self.robot = load_pybullet(FRANKA_URDF, fixed_base=True)
        #assign_link_colors(robot, max_colors=3, s=0.5, v=1.)
        #dump_body(self.robot)
        self.robot_info = PANDA_INFO
        self.EE = link_from_name(self.robot, 'panda_hand')
        #draw_pose(Pose(), parent=self.robot, parent_link=self.EE)
        self.joints = get_movable_joints(self.robot)
        self.ik_joints = get_ik_joints(self.robot, self.robot_info, self.EE)
        #print('Joints', [get_joint_name(self.robot, joint) for joint in self.joints])
        #check_ik_solver(self.robot_info)

        self.Lfinger_link = 9         
        self.Lfinger_joint = 9    
        self.Rfinger_link = 10        
        self.Rfinger_joint = 10        

        # EE to gripper sweet-spot
        self.X_eg = np.array([0, 0, 0.105]) 
        self.R_eg = np.eye(3)

    def reset_robot(self):
        #p.resetBasePositionAndOrientation(self.robot, posObj = np.array([0.1, 0., 0.]), ornObj = np.array([0., 0., 0., 1.]))
        set_joint_positions(self.robot, self.joints, self.default_robot_conf)

    def init_task(self, **kwargs):
        super().init_task()
        self.table_id = p.loadURDF("assets/table.urdf", basePosition=self.center + np.array([0.0, 0.0, -0.35]), baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]), globalScaling=0.4, physicsClientId = self.physicsClientId)
        self.spawn_robot()

        # p.changeDynamics(self.robot, 0, jointLowerLimit = -np.pi, jointUpperLimit = np.pi)
        # p.changeDynamics(self.robot, 1, jointLowerLimit = -np.pi, jointUpperLimit = np.pi)
        # p.changeDynamics(self.robot, 2, jointLowerLimit = -np.pi, jointUpperLimit = np.pi)
        # #p.changeDynamics(self.robot, 3, jointLowerLimit = -2.8, jointUpperLimit = -0.3)
        # p.changeDynamics(self.robot, 4, jointLowerLimit = -np.pi, jointUpperLimit = np.pi)
        # p.changeDynamics(self.robot, 6, jointLowerLimit = -np.pi, jointUpperLimit = np.pi)

    def reset(self, seed = None, step = 1, **kwargs):
        if seed is not None:
            #np.random.seed(seed)
            set_random_seed(seed)
            set_numpy_seed(seed)
        reset_simulation()
        self.init_task(**kwargs)

        self.grasp_constraint = None
        self.grasp_item = None
        self.target_item = None
        self.reset_robot()

        # step sim
        if step:
            step = int(step/self.freq)
            for _ in range(step):
                p.stepSimulation(physicsClientId = self.physicsClientId)

    def debug(self, debug_items = ['grasp']):
        p.removeAllUserDebugItems(physicsClientId = self.physicsClientId)
        for item in debug_items:
            if item == 'ee':
                eeFrame_ID = axiscreator(self.robot, self.EE, physicsClientId = self.physicsClientId)
            elif item == 'grasp':
                graspFrame_ID = axiscreator(self.robot, self.EE, offset = self.X_eg.copy(), physicsClientId = self.physicsClientId)
            elif item == 'mug':
                axiscreator(self.mug_id, physicsClientId = self.physicsClientId)
            elif item == 'handle':
                axiscreator(self.mug_id, offset = self.X_mh.copy(), physicsClientId = self.physicsClientId)


    def get_state(self):
        # Suffix guide
        # g: grasp, e: end-effector

        X_se, R_se = p.getLinkState(self.robot, self.EE, physicsClientId = self.physicsClientId)[:2]
        X_se, R_se = np.array(X_se), Rotation.from_quat(R_se).as_matrix()

        R_sg = R_se @ self.R_eg
        X_sg = X_se + (R_se @ self.X_eg)

        gripper_val_L = 1 - (p.getJointState(self.robot, self.Lfinger_joint, physicsClientId = self.physicsClientId)[0] / self.max_gripper_val)
        gripper_val_R = 1 - (p.getJointState(self.robot, self.Rfinger_joint, physicsClientId = self.physicsClientId)[0] / self.max_gripper_val)

        return {'T_se':(X_se, R_se), 'T_sg':(X_sg, R_sg), 'gripper_vals':(gripper_val_L, gripper_val_R)}

    def gripper_control_lr(self, val_L, val_R, force=100):
        gripper_pose_L = (1-val_L) * self.max_gripper_val
        gripper_pose_R = (1-val_R) * self.max_gripper_val
        p.setJointMotorControl2(self.robot, self.Lfinger_joint, p.POSITION_CONTROL, targetPosition=gripper_pose_L, force=force, physicsClientId = self.physicsClientId)
        p.setJointMotorControl2(self.robot, self.Rfinger_joint, p.POSITION_CONTROL, targetPosition=gripper_pose_R, force=force, physicsClientId = self.physicsClientId)

    def gripper_control(self, val, force=100):
        self.gripper_control_lr(val_L=val, val_R=val, force=force)
    
    def grasp_check(self, item):
        if (len(p.getContactPoints(self.robot, item, self.Lfinger_link, -1)) > 0) and (len(p.getContactPoints(self.robot, item, self.Rfinger_link, -1)) > 0):
            return True
        else:
            return False

    def attach(self, item):
        if self.grasp_check(item) is True:
            if self.grasp_constraint is not None:
                print("already in grasp")
            else:
                self.grasp_constraint = add_fixed_constraint(item, self.robot, robot_link=self.Lfinger_link)
                self.grasp_item = item

    def detach(self):
        if self.grasp_constraint is not None:
            remove_fixed_constraint(self.grasp_item, self.robot, robot_link=self.Lfinger_link)
            self.grasp_constraint = None
            self.grasp_item = None
        else:
            #print('not in grasp')
            pass

    def IK(self, duration, gripper_val, target_T_se, 
           sleep = 0., gripper_force = 300, joint_force = None, init_gripper_val = None, update_state = False):
        duration = int(duration/self.freq)

        target_X_se, target_R_se = target_T_se
        target_gripper_val = gripper_val
        state_dict = self.get_state()
        init_X_se, init_R_se = state_dict['T_se']
        init_q_se, target_q_se = Rotation.from_matrix(init_R_se).as_quat(), Rotation.from_matrix(target_R_se).as_quat()

        if init_gripper_val is not None:
            init_gval_L, init_gval_R = init_gripper_val, init_gripper_val
        else:
            init_gval_L, init_gval_R = state_dict['gripper_vals']

        for t in range(duration):
            if update_state:
                state_dict = self.get_state()
                init_X_se, init_R_se = state_dict['T_se']
                init_q_se = Rotation.from_matrix(init_R_se).as_quat()

            if update_state:
                p_ = 1/(duration-t)
                target_pos = target_X_se * p_ + init_X_se * (1-p_)
                target_orn = p.getQuaternionSlerp(init_q_se, target_q_se, p_)
            else:
                target_pos = (target_X_se - init_X_se) * t/duration   +   init_X_se
                target_orn = p.getQuaternionSlerp(init_q_se, target_q_se, t/duration)
            joint_poses = p.calculateInverseKinematics(self.robot, self.EE, target_pos, target_orn)
            for j in self.ik_joints:
                if joint_force is not None:
                    p.setJointMotorControl2(bodyIndex=self.robot, jointIndex=j, controlMode=p.POSITION_CONTROL, targetPosition=joint_poses[j], physicsClientId = self.physicsClientId, force = joint_force)
                else:
                    p.setJointMotorControl2(bodyIndex=self.robot, jointIndex=j, controlMode=p.POSITION_CONTROL, targetPosition=joint_poses[j], physicsClientId = self.physicsClientId)

            target_gval_L = (target_gripper_val - init_gval_L) * t/duration   +   init_gval_L
            target_gval_R = (target_gripper_val - init_gval_R) * t/duration   +   init_gval_R

            self.gripper_control_lr(val_L=target_gval_L, val_R=target_gval_R, force=gripper_force)
            p.stepSimulation(physicsClientId = self.physicsClientId)
            time.sleep(sleep)

    def teleport(self, configuration):
        if self.grasp_constraint is not None:
            state = self.get_state()
            X_se, R_se = state['T_se']
            X_s_obj, R_s_obj = p.getBasePositionAndOrientation(self.grasp_item)
            X_s_obj, R_s_obj = np.array(X_s_obj), np.array(R_s_obj)
            R_s_obj = Rotation.from_quat(R_s_obj).as_matrix()

            R_e_obj = R_se.T @ R_s_obj
            X_e_obj = R_se.T @ (X_s_obj - X_se)
        set_joint_positions(self.robot, self.ik_joints, configuration)
        if self.grasp_constraint is not None:
            state = self.get_state()
            X_se_new, R_se_new = state['T_se']
            R_enew_objnew, X_enew_objnew = R_e_obj, X_e_obj
            R_s_objnew = R_se_new @ R_enew_objnew
            X_s_objnew = X_se_new + (R_se_new @ X_enew_objnew)
            p.resetBasePositionAndOrientation(self.grasp_item, X_s_objnew, Rotation.from_matrix(R_s_objnew).as_quat(), physicsClientId = self.physicsClientId)

    def controllability(self, conf, min_eig=True, lin_scale = 0.02, ang_scale = np.pi):
        conf = conf + [0., 0.]
        lin, ang = compute_jacobian(self.robot, self.EE, positions=conf)
        lin, ang = np.array(lin)[:-2], np.array(ang)[:-2]

        if min_eig:
            return min(np.linalg.eigvals(lin.T @ lin).min() / lin_scale, np.linalg.eigvals(ang.T @ ang).min() / ang_scale)
        else:
            return np.linalg.eigvals(lin.T @ lin) / lin_scale, np.linalg.eigvals(ang.T @ ang) / ang_scale

    def IKFast_max_controllability(self, target_T_se, max_time=INF, max_candidates=INF, max_attempts=INF, max_distance=INF, verbose = True):
        if target_T_se[1].shape[-2:] == (3,3):
            target_T_se = (target_T_se[0], Rotation.from_matrix(target_T_se[1]).as_quat())
        start_time = time.time()
        generator = ikfast_inverse_kinematics(self.robot, self.robot_info, self.EE, target_T_se, norm=INF, max_attempts=max_attempts, max_time=max_time, max_distance=max_distance, )
        if max_candidates < INF:
            generator = islice(generator, max_candidates)
        solutions = list(generator)
        solutions = sorted(solutions, key=lambda q: -self.controllability(q))
        if verbose:
            if len(solutions) == 0:
                print('Identified {} IK solutions in {:.3f} seconds'.format(len(solutions), time.time() - start_time))
            else:
                eig_max = self.controllability(solutions[0])
                print('Identified {} IK solutions with best controllability of {:.3f} in {:.3f} seconds'.format(
                    len(solutions), eig_max, time.time() - start_time))
        return iter(solutions)

    def IKFast_closest(self, target_T_se, ref_conf, max_time=INF, max_candidates=INF, max_attempts=INF, max_distance=INF, verbose = True):
        max_distance_ = max_distance
        max_distance = INF
        if target_T_se[1].shape[-2:] == (3,3):
            target_T_se = (target_T_se[0], Rotation.from_matrix(target_T_se[1]).as_quat())
        start_time = time.time()
        generator = ikfast_inverse_kinematics(self.robot, self.robot_info, self.EE, target_T_se, norm=INF, max_attempts=max_attempts, max_time=max_time, max_distance=max_distance, )
        if max_candidates < INF:
            generator = islice(generator, max_candidates)
        solutions = list(generator)
        difference_fn = get_difference_fn(self.robot, self.ik_joints) # get_distance_fn
        solutions = sorted(solutions, key=lambda q: get_length(difference_fn(q, ref_conf), norm=INF))

        min_distance = min([INF] + [get_length(difference_fn(q, ref_conf), norm=INF) for q in solutions])
        if min_distance > max_distance_:
            solutions = []
            if verbose:
                print(f'Identified No IK solution within distance {max_distance_} < {min_distance}')
        else:
            if verbose:
                print('Identified {} IK solutions with minimum distance of {:.3f} in {:.3f} seconds'.format(len(solutions), min_distance, time.time() - start_time))
        return iter(solutions)

    def IKFast_(self, target_T_se, IK_time = 1., verbose = True, criteria = 'controllability', ref_conf = None, max_distance = INF):
        max_attempts=int(15000*IK_time)
        max_candidates = 5000
        if criteria == 'controllability':
            sol = self.IKFast_max_controllability(target_T_se=target_T_se, max_attempts=max_attempts, max_candidates=max_candidates, verbose=verbose)
        elif criteria == 'closest':
            if ref_conf is None:
                if target_T_se[1].shape[-2:] == (3,3):
                    target_T_se = (target_T_se[0], Rotation.from_matrix(target_T_se[1]).as_quat())
                sol = either_inverse_kinematics(self.robot, self.robot_info, self.EE, target_T_se, use_pybullet=False,
                                                    max_time=INF, max_candidates=max_candidates, max_attempts=max_attempts, max_distance=max_distance)
            else:
                sol = self.IKFast_closest(target_T_se=target_T_se, max_attempts=max_attempts, max_candidates=max_candidates, verbose=verbose, ref_conf=ref_conf, max_distance=max_distance)

        return sol

    def IK_teleport(self, target_T_se, gripper_val, IK_time = 1., gripper_force = 5., verbose = True, criteria = 'controllability', ref_conf = None, max_distance = INF):
        sol = self.IKFast_(target_T_se, IK_time = IK_time, verbose = verbose, criteria=criteria, ref_conf=ref_conf, max_distance=max_distance)
        conf = next(sol)

        self.teleport(configuration=conf)
        self.gripper_control(gripper_val, gripper_force)

        return conf

    def robot_col_check(self, obj_ids):
        state_id = p.saveState(self.physicsClientId)
        
        p.stepSimulation(self.physicsClientId)
        for obj_id in obj_ids:
            for link in get_links(self.robot):
                if len(p.getContactPoints(self.robot, obj_id, link, -1)):
                    p.restoreState(stateId=state_id)
                    print("Collision Found!!")
                    return True

        p.restoreState(stateId=state_id)
        return False

    def check_feasible(self, pose, gripper_val, col_check_items = [], IK_time = 1., verbose = True, criteria='controllability', ref_conf = None, return_conf = False, max_distance=INF):
        X, R = pose
        z_axis = R[:,2]
        if z_axis[2] > 0.3:
            return False
        else:
            sol = self.IKFast_(pose, IK_time=IK_time, verbose=verbose, criteria=criteria, ref_conf = ref_conf, max_distance=max_distance)
            conf = next(sol)
            if col_check_items:
                self.IK_teleport(target_T_se=pose, gripper_val=gripper_val, IK_time = IK_time, criteria=criteria)
                if self.robot_col_check(obj_ids=col_check_items):
                    return False
        if return_conf is True:
            return conf
        else:
            return True

    def check_feasible_plan(self, plan, IK_time = 1., verbose = True, max_distance=INF):
        item = plan[-1]
        pose = item['pose']
        gripper_val = item['gripper_val']
        col_check_items = item['col_check_items']
        conf = self.check_feasible(pose, gripper_val=gripper_val, col_check_items=col_check_items, IK_time = IK_time, verbose = verbose, criteria='controllability', return_conf=True, max_distance=INF)
        if conf is False:
            return False

        if len(plan) > 1:
            for item in plan[-2::-1]:
                pose = item['pose']
                gripper_val = item['gripper_val']
                col_check_items = item['col_check_items']
                if not self.check_feasible(pose, gripper_val=gripper_val, col_check_items=col_check_items, IK_time = IK_time, verbose = verbose, criteria='closest', ref_conf=conf, max_distance=max_distance):
                    return False

        print("Found plan!!")
        return conf

    def pick_plan(self, grasp):
        X_sdg, R_sdg = grasp

        R_dg_dgpre = np.eye(3)
        R_sdg_pre = R_sdg @ R_dg_dgpre
        X_dg_dgpre = np.array([0., 0., -0.03])
        sX_dg_dgpre = R_sdg @ X_dg_dgpre
        X_sdg_pre = X_sdg + sX_dg_dgpre

        R_sde_pre = R_sdg_pre @ self.R_eg.T
        R_sde = R_sdg @ self.R_eg.T
        X_sde_pre = X_sdg_pre - R_sde_pre@self.X_eg
        X_sde = X_sdg - R_sde@self.X_eg


        pre_grasp = {'pose':(X_sde_pre, R_sde_pre), 'gripper_val': 0., 'col_check_items': [self.table_id, self.target_item]}        
        grasp1 = {'pose':(X_sde, R_sde), 'gripper_val': 0., 'col_check_items': [self.table_id, self.target_item]}                   
        grasp2 = {'pose':(X_sde, R_sde), 'gripper_val': 1., 'col_check_items': []}
        lift = {'pose':(X_sde + np.array([0., 0., 0.2]), R_sde), 'gripper_val': 1., 'col_check_items': []}

        return {'pre_grasp': pre_grasp, 'grasp1':grasp1, 'grasp2':grasp2, 'lift':lift}

    def place_plan(self, place, z_offset = 0.03):
        X_sdg, R_sdg = place

        R_sdg_pre = R_sdg
        X_sdg_pre = X_sdg + np.array([0., 0., z_offset])

        R_sde_pre = R_sdg_pre @ self.R_eg.T
        R_sde = R_sdg @ self.R_eg.T
        X_sde_pre = X_sdg_pre - R_sde_pre@self.X_eg
        X_sde = X_sdg - R_sde@self.X_eg


        pre_place = {'pose':(X_sde_pre, R_sde_pre), 'gripper_val': 1., 'col_check_items': []}
        place = {'pose':(X_sde, R_sde), 'gripper_val': 1., 'col_check_items': []}
        release = {'pose':(X_sde, R_sde), 'gripper_val': 0.5, 'col_check_items': []}


        z_axis = R_sde[:,-1]
        theta = np.arccos(-z_axis[-1])
        rot = np.cross(z_axis, np.array([0,0,-1]))
        if np.linalg.norm(rot) < 0.1:
            dR = np.eye(3)
        else:
            rot = rot/np.linalg.norm(rot)
            dR = Rotation.from_rotvec(rot * theta/2).as_matrix()
        R_sde_fin = dR @ R_sde
        X_sde_fin = X_sde + np.array([0., 0., z_offset])
        retract = {'pose':(X_sde_fin, R_sde_fin), 'gripper_val': 0., 'col_check_items': []}

        return {'pre_place': pre_place, 'place':place, 'release':release, 'retract':retract}

    def pick(self, pre_grasp, grasp,
             sleep = False, reach_force = 50., IK_time = 1.):
        X_sdg, R_sdg = grasp
        target_handles = draw_pose(Pose(X_sdg, p.getEulerFromQuaternion(Rotation.from_matrix(R_sdg).as_quat())))

        plan = self.pick_plan(grasp)
        pre_grasp, grasp1, grasp2, lift = plan['pre_grasp'], plan['grasp1'], plan['grasp2'], plan['lift']

        max_distance = 0.1
        conf = self.check_feasible_plan([pre_grasp, grasp1], IK_time=IK_time, verbose=True, max_distance=max_distance)
        if conf is False:
            raise StopIteration
        # max_distance = 0.08
        # if self.check_feasible_plan([lift, grasp2], IK_time=IK_time, verbose=True) is False:
        #     raise StopIteration
        
        self.IK_teleport(target_T_se=pre_grasp['pose'], gripper_val=pre_grasp['gripper_val'], IK_time = IK_time, criteria='closest', ref_conf=conf, max_distance=max_distance)

        # if self.check_feasible(pose = (X_sde, R_sde), IK_time=IK_time, criteria='controllability') is False:
        #     raise StopIteration
        
        # # Reach Pregrasp Pose
        # self.IK_teleport(target_T_se=(X_sde, R_sde), gripper_val=0., IK_time = IK_time, criteria='controllability')
        # if self.robot_col_check(obj_ids=[self.table_id, self.target_item]):
        #     print("Collision occured!!!")
        #     raise StopIteration
        # self.IK_teleport(target_T_se=(X_sde_pre, R_sde_pre), gripper_val=0., IK_time = IK_time, criteria='closest')
        # if self.robot_col_check(obj_ids=[self.table_id, self.target_item]):
        #     print("Collision occured!!!")
        #     raise StopIteration


        # Pause
        self.IK(duration = 0.5, 
                gripper_val = pre_grasp['gripper_val'], 
                target_T_se = pre_grasp['pose'],
                sleep = 0.000003 * sleep,
        )
        # Reach Grasp pose
        self.IK(duration = 6, 
                gripper_val = grasp1['gripper_val'], 
                target_T_se = grasp1['pose'],
                sleep = 0.000003 * sleep,
                #force = reach_force
        )
        # Grasp
        self.IK(duration = 3, 
                gripper_val = grasp2['gripper_val'], 
                target_T_se = grasp2['pose'],
                sleep = 0.003 * sleep
        )
        # Lift
        self.IK(duration = 2, 
                gripper_val = lift['gripper_val'], 
                target_T_se = lift['pose'],
                sleep = 0.003 * sleep,
                init_gripper_val = lift['gripper_val']
        )
        # Pause
        self.IK(duration = 1, 
                gripper_val = lift['gripper_val'], 
                target_T_se = lift['pose'],
                sleep = 0.003 * sleep,
                init_gripper_val = lift['gripper_val'],
                gripper_force=1000
        )


        if self.target_item is not None:
            self.attach(self.target_item)

        #remove_handles(target_handles)

    def retract_robot(self, gripper_val, IK_time = 1., back = False):
        #target_T_se = (np.array([0.6, 0., 0.6]), np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(np.array([np.pi, 0., 0.])))).reshape(3,3))
        #self.IK_teleport(target_T_se=target_T_se, gripper_val=gripper_val, IK_time = IK_time)
        if back:
            self.teleport(configuration=self.back_robot_conf[:-2])
        else:
            self.teleport(configuration=self.default_robot_conf[:-2])
        self.gripper_control(gripper_val)


    def place(self, pre_place, place, sleep = False, IK_time = 1., z_offset = 0.03, max_distance_plan=(0.05, 1.5)):
        X_sdg, R_sdg = place
        target_handles = draw_pose(Pose(X_sdg, p.getEulerFromQuaternion(Rotation.from_matrix(R_sdg).as_quat())))

        plan = self.place_plan(place, z_offset = z_offset)
        pre_place, place, release, retract = plan['pre_place'], plan['place'], plan['release'], plan['retract']


        # R_sde_pre = R_sdg_pre @ self.R_eg.T
        # R_sde = R_sdg @ self.R_eg.T
        # X_sde_pre = X_sdg_pre - R_sde_pre@self.X_eg
        # X_sde = X_sdg - R_sde@self.X_eg

        # if self.check_feasible(pose = (X_sde, R_sde), IK_time=IK_time, criteria='controllability') is False:
        #     raise StopIteration

        # # Reach Preplace Pose
        # self.IK_teleport(target_T_se=(X_sde, R_sde), gripper_val=1., IK_time = IK_time, criteria='controllability')
        # self.IK_teleport(target_T_se=(X_sde_pre, R_sde_pre), gripper_val=1., IK_time = IK_time, criteria='closest')

        max_distance=max_distance_plan[0]
        conf = self.check_feasible_plan([pre_place, place], IK_time=IK_time, verbose=True, max_distance=max_distance)
        if conf is False:
            raise StopIteration
        max_distance=max_distance_plan[1]
        if self.check_feasible_plan([retract, release], IK_time=IK_time, verbose=True, max_distance=max_distance) is False:
            raise StopIteration
        
        #self.IK_teleport(target_T_se=place['pose'], gripper_val=place['gripper_val'], IK_time = IK_time, criteria='controllability')
        self.IK_teleport(target_T_se=pre_place['pose'], gripper_val=pre_place['gripper_val'], IK_time = IK_time, criteria='closest', max_distance=max_distance, ref_conf=conf)


        # Reach Pre-place pose
        self.IK(duration = 1, 
                gripper_val = pre_place['gripper_val'], 
                target_T_se = pre_place['pose'],
                sleep = 0.003 * sleep,
                gripper_force=300,
        )
        self.detach()

        # Reach place pose
        self.IK(duration = 2, 
                gripper_val = place['gripper_val'], 
                target_T_se = place['pose'],
                sleep = 0.003 * sleep,
                gripper_force=300,
                init_gripper_val=place['gripper_val']
        )

        # Release
        self.IK(duration = 1, 
                gripper_val = release['gripper_val'],
                gripper_force = 300, 
                target_T_se = release['pose'],
                sleep = 0.003 * sleep
        )

        # # Jerk
        # self.IK(duration = 0.3, 
        #         gripper_val=0.,
        #         gripper_force = 1, 
        #         target_T_se = (X_sde + np.array([0., 0., -0.03]), R_sde),
        #         sleep = 0.003 * sleep
        # )
        # self.IK(duration = 0.3, 
        #         gripper_val=0.,
        #         gripper_force = 1, 
        #         target_T_se = (X_sde + np.array([0., 0., 0.03]), R_sde),
        #         sleep = 0.003 * sleep
        # )
        # self.IK(duration = 0.3, 
        #         gripper_val=0.,
        #         gripper_force = 1, 
        #         target_T_se = (X_sde + np.array([0., 0.03, 0.]), R_sde),
        #         sleep = 0.003 * sleep
        # )
        # self.IK(duration = 0.3, 
        #         gripper_val=0.,
        #         gripper_force = 1, 
        #         target_T_se = (X_sde + np.array([0., -0.03, 0.]), R_sde),
        #         sleep = 0.003 * sleep
        # )
        # # Retract
        # self.IK(duration = 0.3, 
        #         gripper_val=0., 
        #         gripper_force = 1, 
        #         target_T_se = (X_sde_pre, R_sde_pre),
        #         sleep = 0.003 * sleep
        # )

        # Retract
        self.IK(duration = 2, 
                gripper_val = retract['gripper_val'], 
                gripper_force = 1, 
                target_T_se = retract['pose'],
                sleep = 0.003 * sleep,
                joint_force=50
        )

        # Retract
        self.IK(duration = 2, 
                gripper_val = retract['gripper_val'], 
                gripper_force = 1, 
                target_T_se = (retract['pose'][0] + np.array([0., 0., 0.02]), retract['pose'][1]),
                sleep = 0.003 * sleep,
                joint_force=50
        )

        #remove_handles(target_handles)
    
    def observe_pick(self):
        state = self.get_state()
        X_sg, R_sg = state['T_sg']

        return observe(self.pickcam_configs, physicsClientId=self.physicsClientId, target_pos = tuple(X_sg), R_sg = R_sg, X_sg = X_sg)

    def observe_pointcloud_pick(self, stride = (1,1)):
        data = self.observe_pick()

        return img_data_to_pointcloud(data, self.xlim_pick, self.ylim_pick, self.zlim_pick, stride=stride)







































































class MugTask(FrankaTask):
    def __init__(self, use_gui = True):
        super().__init__(use_gui=use_gui)

    def init_task(self, mug_type = 'default', distractor = False, use_support = False, **kwargs):
        super().init_task()

        self.mug_scale = 1.
        self.X_mh = np.array([0., 0.055, 0.02]) * (self.mug_scale) # mug to handle
        self.R_mh = np.eye(3)
        self.X_eg = np.array([0, 0, 0.105]) # EE to gripper sweet-spot
        self.R_eg = np.eye(3)
        self.X_m_top = np.array([0., 0., 0.07]) * self.mug_scale
        self.R_m_top = np.eye(3)
        self.branch_length = 0.065
        self.branchLinkId = 1

        if mug_type == 'default':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/mug.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
        elif mug_type == 'cup1':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/mugs/mug1/mug.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            self.X_mh = np.array([0., 0.065, 0.01]) * (self.mug_scale)
        elif mug_type == 'cup2':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/mugs/mug2/mug.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            self.X_mh = np.array([0.0, 0.075, 0.01]) * (self.mug_scale)
        elif mug_type == 'cup3':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/mugs/mug3/mug.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            self.X_mh = np.array([-0.0, 0.07, 0.015]) * (self.mug_scale)
        elif mug_type == 'cup4':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/mugs/mug4/mug.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            self.X_mh = np.array([0.0, 0.07, 0.015]) * (self.mug_scale)
        elif mug_type == 'cup5':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/mugs/mug5/mug.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            self.X_mh = np.array([-0.0, 0.07, 0.01]) * (self.mug_scale)
        elif mug_type == 'cup6':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/mugs/mug6/mug.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            self.X_mh = np.array([0., 0.06, 0.015]) * (self.mug_scale)
        elif mug_type == 'cup7':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/mugs/mug7/mug.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            self.X_mh = np.array([0., 0.08, 0.01]) * (self.mug_scale)
        elif mug_type == 'cup8':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/mugs/mug8/mug.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            self.X_mh = np.array([0.0, 0.06, 0.01]) * (self.mug_scale)
        elif mug_type == 'cup9':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/mugs/mug9/mug.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            self.X_mh = np.array([0.0, 0.06, 0.03]) * (self.mug_scale)
        elif mug_type == 'cup10':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/mugs/mug10/mug.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            self.X_mh = np.array([0.0, 0.05, 0.01]) * (self.mug_scale)
        else:
            raise KeyError
        p.changeDynamics(self.mug_id, -1, lateralFriction=0.8, rollingFriction=0.3, spinningFriction=0.3)
        self.hook_scale = 1.
        self.hook_id = p.loadURDF("assets/hanger.urdf", basePosition=[0.5, 0., 0.], globalScaling=self.hook_scale, physicsClientId = self.physicsClientId, useFixedBase = True)

        if use_support:
            self.support_box_h = 0.3
            self.support_box_id = create_box(w=0.15, l=0.15, h=self.support_box_h, color=(72/255,72/255,72/255,1.))
            p.changeDynamics(self.support_box_id, -1, lateralFriction = 3., rollingFriction=3., spinningFriction=3.)
        else:
            self.support_box_id = None

        if distractor:
            self.lego_scale = 2.5
            self.lego_id = p.loadURDF("assets/distractor/lego.urdf", basePosition=[0.7, -0.2, 0.35], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), globalScaling=self.lego_scale, physicsClientId = self.physicsClientId)
            self.duck_scale = 0.07
            self.duck_id = p.loadURDF("assets/distractor/duck.urdf", basePosition=[0.7, 0.2, 0.35], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), globalScaling=self.duck_scale, physicsClientId = self.physicsClientId)
            self.torus_scale = 0.07
            self.torus_id = p.loadURDF("assets/distractor/torus_textured.urdf", basePosition=[0.5, -0.2, 0.35], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), globalScaling=self.torus_scale, physicsClientId = self.physicsClientId)
            self.bunny_scale = 0.07
            self.bunny_id = p.loadURDF("assets/distractor/bunny.urdf", basePosition=[0.5, 0.2, 0.35], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), globalScaling=self.torus_scale, physicsClientId = self.physicsClientId)


    def reset(self, seed = None, mug_pose = 'upright', mug_type = 'default', distractor = False, use_support = False, step=1):
        if mug_type == 'cups':
            mug_type = f'cup{seed%10 + 1}'
        super().reset(seed = seed, step = False, mug_type=mug_type, distractor=distractor, use_support=use_support)

        randomize_mug_pos = True
        randomize_hook_pos = True

        self.target_item = self.mug_id

        if mug_pose == 'arbitrary':
            #if seed %2 == 0:
            if np.random.rand() > 0.5:
                mug_pose = 'lying'
            else:
                mug_pose = 'upright'


        # Reset cup orientation
        if mug_pose == 'upright':
            mug_orn = np.array([0., 0., np.random.rand()*np.pi*2])
            mug_orn = p.getQuaternionFromEuler(mug_orn)
        elif mug_pose == 'lying':
            mug_orn = np.array([np.pi /2, 0., -np.random.rand()*np.pi])
            mug_orn = p.getQuaternionFromEuler(mug_orn)
            mug_orn = multiply_quats(mug_orn, p.getQuaternionFromEuler(np.array([0., 0., (2*np.random.rand()-1)*np.pi*1/2])))
        else:
            raise KeyError


        # Sample and mug displacement(s) from center
        if randomize_mug_pos:
            if distractor is True:
                disp_x_abs_max = 0.01
                disp_y_abs_max = 0.01
            elif mug_pose == 'upright':
                disp_x_abs_max = 0.05
                disp_y_abs_max = 0.05
            elif mug_pose == 'lying':
                disp_x_abs_max = 0.03
                disp_y_abs_max = 0.03
            mug_disp = np.array([(2*np.random.rand() - 1) * disp_x_abs_max, (2*np.random.rand() - 1) * disp_y_abs_max, 0.])
        else:
            mug_disp = np.array([0., 0., 0.])
        mug_origin = self.center

        if distractor:
            p.changeVisualShape(objectUniqueId=self.bunny_id, linkIndex=-1, rgbaColor = np.concatenate([np.random.rand(3), np.array([1.])]))
            p.changeVisualShape(objectUniqueId=self.lego_id, linkIndex=-1, rgbaColor = np.concatenate([np.random.rand(3), np.array([1.])]))
            p.changeVisualShape(objectUniqueId=self.torus_id, linkIndex=-1, rgbaColor = np.concatenate([np.random.rand(3), np.array([1.])]))
            p.changeVisualShape(objectUniqueId=self.duck_id, linkIndex=-1, rgbaColor = np.concatenate([np.random.rand(3), np.array([1.])]))


        # Sample and distractors displacement(s) from center
        if distractor:
            if randomize_mug_pos:
                disp_x_max = 0.01
                disp_y_max = 0.01
                disps = []
                disps.append(np.array([np.random.rand() * disp_x_max, np.random.rand() * disp_y_max, 0.]))
                disps.append(np.array([-np.random.rand() * disp_x_max, np.random.rand() * disp_y_max, 0.]))
                disps.append(np.array([np.random.rand() * disp_x_max, -np.random.rand() * disp_y_max, 0.]))
                disps.append(np.array([-np.random.rand() * disp_x_max, -np.random.rand() * disp_y_max, 0.]))
            else:
                disps = []
                disps.append(np.array([0., 0., 0.]))
                disps.append(np.array([0., 0., 0.]))
                disps.append(np.array([0., 0., 0.]))
                disps.append(np.array([0., 0., 0.]))


        # Origin pos for mug (and distractors)
        if distractor:
            global_offset = np.array([0.0, 0., 0.])
            dx = 0.12
            dy = 0.12
            origins = [self.center + global_offset + np.array([dx, dy, 0.]),
                       self.center + global_offset + np.array([-dx, dy, 0.]),
                       self.center + global_offset + np.array([dx, -dy, 0.]),
                       self.center + global_offset + np.array([-dx, -dy, 0.])]
            

        # Allocate origin and disp for mugs and distractors
        if distractor:
            idx = list(range(4))
            np.random.shuffle(idx)

            bunny_disp = disps[idx[0]]
            lego_disp = disps[idx[1]]
            duck_disp = disps[idx[2]]
            torus_disp =  disps[idx[3]]

            bunny_origin = origins[idx[0]]
            lego_origin = origins[idx[1]]
            duck_origin = origins[idx[2]]
            torus_origin = origins[idx[3]]



        # Reset mug
        mug_pos = mug_origin + mug_disp
        if self.support_box_id is not None:
            # First reset the support
            support_box_pos = mug_pos + np.array([0., 0., 2.])
            support_box_pos[-1] = stable_z(self.support_box_id, self.table_id) - self.support_box_h + (np.random.rand() * 0.07 * randomize_mug_pos)
            p.resetBasePositionAndOrientation(self.support_box_id, support_box_pos, np.array([0., 0., 0., 1.]), physicsClientId = self.physicsClientId)

            # Reset mug on top of the support
            mug_pos = support_box_pos + np.array([0., 0., 4.])
            p.resetBasePositionAndOrientation(self.mug_id, mug_pos, mug_orn, physicsClientId = self.physicsClientId)
            mug_pos[2] = stable_z(self.mug_id, self.support_box_id)
            p.resetBasePositionAndOrientation(self.mug_id, mug_pos, mug_orn, physicsClientId = self.physicsClientId)
        else:
            mug_pos = mug_pos + np.array([0., 0., 4.])
            p.resetBasePositionAndOrientation(self.mug_id, mug_pos, mug_orn, physicsClientId = self.physicsClientId)
            mug_pos[2] = stable_z(self.mug_id, self.table_id)
            p.resetBasePositionAndOrientation(self.mug_id, mug_pos, mug_orn, physicsClientId = self.physicsClientId)
        

        # Reset distractor
        if distractor:
            bunny_pos = bunny_origin + bunny_disp + np.array([0., 0., 4.])
            bunny_pos[2] = stable_z(self.bunny_id, self.table_id)
            bunny_orn = p.getQuaternionFromEuler([0., 0., np.random.rand()*np.pi*2])

            lego_pos = lego_origin + lego_disp + np.array([0., 0., 4.])
            lego_pos[2] = stable_z(self.lego_id, self.table_id)
            lego_orn = p.getQuaternionFromEuler([0., 0., np.random.rand()*np.pi*2])

            duck_pos = duck_origin + duck_disp + np.array([0., 0., 4.])
            duck_pos[2] = stable_z(self.duck_id, self.table_id)
            duck_orn = p.getQuaternionFromEuler([0., 0., np.random.rand()*np.pi*2])

            torus_pos = torus_origin + torus_disp + np.array([0., 0., 4.])
            torus_pos[2] = stable_z(self.torus_id, self.table_id)
            torus_orn = p.getQuaternionFromEuler([0., 0., np.random.rand()*np.pi*2])

            p.resetBasePositionAndOrientation(self.bunny_id, bunny_pos, bunny_orn, physicsClientId = self.physicsClientId)
            p.resetBasePositionAndOrientation(self.lego_id, lego_pos, lego_orn, physicsClientId = self.physicsClientId)
            p.resetBasePositionAndOrientation(self.duck_id, duck_pos, duck_orn, physicsClientId = self.physicsClientId)
            p.resetBasePositionAndOrientation(self.torus_id, torus_pos, torus_orn, physicsClientId = self.physicsClientId)

        # Reset Hook
        hook_pos = self.center + np.array([0.2, 0.0 ,-0.15])
        if randomize_hook_pos:
            hook_pos = hook_pos + np.array([0., (2*np.random.rand() - 1) * 0.05, 0.])
        hook_orn = np.array([0., 0., np.pi]) + np.array([0., 0., (2*np.random.rand() - 1) * np.pi / 8])
        hook_orn = p.getQuaternionFromEuler(hook_orn)
        p.resetBasePositionAndOrientation(self.hook_id, hook_pos, hook_orn, physicsClientId = self.physicsClientId)


        # Disable collision for visual distractors with robot
        for i in range(12):
            #p.setCollisionFilterPair(self.hook_id, self.robot, -1, i, 0, self.physicsClientId)
            #p.setCollisionFilterPair(self.hook_id, self.robot, 0, i, 0, self.physicsClientId)
            if distractor:
                p.setCollisionFilterPair(self.bunny_id, self.robot, -1, i, 0, self.physicsClientId)
                p.setCollisionFilterPair(self.lego_id, self.robot, -1, i, 0, self.physicsClientId)
                p.setCollisionFilterPair(self.duck_id, self.robot, -1, i, 0, self.physicsClientId)
                p.setCollisionFilterPair(self.torus_id, self.robot, -1, i, 0, self.physicsClientId)
        if distractor:
            p.setCollisionFilterPair(self.bunny_id, self.mug_id, -1, -1, 0, self.physicsClientId)
            p.setCollisionFilterPair(self.lego_id, self.mug_id, -1, -1, 0, self.physicsClientId)
            p.setCollisionFilterPair(self.duck_id, self.mug_id, -1, -1, 0, self.physicsClientId)
            p.setCollisionFilterPair(self.torus_id, self.mug_id, -1, -1, 0, self.physicsClientId)


        p.changeDynamics(self.mug_id, -1, linearDamping = 0.3, angularDamping = 0.3)
        #p.changeDynamics(self.mug_id, -1, contactStiffness = 100000000., contactDamping = 100.)

        # step sim
        if step:
            step = int(step/self.freq)
            for _ in range(step):
                p.stepSimulation(physicsClientId = self.physicsClientId)


    def get_state(self):
        state = super().get_state()

        # Suffix guide
        # ==>  m: Mug, h: Handle, dg: Desired grasp sweet spot, de: Desired End Effector, pre: Pregrasp
        X_sm, R_sm = p.getBasePositionAndOrientation(self.mug_id, physicsClientId = self.physicsClientId)
        X_sm, R_sm = np.array(X_sm), Rotation.from_quat(R_sm).as_matrix()
        X_s_hook, R_s_hook = p.getBasePositionAndOrientation(self.hook_id, physicsClientId = self.physicsClientId)
        X_s_hook, R_s_hook = np.array(X_s_hook), Rotation.from_quat(R_s_hook).as_matrix()
        X_s_branch, R_s_branch = p.getLinkState(self.hook_id, self.branchLinkId)[:2]
        X_s_branch, R_s_branch = np.array(X_s_branch), Rotation.from_quat(R_s_branch).as_matrix()
        R_s_tip = R_s_branch.copy()
        X_s_tip = X_s_branch + (R_s_branch @ np.array([0., 0., self.branch_length]))

        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_sh, R_sh = X_sm + (R_sm @ X_mh), R_sm @ R_mh

        state['T_sm'] = (X_sm, R_sm)
        state['T_sh'] = (X_sh, R_sh)
        state['T_s_hook'] = (X_s_hook, R_s_hook)
        state['T_s_tip'] = (X_s_tip, R_s_tip)
        state['T_s_branch'] = (X_s_branch, R_s_branch)

        return state

#     Z-axis
#       ↑
    #       #
    #       ####
    #       #  # -> Y-axis
    #       ####
    #########


    def oracle_pick_rim(self, mod):
        state = self.get_state()
        X_sm, R_sm = state['T_sm']
        X_se, R_se = state['T_se']

        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_m_top, R_m_top = self.X_m_top, self.R_m_top

        
        #yaw = -np.pi / 2 # 0 ~ pi
        #yaw = yaw + (2*np.random.rand()-1)*np.pi*2/3
        #yaw = yaw + np.pi/2

        yaw_mod_1 = 0.
        yaw_mod_1 += (2*np.random.rand()-1)*np.pi*1/6
        yaw_mod_2 = np.pi 
        yaw_mod_2 += (2*np.random.rand()-1)*np.pi*1/6

        yaw = yaw_mod_1*mod + yaw_mod_2*(1-mod)

        R_top_rim = p.getMatrixFromQuaternion(p.getQuaternionFromEuler(np.array([0., 0., yaw])))
        R_top_rim = np.array(R_top_rim).reshape(3,3)
        rim_X_top_rim = np.array([0.035, 0., 0.])
        X_top_rim = R_top_rim @ rim_X_top_rim
        X_rim_dg = np.array([0., 0., -0.01]) 
        R_rim_dg = np.array([[0. ,1. ,0.],
                             [1. ,0. ,0.],
                             [0. ,0. ,-1.]])

        R_sdg = R_sm @ R_m_top @ R_top_rim @ R_rim_dg
        X_top_dg = X_top_rim + (R_top_rim @ X_rim_dg)
        X_mdg = X_m_top + (R_m_top @ X_top_dg)
        X_sdg = X_sm + (R_sm @ X_mdg)


        perturb_axis = np.random.randn(3)
        perturb_axis = perturb_axis / np.linalg.norm(perturb_axis)
        perturb_angle = (2*np.random.rand()-1) * (np.pi/180 * 5)
        R_perturb = Rotation.from_rotvec(perturb_angle * perturb_axis).as_matrix()
        R_sdg = R_sdg @ R_perturb

        R_dg_dgpre = np.eye(3)
        R_s_dgpre = R_sdg @ R_dg_dgpre
        X_dg_dgpre = np.array([0., 0., -0.03])
        sX_dg_dgpre = R_sdg @ X_dg_dgpre
        X_s_dgpre = X_sdg + sX_dg_dgpre

        pre_grasp = (X_s_dgpre, R_s_dgpre)
        grasp = X_sdg, R_sdg

        return pre_grasp, grasp

    def oracle_pick_handle(self, random_180_flip = False, force_flip = False):
        state = self.get_state()
        X_sm, R_sm = state['T_sm']
        X_se, R_se = state['T_se']

        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_m_top, R_m_top = self.X_m_top, self.R_m_top

        X_hdg = np.array([0., 0.03, 0.0]) * self.mug_scale + (2*np.random.rand(3)-1)*np.array([0., 0., 0.01])
        R_hdg = np.array([[0. ,0. ,1.],
                          [0. ,1. ,0.],
                          [-1. ,0. ,0.]])

        R_sdg = R_sm @ R_mh @ R_hdg

        flip = False
        z_axis = R_sdg[:,-1]
        if z_axis[2] > 0.8:
            flip = True
        elif z_axis[0] < 0:
            flip = True
        if force_flip:
            flip = True

        if flip is True:
            X_hdg = X_hdg - np.array([0.01, 0., 0.])

            R_flip = np.array([[-1. ,0. ,0.],
                               [0. ,1. ,0.],
                               [0. ,0. ,-1.]])
            R_sdg = R_sdg @ R_flip
        else:
            X_hdg = X_hdg + np.array([0.01, 0., 0.])

        X_mdg = X_mh + (R_mh @ X_hdg)
        X_sdg = X_sm + (R_sm @ X_mdg)

        perturb_axis = np.random.randn(3)
        perturb_axis = perturb_axis / np.linalg.norm(perturb_axis)
        perturb_angle = (2*np.random.rand()-1) * (np.pi/180 * 5)
        R_perturb = Rotation.from_rotvec(perturb_angle * perturb_axis).as_matrix()
        R_sdg = R_sdg @ R_perturb

        R_dg_dgpre = np.eye(3)
        R_s_dgpre = R_sdg @ R_dg_dgpre
        X_dg_dgpre = np.array([0., 0., -0.03])
        sX_dg_dgpre = R_sdg @ X_dg_dgpre
        X_s_dgpre = X_sdg + sX_dg_dgpre

        if random_180_flip:
            R_flip = np.array([[-1., 0., 0.],[0., -1., 0.],[0., 0., 1.]]).T
            if np.random.randint(2) == 1:
                R_s_dgpre = R_s_dgpre @ R_flip
                R_sdg = R_sdg @ R_flip
        pre_grasp = (X_s_dgpre, R_s_dgpre)
        grasp = X_sdg, R_sdg

        return pre_grasp, grasp

    def oracle_place_hole(self):
        state = self.get_state()
        X_sm, R_sm = state['T_sm']
        X_se, R_se = state['T_se']
        X_s_tip, R_s_tip = state['T_s_tip']
        R_sg = R_se @ self.R_eg
        X_sg = X_se + (R_se @ self.X_eg)

        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_m_top, R_m_top = self.X_m_top, self.R_m_top

        X_dtop_tip = np.array([0., 0., -0.03])
        R_tip_dtop = np.array([[-1.         ,0.         ,0.],
                               [0.        ,1.         ,0.],
                               [0.         ,0.        ,-1.]])
        X_tip_dtop = -R_tip_dtop @ X_dtop_tip
        R_s_dtop = R_s_tip @ R_tip_dtop
        X_s_dtop = X_s_tip + (R_s_tip @ X_tip_dtop)

        R_s_top = R_sm @ R_m_top
        X_s_top = X_sm + (R_sm @ X_m_top)
        R_top_g = (R_s_top).T @ R_sg
        X_top_g = R_s_top.T @ (X_sg - X_s_top)
        R_dtop_dg, X_dtop_dg = R_top_g, X_top_g

        R_sdg = R_s_dtop @ R_dtop_dg
        X_sdg = X_s_dtop + (R_s_dtop @ X_dtop_dg)

        R_dg_dgpre = np.eye(3)
        R_s_dgpre = R_sdg @ R_dg_dgpre
        X_dg_dgpre = np.array([0., 0., -0.03])
        sX_dg_dgpre = R_sdg @ X_dg_dgpre
        X_s_dgpre = X_sdg + sX_dg_dgpre

        pre_place = (X_s_dgpre, R_s_dgpre)
        place = X_sdg, R_sdg
        #place = X_s_dtop, R_s_dtop
        

        return pre_place, place

    def oracle_place_handle(self, flip_x = False, theta = 0.):
        state = self.get_state()
        X_sm, R_sm = state['T_sm']
        X_se, R_se = state['T_se']
        X_s_tip, R_s_tip = state['T_s_tip']
        R_sg = R_se @ self.R_eg
        X_sg = X_se + (R_se @ self.X_eg)
        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_sh, R_sh = X_sm + (R_sm @ X_mh), R_sm @ R_mh

        X_gh = R_sg.T @ (X_sh - X_sg)
        X_dgdh = X_gh
        R_dgdh = R_sg.T @ R_sh
        X_dhdg = - R_dgdh.T @ X_dgdh

        # R_tip_dh = np.array([[0.        ,0.          ,-1.],
        #                      [0.         ,1.          ,0.],
        #                      [1.         ,0.         ,0.]])

        #theta = np.pi*(3./12)
        R_tip_dh = np.array([[-np.sin(theta)        ,0.0          ,-np.cos(theta)],
                             [0.0         ,-1.0          ,0.0],
                             [-np.cos(theta)         ,0.         ,np.sin(theta)]])
        if flip_x == True:
            R_tip_flip = np.array([[-1.        ,0.          ,0.],
                                    [0.         ,1.          ,0.],
                                    [0.         ,0.         ,-1.]])
            R_tip_dh = R_tip_flip @ R_tip_dh
        X_tip_dh = np.array([0., 0., 0.01]) # Depreceted: X_tip_dh = np.array([0., 0., -0.01])
        R_tip_dg = R_tip_dh @ R_dgdh.T
        X_tip_dg = X_tip_dh + (R_tip_dh @ X_dhdg)
        X_tip_dg = X_tip_dg + np.array([-0.01, 0., 0.])
        X_dg_tip = -(R_tip_dg.T @ X_tip_dg)


        X_gm = R_sg.T @ (X_sm - X_sg)
        if X_gm[1] < 0.:
            y_offset = -0.00 * self.mug_scale
        elif X_gm[1] > 0.:
            y_offset = 0.00 * self.mug_scale
        X_dg_tip = X_dg_tip + np.array([0., y_offset, 0.])

        X_tip_dg = -R_tip_dg @ X_dg_tip
        R_sdg = R_s_tip @ R_tip_dg
        X_sdg = X_s_tip + (R_s_tip @ X_tip_dg)

        R_dg_dgpre = np.eye(3)
        R_s_dgpre = R_sdg @ R_dg_dgpre
        X_dg_dgpre = np.array([0., 0., -0.03])
        sX_dg_dgpre = R_sdg @ X_dg_dgpre
        X_s_dgpre = X_sdg + sX_dg_dgpre

        pre_place = (X_s_dgpre, R_s_dgpre)
        place = X_sdg, R_sdg
        #place = X_s_dtop, R_s_dtop
        
        # R_sdh = R_s_tip @ R_tip_dh
        # X_sdh = X_s_tip + R_s_tip @ X_tip_dh
        # place = (X_sdh, R_sdh)
        raise NotImplementedError # Currently not used. #TODO: remove this
        return pre_place, place

    def oracle_place_handle_horizontal(self, mod, low_var = False):
        state = self.get_state()
        X_sm, R_sm = state['T_sm']
        X_se, R_se = state['T_se']
        X_s_tip, R_s_tip = state['T_s_tip']
        X_s_base, R_s_base = state['T_s_hook']
        R_sg = R_se @ self.R_eg
        X_sg = X_se + (R_se @ self.X_eg)
        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_sh, R_sh = X_sm + (R_sm @ X_mh), R_sm @ R_mh

        X_gh = R_sg.T @ (X_sh - X_sg)
        X_dgdh = X_gh
        R_dgdh = R_sg.T @ R_sh
        X_dhdg = - R_dgdh.T @ X_dgdh

        if mod:
            # theta = -np.pi*np.random.rand() # -pi~0
            offset = np.pi/4
            if low_var:
                theta = -(np.pi-2*offset)*(0.5 + (np.random.rand()-0.5)*0.3) - offset # 30 percent of original variance
            else:
                theta = -(np.pi-2*offset)*np.random.rand() - offset # -pi/4 ~ -3pi/4
            R_base_dh = np.array([[ 0.0,    np.sin(theta),          np.cos(theta)],
                                  [ 0.0,   -np.cos(theta),          np.sin(theta)],
                                  [ 1.0,              0.0,                    0.0]])
        else:
            offset = np.pi/4
            if low_var:
                theta = (np.pi-2*offset)*(0.5 + (np.random.rand()-0.5)*0.3) + offset # 30 percent of original variance
            else:
                theta = (np.pi-2*offset)*np.random.rand() + offset # pi/4 ~ 3pi/4
            R_base_dh = np.array([[ 0.0,   -np.sin(theta),          np.cos(theta)],
                                  [ 0.0,    np.cos(theta),          np.sin(theta)],
                                  [-1.0,              0.0,                    0.0]])

        R_sdg = R_s_base @ R_base_dh @ R_dgdh.T

        sX_tip_dh = np.array([0., 0., 0.00]) # np.array([0., 0., 0.02])
        R_sdh = R_sdg @ R_dgdh
        sX_dhdg = R_sdh @ X_dhdg
        X_sdg = X_s_tip + sX_tip_dh + sX_dhdg

        R_dg_dgpre = np.eye(3)
        R_s_dgpre = R_sdg @ R_dg_dgpre
        X_dg_dgpre = np.array([0., 0., -0.03])
        sX_dg_dgpre = R_sdg @ X_dg_dgpre
        X_s_dgpre = X_sdg + sX_dg_dgpre

        pre_place = (X_s_dgpre, R_s_dgpre)
        place = (X_sdg, R_sdg)

        return pre_place, place

    def oracle_place_handle_deprecated(self):
        state = self.get_state()
        X_sm, R_sm = state['T_sm']
        X_se, R_se = state['T_se']
        X_s_tip, R_s_tip = state['T_s_tip']
        R_sg = R_se @ self.R_eg
        X_sg = X_se + (R_se @ self.X_eg)
        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_sh, R_sh = X_sm + (R_sm @ X_mh), R_sm @ R_mh

        X_gh = R_sg.T @ (X_sh - X_sg)
        X_dgdh = X_gh
        R_dgdh = R_sg.T @ R_sh
        X_dhdg = - R_dgdh.T @ X_dgdh

        R_tip_dg = np.array([[-1.        ,0.          ,0.],
                             [0.         ,1.          ,0.],
                             [0.         ,0.         ,-1.]])
        X_tip_dh = np.array([0., 0., -0.02])
        R_tip_dh = R_tip_dg @ R_dgdh
        X_tip_dg = X_tip_dh + (R_tip_dh @ X_dhdg)
        X_tip_dg = X_tip_dg + np.array([-0.01, 0., 0.])
        X_dg_tip = -(R_tip_dg.T @ X_tip_dg)


        X_gm = R_sg.T @ (X_sm - X_sg)
        if X_gm[1] < 0.:
            y_offset = -0.00 * self.mug_scale
        elif X_gm[1] > 0.:
            y_offset = 0.00 * self.mug_scale
        X_dg_tip = X_dg_tip + np.array([0., y_offset, 0.])

        X_tip_dg = -R_tip_dg @ X_dg_tip
        R_sdg = R_s_tip @ R_tip_dg
        X_sdg = X_s_tip + (R_s_tip @ X_tip_dg)

        R_dg_dgpre = np.eye(3)
        R_s_dgpre = R_sdg @ R_dg_dgpre
        X_dg_dgpre = np.array([0., 0., -0.1])
        sX_dg_dgpre = R_sdg @ X_dg_dgpre
        X_s_dgpre = X_sdg + sX_dg_dgpre

        pre_place = (X_s_dgpre, R_s_dgpre)
        place = X_sdg, R_sdg
        #place = X_s_dtop, R_s_dtop

        return pre_place, place

    def debug(self, debug_items = ['grasp']):
        super().debug(debug_items=debug_items)
        for item in debug_items:
            if item == 'mug':
                mugFrame_ID = axiscreator(self.mug_id, physicsClientId = self.physicsClientId)
            elif item == 'mug_rim':
                topFrame_ID = axiscreator(self.mug_id, offset = self.X_m_top.copy(), physicsClientId = self.physicsClientId)
            elif item == 'mug_handle':
                handleFrame_ID = axiscreator(self.mug_id, offset = self.X_mh.copy(), physicsClientId = self.physicsClientId)
            elif item == 'hook_branch':
                hookFrame_ID = axiscreator(self.hook_id, self.branchLinkId, physicsClientId = self.physicsClientId)
            elif item == 'hook_branch_offset':
                axiscreator(self.hook_id, self.branchLinkId, offset = np.array([0., 0., 0.07]),physicsClientId = self.physicsClientId)
            elif item == 'distractors':
                axiscreator(self.bunny_id, physicsClientId = self.physicsClientId)
                axiscreator(self.duck_id, physicsClientId = self.physicsClientId)
                axiscreator(self.torus_id, physicsClientId = self.physicsClientId)
                axiscreator(self.lego_id, physicsClientId = self.physicsClientId)
            elif item == 'hook_base':
                hookBaseFrame_ID = axiscreator(self.hook_id, physicsClientId = self.physicsClientId)

    def check_pick_success(self):
        state = self.get_state()
        X_sm, R_sm = state['T_sm']
        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_sh, R_sh = X_sm + (R_sm @ X_mh), R_sm @ R_mh
        if X_sh[2] > 0.25:
            return True
        else:
            return False

    def check_place_success(self):
        p.setCollisionFilterPair(self.target_item, self.table_id, -1, -1, 0, self.physicsClientId)
        p.setCollisionFilterPair(self.target_item, self.plane_id, -1, -1, 0, self.physicsClientId)
        if self.support_box_id is not None:
            p.setCollisionFilterPair(self.target_item, self.support_box_id, -1, -1, 0, self.physicsClientId)
        for i in range(12):
            p.setCollisionFilterPair(self.target_item, self.robot, -1, i, 0, self.physicsClientId)

        for _ in range(int(2/self.freq)):
            p.stepSimulation(physicsClientId = self.physicsClientId)

        state = self.get_state()
        X_sh = state['T_sh'][0]
        if X_sh[-1] > 0.:
            result = True
        else:
            result = False

        p.setCollisionFilterPair(self.target_item, self.table_id, -1, -1, 1, self.physicsClientId)
        p.setCollisionFilterPair(self.target_item, self.plane_id, -1, -1, 1, self.physicsClientId)
        if self.support_box_id is not None:
            p.setCollisionFilterPair(self.target_item, self.support_box_id, -1, -1, 1, self.physicsClientId)
        for i in range(12):
            p.setCollisionFilterPair(self.target_item, self.robot, -1, i, 1, self.physicsClientId)

        return result


    # def check_place_success_deprecated(self, thr0 = 0.03, thr1 = 0.05):
    #     state = self.get_state()
    #     X_s_branch, R_s_branch = state['T_s_branch']
    #     X_sh = state['T_sh'][0]

    #     if len(p.getContactPoints(self.hook_id, self.target_item, 0, -1)) != 0:
    #         X_s_branch_0_b = X_s_branch.copy()
    #         X_s_branch_0_a = X_s_branch_0_b + np.array([0., 0., 0.02])
    #         vec0 = X_s_branch_0_b - X_s_branch_0_a
    #         vec1 = X_sh - X_s_branch_0_a
    #         vec1_c = np.dot(vec1, vec0) / np.linalg.norm(vec0)
    #         vec1_perp = vec1 - vec1_c
    #         if vec1_c < 0.:
    #             dist = np.linalg.norm(X_sh - X_s_branch_0_a)
    #         elif vec1_c >= 0. and vec1_c <= np.linalg.norm(vec0):
    #             dist = np.linalg.norm(vec1_perp)
    #         elif vec1_c > np.linalg.norm(vec0):
    #             dist = np.linalg.norm(X_sh - X_s_branch_0_b)
    #         else:
    #             raise NotImplementedError
    #         if dist < thr0:
    #             return True
        
    #     if len(p.getContactPoints(self.hook_id, self.target_item, 1, -1)) != 0:
    #         X_s_branch_1_a = X_s_branch.copy()
    #         X_s_branch_1_b = X_s_branch_1_a + (R_s_branch @ np.array([0., 0., 0.07]))
    #         vec0 = X_s_branch_1_b - X_s_branch_1_a
    #         vec1 = X_sh - X_s_branch_1_a
    #         vec1_c = np.dot(vec1, vec0) / np.linalg.norm(vec0)
    #         vec1_perp = vec1 - vec1_c
    #         if vec1_c < 0.:
    #             dist = np.linalg.norm(X_sh - X_s_branch_1_a)
    #         elif vec1_c >= 0. and vec1_c <= np.linalg.norm(vec0):
    #             dist = np.linalg.norm(vec1_perp)
    #         elif vec1_c > np.linalg.norm(vec0):
    #             dist = np.linalg.norm(X_sh - X_s_branch_1_b)
    #         else:
    #             raise NotImplementedError
    #         if dist < thr1:
    #             return True

    #     return False
    
    # def check_place_success_deprecated(self):
    #     state = self.get_state()
    #     X_s_branch, R_s_branch = state['T_s_branch']
    #     X_s_branch_offset = X_s_branch + (R_s_branch @ np.array([0., 0., 0.03]))

    #     X_sh = state['T_sh'][0]

    #     dist = np.linalg.norm(X_sh - X_s_branch_offset)

    #     if dist < 0.07: # 0.05:
    #         z_offset = X_sh[2] - X_s_branch[2]
    #         if z_offset > -0.02:
    #             return True
    #     return False




















































































class StickTask(FrankaTask):
    def __init__(self, use_gui = True):
        super().__init__(use_gui=use_gui)

    def init_task(self, mug_type = 'default', distractor = False, use_support = False, **kwargs):
        super().init_task()

        self.mug_scale = 1.
        self.X_mh = np.array([0., 0.0, 0.0]) * (self.mug_scale) # mug to handle
        self.R_mh = np.eye(3)
        self.X_eg = np.array([0, 0, 0.105]) # EE to gripper sweet-spot
        self.R_eg = np.eye(3)
        self.X_m_top = np.array([0., 0., 0.0]) * self.mug_scale
        self.R_m_top = np.eye(3)
        self.branch_length = 0.065
        self.branchLinkId = -1

        if mug_type == 'default':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/stick.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
        elif mug_type == 'cup1':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/sticks/stick1.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0., 0.065, 0.01]) * (self.mug_scale)
        elif mug_type == 'cup2':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/sticks/stick2.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0.0, 0.075, 0.01]) * (self.mug_scale)
        elif mug_type == 'cup3':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/sticks/stick3.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([-0.0, 0.07, 0.015]) * (self.mug_scale)
        elif mug_type == 'cup4':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/sticks/stick4.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0.0, 0.07, 0.015]) * (self.mug_scale)
        elif mug_type == 'cup5':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/sticks/stick5.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([-0.0, 0.07, 0.01]) * (self.mug_scale)
        elif mug_type == 'cup6':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/sticks/stick6.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0., 0.06, 0.015]) * (self.mug_scale)
        elif mug_type == 'cup7':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/sticks/stick7.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0., 0.08, 0.01]) * (self.mug_scale)
        elif mug_type == 'cup8':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/sticks/stick8.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0.0, 0.06, 0.01]) * (self.mug_scale)
        elif mug_type == 'cup9':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/sticks/stick9.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0.0, 0.06, 0.03]) * (self.mug_scale)
        elif mug_type == 'cup10':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/sticks/stick10.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0.0, 0.05, 0.01]) * (self.mug_scale)
        else:
            raise KeyError
        p.changeDynamics(self.mug_id, -1, lateralFriction=0.8, rollingFriction=0.3, spinningFriction=0.3)
        self.hook_scale = 1.
        self.hook_id = p.loadURDF("assets/traybox.urdf", basePosition=[0.5, 0., 0.], globalScaling=0.2 * self.hook_scale, physicsClientId = self.physicsClientId, useFixedBase = True)

        if use_support:
            self.support_box_h = 0.3
            self.support_box_id = create_box(w=0.15, l=0.15, h=self.support_box_h, color=(72/255,72/255,72/255,0.))
            p.changeDynamics(self.support_box_id, -1, lateralFriction = 3., rollingFriction=3., spinningFriction=3.)
        else:
            self.support_box_id = None

        if distractor:
            self.lego_scale = 2.5
            self.lego_id = p.loadURDF("assets/distractor/lego.urdf", basePosition=[0.7, -0.2, 0.35], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), globalScaling=self.lego_scale, physicsClientId = self.physicsClientId)
            self.duck_scale = 0.07
            self.duck_id = p.loadURDF("assets/distractor/duck.urdf", basePosition=[0.7, 0.2, 0.35], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), globalScaling=self.duck_scale, physicsClientId = self.physicsClientId)
            self.torus_scale = 0.07
            self.torus_id = p.loadURDF("assets/distractor/torus_textured.urdf", basePosition=[0.5, -0.2, 0.35], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), globalScaling=self.torus_scale, physicsClientId = self.physicsClientId)
            self.bunny_scale = 0.07
            self.bunny_id = p.loadURDF("assets/distractor/bunny.urdf", basePosition=[0.5, 0.2, 0.35], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), globalScaling=self.torus_scale, physicsClientId = self.physicsClientId)


    def reset(self, seed = None, mug_pose = 'upright', mug_type = 'default', distractor = False, use_support = False, step=1):
        if mug_type == 'cups':
            mug_type = f'cup{seed%10 + 1}'
        super().reset(seed = seed, step = False, mug_type=mug_type, distractor=distractor, use_support=use_support)

        randomize_mug_pos = True
        randomize_hook_pos = True

        self.target_item = self.mug_id

        if mug_pose == 'arbitrary' or mug_pose == 'lying':
            arbitrary_tray = True
        else:
            arbitrary_tray = False

        if mug_pose == 'arbitrary':
            #if seed %2 == 0:
            if np.random.rand() > 0.5:
                mug_pose = 'lying'
            else:
                mug_pose = 'upright'


        # Reset cup orientation
        if mug_pose == 'upright':
            mug_orn = np.array([0., 0., np.random.rand()*np.pi*2])
            mug_orn = p.getQuaternionFromEuler(mug_orn)
        elif mug_pose == 'lying':
            mug_orn = np.array([np.pi /2, 0., -np.random.rand()*np.pi])
            mug_orn = p.getQuaternionFromEuler(mug_orn)
            mug_orn = multiply_quats(mug_orn, p.getQuaternionFromEuler(np.array([0., 0., (2*np.random.rand()-1)*np.pi*1/2])))
        else:
            raise KeyError


        # Sample and mug displacement(s) from center
        if randomize_mug_pos:
            if distractor is True:
                disp_x_abs_max = 0.01
                disp_y_abs_max = 0.01
            elif mug_pose == 'upright':
                disp_x_abs_max = 0.05
                disp_y_abs_max = 0.05
            elif mug_pose == 'lying':
                disp_x_abs_max = 0.03
                disp_y_abs_max = 0.03
            mug_disp = np.array([(2*np.random.rand() - 1) * disp_x_abs_max, (2*np.random.rand() - 1) * disp_y_abs_max, 0.])
        else:
            mug_disp = np.array([0., 0., 0.])
        mug_origin = self.center

        if distractor:
            p.changeVisualShape(objectUniqueId=self.bunny_id, linkIndex=-1, rgbaColor = np.concatenate([np.random.rand(3), np.array([1.])]))
            p.changeVisualShape(objectUniqueId=self.lego_id, linkIndex=-1, rgbaColor = np.concatenate([np.random.rand(3), np.array([1.])]))
            p.changeVisualShape(objectUniqueId=self.torus_id, linkIndex=-1, rgbaColor = np.concatenate([np.random.rand(3), np.array([1.])]))
            p.changeVisualShape(objectUniqueId=self.duck_id, linkIndex=-1, rgbaColor = np.concatenate([np.random.rand(3), np.array([1.])]))


        # Sample and distractors displacement(s) from center
        if distractor:
            if randomize_mug_pos:
                disp_x_max = 0.01
                disp_y_max = 0.01
                disps = []
                disps.append(np.array([np.random.rand() * disp_x_max, np.random.rand() * disp_y_max, 0.]))
                disps.append(np.array([-np.random.rand() * disp_x_max, np.random.rand() * disp_y_max, 0.]))
                disps.append(np.array([np.random.rand() * disp_x_max, -np.random.rand() * disp_y_max, 0.]))
                disps.append(np.array([-np.random.rand() * disp_x_max, -np.random.rand() * disp_y_max, 0.]))
            else:
                disps = []
                disps.append(np.array([0., 0., 0.]))
                disps.append(np.array([0., 0., 0.]))
                disps.append(np.array([0., 0., 0.]))
                disps.append(np.array([0., 0., 0.]))


        # Origin pos for mug (and distractors)
        if distractor:
            global_offset = np.array([0.0, 0., 0.])
            dx = 0.12
            dy = 0.12
            origins = [self.center + global_offset + np.array([dx, dy, 0.]),
                       self.center + global_offset + np.array([-dx, dy, 0.]),
                       self.center + global_offset + np.array([dx, -dy, 0.]),
                       self.center + global_offset + np.array([-dx, -dy, 0.])]
            

        # Allocate origin and disp for mugs and distractors
        if distractor:
            idx = list(range(4))
            np.random.shuffle(idx)

            bunny_disp = disps[idx[0]]
            lego_disp = disps[idx[1]]
            duck_disp = disps[idx[2]]
            torus_disp =  disps[idx[3]]

            bunny_origin = origins[idx[0]]
            lego_origin = origins[idx[1]]
            duck_origin = origins[idx[2]]
            torus_origin = origins[idx[3]]



        # Reset mug
        mug_pos = mug_origin + mug_disp
        if self.support_box_id is not None:
            # First reset the support
            support_box_pos = mug_pos + np.array([0., 0., 2.])
            support_box_pos[-1] = stable_z(self.support_box_id, self.table_id) - self.support_box_h + (np.random.rand() * 0.07 * randomize_mug_pos) + (0.08* (mug_pose == 'lying'))
            p.resetBasePositionAndOrientation(self.support_box_id, support_box_pos, np.array([0., 0., 0., 1.]), physicsClientId = self.physicsClientId)

            # Reset mug on top of the support
            mug_pos = support_box_pos + np.array([0., 0., 4.])
            p.resetBasePositionAndOrientation(self.mug_id, mug_pos, mug_orn, physicsClientId = self.physicsClientId)
            mug_pos[2] = stable_z(self.mug_id, self.support_box_id)
            p.resetBasePositionAndOrientation(self.mug_id, mug_pos, mug_orn, physicsClientId = self.physicsClientId)
        else:
            mug_pos = mug_pos + np.array([0., 0., 4.])
            p.resetBasePositionAndOrientation(self.mug_id, mug_pos, mug_orn, physicsClientId = self.physicsClientId)
            mug_pos[2] = stable_z(self.mug_id, self.table_id)
            p.resetBasePositionAndOrientation(self.mug_id, mug_pos, mug_orn, physicsClientId = self.physicsClientId)
        

        # Reset distractor
        if distractor:
            bunny_pos = bunny_origin + bunny_disp + np.array([0., 0., 4.])
            bunny_pos[2] = stable_z(self.bunny_id, self.table_id)
            bunny_orn = p.getQuaternionFromEuler([0., 0., np.random.rand()*np.pi*2])

            lego_pos = lego_origin + lego_disp + np.array([0., 0., 4.])
            lego_pos[2] = stable_z(self.lego_id, self.table_id)
            lego_orn = p.getQuaternionFromEuler([0., 0., np.random.rand()*np.pi*2])

            duck_pos = duck_origin + duck_disp + np.array([0., 0., 4.])
            duck_pos[2] = stable_z(self.duck_id, self.table_id)
            duck_orn = p.getQuaternionFromEuler([0., 0., np.random.rand()*np.pi*2])

            torus_pos = torus_origin + torus_disp + np.array([0., 0., 4.])
            torus_pos[2] = stable_z(self.torus_id, self.table_id)
            torus_orn = p.getQuaternionFromEuler([0., 0., np.random.rand()*np.pi*2])

            p.resetBasePositionAndOrientation(self.bunny_id, bunny_pos, bunny_orn, physicsClientId = self.physicsClientId)
            p.resetBasePositionAndOrientation(self.lego_id, lego_pos, lego_orn, physicsClientId = self.physicsClientId)
            p.resetBasePositionAndOrientation(self.duck_id, duck_pos, duck_orn, physicsClientId = self.physicsClientId)
            p.resetBasePositionAndOrientation(self.torus_id, torus_pos, torus_orn, physicsClientId = self.physicsClientId)

        # Reset Hook
        hook_pos = self.center + np.array([0.13, 0.0 ,0.])
        if randomize_hook_pos:
            hook_pos = hook_pos + np.array([0., (2*np.random.rand() - 1) * 0.05, (2*np.random.rand() - 1) * 0.01])
        hook_orn = np.array([0., 0., np.pi]) + np.array([0., 0., (2*np.random.rand() - 1) * np.pi / 8])
        hook_orn = p.getQuaternionFromEuler(hook_orn)
        if arbitrary_tray:
            theta_ = np.random.rand() * np.pi/12
            dOrn = p.getQuaternionFromEuler(np.array([0, theta_, 0]))
            hook_orn = np.array(p.multiplyTransforms(np.zeros(3),hook_orn, np.zeros(3), dOrn)[-1])
            hook_pos[-1] += 0.03
        p.resetBasePositionAndOrientation(self.hook_id, hook_pos, hook_orn, physicsClientId = self.physicsClientId)


        # Disable collision for visual distractors with robot
        for i in range(12):
            #p.setCollisionFilterPair(self.hook_id, self.robot, -1, i, 0, self.physicsClientId)
            #p.setCollisionFilterPair(self.hook_id, self.robot, 0, i, 0, self.physicsClientId)
            if self.support_box_id is not None:
                p.setCollisionFilterPair(self.support_box_id, self.robot, -1, i, 0, self.physicsClientId)
            if distractor:
                p.setCollisionFilterPair(self.bunny_id, self.robot, -1, i, 0, self.physicsClientId)
                p.setCollisionFilterPair(self.lego_id, self.robot, -1, i, 0, self.physicsClientId)
                p.setCollisionFilterPair(self.duck_id, self.robot, -1, i, 0, self.physicsClientId)
                p.setCollisionFilterPair(self.torus_id, self.robot, -1, i, 0, self.physicsClientId)
        if distractor:
            p.setCollisionFilterPair(self.bunny_id, self.mug_id, -1, -1, 0, self.physicsClientId)
            p.setCollisionFilterPair(self.lego_id, self.mug_id, -1, -1, 0, self.physicsClientId)
            p.setCollisionFilterPair(self.duck_id, self.mug_id, -1, -1, 0, self.physicsClientId)
            p.setCollisionFilterPair(self.torus_id, self.mug_id, -1, -1, 0, self.physicsClientId)


        p.changeDynamics(self.mug_id, -1, linearDamping = 0.3, angularDamping = 0.3)
        #p.changeDynamics(self.mug_id, -1, contactStiffness = 100000000., contactDamping = 100.)

        # step sim
        if step:
            step = int(step/self.freq)
            for _ in range(step):
                p.stepSimulation(physicsClientId = self.physicsClientId)


    def get_state(self):
        state = super().get_state()

        # Suffix guide
        # ==>  m: Mug, h: Handle, dg: Desired grasp sweet spot, de: Desired End Effector, pre: Pregrasp
        X_sm, R_sm = p.getBasePositionAndOrientation(self.mug_id, physicsClientId = self.physicsClientId)
        X_sm, R_sm = np.array(X_sm), Rotation.from_quat(R_sm).as_matrix()
        X_s_hook, R_s_hook = p.getBasePositionAndOrientation(self.hook_id, physicsClientId = self.physicsClientId)
        X_s_hook, R_s_hook = np.array(X_s_hook), Rotation.from_quat(R_s_hook).as_matrix()
        # X_s_branch, R_s_branch = p.getLinkState(self.hook_id, self.branchLinkId)[:2]
        # X_s_branch, R_s_branch = np.array(X_s_branch), Rotation.from_quat(R_s_branch).as_matrix()
        # R_s_tip = R_s_branch.copy()
        # X_s_tip = X_s_branch + (R_s_branch @ np.array([0., 0., self.branch_length]))
        X_s_branch, R_s_branch = X_s_hook.copy(), R_s_hook.copy()
        R_s_tip, X_s_tip = R_s_hook.copy(), X_s_hook.copy()

        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_sh, R_sh = X_sm + (R_sm @ X_mh), R_sm @ R_mh

        state['T_sm'] = (X_sm, R_sm)
        state['T_sh'] = (X_sh, R_sh)
        state['T_s_hook'] = (X_s_hook, R_s_hook)
        state['T_s_tip'] = (X_s_tip, R_s_tip)
        state['T_s_branch'] = (X_s_branch, R_s_branch)

        return state

#     Z-axis
#       ↑
    #       #
    #       ####
    #       #  # -> Y-axis
    #       ####
    #########


    def oracle_pick_rim(self, mod):
        state = self.get_state()
        X_sm, R_sm = state['T_sm']
        X_se, R_se = state['T_se']

        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_m_top, R_m_top = self.X_m_top, self.R_m_top

        
        #yaw = -np.pi / 2 # 0 ~ pi
        #yaw = yaw + (2*np.random.rand()-1)*np.pi*2/3
        #yaw = yaw + np.pi/2

        # yaw_mod_1 = 0.
        # yaw_mod_1 += (2*np.random.rand()-1)*np.pi*1/6
        # yaw_mod_2 = np.pi 
        # yaw_mod_2 += (2*np.random.rand()-1)*np.pi*1/6

        # yaw = yaw_mod_1*mod + yaw_mod_2*(1-mod)

        # R_top_rim = p.getMatrixFromQuaternion(p.getQuaternionFromEuler(np.array([0., 0., yaw])))
        # R_top_rim = np.array(R_top_rim).reshape(3,3)
        # rim_X_top_rim = np.array([0.035, 0., 0.])
        # X_top_rim = R_top_rim @ rim_X_top_rim
        # X_rim_dg = np.array([0., 0., -0.01]) 
        # R_rim_dg = np.array([[0. ,1. ,0.],
        #                      [1. ,0. ,0.],
        #                      [0. ,0. ,-1.]])

        # R_sdg = R_sm @ R_m_top @ R_top_rim @ R_rim_dg
        # X_top_dg = X_top_rim + (R_top_rim @ X_rim_dg)
        # X_mdg = X_m_top + (R_m_top @ X_top_dg)
        # X_sdg = X_sm + (R_sm @ X_mdg)

        # if mod == 0:
        #     R_mdg = np.array([[-1,0,0],[0,1,0],[0,0,-1]]).T
        # if mod == 1:
        #     R_mdg = np.array([[0,1,0],[1,0,0],[0,0,-1]]).T
        # if mod == 2:
        #     R_mdg = np.array([[1,0,0],[0,-1,0],[0,0,-1]]).T
        # if mod == 3:
        #     R_mdg = np.array([[0,-1,0],[-1,0,0],[0,0,-1]]).T


        # theta = 0
        # R_mdg = np.array([[0,0,1],[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),-1]]).T




        # theta = (2*np.random.rand() - 1) * np.pi / 2
        # R_sdg = np.array([[0,0,1],[np.sin(theta),-np.cos(theta),0],[np.cos(theta), np.sin(theta), 0]]).T
        theta = np.random.choice([0,1,2,3]) * np.pi/2
        R_mdg = np.array([[0,0,1],[np.sin(theta),-np.cos(theta),0],[np.cos(theta), np.sin(theta), 0]]).T
        R_sdg = R_sm @ R_mdg

        X_mdg = np.array([0., 0., (2*np.random.rand() - 1) *0.01])

        X_sdg = X_sm + X_mdg
        #R_sdg = R_sm @ R_mdg


        perturb_axis = np.random.randn(3)
        perturb_axis = perturb_axis / np.linalg.norm(perturb_axis)
        perturb_angle = (2*np.random.rand()-1) * (np.pi/180 * 5)
        R_perturb = Rotation.from_rotvec(perturb_angle * perturb_axis).as_matrix()
        R_sdg = R_sdg @ R_perturb


        pre_grasp = None
        grasp = X_sdg, R_sdg

        return pre_grasp, grasp

    def oracle_pick_handle(self, random_180_flip = False, force_flip = False):
        state = self.get_state()
        X_sm, R_sm = state['T_sm']
        X_se, R_se = state['T_se']

        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_m_top, R_m_top = self.X_m_top, self.R_m_top

        X_hdg = np.array([0., 0.03, 0.0]) * self.mug_scale + (2*np.random.rand(3)-1)*np.array([0., 0., 0.01])
        R_hdg = np.array([[0. ,0. ,1.],
                          [0. ,1. ,0.],
                          [-1. ,0. ,0.]])

        R_sdg = R_sm @ R_mh @ R_hdg

        flip = False
        z_axis = R_sdg[:,-1]
        if z_axis[2] > 0.8:
            flip = True
        elif z_axis[0] < 0:
            flip = True
        if force_flip:
            flip = True

        if flip is True:
            X_hdg = X_hdg - np.array([0.01, 0., 0.])

            R_flip = np.array([[-1. ,0. ,0.],
                               [0. ,1. ,0.],
                               [0. ,0. ,-1.]])
            R_sdg = R_sdg @ R_flip
        else:
            X_hdg = X_hdg + np.array([0.01, 0., 0.])

        X_mdg = X_mh + (R_mh @ X_hdg)
        X_sdg = X_sm + (R_sm @ X_mdg)

        perturb_axis = np.random.randn(3)
        perturb_axis = perturb_axis / np.linalg.norm(perturb_axis)
        perturb_angle = (2*np.random.rand()-1) * (np.pi/180 * 5)
        R_perturb = Rotation.from_rotvec(perturb_angle * perturb_axis).as_matrix()
        R_sdg = R_sdg @ R_perturb

        R_dg_dgpre = np.eye(3)
        R_s_dgpre = R_sdg @ R_dg_dgpre
        X_dg_dgpre = np.array([0., 0., -0.03])
        sX_dg_dgpre = R_sdg @ X_dg_dgpre
        X_s_dgpre = X_sdg + sX_dg_dgpre

        if random_180_flip:
            R_flip = np.array([[-1., 0., 0.],[0., -1., 0.],[0., 0., 1.]]).T
            if np.random.randint(2) == 1:
                R_s_dgpre = R_s_dgpre @ R_flip
                R_sdg = R_sdg @ R_flip
        pre_grasp = (X_s_dgpre, R_s_dgpre)
        grasp = X_sdg, R_sdg

        return pre_grasp, grasp

    def oracle_place_hole(self):
        state = self.get_state()
        X_sm, R_sm = state['T_sm']
        X_se, R_se = state['T_se']
        X_s_tip, R_s_tip = state['T_s_tip']
        R_sg = R_se @ self.R_eg
        X_sg = X_se + (R_se @ self.X_eg)

        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_m_top, R_m_top = self.X_m_top, self.R_m_top

        X_dtop_tip = np.array([0., 0., -0.03])
        R_tip_dtop = np.array([[-1.         ,0.         ,0.],
                               [0.        ,1.         ,0.],
                               [0.         ,0.        ,-1.]])
        X_tip_dtop = -R_tip_dtop @ X_dtop_tip
        R_s_dtop = R_s_tip @ R_tip_dtop
        X_s_dtop = X_s_tip + (R_s_tip @ X_tip_dtop)

        R_s_top = R_sm @ R_m_top
        X_s_top = X_sm + (R_sm @ X_m_top)
        R_top_g = (R_s_top).T @ R_sg
        X_top_g = R_s_top.T @ (X_sg - X_s_top)
        R_dtop_dg, X_dtop_dg = R_top_g, X_top_g

        R_sdg = R_s_dtop @ R_dtop_dg
        X_sdg = X_s_dtop + (R_s_dtop @ X_dtop_dg)

        R_dg_dgpre = np.eye(3)
        R_s_dgpre = R_sdg @ R_dg_dgpre
        X_dg_dgpre = np.array([0., 0., -0.03])
        sX_dg_dgpre = R_sdg @ X_dg_dgpre
        X_s_dgpre = X_sdg + sX_dg_dgpre

        pre_place = (X_s_dgpre, R_s_dgpre)
        place = X_sdg, R_sdg
        #place = X_s_dtop, R_s_dtop
        

        return pre_place, place

    def oracle_place_handle(self, flip_x = False, theta = 0.):
        state = self.get_state()
        X_sm, R_sm = state['T_sm']
        X_se, R_se = state['T_se']
        X_s_tip, R_s_tip = state['T_s_tip']
        R_sg = R_se @ self.R_eg
        X_sg = X_se + (R_se @ self.X_eg)
        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_sh, R_sh = X_sm + (R_sm @ X_mh), R_sm @ R_mh

        X_gh = R_sg.T @ (X_sh - X_sg)
        X_dgdh = X_gh
        R_dgdh = R_sg.T @ R_sh
        X_dhdg = - R_dgdh.T @ X_dgdh

        # R_tip_dh = np.array([[0.        ,0.          ,-1.],
        #                      [0.         ,1.          ,0.],
        #                      [1.         ,0.         ,0.]])

        #theta = np.pi*(3./12)
        R_tip_dh = np.array([[-np.sin(theta)        ,0.0          ,-np.cos(theta)],
                             [0.0         ,-1.0          ,0.0],
                             [-np.cos(theta)         ,0.         ,np.sin(theta)]])
        if flip_x == True:
            R_tip_flip = np.array([[-1.        ,0.          ,0.],
                                    [0.         ,1.          ,0.],
                                    [0.         ,0.         ,-1.]])
            R_tip_dh = R_tip_flip @ R_tip_dh
        X_tip_dh = np.array([0., 0., 0.01]) # Depreceted: X_tip_dh = np.array([0., 0., -0.01])
        R_tip_dg = R_tip_dh @ R_dgdh.T
        X_tip_dg = X_tip_dh + (R_tip_dh @ X_dhdg)
        X_tip_dg = X_tip_dg + np.array([-0.01, 0., 0.])
        X_dg_tip = -(R_tip_dg.T @ X_tip_dg)


        X_gm = R_sg.T @ (X_sm - X_sg)
        if X_gm[1] < 0.:
            y_offset = -0.00 * self.mug_scale
        elif X_gm[1] > 0.:
            y_offset = 0.00 * self.mug_scale
        X_dg_tip = X_dg_tip + np.array([0., y_offset, 0.])

        X_tip_dg = -R_tip_dg @ X_dg_tip
        R_sdg = R_s_tip @ R_tip_dg
        X_sdg = X_s_tip + (R_s_tip @ X_tip_dg)

        R_dg_dgpre = np.eye(3)
        R_s_dgpre = R_sdg @ R_dg_dgpre
        X_dg_dgpre = np.array([0., 0., -0.03])
        sX_dg_dgpre = R_sdg @ X_dg_dgpre
        X_s_dgpre = X_sdg + sX_dg_dgpre

        pre_place = (X_s_dgpre, R_s_dgpre)
        place = X_sdg, R_sdg
        #place = X_s_dtop, R_s_dtop
        
        # R_sdh = R_s_tip @ R_tip_dh
        # X_sdh = X_s_tip + R_s_tip @ X_tip_dh
        # place = (X_sdh, R_sdh)
        raise NotImplementedError # Currently not used. #TODO: remove this
        return pre_place, place

    def oracle_place_handle_horizontal(self, mod, low_var = False):
        state = self.get_state()
        X_sm, R_sm = state['T_sm']
        X_se, R_se = state['T_se']
        X_s_tip, R_s_tip = state['T_s_tip']
        X_s_base, R_s_base = state['T_s_hook']
        R_sg = R_se @ self.R_eg
        X_sg = X_se + (R_se @ self.X_eg)
        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_sh, R_sh = X_sm + (R_sm @ X_mh), R_sm @ R_mh

        X_gh = R_sg.T @ (X_sh - X_sg)
        X_dgdh = X_gh
        R_dgdh = R_sg.T @ R_sh
        X_dhdg = - R_dgdh.T @ X_dgdh

        # if mod:
        #     # theta = -np.pi*np.random.rand() # -pi~0
        #     offset = np.pi/4
        #     if low_var:
        #         theta = -(np.pi-2*offset)*(0.5 + (np.random.rand()-0.5)*0.3) - offset # 30 percent of original variance
        #     else:
        #         theta = -(np.pi-2*offset)*np.random.rand() - offset # -pi/4 ~ -3pi/4
        #     R_base_dh = np.array([[ 0.0,    np.sin(theta),          np.cos(theta)],
        #                           [ 0.0,   -np.cos(theta),          np.sin(theta)],
        #                           [ 1.0,              0.0,                    0.0]])
        # else:
        #     offset = np.pi/4
        #     if low_var:
        #         theta = (np.pi-2*offset)*(0.5 + (np.random.rand()-0.5)*0.3) + offset # 30 percent of original variance
        #     else:
        #         theta = (np.pi-2*offset)*np.random.rand() + offset # pi/4 ~ 3pi/4
        #     R_base_dh = np.array([[ 0.0,   -np.sin(theta),          np.cos(theta)],
        #                           [ 0.0,    np.cos(theta),          np.sin(theta)],
        #                           [-1.0,              0.0,                    0.0]])

        # R_sdg = R_s_base @ R_base_dh @ R_dgdh.T

        theta = (2*np.random.rand() - 1) * 2* np.pi
        R_sdg = np.array([[ -np.cos(theta),   -np.sin(theta),          0],
                                [ -np.sin(theta),    np.cos(theta),          0],
                                [0.0,              0.0,                    -1.0]])
        X_sdg = X_s_base + np.array([0., 0., 0.03]) + np.array([0., 0., 0.01]) * (2*np.random.rand() - 1)



        # sX_tip_dh = np.array([0., 0., 0.00]) # np.array([0., 0., 0.02])
        # R_sdh = R_sdg @ R_dgdh
        # sX_dhdg = R_sdh @ X_dhdg
        # X_sdg = X_s_tip + sX_tip_dh + sX_dhdg

        # R_dg_dgpre = np.eye(3)
        # R_s_dgpre = R_sdg @ R_dg_dgpre
        # X_dg_dgpre = np.array([0., 0., -0.03])
        # sX_dg_dgpre = R_sdg @ X_dg_dgpre
        # X_s_dgpre = X_sdg + sX_dg_dgpre

        # pre_place = (X_s_dgpre, R_s_dgpre)
        pre_place = None
        place = (X_sdg, R_sdg)

        return pre_place, place

    def oracle_place_handle_deprecated(self):
        state = self.get_state()
        X_sm, R_sm = state['T_sm']
        X_se, R_se = state['T_se']
        X_s_tip, R_s_tip = state['T_s_tip']
        R_sg = R_se @ self.R_eg
        X_sg = X_se + (R_se @ self.X_eg)
        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_sh, R_sh = X_sm + (R_sm @ X_mh), R_sm @ R_mh

        X_gh = R_sg.T @ (X_sh - X_sg)
        X_dgdh = X_gh
        R_dgdh = R_sg.T @ R_sh
        X_dhdg = - R_dgdh.T @ X_dgdh

        R_tip_dg = np.array([[-1.        ,0.          ,0.],
                             [0.         ,1.          ,0.],
                             [0.         ,0.         ,-1.]])
        X_tip_dh = np.array([0., 0., -0.02])
        R_tip_dh = R_tip_dg @ R_dgdh
        X_tip_dg = X_tip_dh + (R_tip_dh @ X_dhdg)
        X_tip_dg = X_tip_dg + np.array([-0.01, 0., 0.])
        X_dg_tip = -(R_tip_dg.T @ X_tip_dg)


        X_gm = R_sg.T @ (X_sm - X_sg)
        if X_gm[1] < 0.:
            y_offset = -0.00 * self.mug_scale
        elif X_gm[1] > 0.:
            y_offset = 0.00 * self.mug_scale
        X_dg_tip = X_dg_tip + np.array([0., y_offset, 0.])

        X_tip_dg = -R_tip_dg @ X_dg_tip
        R_sdg = R_s_tip @ R_tip_dg
        X_sdg = X_s_tip + (R_s_tip @ X_tip_dg)

        R_dg_dgpre = np.eye(3)
        R_s_dgpre = R_sdg @ R_dg_dgpre
        X_dg_dgpre = np.array([0., 0., -0.1])
        sX_dg_dgpre = R_sdg @ X_dg_dgpre
        X_s_dgpre = X_sdg + sX_dg_dgpre

        pre_place = (X_s_dgpre, R_s_dgpre)
        place = X_sdg, R_sdg
        #place = X_s_dtop, R_s_dtop

        return pre_place, place

    def debug(self, debug_items = ['grasp']):
        super().debug(debug_items=debug_items)
        for item in debug_items:
            if item == 'mug':
                mugFrame_ID = axiscreator(self.mug_id, physicsClientId = self.physicsClientId)
            elif item == 'mug_rim':
                topFrame_ID = axiscreator(self.mug_id, offset = self.X_m_top.copy(), physicsClientId = self.physicsClientId)
            elif item == 'mug_handle':
                handleFrame_ID = axiscreator(self.mug_id, offset = self.X_mh.copy(), physicsClientId = self.physicsClientId)
            elif item == 'hook_branch':
                hookFrame_ID = axiscreator(self.hook_id, self.branchLinkId, physicsClientId = self.physicsClientId)
            elif item == 'hook_branch_offset':
                axiscreator(self.hook_id, self.branchLinkId, offset = np.array([0., 0., 0.07]),physicsClientId = self.physicsClientId)
            elif item == 'distractors':
                axiscreator(self.bunny_id, physicsClientId = self.physicsClientId)
                axiscreator(self.duck_id, physicsClientId = self.physicsClientId)
                axiscreator(self.torus_id, physicsClientId = self.physicsClientId)
                axiscreator(self.lego_id, physicsClientId = self.physicsClientId)
            elif item == 'hook_base':
                hookBaseFrame_ID = axiscreator(self.hook_id, physicsClientId = self.physicsClientId)

    def check_pick_success(self):
        state = self.get_state()
        X_sm, R_sm = state['T_sm']
        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_sh, R_sh = X_sm + (R_sm @ X_mh), R_sm @ R_mh
        if X_sh[2] > 0.25:
            return True
        else:
            return False

    def check_place_success(self):
        p.setCollisionFilterPair(self.target_item, self.table_id, -1, -1, 0, self.physicsClientId)
        p.setCollisionFilterPair(self.target_item, self.plane_id, -1, -1, 0, self.physicsClientId)
        if self.support_box_id is not None:
            p.setCollisionFilterPair(self.target_item, self.support_box_id, -1, -1, 0, self.physicsClientId)
        for i in range(12):
            p.setCollisionFilterPair(self.target_item, self.robot, -1, i, 0, self.physicsClientId)

        for _ in range(int(2/self.freq)):
            p.stepSimulation(physicsClientId = self.physicsClientId)

        state = self.get_state()
        X_sh = state['T_sh'][0]
        if X_sh[-1] > 0.:
            result = True
        else:
            result = False

        p.setCollisionFilterPair(self.target_item, self.table_id, -1, -1, 1, self.physicsClientId)
        p.setCollisionFilterPair(self.target_item, self.plane_id, -1, -1, 1, self.physicsClientId)
        if self.support_box_id is not None:
            p.setCollisionFilterPair(self.target_item, self.support_box_id, -1, -1, 1, self.physicsClientId)
        for i in range(12):
            p.setCollisionFilterPair(self.target_item, self.robot, -1, i, 1, self.physicsClientId)

        return result











































































class BowlTask(FrankaTask):
    def __init__(self, use_gui = True):
        super().__init__(use_gui=use_gui)

    def init_task(self, mug_type = 'default', distractor = False, use_support = False, **kwargs):
        super().init_task()

        self.mug_scale = 1.
        self.X_mh = np.array([0., 0.0, -0.03]) * (self.mug_scale) # mug to handle
        self.R_mh = np.eye(3)
        self.X_eg = np.array([0, 0, 0.105]) # EE to gripper sweet-spot
        self.R_eg = np.eye(3)
        self.X_m_top = np.array([0., 0., 0.01]) * self.mug_scale
        self.R_m_top = np.eye(3)
        self.branch_length = 0.065
        self.branchLinkId = -1

        if mug_type == 'default':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bowls/bowl0/bowl.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
        elif mug_type == 'cup1':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bowls/bowl1/bowl.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0., 0.065, 0.01]) * (self.mug_scale)
        elif mug_type == 'cup2':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bowls/bowl2/bowl.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0.0, 0.075, 0.01]) * (self.mug_scale)
        elif mug_type == 'cup3':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bowls/bowl3/bowl.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([-0.0, 0.07, 0.015]) * (self.mug_scale)
        elif mug_type == 'cup4':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bowls/bowl4/bowl.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0.0, 0.07, 0.015]) * (self.mug_scale)
        elif mug_type == 'cup5':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bowls/bowl5/bowl.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([-0.0, 0.07, 0.01]) * (self.mug_scale)
        elif mug_type == 'cup6':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bowls/bowl6/bowl.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0., 0.06, 0.015]) * (self.mug_scale)
        elif mug_type == 'cup7':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bowls/bowl7/bowl.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0., 0.08, 0.01]) * (self.mug_scale)
        elif mug_type == 'cup8':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bowls/bowl8/bowl.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0.0, 0.06, 0.01]) * (self.mug_scale)
        elif mug_type == 'cup9':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bowls/bowl9/bowl.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0.0, 0.06, 0.03]) * (self.mug_scale)
        elif mug_type == 'cup10':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bowls/bowl10/bowl.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0.0, 0.05, 0.01]) * (self.mug_scale)
        else:
            raise KeyError
        p.changeDynamics(self.mug_id, -1, lateralFriction=0.8, rollingFriction=0.3, spinningFriction=0.3)
        self.hook_scale = 1.
        self.hook_id = p.loadURDF("assets/traybox.urdf", basePosition=[0.5, 0., 0.], globalScaling=0.15 * self.hook_scale, physicsClientId = self.physicsClientId, useFixedBase = True)

        if use_support:
            # self.support_box_id = p.loadURDF("assets/traysupport.urdf", basePosition=[0.5, 0., 0.3], globalScaling=0.2, physicsClientId = self.physicsClientId, useFixedBase = True)
            self.support_box_h = 0.3
            self.support_box_id = create_box(w=0.15, l=0.15, h=self.support_box_h, color=(72/255,72/255,72/255,1.))
            p.changeDynamics(self.support_box_id, -1, lateralFriction = 3., rollingFriction=3., spinningFriction=3.)
        else:
            self.support_box_id = None

        if distractor:
            self.lego_scale = 2.5
            self.lego_id = p.loadURDF("assets/distractor/lego.urdf", basePosition=[0.7, -0.2, 0.35], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), globalScaling=self.lego_scale, physicsClientId = self.physicsClientId)
            self.duck_scale = 0.07
            self.duck_id = p.loadURDF("assets/distractor/duck.urdf", basePosition=[0.7, 0.2, 0.35], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), globalScaling=self.duck_scale, physicsClientId = self.physicsClientId)
            self.torus_scale = 0.07
            self.torus_id = p.loadURDF("assets/distractor/torus_textured.urdf", basePosition=[0.5, -0.2, 0.35], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), globalScaling=self.torus_scale, physicsClientId = self.physicsClientId)
            self.bunny_scale = 0.07
            self.bunny_id = p.loadURDF("assets/distractor/bunny.urdf", basePosition=[0.5, 0.2, 0.35], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), globalScaling=self.torus_scale, physicsClientId = self.physicsClientId)


    def reset(self, seed = None, mug_pose = 'upright', mug_type = 'default', distractor = False, use_support = False, step=1):
        if mug_type == 'cups':
            mug_type = f'cup{seed%10 + 1}'
        super().reset(seed = seed, step = False, mug_type=mug_type, distractor=distractor, use_support=use_support)

        randomize_mug_pos = True
        randomize_hook_pos = True

        self.target_item = self.mug_id

        if mug_pose == 'arbitrary' or mug_pose == 'lying':
            arbitrary_tray = True
        else:
            arbitrary_tray = False

        if mug_pose == 'arbitrary':
            #if seed %2 == 0:
            if np.random.rand() > 0.5:
                mug_pose = 'lying'
            else:
                mug_pose = 'upright'


        # Reset cup orientation
        if mug_pose == 'upright':
            mug_orn = np.array([0., 0., np.random.rand()*np.pi*2])
            mug_orn = p.getQuaternionFromEuler(mug_orn)
        elif mug_pose == 'lying':
            mug_orn = np.array([np.pi /2, 0., -np.random.rand()*np.pi])
            mug_orn = p.getQuaternionFromEuler(mug_orn)
            mug_orn = multiply_quats(mug_orn, p.getQuaternionFromEuler(np.array([0., 0., (2*np.random.rand()-1)*np.pi*1/2])))
        else:
            raise KeyError


        # Sample and mug displacement(s) from center
        if randomize_mug_pos:
            if distractor is True:
                disp_x_abs_max = 0.01
                disp_y_abs_max = 0.01
            elif mug_pose == 'upright':
                disp_x_abs_max = 0.05
                disp_y_abs_max = 0.05
            elif mug_pose == 'lying':
                disp_x_abs_max = 0.03
                disp_y_abs_max = 0.03
            mug_disp = np.array([(2*np.random.rand() - 1) * disp_x_abs_max, (2*np.random.rand() - 1) * disp_y_abs_max, 0.])
        else:
            mug_disp = np.array([0., 0., 0.])
        mug_origin = self.center + np.array([-0.05, 0., 0.])

        if distractor:
            p.changeVisualShape(objectUniqueId=self.bunny_id, linkIndex=-1, rgbaColor = np.concatenate([np.random.rand(3), np.array([1.])]))
            p.changeVisualShape(objectUniqueId=self.lego_id, linkIndex=-1, rgbaColor = np.concatenate([np.random.rand(3), np.array([1.])]))
            p.changeVisualShape(objectUniqueId=self.torus_id, linkIndex=-1, rgbaColor = np.concatenate([np.random.rand(3), np.array([1.])]))
            p.changeVisualShape(objectUniqueId=self.duck_id, linkIndex=-1, rgbaColor = np.concatenate([np.random.rand(3), np.array([1.])]))


        # Sample and distractors displacement(s) from center
        if distractor:
            if randomize_mug_pos:
                disp_x_max = 0.01
                disp_y_max = 0.01
                disps = []
                disps.append(np.array([np.random.rand() * disp_x_max, np.random.rand() * disp_y_max, 0.]))
                disps.append(np.array([-np.random.rand() * disp_x_max, np.random.rand() * disp_y_max, 0.]))
                disps.append(np.array([np.random.rand() * disp_x_max, -np.random.rand() * disp_y_max, 0.]))
                disps.append(np.array([-np.random.rand() * disp_x_max, -np.random.rand() * disp_y_max, 0.]))
            else:
                disps = []
                disps.append(np.array([0., 0., 0.]))
                disps.append(np.array([0., 0., 0.]))
                disps.append(np.array([0., 0., 0.]))
                disps.append(np.array([0., 0., 0.]))


        # Origin pos for mug (and distractors)
        if distractor:
            global_offset = np.array([0.0, 0., 0.])
            dx = 0.12
            dy = 0.12
            origins = [self.center + global_offset + np.array([dx, dy, 0.]),
                       self.center + global_offset + np.array([-dx, dy, 0.]),
                       self.center + global_offset + np.array([dx, -dy, 0.]),
                       self.center + global_offset + np.array([-dx, -dy, 0.])]
            

        # Allocate origin and disp for mugs and distractors
        if distractor:
            idx = list(range(4))
            np.random.shuffle(idx)

            bunny_disp = disps[idx[0]]
            lego_disp = disps[idx[1]]
            duck_disp = disps[idx[2]]
            torus_disp =  disps[idx[3]]

            bunny_origin = origins[idx[0]]
            lego_origin = origins[idx[1]]
            duck_origin = origins[idx[2]]
            torus_origin = origins[idx[3]]



        # Reset mug
        mug_pos = mug_origin + mug_disp
        if self.support_box_id is not None:
            # First reset the support
            support_box_pos = mug_pos + np.array([0., 0., 2.])
            # support_box_pos[-1] = stable_z(self.support_box_id, self.table_id) + (np.random.rand() * 0.04 * randomize_mug_pos)
            support_box_pos[-1] = stable_z(self.support_box_id, self.table_id) - self.support_box_h + (np.random.rand() * 0.07 * randomize_mug_pos)
            p.resetBasePositionAndOrientation(self.support_box_id, support_box_pos, np.array([0., 0., 0., 1.]), physicsClientId = self.physicsClientId)

            # Reset mug on top of the support
            mug_pos = support_box_pos + np.array([(2*np.random.rand() -1)*0.01, (2*np.random.rand() -1)*0.01, 4.])
            p.resetBasePositionAndOrientation(self.mug_id, mug_pos, mug_orn, physicsClientId = self.physicsClientId)
            mug_pos[2] = stable_z(self.mug_id, self.support_box_id)
            p.resetBasePositionAndOrientation(self.mug_id, mug_pos, mug_orn, physicsClientId = self.physicsClientId)
        else:
            mug_pos = mug_pos + np.array([0., 0., 4.])
            p.resetBasePositionAndOrientation(self.mug_id, mug_pos, mug_orn, physicsClientId = self.physicsClientId)
            mug_pos[2] = stable_z(self.mug_id, self.table_id)
            p.resetBasePositionAndOrientation(self.mug_id, mug_pos, mug_orn, physicsClientId = self.physicsClientId)
        

        # Reset distractor
        if distractor:
            bunny_pos = bunny_origin + bunny_disp + np.array([0., 0., 4.])
            bunny_pos[2] = stable_z(self.bunny_id, self.table_id)
            bunny_orn = p.getQuaternionFromEuler([0., 0., np.random.rand()*np.pi*2])

            lego_pos = lego_origin + lego_disp + np.array([0., 0., 4.])
            lego_pos[2] = stable_z(self.lego_id, self.table_id)
            lego_orn = p.getQuaternionFromEuler([0., 0., np.random.rand()*np.pi*2])

            duck_pos = duck_origin + duck_disp + np.array([0., 0., 4.])
            duck_pos[2] = stable_z(self.duck_id, self.table_id)
            duck_orn = p.getQuaternionFromEuler([0., 0., np.random.rand()*np.pi*2])

            torus_pos = torus_origin + torus_disp + np.array([0., 0., 4.])
            torus_pos[2] = stable_z(self.torus_id, self.table_id)
            torus_orn = p.getQuaternionFromEuler([0., 0., np.random.rand()*np.pi*2])

            p.resetBasePositionAndOrientation(self.bunny_id, bunny_pos, bunny_orn, physicsClientId = self.physicsClientId)
            p.resetBasePositionAndOrientation(self.lego_id, lego_pos, lego_orn, physicsClientId = self.physicsClientId)
            p.resetBasePositionAndOrientation(self.duck_id, duck_pos, duck_orn, physicsClientId = self.physicsClientId)
            p.resetBasePositionAndOrientation(self.torus_id, torus_pos, torus_orn, physicsClientId = self.physicsClientId)

        # Reset Hook
        hook_pos = self.center + np.array([0.13, 0.0 ,0.])
        if randomize_hook_pos:
            hook_pos = hook_pos + np.array([0., (2*np.random.rand() - 1) * 0.05, (2*np.random.rand() - 1) * 0.01])
        hook_orn = np.array([0., 0., np.pi]) + np.array([0., 0., (2*np.random.rand() - 1) * np.pi / 8])
        hook_orn = p.getQuaternionFromEuler(hook_orn)
        if arbitrary_tray:
            theta_ = np.random.rand() * np.pi/12
            dOrn = p.getQuaternionFromEuler(np.array([0, theta_, 0]))
            hook_orn = np.array(p.multiplyTransforms(np.zeros(3),hook_orn, np.zeros(3), dOrn)[-1])
            hook_pos[-1] += 0.0
        p.resetBasePositionAndOrientation(self.hook_id, hook_pos, hook_orn, physicsClientId = self.physicsClientId)


        # Disable collision for visual distractors with robot
        for i in range(12):
            #p.setCollisionFilterPair(self.hook_id, self.robot, -1, i, 0, self.physicsClientId)
            #p.setCollisionFilterPair(self.hook_id, self.robot, 0, i, 0, self.physicsClientId)
            if self.support_box_id is not None:
                p.setCollisionFilterPair(self.support_box_id, self.robot, -1, i, 0, self.physicsClientId)
            if distractor:
                p.setCollisionFilterPair(self.bunny_id, self.robot, -1, i, 0, self.physicsClientId)
                p.setCollisionFilterPair(self.lego_id, self.robot, -1, i, 0, self.physicsClientId)
                p.setCollisionFilterPair(self.duck_id, self.robot, -1, i, 0, self.physicsClientId)
                p.setCollisionFilterPair(self.torus_id, self.robot, -1, i, 0, self.physicsClientId)
        if distractor:
            p.setCollisionFilterPair(self.bunny_id, self.mug_id, -1, -1, 0, self.physicsClientId)
            p.setCollisionFilterPair(self.lego_id, self.mug_id, -1, -1, 0, self.physicsClientId)
            p.setCollisionFilterPair(self.duck_id, self.mug_id, -1, -1, 0, self.physicsClientId)
            p.setCollisionFilterPair(self.torus_id, self.mug_id, -1, -1, 0, self.physicsClientId)


        p.changeDynamics(self.mug_id, -1, linearDamping = 0.3, angularDamping = 0.3)
        #p.changeDynamics(self.mug_id, -1, contactStiffness = 100000000., contactDamping = 100.)

        # step sim
        if step:
            step = int(step/self.freq)
            for _ in range(step):
                p.stepSimulation(physicsClientId = self.physicsClientId)

    def get_state(self):
        state = super().get_state()

        # Suffix guide
        # ==>  m: Mug, h: Handle, dg: Desired grasp sweet spot, de: Desired End Effector, pre: Pregrasp
        X_sm, R_sm = p.getBasePositionAndOrientation(self.mug_id, physicsClientId = self.physicsClientId)
        X_sm, R_sm = np.array(X_sm), Rotation.from_quat(R_sm).as_matrix()
        X_s_hook, R_s_hook = p.getBasePositionAndOrientation(self.hook_id, physicsClientId = self.physicsClientId)
        X_s_hook, R_s_hook = np.array(X_s_hook), Rotation.from_quat(R_s_hook).as_matrix()
        # X_s_branch, R_s_branch = p.getLinkState(self.hook_id, self.branchLinkId)[:2]
        # X_s_branch, R_s_branch = np.array(X_s_branch), Rotation.from_quat(R_s_branch).as_matrix()
        # R_s_tip = R_s_branch.copy()
        # X_s_tip = X_s_branch + (R_s_branch @ np.array([0., 0., self.branch_length]))
        X_s_branch, R_s_branch = X_s_hook.copy(), R_s_hook.copy()
        R_s_tip, X_s_tip = R_s_hook.copy(), X_s_hook.copy()

        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_sh, R_sh = X_sm + (R_sm @ X_mh), R_sm @ R_mh

        state['T_sm'] = (X_sm, R_sm)
        state['T_sh'] = (X_sh, R_sh)
        state['T_s_hook'] = (X_s_hook, R_s_hook)
        state['T_s_tip'] = (X_s_tip, R_s_tip)
        state['T_s_branch'] = (X_s_branch, R_s_branch)

        return state

#     Z-axis
#       ↑
    #       #
    #       ####
    #       #  # -> Y-axis
    #       ####
    #########


    def oracle_pick_rim(self, mod):
        state = self.get_state()
        X_sm, R_sm = state['T_sm']
        X_se, R_se = state['T_se']

        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_m_top, R_m_top = self.X_m_top, self.R_m_top

        
        #yaw = -np.pi / 2 # 0 ~ pi
        #yaw = yaw + (2*np.random.rand()-1)*np.pi*2/3
        #yaw = yaw + np.pi/2

        # yaw_mod_1 = 0.
        # yaw_mod_1 += (2*np.random.rand()-1)*np.pi*1/6
        # yaw_mod_2 = np.pi 
        # yaw_mod_2 += (2*np.random.rand()-1)*np.pi*1/6

        # yaw = yaw_mod_1*mod + yaw_mod_2*(1-mod)
        yaw = (2*np.random.rand()-1)*np.pi

        R_top_rim = p.getMatrixFromQuaternion(p.getQuaternionFromEuler(np.array([0., 0., yaw])))
        R_top_rim = np.array(R_top_rim).reshape(3,3)
        rim_X_top_rim = np.array([0.05, 0., 0.])
        X_top_rim = R_top_rim @ rim_X_top_rim
        X_rim_dg = np.array([0., 0., 0.]) 
        R_rim_dg = np.array([[0. ,1. ,0.],
                             [1. ,0. ,0.],
                             [0. ,0. ,-1.]])

        R_sdg = R_sm @ R_m_top @ R_top_rim @ R_rim_dg
        X_top_dg = X_top_rim + (R_top_rim @ X_rim_dg)
        X_mdg = X_m_top + (R_m_top @ X_top_dg)
        X_sdg = X_sm + (R_sm @ X_mdg)


        perturb_axis = np.random.randn(3)
        perturb_axis = perturb_axis / np.linalg.norm(perturb_axis)
        perturb_angle = (2*np.random.rand()-1) * (np.pi/180 * 5)
        R_perturb = Rotation.from_rotvec(perturb_angle * perturb_axis).as_matrix()
        R_sdg = R_sdg @ R_perturb

        R_dg_dgpre = np.eye(3)
        R_s_dgpre = R_sdg @ R_dg_dgpre
        X_dg_dgpre = np.array([0., 0., -0.03])
        sX_dg_dgpre = R_sdg @ X_dg_dgpre
        X_s_dgpre = X_sdg + sX_dg_dgpre

        pre_grasp = (X_s_dgpre, R_s_dgpre)
        grasp = X_sdg, R_sdg

        return pre_grasp, grasp

    def oracle_place_handle_horizontal(self, mod, low_var = False):
        state = self.get_state()
        X_sm, R_sm = state['T_sm']
        X_se, R_se = state['T_se']
        X_s_tip, R_s_tip = state['T_s_tip']
        X_s_base, R_s_base = state['T_s_hook']
        R_sg = R_se @ self.R_eg
        X_sg = X_se + (R_se @ self.X_eg)
        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_sh, R_sh = X_sm + (R_sm @ X_mh), R_sm @ R_mh

        X_gh = R_sg.T @ (X_sh - X_sg)
        X_dgdh = X_gh
        R_dgdh = R_sg.T @ R_sh
        X_dhdg = - R_dgdh.T @ X_dgdh

        # if mod:
        #     # theta = -np.pi*np.random.rand() # -pi~0
        #     offset = np.pi/4
        #     if low_var:
        #         theta = -(np.pi-2*offset)*(0.5 + (np.random.rand()-0.5)*0.3) - offset # 30 percent of original variance
        #     else:
        #         theta = -(np.pi-2*offset)*np.random.rand() - offset # -pi/4 ~ -3pi/4
        #     R_base_dh = np.array([[ 0.0,    np.sin(theta),          np.cos(theta)],
        #                           [ 0.0,   -np.cos(theta),          np.sin(theta)],
        #                           [ 1.0,              0.0,                    0.0]])
        # else:
        #     offset = np.pi/4
        #     if low_var:
        #         theta = (np.pi-2*offset)*(0.5 + (np.random.rand()-0.5)*0.3) + offset # 30 percent of original variance
        #     else:
        #         theta = (np.pi-2*offset)*np.random.rand() + offset # pi/4 ~ 3pi/4
        #     R_base_dh = np.array([[ 0.0,   -np.sin(theta),          np.cos(theta)],
        #                           [ 0.0,    np.cos(theta),          np.sin(theta)],
        #                           [-1.0,              0.0,                    0.0]])

        theta = np.random.rand() * 2 * np.pi
        R_base_dh = np.array([[ np.cos(theta),   -np.sin(theta),       0.0],
                              [ np.sin(theta),   np.cos(theta),        0.0],
                              [ 0.0,              0.0,                 1.0]])
                                

        R_sdg = R_s_base @ R_base_dh @ R_dgdh.T

        sX_tip_dh = np.array([0., 0., 0.00]) # np.array([0., 0., 0.02])
        R_sdh = R_sdg @ R_dgdh
        sX_dhdg = R_sdh @ X_dhdg
        X_sdg = X_s_tip + sX_tip_dh + sX_dhdg

        R_dg_dgpre = np.eye(3)
        R_s_dgpre = R_sdg @ R_dg_dgpre
        X_dg_dgpre = np.array([0., 0., -0.03])
        sX_dg_dgpre = R_sdg @ X_dg_dgpre
        X_s_dgpre = X_sdg + sX_dg_dgpre

        pre_place = (X_s_dgpre, R_s_dgpre)
        place = (X_sdg, R_sdg)

        return pre_place, place


    def debug(self, debug_items = ['grasp']):
        super().debug(debug_items=debug_items)
        for item in debug_items:
            if item == 'mug':
                mugFrame_ID = axiscreator(self.mug_id, physicsClientId = self.physicsClientId)
            elif item == 'mug_rim':
                topFrame_ID = axiscreator(self.mug_id, offset = self.X_m_top.copy(), physicsClientId = self.physicsClientId)
            elif item == 'mug_handle':
                handleFrame_ID = axiscreator(self.mug_id, offset = self.X_mh.copy(), physicsClientId = self.physicsClientId)
            elif item == 'hook_branch':
                hookFrame_ID = axiscreator(self.hook_id, self.branchLinkId, physicsClientId = self.physicsClientId)
            elif item == 'hook_branch_offset':
                axiscreator(self.hook_id, self.branchLinkId, offset = np.array([0., 0., 0.07]),physicsClientId = self.physicsClientId)
            elif item == 'distractors':
                axiscreator(self.bunny_id, physicsClientId = self.physicsClientId)
                axiscreator(self.duck_id, physicsClientId = self.physicsClientId)
                axiscreator(self.torus_id, physicsClientId = self.physicsClientId)
                axiscreator(self.lego_id, physicsClientId = self.physicsClientId)
            elif item == 'hook_base':
                hookBaseFrame_ID = axiscreator(self.hook_id, physicsClientId = self.physicsClientId)

    def check_pick_success(self):
        state = self.get_state()
        X_sm, R_sm = state['T_sm']
        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_sh, R_sh = X_sm + (R_sm @ X_mh), R_sm @ R_mh
        if X_sh[2] > 0.22:
            return True
        else:
            return False

    def check_place_success(self):
        p.setCollisionFilterPair(self.target_item, self.table_id, -1, -1, 0, self.physicsClientId)
        p.setCollisionFilterPair(self.target_item, self.plane_id, -1, -1, 0, self.physicsClientId)
        if self.support_box_id is not None:
            p.setCollisionFilterPair(self.target_item, self.support_box_id, -1, -1, 0, self.physicsClientId)
        for i in range(12):
            p.setCollisionFilterPair(self.target_item, self.robot, -1, i, 0, self.physicsClientId)

        for _ in range(int(2/self.freq)):
            p.stepSimulation(physicsClientId = self.physicsClientId)

        state = self.get_state()
        X_sh = state['T_sh'][0]
        if X_sh[-1] > 0.:
            result = True
        else:
            result = False

        p.setCollisionFilterPair(self.target_item, self.table_id, -1, -1, 1, self.physicsClientId)
        p.setCollisionFilterPair(self.target_item, self.plane_id, -1, -1, 1, self.physicsClientId)
        if self.support_box_id is not None:
            p.setCollisionFilterPair(self.target_item, self.support_box_id, -1, -1, 1, self.physicsClientId)
        for i in range(12):
            p.setCollisionFilterPair(self.target_item, self.robot, -1, i, 1, self.physicsClientId)

        return result

    def pick_plan(self, grasp): 
        X_sdg, R_sdg = grasp

        R_dg_dgpre = np.eye(3)
        R_sdg_pre = R_sdg @ R_dg_dgpre
        X_dg_dgpre = np.array([0., 0., -0.03])
        sX_dg_dgpre = R_sdg @ X_dg_dgpre
        X_sdg_pre = X_sdg + sX_dg_dgpre

        R_sde_pre = R_sdg_pre @ self.R_eg.T
        R_sde = R_sdg @ self.R_eg.T
        X_sde_pre = X_sdg_pre - R_sde_pre@self.X_eg
        X_sde = X_sdg - R_sde@self.X_eg


        pre_grasp = {'pose':(X_sde_pre, R_sde_pre), 'gripper_val': 0., 'col_check_items': [self.table_id, self.target_item, self.hook_id]}        # Changes
        grasp1 = {'pose':(X_sde, R_sde), 'gripper_val': 0., 'col_check_items': [self.table_id, self.target_item, self.hook_id]}                   # Changes
        grasp2 = {'pose':(X_sde, R_sde), 'gripper_val': 1., 'col_check_items': []}
        lift = {'pose':(X_sde + np.array([0., 0., 0.2]), R_sde), 'gripper_val': 1., 'col_check_items': []}

        return {'pre_grasp': pre_grasp, 'grasp1':grasp1, 'grasp2':grasp2, 'lift':lift}























































































































































































class BottleTask(FrankaTask):
    def __init__(self, use_gui = True):
        super().__init__(use_gui=use_gui)

    def init_task(self, mug_type = 'default0', distractor = False, use_support = False, **kwargs):
        super().init_task()

        self.mug_scale = 1.
        self.X_mh = np.array([0., 0.0, -0.03]) * (self.mug_scale) # mug to handle
        self.R_mh = np.eye(3)
        self.X_eg = np.array([0, 0, 0.105]) # EE to gripper sweet-spot
        self.R_eg = np.eye(3)
        self.X_m_top = np.array([0., 0., 0.15]) * self.mug_scale
        self.R_m_top = np.eye(3)
        self.branch_length = 0.065
        self.branchLinkId = -1

        if mug_type == 'default0':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bottles/bottle_train0/bottle.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            self.X_m_top = np.array([0., 0., 0.14]) * self.mug_scale
        elif mug_type == 'default1':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bottles/bottle_train1/bottle.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            self.X_m_top = np.array([0., 0., 0.14]) * self.mug_scale
        elif mug_type == 'default2':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bottles/bottle_train2/bottle.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            self.X_m_top = np.array([0., 0., 0.13]) * self.mug_scale
        elif mug_type == 'default3':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bottles/bottle_train3/bottle.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            self.X_m_top = np.array([0., 0., 0.13]) * self.mug_scale
        elif mug_type == 'default4':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bottles/bottle_train4/bottle.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            self.X_m_top = np.array([0., 0., 0.13]) * self.mug_scale
        elif mug_type == 'cup1':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bottles/bottle1/bottle.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0., 0.065, 0.01]) * (self.mug_scale)
        elif mug_type == 'cup2':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bottles/bottle2/bottle.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0.0, 0.075, 0.01]) * (self.mug_scale)
        elif mug_type == 'cup3':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bottles/bottle3/bottle.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([-0.0, 0.07, 0.015]) * (self.mug_scale)
        elif mug_type == 'cup4':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bottles/bottle4/bottle.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0.0, 0.07, 0.015]) * (self.mug_scale)
        elif mug_type == 'cup5':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bottles/bottle5/bottle.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([-0.0, 0.07, 0.01]) * (self.mug_scale)
        elif mug_type == 'cup6':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bottles/bottle6/bottle.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0., 0.06, 0.015]) * (self.mug_scale)
        elif mug_type == 'cup7':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bottles/bottle7/bottle.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0., 0.08, 0.01]) * (self.mug_scale)
        elif mug_type == 'cup8':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bottles/bottle8/bottle.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0.0, 0.06, 0.01]) * (self.mug_scale)
        elif mug_type == 'cup9':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bottles/bottle9/bottle.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0.0, 0.06, 0.03]) * (self.mug_scale)
        elif mug_type == 'cup10':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bottles/bottle10/bottle.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0.0, 0.05, 0.01]) * (self.mug_scale)
        else:
            raise KeyError
        p.changeDynamics(self.mug_id, -1, lateralFriction=0.8, rollingFriction=0.3, spinningFriction=0.3)
        self.hook_scale = 1.
        self.hook_id = p.loadURDF("assets/traybox.urdf", basePosition=[0.5, 0., 0.], globalScaling=0.24 * self.hook_scale, physicsClientId = self.physicsClientId, useFixedBase = True)
        p.changeDynamics(self.hook_id, -1, lateralFriction = 3., rollingFriction=3., spinningFriction=3.)

        if use_support:
            self.support_box_id = p.loadURDF("assets/bottle_support.urdf", basePosition=[0.5, 0., 0.3], globalScaling=0.15, physicsClientId = self.physicsClientId, useFixedBase = True)
            # self.support_box_h = 0.3
            # self.support_box_id = create_box(w=0.15, l=0.15, h=self.support_box_h, color=(72/255,72/255,72/255,1.))
            p.changeDynamics(self.support_box_id, -1, lateralFriction = 3., rollingFriction=3., spinningFriction=3.)
        else:
            self.support_box_id = None

        if distractor:
            self.lego_scale = 2.5
            self.lego_id = p.loadURDF("assets/distractor/lego.urdf", basePosition=[0.7, -0.2, 0.35], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), globalScaling=self.lego_scale, physicsClientId = self.physicsClientId)
            self.duck_scale = 0.07
            self.duck_id = p.loadURDF("assets/distractor/duck.urdf", basePosition=[0.7, 0.2, 0.35], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), globalScaling=self.duck_scale, physicsClientId = self.physicsClientId)
            self.torus_scale = 0.07
            self.torus_id = p.loadURDF("assets/distractor/torus_textured.urdf", basePosition=[0.5, -0.2, 0.35], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), globalScaling=self.torus_scale, physicsClientId = self.physicsClientId)
            self.bunny_scale = 0.07
            self.bunny_id = p.loadURDF("assets/distractor/bunny.urdf", basePosition=[0.5, 0.2, 0.35], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), globalScaling=self.torus_scale, physicsClientId = self.physicsClientId)


    def reset(self, seed = None, mug_pose = 'upright', mug_type = 'default0', distractor = False, use_support = False, step=1):
        if mug_type == 'cups':
            mug_type = f'cup{seed%10 + 1}'
        if mug_type == 'train' or mug_type == 'default':
            mug_type = f'default{seed%5}'
        super().reset(seed = seed, step = False, mug_type=mug_type, distractor=distractor, use_support=use_support)

        randomize_mug_pos = True
        randomize_hook_pos = True

        self.target_item = self.mug_id

        if mug_pose == 'arbitrary' or mug_pose == 'lying':
            arbitrary_tray = True
        else:
            arbitrary_tray = False

        if mug_pose == 'arbitrary':
            #if seed %2 == 0:
            if np.random.rand() > 0.5:
                mug_pose = 'lying'
            else:
                mug_pose = 'upright'


        # Reset cup orientation
        if mug_pose == 'upright':
            mug_orn = np.array([0., 0., np.random.rand()*np.pi*2])
            mug_orn = p.getQuaternionFromEuler(mug_orn)
        elif mug_pose == 'lying':
            mug_orn = np.array([np.pi /4, 0 , -np.random.rand()*np.pi])
            mug_orn = p.getQuaternionFromEuler(mug_orn)
            mug_orn = multiply_quats(mug_orn, p.getQuaternionFromEuler(np.array([0., 0., (2*np.random.rand()-1)*np.pi*1/2])))
        else:
            raise KeyError


        # Sample and mug displacement(s) from center
        if randomize_mug_pos:
            if distractor is True:
                disp_x_abs_max = 0.01
                disp_y_abs_max = 0.01
            elif mug_pose == 'upright':
                disp_x_abs_max = 0.05
                disp_y_abs_max = 0.05
            elif mug_pose == 'lying':
                disp_x_abs_max = 0.03
                disp_y_abs_max = 0.03
            mug_disp = np.array([(2*np.random.rand() - 1) * disp_x_abs_max, (2*np.random.rand() - 1) * disp_y_abs_max, 0.])
        else:
            mug_disp = np.array([0., 0., 0.])
        mug_origin = self.center + np.array([-0.05, 0., 0.])
        if mug_pose == 'lying':
            mug_origin = self.center + np.array([-0.01, 0., 0.])

        if distractor:
            p.changeVisualShape(objectUniqueId=self.bunny_id, linkIndex=-1, rgbaColor = np.concatenate([np.random.rand(3), np.array([1.])]))
            p.changeVisualShape(objectUniqueId=self.lego_id, linkIndex=-1, rgbaColor = np.concatenate([np.random.rand(3), np.array([1.])]))
            p.changeVisualShape(objectUniqueId=self.torus_id, linkIndex=-1, rgbaColor = np.concatenate([np.random.rand(3), np.array([1.])]))
            p.changeVisualShape(objectUniqueId=self.duck_id, linkIndex=-1, rgbaColor = np.concatenate([np.random.rand(3), np.array([1.])]))


        # Sample and distractors displacement(s) from center
        if distractor:
            if randomize_mug_pos:
                disp_x_max = 0.0
                disp_y_max = 0.0
                disps = []
                disps.append(np.array([np.random.rand() * disp_x_max, np.random.rand() * disp_y_max, 0.]))
                disps.append(np.array([-np.random.rand() * disp_x_max, np.random.rand() * disp_y_max, 0.]))
                disps.append(np.array([np.random.rand() * disp_x_max, -np.random.rand() * disp_y_max, 0.]))
                disps.append(np.array([-np.random.rand() * disp_x_max, -np.random.rand() * disp_y_max, 0.]))
            else:
                disps = []
                disps.append(np.array([0., 0., 0.]))
                disps.append(np.array([0., 0., 0.]))
                disps.append(np.array([0., 0., 0.]))
                disps.append(np.array([0., 0., 0.]))


        # Origin pos for mug (and distractors)
        if distractor:
            global_offset = np.array([0.0, 0., 0.])
            dx = 0.13
            dy = 0.13
            origins = [self.center + global_offset + np.array([dx, dy, 0.]),
                       self.center + global_offset + np.array([-dx, dy, 0.]),
                       self.center + global_offset + np.array([dx, -dy, 0.]),
                       self.center + global_offset + np.array([-dx, -dy, 0.])]
            

        # Allocate origin and disp for mugs and distractors
        if distractor:
            idx = list(range(4))
            np.random.shuffle(idx)

            bunny_disp = disps[idx[0]]
            lego_disp = disps[idx[1]]
            duck_disp = disps[idx[2]]
            torus_disp =  disps[idx[3]]

            bunny_origin = origins[idx[0]]
            lego_origin = origins[idx[1]]
            duck_origin = origins[idx[2]]
            torus_origin = origins[idx[3]]



        # Reset mug
        mug_pos = mug_origin + mug_disp*(mug_pose != 'lying')
        if self.support_box_id is not None:
            # First reset the support
            support_box_pos = mug_pos + np.array([0., 0., 2.])
            support_box_pos[-1] = stable_z(self.support_box_id, self.table_id) + (np.random.rand() * 0.01 * randomize_mug_pos)
            # support_box_pos[-1] = stable_z(self.support_box_id, self.table_id) - self.support_box_h + (np.random.rand() * 0.07 * randomize_mug_pos)
            p.resetBasePositionAndOrientation(self.support_box_id, support_box_pos, np.array([0., 0., 0., 1.]), physicsClientId = self.physicsClientId)

            # Reset mug on top of the support
            mug_pos = support_box_pos + np.array([(2*np.random.rand() -1)*0.01*(mug_pose != 'lying'), (2*np.random.rand() -1)*0.01*(mug_pose != 'lying'), 4.])
            p.resetBasePositionAndOrientation(self.mug_id, mug_pos, mug_orn, physicsClientId = self.physicsClientId)
            mug_pos[2] = stable_z(self.mug_id, self.support_box_id)
            p.resetBasePositionAndOrientation(self.mug_id, mug_pos, mug_orn, physicsClientId = self.physicsClientId)
        else:
            mug_pos = mug_pos + np.array([0., 0., 4.])
            p.resetBasePositionAndOrientation(self.mug_id, mug_pos, mug_orn, physicsClientId = self.physicsClientId)
            mug_pos[2] = stable_z(self.mug_id, self.table_id)
            p.resetBasePositionAndOrientation(self.mug_id, mug_pos, mug_orn, physicsClientId = self.physicsClientId)
        

        # Reset distractor
        if distractor:
            bunny_pos = bunny_origin + bunny_disp + np.array([0., 0., 4.])
            bunny_pos[2] = stable_z(self.bunny_id, self.table_id)
            bunny_orn = p.getQuaternionFromEuler([0., 0., np.random.rand()*np.pi*2])

            lego_pos = lego_origin + lego_disp + np.array([0., 0., 4.])
            lego_pos[2] = stable_z(self.lego_id, self.table_id)
            lego_orn = p.getQuaternionFromEuler([0., 0., np.random.rand()*np.pi*2])

            duck_pos = duck_origin + duck_disp + np.array([0., 0., 4.])
            duck_pos[2] = stable_z(self.duck_id, self.table_id)
            duck_orn = p.getQuaternionFromEuler([0., 0., np.random.rand()*np.pi*2])

            torus_pos = torus_origin + torus_disp + np.array([0., 0., 4.])
            torus_pos[2] = stable_z(self.torus_id, self.table_id)
            torus_orn = p.getQuaternionFromEuler([0., 0., np.random.rand()*np.pi*2])

            p.resetBasePositionAndOrientation(self.bunny_id, bunny_pos, bunny_orn, physicsClientId = self.physicsClientId)
            p.resetBasePositionAndOrientation(self.lego_id, lego_pos, lego_orn, physicsClientId = self.physicsClientId)
            p.resetBasePositionAndOrientation(self.duck_id, duck_pos, duck_orn, physicsClientId = self.physicsClientId)
            p.resetBasePositionAndOrientation(self.torus_id, torus_pos, torus_orn, physicsClientId = self.physicsClientId)

        # Reset Hook
        hook_pos = self.center + np.array([0.13, 0.0 , - 0.03])
        if randomize_hook_pos:
            if distractor:
                hook_pos = hook_pos + np.array([0., (2*np.random.rand() - 1) * 0.0, -(2*np.random.rand() - 1) * 0.01])
            else:
                hook_pos = hook_pos + np.array([0., (2*np.random.rand() - 1) * 0.0, -(2*np.random.rand() - 1) * 0.01])
        hook_orn = np.array([0., 0., np.pi]) + np.array([0., 0., (2*np.random.rand() - 1) * np.pi / 8])
        hook_orn = p.getQuaternionFromEuler(hook_orn)
        if arbitrary_tray:
            theta_ = np.random.rand() * np.pi/12
            dOrn = p.getQuaternionFromEuler(np.array([0, theta_, 0]))
            hook_orn = np.array(p.multiplyTransforms(np.zeros(3),hook_orn, np.zeros(3), dOrn)[-1])
            hook_pos[-1] += 0.0
        p.resetBasePositionAndOrientation(self.hook_id, hook_pos, hook_orn, physicsClientId = self.physicsClientId)


        # Disable collision for visual distractors with robot
        for i in range(12):
            #p.setCollisionFilterPair(self.hook_id, self.robot, -1, i, 0, self.physicsClientId)
            #p.setCollisionFilterPair(self.hook_id, self.robot, 0, i, 0, self.physicsClientId)
            if self.support_box_id is not None:
                p.setCollisionFilterPair(self.support_box_id, self.robot, -1, i, 0, self.physicsClientId)
            if distractor:
                p.setCollisionFilterPair(self.bunny_id, self.robot, -1, i, 0, self.physicsClientId)
                p.setCollisionFilterPair(self.lego_id, self.robot, -1, i, 0, self.physicsClientId)
                p.setCollisionFilterPair(self.duck_id, self.robot, -1, i, 0, self.physicsClientId)
                p.setCollisionFilterPair(self.torus_id, self.robot, -1, i, 0, self.physicsClientId)
        if distractor:
            p.setCollisionFilterPair(self.bunny_id, self.mug_id, -1, -1, 0, self.physicsClientId)
            p.setCollisionFilterPair(self.lego_id, self.mug_id, -1, -1, 0, self.physicsClientId)
            p.setCollisionFilterPair(self.duck_id, self.mug_id, -1, -1, 0, self.physicsClientId)
            p.setCollisionFilterPair(self.torus_id, self.mug_id, -1, -1, 0, self.physicsClientId)


        p.changeDynamics(self.mug_id, -1, linearDamping = 0.3, angularDamping = 0.3)
        #p.changeDynamics(self.mug_id, -1, contactStiffness = 100000000., contactDamping = 100.)

        # step sim
        if step:
            step = int(step/self.freq)
            for _ in range(step):
                p.stepSimulation(physicsClientId = self.physicsClientId)

    def get_state(self):
        state = super().get_state()

        # Suffix guide
        # ==>  m: Mug, h: Handle, dg: Desired grasp sweet spot, de: Desired End Effector, pre: Pregrasp
        X_sm, R_sm = p.getBasePositionAndOrientation(self.mug_id, physicsClientId = self.physicsClientId)
        X_sm, R_sm = np.array(X_sm), Rotation.from_quat(R_sm).as_matrix()
        X_s_hook, R_s_hook = p.getBasePositionAndOrientation(self.hook_id, physicsClientId = self.physicsClientId)
        X_s_hook, R_s_hook = np.array(X_s_hook), Rotation.from_quat(R_s_hook).as_matrix()
        # X_s_branch, R_s_branch = p.getLinkState(self.hook_id, self.branchLinkId)[:2]
        # X_s_branch, R_s_branch = np.array(X_s_branch), Rotation.from_quat(R_s_branch).as_matrix()
        # R_s_tip = R_s_branch.copy()
        # X_s_tip = X_s_branch + (R_s_branch @ np.array([0., 0., self.branch_length]))
        X_s_branch, R_s_branch = X_s_hook.copy(), R_s_hook.copy()
        R_s_tip, X_s_tip = R_s_hook.copy(), X_s_hook.copy()

        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_sh, R_sh = X_sm + (R_sm @ X_mh), R_sm @ R_mh

        state['T_sm'] = (X_sm, R_sm)
        state['T_sh'] = (X_sh, R_sh)
        state['T_s_hook'] = (X_s_hook, R_s_hook)
        state['T_s_tip'] = (X_s_tip, R_s_tip)
        state['T_s_branch'] = (X_s_branch, R_s_branch)

        return state

#     Z-axis
#       ↑
    #       #
    #       ####
    #       #  # -> Y-axis
    #       ####
    #########

    def oracle_pick_rim(self, mod):
        state = self.get_state()
        X_sm, R_sm = state['T_sm']
        X_se, R_se = state['T_se']

        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_m_top, R_m_top = self.X_m_top, self.R_m_top

        
        #yaw = -np.pi / 2 # 0 ~ pi
        #yaw = yaw + (2*np.random.rand()-1)*np.pi*2/3
        #yaw = yaw + np.pi/2

        # yaw_mod_1 = 0.
        # yaw_mod_1 += (2*np.random.rand()-1)*np.pi*1/6
        # yaw_mod_2 = np.pi 
        # yaw_mod_2 += (2*np.random.rand()-1)*np.pi*1/6

        # yaw = yaw_mod_1*mod + yaw_mod_2*(1-mod)
        yaw = (2*np.random.rand()-1)*np.pi

        # R_top_rim = multiply_quats(p.getQuaternionFromEuler(np.array([0., 0., yaw])), p.getQuaternionFromEuler(np.array([0., np.pi/2., 0.])), p.getQuaternionFromEuler(np.array([0., 0., np.pi * ((np.random.rand() > 0.5)-0.5)   ])))
        R_top_rim = multiply_quats(p.getQuaternionFromEuler(np.array([0., 0., yaw])), p.getQuaternionFromEuler(np.array([0., np.pi/2., 0.])), p.getQuaternionFromEuler(np.array([0., 0., np.pi /2   ])))
        R_top_rim = p.getMatrixFromQuaternion(R_top_rim)
        R_top_rim = np.array(R_top_rim).reshape(3,3)
        rim_X_top_rim = np.array([0., 0., 0.])
        X_top_rim = R_top_rim @ rim_X_top_rim
        X_rim_dg = np.array([0., 0., 0.]) 
        R_rim_dg = np.array([[0. ,1. ,0.],
                             [1. ,0. ,0.],
                             [0. ,0. ,-1.]])

        R_sdg = R_sm @ R_m_top @ R_top_rim @ R_rim_dg
        X_top_dg = X_top_rim + (R_top_rim @ X_rim_dg)
        X_mdg = X_m_top + (R_m_top @ X_top_dg)
        X_sdg = X_sm + (R_sm @ X_mdg)


        perturb_axis = np.random.randn(3)
        perturb_axis = perturb_axis / np.linalg.norm(perturb_axis)
        perturb_angle = (2*np.random.rand()-1) * (np.pi/180 * 5)
        R_perturb = Rotation.from_rotvec(perturb_angle * perturb_axis).as_matrix()
        R_sdg = R_sdg @ R_perturb

        R_dg_dgpre = np.eye(3)
        R_s_dgpre = R_sdg @ R_dg_dgpre
        X_dg_dgpre = np.array([0., 0., -0.03])
        sX_dg_dgpre = R_sdg @ X_dg_dgpre
        X_s_dgpre = X_sdg + sX_dg_dgpre

        pre_grasp = (X_s_dgpre, R_s_dgpre)
        grasp = X_sdg, R_sdg

        return pre_grasp, grasp

    def oracle_place_handle_horizontal(self, mod, low_var = False):
        state = self.get_state()
        X_sm, R_sm = state['T_sm']
        X_se, R_se = state['T_se']
        X_s_tip, R_s_tip = state['T_s_tip']
        X_s_base, R_s_base = state['T_s_hook']
        R_sg = R_se @ self.R_eg
        X_sg = X_se + (R_se @ self.X_eg)
        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_sh, R_sh = X_sm + (R_sm @ X_mh), R_sm @ R_mh

        X_gh = R_sg.T @ (X_sh - X_sg)
        X_dgdh = X_gh
        R_dgdh = R_sg.T @ R_sh
        X_dhdg = - R_dgdh.T @ X_dgdh

        # if mod:
        #     # theta = -np.pi*np.random.rand() # -pi~0
        #     offset = np.pi/4
        #     if low_var:
        #         theta = -(np.pi-2*offset)*(0.5 + (np.random.rand()-0.5)*0.3) - offset # 30 percent of original variance
        #     else:
        #         theta = -(np.pi-2*offset)*np.random.rand() - offset # -pi/4 ~ -3pi/4
        #     R_base_dh = np.array([[ 0.0,    np.sin(theta),          np.cos(theta)],
        #                           [ 0.0,   -np.cos(theta),          np.sin(theta)],
        #                           [ 1.0,              0.0,                    0.0]])
        # else:
        #     offset = np.pi/4
        #     if low_var:
        #         theta = (np.pi-2*offset)*(0.5 + (np.random.rand()-0.5)*0.3) + offset # 30 percent of original variance
        #     else:
        #         theta = (np.pi-2*offset)*np.random.rand() + offset # pi/4 ~ 3pi/4
        #     R_base_dh = np.array([[ 0.0,   -np.sin(theta),          np.cos(theta)],
        #                           [ 0.0,    np.cos(theta),          np.sin(theta)],
        #                           [-1.0,              0.0,                    0.0]])

        theta = np.random.rand() * 2 * np.pi
        R_base_dh = np.array([[ np.cos(theta),   -np.sin(theta),       0.0],
                              [ np.sin(theta),   np.cos(theta),        0.0],
                              [ 0.0,              0.0,                 1.0]])
                                

        R_sdg = R_s_base @ R_base_dh @ R_dgdh.T

        sX_tip_dh = np.array([0., 0., 0.02]) # np.array([0., 0., 0.02])
        R_sdh = R_sdg @ R_dgdh
        sX_dhdg = R_sdh @ X_dhdg
        X_sdg = X_s_tip + sX_tip_dh + sX_dhdg

        R_dg_dgpre = np.eye(3)
        R_s_dgpre = R_sdg @ R_dg_dgpre
        X_dg_dgpre = np.array([0., 0., -0.03])
        sX_dg_dgpre = R_sdg @ X_dg_dgpre
        X_s_dgpre = X_sdg + sX_dg_dgpre

        pre_place = (X_s_dgpre, R_s_dgpre)
        place = (X_sdg, R_sdg)

        return pre_place, place


    def debug(self, debug_items = ['grasp']):
        super().debug(debug_items=debug_items)
        for item in debug_items:
            if item == 'mug':
                mugFrame_ID = axiscreator(self.mug_id, physicsClientId = self.physicsClientId)
            elif item == 'mug_rim':
                topFrame_ID = axiscreator(self.mug_id, offset = self.X_m_top.copy(), physicsClientId = self.physicsClientId)
            elif item == 'mug_handle':
                handleFrame_ID = axiscreator(self.mug_id, offset = self.X_mh.copy(), physicsClientId = self.physicsClientId)
            elif item == 'hook_branch':
                hookFrame_ID = axiscreator(self.hook_id, self.branchLinkId, physicsClientId = self.physicsClientId)
            elif item == 'hook_branch_offset':
                axiscreator(self.hook_id, self.branchLinkId, offset = np.array([0., 0., 0.07]),physicsClientId = self.physicsClientId)
            elif item == 'distractors':
                axiscreator(self.bunny_id, physicsClientId = self.physicsClientId)
                axiscreator(self.duck_id, physicsClientId = self.physicsClientId)
                axiscreator(self.torus_id, physicsClientId = self.physicsClientId)
                axiscreator(self.lego_id, physicsClientId = self.physicsClientId)
            elif item == 'hook_base':
                hookBaseFrame_ID = axiscreator(self.hook_id, physicsClientId = self.physicsClientId)

    def check_pick_success(self):
        state = self.get_state()
        X_sm, R_sm = state['T_sm']
        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_sh, R_sh = X_sm + (R_sm @ X_mh), R_sm @ R_mh
        if X_sh[2] > 0.22:
            return True
        else:
            return False

    def check_place_success(self):
        p.setCollisionFilterPair(self.target_item, self.table_id, -1, -1, 0, self.physicsClientId)
        p.setCollisionFilterPair(self.target_item, self.plane_id, -1, -1, 0, self.physicsClientId)
        if self.support_box_id is not None:
            p.setCollisionFilterPair(self.target_item, self.support_box_id, -1, -1, 0, self.physicsClientId)
        for i in range(12):
            p.setCollisionFilterPair(self.target_item, self.robot, -1, i, 0, self.physicsClientId)

        for _ in range(int(2/self.freq)):
            p.stepSimulation(physicsClientId = self.physicsClientId)

        state = self.get_state()
        X_sh = state['T_sh'][0]
        if X_sh[-1] > 0.:
            result = True
        else:
            result = False

        p.setCollisionFilterPair(self.target_item, self.table_id, -1, -1, 1, self.physicsClientId)
        p.setCollisionFilterPair(self.target_item, self.plane_id, -1, -1, 1, self.physicsClientId)
        if self.support_box_id is not None:
            p.setCollisionFilterPair(self.target_item, self.support_box_id, -1, -1, 1, self.physicsClientId)
        for i in range(12):
            p.setCollisionFilterPair(self.target_item, self.robot, -1, i, 1, self.physicsClientId)

        return result

    def pick_plan(self, grasp): 
        X_sdg, R_sdg = grasp

        R_dg_dgpre = np.eye(3)
        R_sdg_pre = R_sdg @ R_dg_dgpre
        X_dg_dgpre = np.array([0., 0., -0.03])
        sX_dg_dgpre = R_sdg @ X_dg_dgpre
        X_sdg_pre = X_sdg + sX_dg_dgpre

        R_sde_pre = R_sdg_pre @ self.R_eg.T
        R_sde = R_sdg @ self.R_eg.T
        X_sde_pre = X_sdg_pre - R_sde_pre@self.X_eg
        X_sde = X_sdg - R_sde@self.X_eg


        pre_grasp = {'pose':(X_sde_pre, R_sde_pre), 'gripper_val': 0., 'col_check_items': [self.table_id, self.target_item, self.hook_id]}        # Changes
        grasp1 = {'pose':(X_sde, R_sde), 'gripper_val': 0., 'col_check_items': [self.table_id, self.target_item, self.hook_id]}                   # Changes
        grasp2 = {'pose':(X_sde, R_sde), 'gripper_val': 1., 'col_check_items': []}
        lift = {'pose':(X_sde + np.array([0., 0., 0.2]), R_sde), 'gripper_val': 1., 'col_check_items': []}

        return {'pre_grasp': pre_grasp, 'grasp1':grasp1, 'grasp2':grasp2, 'lift':lift}

    def place(self, pre_place, place, sleep = False, IK_time = 1., z_offset = 0.03, max_distance_plan=(0.05, 1.5)):
        max_distance_plan=(0.1, 1.5)
        X_sdg, R_sdg = place
        target_handles = draw_pose(Pose(X_sdg, p.getEulerFromQuaternion(Rotation.from_matrix(R_sdg).as_quat())))

        plan = self.place_plan(place, z_offset = z_offset)
        pre_place, place, release, retract = plan['pre_place'], plan['place'], plan['release'], plan['retract']


        # R_sde_pre = R_sdg_pre @ self.R_eg.T
        # R_sde = R_sdg @ self.R_eg.T
        # X_sde_pre = X_sdg_pre - R_sde_pre@self.X_eg
        # X_sde = X_sdg - R_sde@self.X_eg

        # if self.check_feasible(pose = (X_sde, R_sde), IK_time=IK_time, criteria='controllability') is False:
        #     raise StopIteration

        # # Reach Preplace Pose
        # self.IK_teleport(target_T_se=(X_sde, R_sde), gripper_val=1., IK_time = IK_time, criteria='controllability')
        # self.IK_teleport(target_T_se=(X_sde_pre, R_sde_pre), gripper_val=1., IK_time = IK_time, criteria='closest')

        max_distance=max_distance_plan[0]
        conf = self.check_feasible_plan([pre_place, place], IK_time=IK_time, verbose=True, max_distance=max_distance)
        if conf is False:
            raise StopIteration
        max_distance=max_distance_plan[1]
        if self.check_feasible_plan([retract, release], IK_time=IK_time, verbose=True, max_distance=max_distance) is False:
            raise StopIteration
        
        #self.IK_teleport(target_T_se=place['pose'], gripper_val=place['gripper_val'], IK_time = IK_time, criteria='controllability')
        self.IK_teleport(target_T_se=pre_place['pose'], gripper_val=pre_place['gripper_val'], IK_time = IK_time, criteria='closest', max_distance=max_distance, ref_conf=conf)


        # Reach Pre-place pose
        self.IK(duration = 1, 
                gripper_val = pre_place['gripper_val'], 
                target_T_se = pre_place['pose'],
                sleep = 0.003 * sleep,
                gripper_force=300,
        )
        self.detach()

        # Reach place pose
        self.IK(duration = 2, 
                gripper_val = place['gripper_val'], 
                target_T_se = place['pose'],
                sleep = 0.003 * sleep,
                gripper_force=300,
                init_gripper_val=place['gripper_val']
        )

        # Release
        self.IK(duration = 1, 
                gripper_val = release['gripper_val'],
                gripper_force = 300, 
                target_T_se = release['pose'],
                sleep = 0.003 * sleep
        )


        # Retract
        self.IK(duration = 2, 
                gripper_val = retract['gripper_val'], 
                gripper_force = 1, 
                target_T_se = (retract['pose'][0] + np.array([0., 0., 0.05]), retract['pose'][1]),
                sleep = 0.003 * sleep,
                joint_force=50
        )

        # Retract
        self.IK(duration = 2, 
                gripper_val = retract['gripper_val'], 
                gripper_force = 1, 
                target_T_se = (retract['pose'][0] + np.array([0., 0., 0.05]), retract['pose'][1]),
                sleep = 0.003 * sleep,
                joint_force=50
        )

        #remove_handles(target_handles)










































































































































































































































































































































class BottleTaskOld(FrankaTask):
    def __init__(self, use_gui = True):
        super().__init__(use_gui=use_gui)

    def init_task(self, mug_type = 'default', distractor = False, use_support = False, **kwargs):
        super().init_task()

        self.mug_scale = 1.
        self.X_mh = np.array([0., 0.0, -0.03]) * (self.mug_scale) # mug to handle
        self.R_mh = np.eye(3)
        self.X_eg = np.array([0, 0, 0.105]) # EE to gripper sweet-spot
        self.R_eg = np.eye(3)
        self.X_m_top = np.array([0., 0., 0.15]) * self.mug_scale
        self.R_m_top = np.eye(3)
        self.branch_length = 0.065
        self.branchLinkId = -1

        if mug_type == 'default':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bottles/bottle0/bottle.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
        elif mug_type == 'cup1':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bottles/bottle1/bottle.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0., 0.065, 0.01]) * (self.mug_scale)
        elif mug_type == 'cup2':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bottles/bottle2/bottle.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0.0, 0.075, 0.01]) * (self.mug_scale)
        elif mug_type == 'cup3':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bottles/bottle3/bottle.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([-0.0, 0.07, 0.015]) * (self.mug_scale)
        elif mug_type == 'cup4':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bottles/bottle4/bottle.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0.0, 0.07, 0.015]) * (self.mug_scale)
        elif mug_type == 'cup5':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bottles/bottle5/bottle.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([-0.0, 0.07, 0.01]) * (self.mug_scale)
        elif mug_type == 'cup6':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bottles/bottle6/bottle.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0., 0.06, 0.015]) * (self.mug_scale)
        elif mug_type == 'cup7':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bottles/bottle7/bottle.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0., 0.08, 0.01]) * (self.mug_scale)
        elif mug_type == 'cup8':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bottles/bottle8/bottle.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0.0, 0.06, 0.01]) * (self.mug_scale)
        elif mug_type == 'cup9':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bottles/bottle9/bottle.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0.0, 0.06, 0.03]) * (self.mug_scale)
        elif mug_type == 'cup10':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/bottles/bottle10/bottle.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            #self.X_mh = np.array([0.0, 0.05, 0.01]) * (self.mug_scale)
        else:
            raise KeyError
        p.changeDynamics(self.mug_id, -1, lateralFriction=0.8, rollingFriction=0.3, spinningFriction=0.3)
        self.hook_scale = 1.
        self.hook_id = p.loadURDF("assets/traybox.urdf", basePosition=[0.5, 0., 0.], globalScaling=0.2 * self.hook_scale, physicsClientId = self.physicsClientId, useFixedBase = True)
        p.changeDynamics(self.hook_id, -1, lateralFriction = 3., rollingFriction=3., spinningFriction=3.)

        if use_support:
            # self.support_box_id = p.loadURDF("assets/traysupport.urdf", basePosition=[0.5, 0., 0.3], globalScaling=0.2, physicsClientId = self.physicsClientId, useFixedBase = True)
            self.support_box_h = 0.3
            self.support_box_id = create_box(w=0.15, l=0.15, h=self.support_box_h, color=(72/255,72/255,72/255,1.))
            p.changeDynamics(self.support_box_id, -1, lateralFriction = 3., rollingFriction=3., spinningFriction=3.)
        else:
            self.support_box_id = None

        if distractor:
            self.lego_scale = 2.5
            self.lego_id = p.loadURDF("assets/distractor/lego.urdf", basePosition=[0.7, -0.2, 0.35], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), globalScaling=self.lego_scale, physicsClientId = self.physicsClientId)
            self.duck_scale = 0.07
            self.duck_id = p.loadURDF("assets/distractor/duck.urdf", basePosition=[0.7, 0.2, 0.35], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), globalScaling=self.duck_scale, physicsClientId = self.physicsClientId)
            self.torus_scale = 0.07
            self.torus_id = p.loadURDF("assets/distractor/torus_textured.urdf", basePosition=[0.5, -0.2, 0.35], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), globalScaling=self.torus_scale, physicsClientId = self.physicsClientId)
            self.bunny_scale = 0.07
            self.bunny_id = p.loadURDF("assets/distractor/bunny.urdf", basePosition=[0.5, 0.2, 0.35], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), globalScaling=self.torus_scale, physicsClientId = self.physicsClientId)


    def reset(self, seed = None, mug_pose = 'upright', mug_type = 'default', distractor = False, use_support = False, step=1):
        if mug_type == 'cups':
            mug_type = f'cup{seed%10 + 1}'
        super().reset(seed = seed, step = False, mug_type=mug_type, distractor=distractor, use_support=use_support)

        randomize_mug_pos = True
        randomize_hook_pos = True

        self.target_item = self.mug_id

        if mug_pose == 'arbitrary' or mug_pose == 'lying':
            arbitrary_tray = True
        else:
            arbitrary_tray = False

        if mug_pose == 'arbitrary':
            #if seed %2 == 0:
            if np.random.rand() > 0.5:
                mug_pose = 'lying'
            else:
                mug_pose = 'upright'


        # Reset cup orientation
        if mug_pose == 'upright':
            mug_orn = np.array([0., 0., np.random.rand()*np.pi*2])
            mug_orn = p.getQuaternionFromEuler(mug_orn)
        elif mug_pose == 'lying':
            mug_orn = np.array([np.pi /2, 0., -np.random.rand()*np.pi])
            mug_orn = p.getQuaternionFromEuler(mug_orn)
            mug_orn = multiply_quats(mug_orn, p.getQuaternionFromEuler(np.array([0., 0., (2*np.random.rand()-1)*np.pi*1/2])))
        else:
            raise KeyError


        # Sample and mug displacement(s) from center
        if randomize_mug_pos:
            if distractor is True:
                disp_x_abs_max = 0.01
                disp_y_abs_max = 0.01
            elif mug_pose == 'upright':
                disp_x_abs_max = 0.05
                disp_y_abs_max = 0.05
            elif mug_pose == 'lying':
                disp_x_abs_max = 0.03
                disp_y_abs_max = 0.03
            mug_disp = np.array([(2*np.random.rand() - 1) * disp_x_abs_max, (2*np.random.rand() - 1) * disp_y_abs_max, 0.])
        else:
            mug_disp = np.array([0., 0., 0.])
        mug_origin = self.center + np.array([-0.05, 0., 0.])

        if distractor:
            p.changeVisualShape(objectUniqueId=self.bunny_id, linkIndex=-1, rgbaColor = np.concatenate([np.random.rand(3), np.array([1.])]))
            p.changeVisualShape(objectUniqueId=self.lego_id, linkIndex=-1, rgbaColor = np.concatenate([np.random.rand(3), np.array([1.])]))
            p.changeVisualShape(objectUniqueId=self.torus_id, linkIndex=-1, rgbaColor = np.concatenate([np.random.rand(3), np.array([1.])]))
            p.changeVisualShape(objectUniqueId=self.duck_id, linkIndex=-1, rgbaColor = np.concatenate([np.random.rand(3), np.array([1.])]))


        # Sample and distractors displacement(s) from center
        if distractor:
            if randomize_mug_pos:
                disp_x_max = 0.01
                disp_y_max = 0.01
                disps = []
                disps.append(np.array([np.random.rand() * disp_x_max, np.random.rand() * disp_y_max, 0.]))
                disps.append(np.array([-np.random.rand() * disp_x_max, np.random.rand() * disp_y_max, 0.]))
                disps.append(np.array([np.random.rand() * disp_x_max, -np.random.rand() * disp_y_max, 0.]))
                disps.append(np.array([-np.random.rand() * disp_x_max, -np.random.rand() * disp_y_max, 0.]))
            else:
                disps = []
                disps.append(np.array([0., 0., 0.]))
                disps.append(np.array([0., 0., 0.]))
                disps.append(np.array([0., 0., 0.]))
                disps.append(np.array([0., 0., 0.]))


        # Origin pos for mug (and distractors)
        if distractor:
            global_offset = np.array([0.0, 0., 0.])
            dx = 0.12
            dy = 0.12
            origins = [self.center + global_offset + np.array([dx, dy, 0.]),
                       self.center + global_offset + np.array([-dx, dy, 0.]),
                       self.center + global_offset + np.array([dx, -dy, 0.]),
                       self.center + global_offset + np.array([-dx, -dy, 0.])]
            

        # Allocate origin and disp for mugs and distractors
        if distractor:
            idx = list(range(4))
            np.random.shuffle(idx)

            bunny_disp = disps[idx[0]]
            lego_disp = disps[idx[1]]
            duck_disp = disps[idx[2]]
            torus_disp =  disps[idx[3]]

            bunny_origin = origins[idx[0]]
            lego_origin = origins[idx[1]]
            duck_origin = origins[idx[2]]
            torus_origin = origins[idx[3]]



        # Reset mug
        mug_pos = mug_origin + mug_disp
        if self.support_box_id is not None:
            # First reset the support
            support_box_pos = mug_pos + np.array([0., 0., 2.])
            # support_box_pos[-1] = stable_z(self.support_box_id, self.table_id) + (np.random.rand() * 0.04 * randomize_mug_pos)
            support_box_pos[-1] = stable_z(self.support_box_id, self.table_id) - self.support_box_h + (np.random.rand() * 0.07 * randomize_mug_pos)
            p.resetBasePositionAndOrientation(self.support_box_id, support_box_pos, np.array([0., 0., 0., 1.]), physicsClientId = self.physicsClientId)

            # Reset mug on top of the support
            mug_pos = support_box_pos + np.array([(2*np.random.rand() -1)*0.01, (2*np.random.rand() -1)*0.01, 4.])
            p.resetBasePositionAndOrientation(self.mug_id, mug_pos, mug_orn, physicsClientId = self.physicsClientId)
            mug_pos[2] = stable_z(self.mug_id, self.support_box_id)
            p.resetBasePositionAndOrientation(self.mug_id, mug_pos, mug_orn, physicsClientId = self.physicsClientId)
        else:
            mug_pos = mug_pos + np.array([0., 0., 4.])
            p.resetBasePositionAndOrientation(self.mug_id, mug_pos, mug_orn, physicsClientId = self.physicsClientId)
            mug_pos[2] = stable_z(self.mug_id, self.table_id)
            p.resetBasePositionAndOrientation(self.mug_id, mug_pos, mug_orn, physicsClientId = self.physicsClientId)
        

        # Reset distractor
        if distractor:
            bunny_pos = bunny_origin + bunny_disp + np.array([0., 0., 4.])
            bunny_pos[2] = stable_z(self.bunny_id, self.table_id)
            bunny_orn = p.getQuaternionFromEuler([0., 0., np.random.rand()*np.pi*2])

            lego_pos = lego_origin + lego_disp + np.array([0., 0., 4.])
            lego_pos[2] = stable_z(self.lego_id, self.table_id)
            lego_orn = p.getQuaternionFromEuler([0., 0., np.random.rand()*np.pi*2])

            duck_pos = duck_origin + duck_disp + np.array([0., 0., 4.])
            duck_pos[2] = stable_z(self.duck_id, self.table_id)
            duck_orn = p.getQuaternionFromEuler([0., 0., np.random.rand()*np.pi*2])

            torus_pos = torus_origin + torus_disp + np.array([0., 0., 4.])
            torus_pos[2] = stable_z(self.torus_id, self.table_id)
            torus_orn = p.getQuaternionFromEuler([0., 0., np.random.rand()*np.pi*2])

            p.resetBasePositionAndOrientation(self.bunny_id, bunny_pos, bunny_orn, physicsClientId = self.physicsClientId)
            p.resetBasePositionAndOrientation(self.lego_id, lego_pos, lego_orn, physicsClientId = self.physicsClientId)
            p.resetBasePositionAndOrientation(self.duck_id, duck_pos, duck_orn, physicsClientId = self.physicsClientId)
            p.resetBasePositionAndOrientation(self.torus_id, torus_pos, torus_orn, physicsClientId = self.physicsClientId)

        # Reset Hook
        hook_pos = self.center + np.array([0.13, 0.0 , - 0.0])
        if randomize_hook_pos:
            if distractor:
                hook_pos = hook_pos + np.array([0., (2*np.random.rand() - 1) * 0.01, -(2*np.random.rand() - 1) * 0.01])
            else:
                hook_pos = hook_pos + np.array([0., (2*np.random.rand() - 1) * 0.05, -(2*np.random.rand() - 1) * 0.01])
        hook_orn = np.array([0., 0., np.pi]) + np.array([0., 0., (2*np.random.rand() - 1) * np.pi / 8])
        hook_orn = p.getQuaternionFromEuler(hook_orn)
        if arbitrary_tray:
            theta_ = np.random.rand() * np.pi/12
            dOrn = p.getQuaternionFromEuler(np.array([0, theta_, 0]))
            hook_orn = np.array(p.multiplyTransforms(np.zeros(3),hook_orn, np.zeros(3), dOrn)[-1])
            hook_pos[-1] += 0.0
        p.resetBasePositionAndOrientation(self.hook_id, hook_pos, hook_orn, physicsClientId = self.physicsClientId)


        # Disable collision for visual distractors with robot
        for i in range(12):
            #p.setCollisionFilterPair(self.hook_id, self.robot, -1, i, 0, self.physicsClientId)
            #p.setCollisionFilterPair(self.hook_id, self.robot, 0, i, 0, self.physicsClientId)
            if self.support_box_id is not None:
                p.setCollisionFilterPair(self.support_box_id, self.robot, -1, i, 0, self.physicsClientId)
            if distractor:
                p.setCollisionFilterPair(self.bunny_id, self.robot, -1, i, 0, self.physicsClientId)
                p.setCollisionFilterPair(self.lego_id, self.robot, -1, i, 0, self.physicsClientId)
                p.setCollisionFilterPair(self.duck_id, self.robot, -1, i, 0, self.physicsClientId)
                p.setCollisionFilterPair(self.torus_id, self.robot, -1, i, 0, self.physicsClientId)
        if distractor:
            p.setCollisionFilterPair(self.bunny_id, self.mug_id, -1, -1, 0, self.physicsClientId)
            p.setCollisionFilterPair(self.lego_id, self.mug_id, -1, -1, 0, self.physicsClientId)
            p.setCollisionFilterPair(self.duck_id, self.mug_id, -1, -1, 0, self.physicsClientId)
            p.setCollisionFilterPair(self.torus_id, self.mug_id, -1, -1, 0, self.physicsClientId)


        p.changeDynamics(self.mug_id, -1, linearDamping = 0.3, angularDamping = 0.3)
        #p.changeDynamics(self.mug_id, -1, contactStiffness = 100000000., contactDamping = 100.)

        # step sim
        if step:
            step = int(step/self.freq)
            for _ in range(step):
                p.stepSimulation(physicsClientId = self.physicsClientId)

    def get_state(self):
        state = super().get_state()

        # Suffix guide
        # ==>  m: Mug, h: Handle, dg: Desired grasp sweet spot, de: Desired End Effector, pre: Pregrasp
        X_sm, R_sm = p.getBasePositionAndOrientation(self.mug_id, physicsClientId = self.physicsClientId)
        X_sm, R_sm = np.array(X_sm), Rotation.from_quat(R_sm).as_matrix()
        X_s_hook, R_s_hook = p.getBasePositionAndOrientation(self.hook_id, physicsClientId = self.physicsClientId)
        X_s_hook, R_s_hook = np.array(X_s_hook), Rotation.from_quat(R_s_hook).as_matrix()
        # X_s_branch, R_s_branch = p.getLinkState(self.hook_id, self.branchLinkId)[:2]
        # X_s_branch, R_s_branch = np.array(X_s_branch), Rotation.from_quat(R_s_branch).as_matrix()
        # R_s_tip = R_s_branch.copy()
        # X_s_tip = X_s_branch + (R_s_branch @ np.array([0., 0., self.branch_length]))
        X_s_branch, R_s_branch = X_s_hook.copy(), R_s_hook.copy()
        R_s_tip, X_s_tip = R_s_hook.copy(), X_s_hook.copy()

        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_sh, R_sh = X_sm + (R_sm @ X_mh), R_sm @ R_mh

        state['T_sm'] = (X_sm, R_sm)
        state['T_sh'] = (X_sh, R_sh)
        state['T_s_hook'] = (X_s_hook, R_s_hook)
        state['T_s_tip'] = (X_s_tip, R_s_tip)
        state['T_s_branch'] = (X_s_branch, R_s_branch)

        return state

#     Z-axis
#       ↑
    #       #
    #       ####
    #       #  # -> Y-axis
    #       ####
    #########

    def oracle_pick_rim(self, mod):
        state = self.get_state()
        X_sm, R_sm = state['T_sm']
        X_se, R_se = state['T_se']

        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_m_top, R_m_top = self.X_m_top, self.R_m_top

        
        #yaw = -np.pi / 2 # 0 ~ pi
        #yaw = yaw + (2*np.random.rand()-1)*np.pi*2/3
        #yaw = yaw + np.pi/2

        # yaw_mod_1 = 0.
        # yaw_mod_1 += (2*np.random.rand()-1)*np.pi*1/6
        # yaw_mod_2 = np.pi 
        # yaw_mod_2 += (2*np.random.rand()-1)*np.pi*1/6

        # yaw = yaw_mod_1*mod + yaw_mod_2*(1-mod)
        yaw = (2*np.random.rand()-1)*np.pi

        R_top_rim = multiply_quats(p.getQuaternionFromEuler(np.array([0., 0., yaw])), p.getQuaternionFromEuler(np.array([0., np.pi/2., 0.])), p.getQuaternionFromEuler(np.array([0., 0., np.pi * ((np.random.rand() > 0.5)-0.5)   ])))
        R_top_rim = p.getMatrixFromQuaternion(R_top_rim)
        R_top_rim = np.array(R_top_rim).reshape(3,3)
        rim_X_top_rim = np.array([0., 0., 0.])
        X_top_rim = R_top_rim @ rim_X_top_rim
        X_rim_dg = np.array([0., 0., 0.]) 
        R_rim_dg = np.array([[0. ,1. ,0.],
                             [1. ,0. ,0.],
                             [0. ,0. ,-1.]])

        R_sdg = R_sm @ R_m_top @ R_top_rim @ R_rim_dg
        X_top_dg = X_top_rim + (R_top_rim @ X_rim_dg)
        X_mdg = X_m_top + (R_m_top @ X_top_dg)
        X_sdg = X_sm + (R_sm @ X_mdg)


        perturb_axis = np.random.randn(3)
        perturb_axis = perturb_axis / np.linalg.norm(perturb_axis)
        perturb_angle = (2*np.random.rand()-1) * (np.pi/180 * 5)
        R_perturb = Rotation.from_rotvec(perturb_angle * perturb_axis).as_matrix()
        R_sdg = R_sdg @ R_perturb

        R_dg_dgpre = np.eye(3)
        R_s_dgpre = R_sdg @ R_dg_dgpre
        X_dg_dgpre = np.array([0., 0., -0.03])
        sX_dg_dgpre = R_sdg @ X_dg_dgpre
        X_s_dgpre = X_sdg + sX_dg_dgpre

        pre_grasp = (X_s_dgpre, R_s_dgpre)
        grasp = X_sdg, R_sdg

        return pre_grasp, grasp

    def oracle_place_handle_horizontal(self, mod, low_var = False):
        state = self.get_state()
        X_sm, R_sm = state['T_sm']
        X_se, R_se = state['T_se']
        X_s_tip, R_s_tip = state['T_s_tip']
        X_s_base, R_s_base = state['T_s_hook']
        R_sg = R_se @ self.R_eg
        X_sg = X_se + (R_se @ self.X_eg)
        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_sh, R_sh = X_sm + (R_sm @ X_mh), R_sm @ R_mh

        X_gh = R_sg.T @ (X_sh - X_sg)
        X_dgdh = X_gh
        R_dgdh = R_sg.T @ R_sh
        X_dhdg = - R_dgdh.T @ X_dgdh

        # if mod:
        #     # theta = -np.pi*np.random.rand() # -pi~0
        #     offset = np.pi/4
        #     if low_var:
        #         theta = -(np.pi-2*offset)*(0.5 + (np.random.rand()-0.5)*0.3) - offset # 30 percent of original variance
        #     else:
        #         theta = -(np.pi-2*offset)*np.random.rand() - offset # -pi/4 ~ -3pi/4
        #     R_base_dh = np.array([[ 0.0,    np.sin(theta),          np.cos(theta)],
        #                           [ 0.0,   -np.cos(theta),          np.sin(theta)],
        #                           [ 1.0,              0.0,                    0.0]])
        # else:
        #     offset = np.pi/4
        #     if low_var:
        #         theta = (np.pi-2*offset)*(0.5 + (np.random.rand()-0.5)*0.3) + offset # 30 percent of original variance
        #     else:
        #         theta = (np.pi-2*offset)*np.random.rand() + offset # pi/4 ~ 3pi/4
        #     R_base_dh = np.array([[ 0.0,   -np.sin(theta),          np.cos(theta)],
        #                           [ 0.0,    np.cos(theta),          np.sin(theta)],
        #                           [-1.0,              0.0,                    0.0]])

        theta = np.random.rand() * 2 * np.pi
        R_base_dh = np.array([[ np.cos(theta),   -np.sin(theta),       0.0],
                              [ np.sin(theta),   np.cos(theta),        0.0],
                              [ 0.0,              0.0,                 1.0]])
                                

        R_sdg = R_s_base @ R_base_dh @ R_dgdh.T

        sX_tip_dh = np.array([0., 0., 0.00]) # np.array([0., 0., 0.02])
        R_sdh = R_sdg @ R_dgdh
        sX_dhdg = R_sdh @ X_dhdg
        X_sdg = X_s_tip + sX_tip_dh + sX_dhdg

        R_dg_dgpre = np.eye(3)
        R_s_dgpre = R_sdg @ R_dg_dgpre
        X_dg_dgpre = np.array([0., 0., -0.03])
        sX_dg_dgpre = R_sdg @ X_dg_dgpre
        X_s_dgpre = X_sdg + sX_dg_dgpre

        pre_place = (X_s_dgpre, R_s_dgpre)
        place = (X_sdg, R_sdg)

        return pre_place, place


    def debug(self, debug_items = ['grasp']):
        super().debug(debug_items=debug_items)
        for item in debug_items:
            if item == 'mug':
                mugFrame_ID = axiscreator(self.mug_id, physicsClientId = self.physicsClientId)
            elif item == 'mug_rim':
                topFrame_ID = axiscreator(self.mug_id, offset = self.X_m_top.copy(), physicsClientId = self.physicsClientId)
            elif item == 'mug_handle':
                handleFrame_ID = axiscreator(self.mug_id, offset = self.X_mh.copy(), physicsClientId = self.physicsClientId)
            elif item == 'hook_branch':
                hookFrame_ID = axiscreator(self.hook_id, self.branchLinkId, physicsClientId = self.physicsClientId)
            elif item == 'hook_branch_offset':
                axiscreator(self.hook_id, self.branchLinkId, offset = np.array([0., 0., 0.07]),physicsClientId = self.physicsClientId)
            elif item == 'distractors':
                axiscreator(self.bunny_id, physicsClientId = self.physicsClientId)
                axiscreator(self.duck_id, physicsClientId = self.physicsClientId)
                axiscreator(self.torus_id, physicsClientId = self.physicsClientId)
                axiscreator(self.lego_id, physicsClientId = self.physicsClientId)
            elif item == 'hook_base':
                hookBaseFrame_ID = axiscreator(self.hook_id, physicsClientId = self.physicsClientId)

    def check_pick_success(self):
        state = self.get_state()
        X_sm, R_sm = state['T_sm']
        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_sh, R_sh = X_sm + (R_sm @ X_mh), R_sm @ R_mh
        if X_sh[2] > 0.22:
            return True
        else:
            return False

    def check_place_success(self):
        p.setCollisionFilterPair(self.target_item, self.table_id, -1, -1, 0, self.physicsClientId)
        p.setCollisionFilterPair(self.target_item, self.plane_id, -1, -1, 0, self.physicsClientId)
        if self.support_box_id is not None:
            p.setCollisionFilterPair(self.target_item, self.support_box_id, -1, -1, 0, self.physicsClientId)
        for i in range(12):
            p.setCollisionFilterPair(self.target_item, self.robot, -1, i, 0, self.physicsClientId)

        for _ in range(int(2/self.freq)):
            p.stepSimulation(physicsClientId = self.physicsClientId)

        state = self.get_state()
        X_sh = state['T_sh'][0]
        if X_sh[-1] > 0.:
            result = True
        else:
            result = False

        p.setCollisionFilterPair(self.target_item, self.table_id, -1, -1, 1, self.physicsClientId)
        p.setCollisionFilterPair(self.target_item, self.plane_id, -1, -1, 1, self.physicsClientId)
        if self.support_box_id is not None:
            p.setCollisionFilterPair(self.target_item, self.support_box_id, -1, -1, 1, self.physicsClientId)
        for i in range(12):
            p.setCollisionFilterPair(self.target_item, self.robot, -1, i, 1, self.physicsClientId)

        return result

    def pick_plan(self, grasp): 
        X_sdg, R_sdg = grasp

        R_dg_dgpre = np.eye(3)
        R_sdg_pre = R_sdg @ R_dg_dgpre
        X_dg_dgpre = np.array([0., 0., -0.03])
        sX_dg_dgpre = R_sdg @ X_dg_dgpre
        X_sdg_pre = X_sdg + sX_dg_dgpre

        R_sde_pre = R_sdg_pre @ self.R_eg.T
        R_sde = R_sdg @ self.R_eg.T
        X_sde_pre = X_sdg_pre - R_sde_pre@self.X_eg
        X_sde = X_sdg - R_sde@self.X_eg


        pre_grasp = {'pose':(X_sde_pre, R_sde_pre), 'gripper_val': 0., 'col_check_items': [self.table_id, self.target_item, self.hook_id]}        # Changes
        grasp1 = {'pose':(X_sde, R_sde), 'gripper_val': 0., 'col_check_items': [self.table_id, self.target_item, self.hook_id]}                   # Changes
        grasp2 = {'pose':(X_sde, R_sde), 'gripper_val': 1., 'col_check_items': []}
        lift = {'pose':(X_sde + np.array([0., 0., 0.2]), R_sde), 'gripper_val': 1., 'col_check_items': []}

        return {'pre_grasp': pre_grasp, 'grasp1':grasp1, 'grasp2':grasp2, 'lift':lift}

    def place(self, pre_place, place, sleep = False, IK_time = 1., z_offset = 0.03, max_distance_plan=(0.1, 1.5)):
        super().place(pre_place, place, sleep = False, IK_time = 1., z_offset = 0.03, max_distance_plan=(0.1, 1.5))