import time
import os
from matplotlib import use

import numpy as np

import pybullet as p
import pybullet_data

from scipy.spatial.transform import Rotation
from pybullet_tools.utils import add_data_path, connect, dump_body, disconnect, wait_for_user, \
    get_movable_joints, get_sample_fn, set_joint_positions, get_joint_name, LockRenderer, link_from_name, get_link_pose, \
    multiply, Pose, Point, interpolate_poses, HideOutput, draw_pose, set_camera_pose, load_pybullet, \
    assign_link_colors, add_line, point_from_pose, remove_handles, BLUE, INF, set_client, stable_z, create_box, set_point, reset_simulation, \
    add_fixed_constraint, remove_fixed_constraint, set_numpy_seed, set_random_seed
from pybullet_tools.ikfast.ikfast import get_ik_joints, either_inverse_kinematics, check_ik_solver

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
        p.setTimeStep(1/480)

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
                   'ypr': (90, -40, 0),
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
                        'ypr': (90+135, -40, 0),
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
                        'ypr': (90+135+90, -40, 0),
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

        self.pickcam_configs =  [pickcam_1_config, pickcam_2_config, pickcam_3_config]# + [pickcam_4_config, pickcam_5_config, pickcam_6_config]
        self.xlim_pick = np.array([-0.15, 0.15]) 
        self.ylim_pick = np.array([-0.15, 0.15]) 
        self.zlim_pick = np.array([-0.15, 0.15])

        self.max_gripper_val = 0.025

        self.default_robot_conf = np.array([-0.0005458792039334526,    # 0
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
        set_joint_positions(self.robot, self.joints, self.default_robot_conf)

    def init_task(self, **kwargs):
        super().init_task()
        self.table_id = p.loadURDF("assets/table.urdf", basePosition=self.center + np.array([0.0, 0.0, -0.35]), baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]), globalScaling=0.4, physicsClientId = self.physicsClientId)
        self.spawn_robot()

    def reset(self, seed = None, step = 1000, **kwargs):
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
            for _ in range(step):
                p.stepSimulation(physicsClientId = self.physicsClientId)

    def debug(self, debug_items = ['grasp']):
        p.removeAllUserDebugItems(physicsClientId = self.physicsClientId)
        for item in debug_items:
            if item == 'ee':
                eeFrame_ID = axiscreator(self.robot, self.EE, physicsCliendId = self.physicsClientId)
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
           sleep = 0., gripper_force = 300, joint_force = None, init_gripper_val = None):

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

    def IK_teleport(self, target_T_se, gripper_val, IK_time = 1., gripper_force = 5.):
        if target_T_se[1].shape[-2:] == (3,3):
            target_T_se = (target_T_se[0], Rotation.from_matrix(target_T_se[1]).as_quat())

        sol = either_inverse_kinematics(self.robot, self.robot_info, self.EE, target_T_se, use_pybullet=False,
                                            max_distance=INF, max_time=INF, max_candidates=INF, max_attempts=int(15000*IK_time))
        conf = next(sol)
        if conf is None:
            return False

        self.teleport(configuration=conf)
        self.gripper_control(gripper_val, gripper_force)

        return conf

    def pick(self, pre_grasp, grasp,
             sleep = False, reach_force = 50., IK_time = 1.):
        X_sdg_pre, R_sdg_pre = pre_grasp
        X_sdg, R_sdg = grasp
        target_handles = draw_pose(Pose(X_sdg, p.getEulerFromQuaternion(Rotation.from_matrix(R_sdg).as_quat())))

        R_sde_pre = R_sdg_pre @ self.R_eg.T
        R_sde = R_sdg @ self.R_eg.T
        X_sde_pre = X_sdg_pre - R_sde_pre@self.X_eg
        X_sde = X_sdg - R_sde@self.X_eg

        
        # Reach Pregrasp Pose
        self.IK_teleport(target_T_se=(X_sde_pre, R_sde_pre), gripper_val=0., IK_time = IK_time)
        
        # Reach Grasp pose
        self.IK(duration = 4000, 
                gripper_val=0., 
                target_T_se = (X_sde, R_sde),
                sleep = 0.000003 * sleep,
                #force = reach_force
        )
        # Grasp
        self.IK(duration = 600, 
                gripper_val=1., 
                target_T_se = (X_sde, R_sde),
                sleep = 0.003 * sleep
        )
        # Lift

        self.IK(duration = 2000, 
                gripper_val=1., 
                target_T_se = (X_sde + np.array([0., 0., 0.2]), R_sde),
                sleep = 0.003 * sleep,
                init_gripper_val=1.
        )
        if self.target_item is not None:
            self.attach(self.target_item)

        #remove_handles(target_handles)

    def retract_robot(self, gripper_val, IK_time = 1.):
        #target_T_se = (np.array([0.6, 0., 0.6]), np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(np.array([np.pi, 0., 0.])))).reshape(3,3))
        #self.IK_teleport(target_T_se=target_T_se, gripper_val=gripper_val, IK_time = IK_time)
        self.teleport(configuration=self.default_robot_conf[:-2])
        self.gripper_control(gripper_val)

    def place(self, pre_place, place, sleep = False, IK_time = 1.):
        X_sdg_pre, R_sdg_pre = pre_place
        X_sdg, R_sdg = place
        target_handles = draw_pose(Pose(X_sdg, p.getEulerFromQuaternion(Rotation.from_matrix(R_sdg).as_quat())))

        R_sde_pre = R_sdg_pre @ self.R_eg.T
        R_sde = R_sdg @ self.R_eg.T
        X_sde_pre = X_sdg_pre - R_sde_pre@self.X_eg
        X_sde = X_sdg - R_sde@self.X_eg


        # Reach Preplace Pose
        self.IK_teleport(target_T_se=(X_sde_pre, R_sde_pre), gripper_val=1., IK_time = IK_time)

        self.IK(duration = 1000, 
                gripper_val=1., 
                target_T_se = (X_sde_pre, R_sde_pre),
                sleep = 0.003 * sleep,
                gripper_force=300,
        )
        self.detach()

        # Reach place pose
        self.IK(duration = 2000, 
                gripper_val=1., 
                target_T_se = (X_sde, R_sde),
                sleep = 0.003 * sleep,
                gripper_force=300,
                init_gripper_val=1.
        )

        # release
        self.IK(duration = 2000, 
                gripper_val=0.4,
                gripper_force = 1, 
                target_T_se = (X_sde, R_sde),
                sleep = 0.003 * sleep
        )

        # Retract
        self.IK(duration = 2000, 
                gripper_val=0.4, 
                gripper_force = 1, 
                target_T_se = (X_sde_pre, R_sde_pre),
                sleep = 0.003 * sleep
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

    def init_task(self, eval_mode = False, use_box = False, **kwargs):
        super().init_task()

        self.mug_scale = 1.
        self.X_mh = np.array([0., 0.055, 0.02]) * (self.mug_scale) # mug to handle
        self.R_mh = np.eye(3)
        self.X_eg = np.array([0, 0, 0.105]) # EE to gripper sweet-spot
        self.R_eg = np.eye(3)
        self.X_m_top = np.array([0., 0., 0.07]) * self.mug_scale
        self.R_m_top = np.eye(3)


        if eval_mode == 'cup1':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/hongin/mug.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            self.X_mh = np.array([0., 0.085, 0.02]) * (self.mug_scale)
        elif eval_mode == 'cup2':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/hongin/tea_cup.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            self.X_mh = np.array([0.08, 0., 0.08]) * (self.mug_scale)
        elif eval_mode == 'cup3':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/hongin/widecup.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            self.X_mh = np.array([-0.08, 0., 0.05]) * (self.mug_scale)
        elif eval_mode == 'cup4':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/hongin/coffee_cup_sharp.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
            self.X_mh = np.array([0.065, 0., 0.065]) * (self.mug_scale)
        elif eval_mode == False or eval_mode == 'distractor':
            self.mug_scale = 1.
            self.mug_id = p.loadURDF("assets/mug.urdf", basePosition=[0.5, -0.4, 0.05], globalScaling=self.mug_scale, physicsClientId = self.physicsClientId)
        else:
            raise KeyError
        self.hook_scale = 1.
        self.hook_id = p.loadURDF("assets/hanger.urdf", basePosition=[0.5, 0., 0.], globalScaling=self.hook_scale, physicsClientId = self.physicsClientId, useFixedBase = True)

        if use_box:
            self.box_h = 0.3
            self.box_id = create_box(w=0.1, l=0.1, h=self.box_h, color=(72/255,72/255,72/255,1.))
        else:
            self.box_id = None

        if eval_mode == 'distractor':
            self.lego_scale = 3.
            self.lego_id = p.loadURDF("assets/hongin/lego.urdf", basePosition=[0.7, -0.2, 0.35], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), globalScaling=self.lego_scale, physicsClientId = self.physicsClientId)
            self.duck_scale = 0.1
            self.duck_id = p.loadURDF("assets/hongin/duck.urdf", basePosition=[0.7, 0.2, 0.35], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), globalScaling=self.duck_scale, physicsClientId = self.physicsClientId)
            self.torus_scale = 0.1
            self.torus_id = p.loadURDF("assets/hongin/torus_textured.urdf", basePosition=[0.5, -0.2, 0.35], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), globalScaling=self.torus_scale, physicsClientId = self.physicsClientId)


    def reset(self, seed = None, eval_mode = False, cup_pose='upright', use_box = False, step=1000):
        super().reset(seed = seed, step = False, eval_mode = eval_mode, use_box = use_box)
        self.target_item = self.mug_id

        randomize_mug_pos = True
        randomize_hook_pos = True
        if eval_mode == 'distractor':
            #randomize_mug_pos, randomize_hook_pos = False, False
            pass

        # reset mug
        #mug_orn = np.array([0., 0., np.pi / 2])
        #mug_orn = np.array([0., 0., np.pi * 0.5])
        #mug_orn = np.array([0., 0., np.random.rand()*np.pi*2])
        #mug_orn = np.array([np.pi /2, np.random.rand()*np.pi*2, 0.])
        #mug_orn = np.random.randn(3)*100

        if cup_pose == 'upright':
            mug_orn = np.array([0., 0., np.random.rand()*np.pi*2])
        elif cup_pose == 'lying':
            mug_orn = np.array([np.pi /2, 0., np.random.rand()*np.pi*2])
            #mug_orn = np.array([np.pi /2, np.random.rand()*np.pi*2, 0.])
        else:
            raise KeyError

        if self.box_id is not None:
            box_pos = self.center + np.array([0., 0., 2.])
            if randomize_mug_pos:
                box_pos = box_pos + np.array([(2*np.random.rand() - 1) * 0.05, (2*np.random.rand() - 1) * 0.1, 0.])
            
            box_pos[-1] = stable_z(self.box_id, self.table_id)
            p.resetBasePositionAndOrientation(self.box_id, box_pos, np.array([0., 0., 0., 1.]), physicsClientId = self.physicsClientId)

            mug_orn = p.getQuaternionFromEuler(mug_orn)
            mug_pos = box_pos + np.array([0., 0., 4.])
            p.resetBasePositionAndOrientation(self.mug_id, mug_pos, mug_orn, physicsClientId = self.physicsClientId)

            mug_pos[2] = stable_z(self.mug_id, self.box_id)
            p.resetBasePositionAndOrientation(self.mug_id, mug_pos, mug_orn, physicsClientId = self.physicsClientId)
        else:
            mug_orn = p.getQuaternionFromEuler(mug_orn)
            mug_pos = self.center + np.array([0., 0., 4.])
            if randomize_mug_pos:
                mug_pos = mug_pos + np.array([(2*np.random.rand() - 1) * 0.05, (2*np.random.rand() - 1) * 0.1, 0.])
            p.resetBasePositionAndOrientation(self.mug_id, mug_pos, mug_orn, physicsClientId = self.physicsClientId)

            mug_pos[2] = stable_z(self.mug_id, self.table_id)
            p.resetBasePositionAndOrientation(self.mug_id, mug_pos, mug_orn, physicsClientId = self.physicsClientId)




        if eval_mode == 'distractor':
            lego_pos = self.center + np.array([0.1, -0.1, 0.]) + np.random.randn(3)*0.01
            lego_pos[2] = stable_z(self.lego_id, self.table_id)
            lego_orn = p.getQuaternionFromEuler([0., 0., np.random.rand()*np.pi*2])
            duck_pos = self.center + np.array([0.1, 0.1, 0.]) + np.random.randn(3)*0.01
            duck_pos[2] = stable_z(self.duck_id, self.table_id)
            duck_orn = p.getQuaternionFromEuler([0., 0., np.random.rand()*np.pi*2])
            torus_pos = self.center + np.array([-0.1, -0.15, 0.]) + np.random.randn(3)*0.01
            torus_pos[2] = stable_z(self.torus_id, self.table_id)
            torus_orn = p.getQuaternionFromEuler([0., 0., np.random.rand()*np.pi*2])

            p.resetBasePositionAndOrientation(self.lego_id, lego_pos, lego_orn, physicsClientId = self.physicsClientId)
            p.resetBasePositionAndOrientation(self.duck_id, duck_pos, duck_orn, physicsClientId = self.physicsClientId)
            p.resetBasePositionAndOrientation(self.torus_id, torus_pos, torus_orn, physicsClientId = self.physicsClientId)

        hook_pos = self.center + np.array([0.2, 0. ,-0.15]) 
        if randomize_hook_pos:
            hook_pos = hook_pos + np.array([0., (2*np.random.rand() - 1) * 0.1, 0.])
        hook_orn = np.array([0., 0., np.pi]) + np.array([0., 0., (2*np.random.rand() - 1) * np.pi / 4])
        #hook_pos = self.center + np.array([0.2, 0. ,-0.15])
        #hook_orn = np.array([0., 0., np.pi])
        hook_orn = p.getQuaternionFromEuler(hook_orn)
        p.resetBasePositionAndOrientation(self.hook_id, hook_pos, hook_orn, physicsClientId = self.physicsClientId)

        for i in range(12):
            p.setCollisionFilterPair(self.hook_id, self.robot, -1, i, 0, self.physicsClientId)
            p.setCollisionFilterPair(self.hook_id, self.robot, 0, i, 0, self.physicsClientId)
            if eval_mode == 'distractor':
                p.setCollisionFilterPair(self.lego_id, self.robot, -1, i, 0, self.physicsClientId)
                p.setCollisionFilterPair(self.duck_id, self.robot, -1, i, 0, self.physicsClientId)
                p.setCollisionFilterPair(self.torus_id, self.robot, -1, i, 0, self.physicsClientId)
        if eval_mode == 'distractor':
            p.setCollisionFilterPair(self.lego_id, self.mug_id, -1, -1, 0, self.physicsClientId)
            p.setCollisionFilterPair(self.duck_id, self.mug_id, -1, -1, 0, self.physicsClientId)
            p.setCollisionFilterPair(self.torus_id, self.mug_id, -1, -1, 0, self.physicsClientId)

        # step sim
        if step:
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
        X_s_branch, R_s_branch = p.getLinkState(self.hook_id, 0)[:2]
        X_s_branch, R_s_branch = np.array(X_s_branch), Rotation.from_quat(R_s_branch).as_matrix()
        R_s_tip = R_s_branch.copy()
        X_s_tip = X_s_branch + (R_s_branch @ np.array([0., 0., 0.1]))

        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_sh, R_sh = X_sm + (R_sm @ X_mh), R_sm @ R_mh

        state['T_sm'] = (X_sm, R_sm)
        state['T_sh'] = (X_sh, R_sh)
        state['T_s_hook'] = (X_s_hook, R_s_hook)
        state['T_s_tip'] = (X_s_tip, R_s_tip)
        state['T_s_branch'] = (X_s_branch, R_s_branch)

        return state

    def oracle_pick_rim(self, random_180_flip = False):
        state = self.get_state()
        X_sm, R_sm = state['T_sm']
        X_se, R_se = state['T_se']

        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_m_top, R_m_top = self.X_m_top, self.R_m_top

        yaw = -np.pi / 2 # 0 ~ pi
        R_top_rim = p.getMatrixFromQuaternion(p.getQuaternionFromEuler(np.array([0., 0., yaw])))
        R_top_rim = np.array(R_top_rim).reshape(3,3)
        rim_X_top_rim = np.array([0.035, 0., 0.])
        X_top_rim = R_top_rim @ rim_X_top_rim
        X_rim_dg = np.array([0., 0., -0.02])
        R_rim_dg = np.array([[0. ,1. ,0.],
                             [1. ,0. ,0.],
                             [0. ,0. ,-1.]])

        R_sdg = R_sm @ R_m_top @ R_top_rim @ R_rim_dg
        X_top_dg = X_top_rim + (R_top_rim @ X_rim_dg)
        X_mdg = X_m_top + (R_m_top @ X_top_dg)
        X_sdg = X_sm + (R_sm @ X_mdg)

        R_dg_dgpre = np.eye(3)
        R_s_dgpre = R_sdg @ R_dg_dgpre
        X_dg_dgpre = np.array([0., 0., -0.1])
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

    def oracle_pick_handle(self, random_180_flip = False):
        state = self.get_state()
        X_sm, R_sm = state['T_sm']
        X_se, R_se = state['T_se']

        X_mh, R_mh = self.X_mh.copy(), self.R_mh.copy()
        X_m_top, R_m_top = self.X_m_top, self.R_m_top

        X_hdg = np.array([0., 0.03, 0.0]) * self.mug_scale
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
        if flip is True:
            R_flip = np.array([[-1. ,0. ,0.],
                               [0. ,1. ,0.],
                               [0. ,0. ,-1.]])
            R_sdg = R_sdg @ R_flip

        X_mdg = X_mh + (R_mh @ X_hdg)
        X_sdg = X_sm + (R_sm @ X_mdg)

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
        X_dg_dgpre = np.array([0., 0., -0.1])
        sX_dg_dgpre = R_sdg @ X_dg_dgpre
        X_s_dgpre = X_sdg + sX_dg_dgpre

        pre_place = (X_s_dgpre, R_s_dgpre)
        place = X_sdg, R_sdg
        #place = X_s_dtop, R_s_dtop
        

        return pre_place, place

    def oracle_place_handle(self):
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
        X_tip_dh = np.array([0., 0., -0.01])
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
                hookFrame_ID = axiscreator(self.hook_id, 0, physicsClientId = self.physicsClientId)
            elif item == 'hook_branch_offset':
                axiscreator(self.hook_id, 0, offset = np.array([0., 0., 0.03]),physicsClientId = self.physicsClientId)
            elif item == 'distractors':
                axiscreator(self.duck_id, physicsClientId = self.physicsClientId)
                axiscreator(self.torus_id, physicsClientId = self.physicsClientId)
                axiscreator(self.lego_id, physicsClientId = self.physicsClientId)

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
        state = self.get_state()
        X_s_branch, R_s_branch = state['T_s_branch']
        X_s_branch_offset = X_s_branch + (R_s_branch @ np.array([0., 0., 0.03]))

        X_sh = state['T_sh'][0]

        dist = np.linalg.norm(X_sh - X_s_branch_offset)

        if dist < 0.07: # 0.05:
            z_offset = X_sh[2] - X_s_branch[2]
            if z_offset > -0.02:
                return True
        return False