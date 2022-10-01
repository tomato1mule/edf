import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append('..')
import time
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
from .sixdof_non_equi_transport import Transport
# from raven import cameras
# from raven import utils
# import tensorflow as tf
from .sixdof_non_equi_attention import Attention


class TransporterAgent:
    """Agent that uses Transporter Networks."""

    def __init__(self, name, task, root_dir, device=1, n_rotations=36, load=False, crop_size = 16*14, pix_size = 0.00125, bounds = np.array([[0.4, 0.8],[-0.2, 0.2], [0., 0.4]]), H = 320, W = 320, lr = 1e-4):
        self.name = name
        self.task = task
        self.total_steps = 0
        self.crop_size = crop_size
        self.n_rotations = n_rotations
        self.pix_size = pix_size
        self.in_shape = (H, W, 6)
        #self.cam_config = cameras.RealSenseD415.CONFIG
        self.models_dir = os.path.join(root_dir, 'checkpoints_non_equi', self.name)
        self.bounds = np.array([[0.4, 0.8],[-0.2, 0.2], [0., 0.4]])

        if device == 1:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == 'cuda':
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        self.device = device

        self.attention = Attention(
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            preprocess=None, # utils.preprocess,
            device=self.device, lr=lr)

        self.transport = Transport(
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=None, # utils.preprocess,
            device=self.device, lr=lr)

        if load != False:
            # print('load pretained model checkpoint at {} step'.format(load))
            self.load(load)
            self.total_steps = load

    # def get_image(self, obs):
    #     """Stack color and height images image."""

    #     # if self.use_goal_image:
    #     #   colormap_g, heightmap_g = utils.get_fused_heightmap(goal, configs)
    #     #   goal_image = self.concatenate_c_h(colormap_g, heightmap_g)
    #     #   input_image = np.concatenate((input_image, goal_image), axis=2)
    #     #   assert input_image.shape[2] == 12, input_image.shape

    #     # Get color and height maps from RGB-D images.
    #     cmap, hmap = utils.get_fused_heightmap(
    #         obs, self.cam_config, self.bounds, self.pix_size)
    #     # print('depth image',hmap.shape, hmap[Ellipsis, None].shape)
    #     img = np.concatenate((cmap,
    #                           hmap[Ellipsis, None],
    #                           hmap[Ellipsis, None],
    #                           hmap[Ellipsis, None]), axis=2)
    #     assert img.shape == self.in_shape, img.shape
    #     return img

    # def get_sample(self, dataset, augment=True):
    #     """Get a dataset sample.

    #     Args:
    #       dataset: a ravens.Dataset (train or validation)
    #       augment: if True, perform data augmentation.

    #     Returns:
    #       tuple of data for training:
    #         (input_image, p0, p0_theta, p1, p1_theta)
    #       tuple additionally includes (z, roll, pitch) if self.six_dof
    #       if self.use_goal_image, then the goal image is stacked with the
    #       current image in `input_image`. If splitting up current and goal
    #       images is desired, it should be done outside this method.
    #     """

    #     (obs, act, _, _), _ = dataset.sample()
    #     img = self.get_image(obs)
    #     #print('img',img.shape)

    #     # Get training labels from data sample.
    #     p0_xyz, p0_xyzw = act['pose0']
    #     p1_xyz, p1_xyzw = act['pose1']

    #     # p0 theta is the global angle
    #     p0 = utils.xyz_to_pix(p0_xyz, self.bounds, self.pix_size)
    #     p0_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p0_xyzw)[2])
    #     p0_theta = (2*np.pi + p0_theta)%(2 * np.pi)
        
    #     # p1 theta is the difference (no change during transform)
    #     p1 = utils.xyz_to_pix(p1_xyz, self.bounds, self.pix_size)
    #     p1_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p1_xyzw)[2])
        
        

    #     # Data augmentation.
    #     if augment:
    #         img, _, (p0, p1), (theta, trans, pivot) = utils.perturb(img, [p0, p1])
    #         p0_theta = p0_theta - theta
    #     # print('input',img.shape)
    #     # print('label,p0,p0_theta,p1,p1_theta',p0, p0_theta, p1, p1_theta)
    #     return img, p0, p0_theta, p1, p1_theta

    def get_sample(self, data):
        img, pick, place = data
        p0 = pick[0]
        p0_theta = pick[1]
        p0_z = pick[2]
        p0_roll = pick[3]
        p0_pitch = pick[4]
        p1 = place[0]
        p1_theta = place[1] - p0_theta
        p1_theta = (2*np.pi + p1_theta)%(2 * np.pi)
        p1_z = place[2]
        p1_roll = place[3]
        p1_pitch = place[4]

        return img, p0, p0_theta, p0_z, p0_roll, p0_pitch, p1, p1_theta, p1_z, p1_roll, p1_pitch

    def train(self, data):
        """Train on a dataset sample for 1 iteration.

        Args:
          dataset: a ravens.Dataset.
          writer: a TF summary writer (for tensorboard).
        """
        time_0 = time.time()
        img, p0, p0_theta, p0_z, p0_roll, p0_pitch, p1, p1_theta, p1_z, p1_roll, p1_pitch = self.get_sample(data)
        # print('img',img.shape)
        # print('p0',p0)
        # print('p0_theta',p0_theta)
        # print('p1', p1)
        # print('p1_theta', p1_theta)
        # Get training losses.
        step = self.total_steps + 1
        loss0 = self.attention.train(img, p0, p0_theta, p0_z, p0_roll, p0_pitch)
        loss1 = self.transport.train(img, p0, p1, p1_theta, p1_z, p1_roll, p1_pitch, p0_z, p0_roll, p0_pitch)
        time_0 = time.time() - time_0
        print(f'Train Iter: {step} Loss: {loss0:.20f} {loss1:.20f} time: {time_0:.4f}', flush=True)
        self.total_steps = step

    # def validate(self, dataset, writer=None):  # pylint: disable=unused-argument
    #     """Test on a validation dataset for 10 iterations."""
    #     print('Skipping validation.')
    #     # tf.keras.backend.set_learning_phase(0)
    #     # n_iter = 10
    #     # loss0, loss1 = 0, 0
    #     # for i in range(n_iter):
    #     #   img, p0, p0_theta, p1, p1_theta = self.get_sample(dataset, False)

    #     #   # Get validation losses. Do not backpropagate.
    #     #   loss0 += self.attention.train(img, p0, p0_theta, backprop=False)
    #     #   if isinstance(self.transport, Attention):
    #     #     loss1 += self.transport.train(img, p1, p1_theta, backprop=False)
    #     #   else:
    #     #     loss1 += self.transport.train(img, p0, p1, p1_theta, backprop=False)
    #     # loss0 /= n_iter
    #     # loss1 /= n_iter
    #     # with writer.as_default():
    #     #   sc = tf.summary.scalar
    #     #   sc('test_loss/attention', loss0, self.total_steps)
    #     #   sc('test_loss/transport', loss1, self.total_steps)
    #     # print(f'Validation Loss: {loss0:.4f} {loss1:.4f}')

    def act(self, img, info=None, goal=None, return_output=False, argmax_policy = True, gt_data = None):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        pick_conf, zrp, zrp_log_std = self.attention.forward(img, train=False) # (H,W,nRot), (H,W,nRot,3), (H,W,nRot,3)

        if argmax_policy:
            argmax = np.argmax(pick_conf)
            argmax = np.unravel_index(argmax, shape=pick_conf.shape)
            p0_pix = argmax[:2]
            p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

            p0_zrp = zrp[p0_pix[0], p0_pix[1], argmax[2]] # (3,)
            p0_z, p0_roll, p0_pitch = p0_zrp.cpu().numpy()

            if gt_data:
                _, p0_pix, p0_theta, p0_z, p0_roll, p0_pitch, __, p1_theta_gt, ____, _____, ______ = self.get_sample(gt_data)
                

            if return_output:
                place_conf, zrp_place, zrp_log_std_place, crop  = self.transport.forward(img, p0_pix, p0_z, p0_roll, p0_pitch, train=False, return_crop=True) # (H,W,nRot), (H,W,nRot,3), (H,W,nRot,3)
            else:
                place_conf, zrp_place, zrp_log_std_place  = self.transport.forward(img, p0_pix, p0_z, p0_roll, p0_pitch, train=False) # (H,W,nRot), (H,W,nRot,3), (H,W,nRot,3)


            argmax = np.argmax(place_conf)
            argmax = np.unravel_index(argmax, shape=place_conf.shape)
            p1_pix = argmax[:2]
            p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])
            p1_theta = p1_theta + p0_theta

            p1_zrp = zrp_place[p1_pix[0], p1_pix[1], argmax[2]] # (3,)
            p1_z, p1_roll, p1_pitch = p1_zrp.cpu().numpy()

            if return_output:
                # print(crop.shape)
                # crop = crop[argmax[2], :3,...].detach().cpu().permute(1,2,0)
                crop = crop[int(p1_theta_gt), :3,...].detach().cpu().permute(1,2,0)
                crop = crop - crop.min()
                crop = crop / crop.max()
                # print(f"argmax: {argmax[2]}")
                # crop = crop[:, :3,...].detach().cpu()

            if return_output is True:
                return (p0_pix, p0_theta, p0_z, p0_roll, p0_pitch), (p1_pix, p1_theta, p1_z, p1_roll, p1_pitch), (pick_conf, place_conf, crop.numpy())
            else:
                return (p0_pix, p0_theta, p0_z, p0_roll, p0_pitch), (p1_pix, p1_theta, p1_z, p1_roll, p1_pitch)
        else:
            raise NotImplementedError


        # # Pixels to end effector poses.
        # hmap = img[:, :, 3]
        # p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
        # p1_xyz = utils.pix_to_xyz(p1_pix, hmap, self.bounds, self.pix_size)
        # # Todo change z value for block-insertion
        # p0_xyz = list(p0_xyz)
        # p0_xyz[2] = 0.03 + 0.04  # cube is 0.08X0.03 X 0.04
        # p0_xyz = tuple(p0_xyz)
        # p1_xyz = list(p1_xyz)
        # p1_xyz[2] = 0.03 + 0.04  # cube is 0.08X0.03 X 0.04
        # p1_xyz = tuple(p1_xyz)
        # p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
        # p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))
        # return {
        #     'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
        #     'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw))
        # }

    def act_pick(self, img):
        pick_conf, zrp, zrp_log_std = self.attention.forward(img, train=False) # (H,W,nRot), (H,W,nRot,3), (H,W,nRot,3)

        return pick_conf, zrp, zrp_log_std # (H,W,nRot), (H,W,nRot,3), (H,W,nRot,3)

    def act_place(self, img, p0_pix, p0_z, p0_roll, p0_pitch):
        place_conf, zrp_place, zrp_log_std_place = self.transport.forward(img, p0_pix, p0_z, p0_roll, p0_pitch, train=False) # (H,W,nRot), (H,W,nRot,3), (H,W,nRot,3)

        return place_conf, zrp_place, zrp_log_std_place # (H,W,nRot), (H,W,nRot,3), (H,W,nRot,3)


    def load(self, n_iter):
        """Load pre-trained models."""
        print(f'Loading pre-trained model at {n_iter} iterations.')
        attention_fname = 'attention-ckpt-%d.pt' % n_iter
        transport_fname1 = 'transport-ckpt1-%d.pt' % n_iter
        transport_fname2 = 'transport-ckpt2-%d.pt' % n_iter
        attention_fname = os.path.join(self.models_dir, attention_fname)
        transport_fname1 = os.path.join(self.models_dir, transport_fname1)
        transport_fname2 = os.path.join(self.models_dir, transport_fname2)
        print(transport_fname2)
        self.attention.load(attention_fname)
        self.transport.load(transport_fname1, transport_fname2)
        self.total_steps = n_iter

    def save(self):
        """Save models."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        attention_fname = 'attention-ckpt-%d.pt' % self.total_steps
        transport_fname1 = 'transport-ckpt1-%d.pt' % self.total_steps
        transport_fname2 = 'transport-ckpt2-%d.pt' % self.total_steps
        attention_fname = os.path.join(self.models_dir, attention_fname)
        transport_fname1 = os.path.join(self.models_dir, transport_fname1)
        transport_fname2 = os.path.join(self.models_dir, transport_fname2)
        self.attention.save(attention_fname)
        self.transport.save(transport_fname1, transport_fname2)
        print('save the snapshot to {}'.format(self.models_dir))
