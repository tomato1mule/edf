import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append('..')
import time
import numpy as np
import os
import torch
import e2cnn
from e2cnn import gspaces
import torch.nn.functional as F
import e2cnn.nn as enn
from equ_transport_tail import Transport
from equ_attention import Attention
from raven import cameras
from raven import utils
# to use tensorboard
import tensorflow as tf



class TransporterAgent:
  """Agent that uses Transporter Networks."""

  def __init__(self, name, task, root_dir, device=1, n_rotations=36,lite=False,load=False, angle_lite=False,init=False):
    self.name = name
    self.task = task
    self.total_steps = 0
    self.crop_size = 64
    self.n_rotations = n_rotations
    self.pix_size = 0.003125
    self.in_shape = (320, 160, 6)
    self.cam_config = cameras.RealSenseD415.CONFIG
    self.models_dir = os.path.join(root_dir, 'checkpoints_tail', self.name)
    self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
    if device==1:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
      device = torch.device('cpu')

    self.device =device

    self.attention = Attention(
      in_shape=self.in_shape,
      n_rotations=1,
      preprocess=utils.preprocess,
      device=self.device,
      lite = lite,
      angle_lite = angle_lite,
      init=init)
                
    self.transport = Transport(
      in_shape=self.in_shape,
      n_rotations=self.n_rotations,
      crop_size=self.crop_size,
      preprocess=utils.preprocess,
      device=self.device,
      lite = lite,
      init=init)
      
    if load!=False:
      #print('load pretained model checkpoint at {} step'.format(load))
      self.load(load)
      self.total_steps = load

  def get_image(self, obs):
    """Stack color and height images image."""

    # if self.use_goal_image:
    #   colormap_g, heightmap_g = utils.get_fused_heightmap(goal, configs)
    #   goal_image = self.concatenate_c_h(colormap_g, heightmap_g)
    #   input_image = np.concatenate((input_image, goal_image), axis=2)
    #   assert input_image.shape[2] == 12, input_image.shape

    # Get color and height maps from RGB-D images.
    cmap, hmap = utils.get_fused_heightmap(
        obs, self.cam_config, self.bounds, self.pix_size)
    #print('depth image',hmap.shape, hmap[Ellipsis, None].shape)
    img = np.concatenate((cmap,
                          hmap[Ellipsis, None],
                          hmap[Ellipsis, None],
                          hmap[Ellipsis, None]), axis=2)
    assert img.shape == self.in_shape, img.shape
    return img

  def get_sample(self, dataset, augment=True):
    """Get a dataset sample.

    Args:
      dataset: a ravens.Dataset (train or validation)
      augment: if True, perform data augmentation.

    Returns:
      tuple of data for training:
        (input_image, p0, p0_theta, p1, p1_theta)
      tuple additionally includes (z, roll, pitch) if self.six_dof
      if self.use_goal_image, then the goal image is stacked with the
      current image in `input_image`. If splitting up current and goal
      images is desired, it should be done outside this method.
    """

    (obs, act, _, _), _ = dataset.sample()
    img = self.get_image(obs)
    # Get training labels from data sample.
    p0_xyz, p0_xyzw = act['pose0']
    p1_xyz, p1_xyzw = act['pose1']
    
    #p0 theta is the global angle
    p0 = utils.xyz_to_pix(p0_xyz, self.bounds, self.pix_size)
    p0_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p0_xyzw)[2])
    p0_theta = (2*np.pi + p0_theta)%(2 * np.pi)
    #p1 theta is the difference (no change during transform)
    p1 = utils.xyz_to_pix(p1_xyz, self.bounds, self.pix_size)
    p1_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p1_xyzw)[2])
    #p1_theta = (p1_theta + 2*np.pi)%(2*np.pi)
    # Data augmentation.
    if augment:
      img, _, (p0, p1), (theta,trans,pivot) = utils.perturb(img, [p0, p1])
      p0_theta = p0_theta - theta
    #print('input',img.shape)
    #print('label,p0,p0_theta,p1,p1_theta',p0, p0_theta, p1, p1_theta)
    return img, p0, p0_theta, p1, p1_theta

  def train(self, dataset, writer=None):
    """Train on a dataset sample for 1 iteration.

    Args:
      dataset: a ravens.Dataset.
      writer: a TF summary writer (for tensorboard).
    """
    time_0 = time.time()
    img, p0, p0_theta, p1, p1_theta = self.get_sample(dataset)
    #print('img',img.shape)
    #print('p0',p0)
    #print('p0_theta',p0_theta)
    #print('p1', p1)
    #print('p1_theta', p1_theta)
    # Get training losses.
    step = self.total_steps + 1
    loss00,loss01 = self.attention.train(img, p0, p0_theta)
    loss1 = self.transport.train(img, p0, p1, p1_theta)
    # TODO by Haojie, ADD THE WRITER
    if writer!=None:
      with writer.as_default():
        sc = tf.summary.scalar
        sc('train_loss/attention', loss00, step)
        sc('train_loss/attention2', loss01, step)
        sc('train_loss/transport', loss1, step)
    time_0 = time.time() -time_0
    print(f'Train Iter: {step} Loss: {loss00:.4f} {loss01:.4f} {loss1:.4f} time: {time_0:.4f}')
    self.total_steps = step

  def validate(self, dataset, writer=None):  # pylint: disable=unused-argument
    """Test on a validation dataset for 10 iterations."""
    print('Skipping validation.')
    # tf.keras.backend.set_learning_phase(0)
    # n_iter = 10
    # loss0, loss1 = 0, 0
    # for i in range(n_iter):
    #   img, p0, p0_theta, p1, p1_theta = self.get_sample(dataset, False)

    #   # Get validation losses. Do not backpropagate.
    #   loss0 += self.attention.train(img, p0, p0_theta, backprop=False)
    #   if isinstance(self.transport, Attention):
    #     loss1 += self.transport.train(img, p1, p1_theta, backprop=False)
    #   else:
    #     loss1 += self.transport.train(img, p0, p1, p1_theta, backprop=False)
    # loss0 /= n_iter
    # loss1 /= n_iter
    # with writer.as_default():
    #   sc = tf.summary.scalar
    #   sc('test_loss/attention', loss0, self.total_steps)
    #   sc('test_loss/transport', loss1, self.total_steps)
    # print(f'Validation Loss: {loss0:.4f} {loss1:.4f}')

  def act(self, obs, info=None, goal=None):  # pylint: disable=unused-argument
    """Run inference and return best action given visual observations."""
    # Get heightmap from RGB-D images.
    img = self.get_image(obs)

    # Attention model forward pass.
    time1 = time.time()
    pick_conf, p0_theta, _ = self.attention.forward(img,train=False)
    #print('pick_conf',pick_conf.shape)
    
    #print('save pick conf')
    #np.save('pick_conf.npy',pick_conf)
    argmax = np.argmax(pick_conf)
    argmax = np.unravel_index(argmax, shape=pick_conf.shape)
    p0_pix = argmax[:2]
    #print('pick location',p0_pix)
    p0_theta = np.argmax(p0_theta)
    #print('p0_theta_index',p0_theta)
    p0_theta =  p0_theta * (2 * np.pi / self.n_rotations)

    # Transport model forward pass.
    place_conf = self.transport.forward(img, p0_pix,train=False)
    #print('place_conf',place_conf.shape)
    argmax = np.argmax(place_conf)
    argmax = np.unravel_index(argmax, shape=place_conf.shape)
    p1_pix = argmax[:2]
    p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])
    
    #print('time:',time.time()-time1)
    
    #print('place location and theta', p1_pix,p1_theta)
    
    #print('img',img.shape)
    #print('p0',p0_pix)
    #print('p0_theta',p0_theta)
    #print('p1', p1_pix)
    #print('p1_theta', p1_theta)
    # Get training losses.

    # Pixels to end effector poses.
    hmap = img[:, :, 3]
    p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
    p1_xyz = utils.pix_to_xyz(p1_pix, hmap, self.bounds, self.pix_size)
    
    # change z value for block-insertion
    p0_xyz = list(p0_xyz)
    p0_xyz[2] = 0.03 + 0.04  # cube is 0.08X0.03 X 0.04
    p0_xyz = tuple(p0_xyz)

    p1_xyz = list(p1_xyz)
    p1_xyz[2] = 0.03 + 0.04  # cube is 0.08X0.03 X 0.04
    p1_xyz = tuple(p1_xyz)
    
    # get pose
    p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
    p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))

    return {
        'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
        'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw))
    }

  def load(self, n_iter):
    """Load pre-trained models."""
    print(f'Loading pre-trained model at {n_iter} iterations.')
    attention_fname1 = 'attention-ckpt1-%d.pt' % n_iter
    attention_fname2 = 'attention-ckpt2-%d.pt' % n_iter
    transport_fname1 = 'transport-ckpt1-%d.pt' % n_iter
    transport_fname2 = 'transport-ckpt2-%d.pt' % n_iter
    transport_fname3 = 'transport-ckpt3-%d.pt' % n_iter
    
    attention_fname1 = os.path.join(self.models_dir, attention_fname1)
    attention_fname2 = os.path.join(self.models_dir, attention_fname2)
    transport_fname1 = os.path.join(self.models_dir, transport_fname1)
    transport_fname2 = os.path.join(self.models_dir, transport_fname2)
    transport_fname3 = os.path.join(self.models_dir, transport_fname3)
    print(transport_fname2)
    self.attention.load(attention_fname1,attention_fname2)
    self.transport.load(transport_fname1,transport_fname2,transport_fname3)
    self.total_steps = n_iter

  def save(self):
    """Save models."""
    if not os.path.exists(self.models_dir):
      os.makedirs(self.models_dir)
    attention_fname1 = 'attention-ckpt1-%d.pt' % self.total_steps
    attention_fname2 = 'attention-ckpt2-%d.pt' % self.total_steps
    transport_fname1 = 'transport-ckpt1-%d.pt' % self.total_steps
    transport_fname2 = 'transport-ckpt2-%d.pt' % self.total_steps
    transport_fname3 = 'transport-ckpt3-%d.pt' % self.total_steps
    
    attention_fname1 = os.path.join(self.models_dir, attention_fname1)
    attention_fname2 = os.path.join(self.models_dir, attention_fname2)
    transport_fname1 = os.path.join(self.models_dir, transport_fname1)
    transport_fname2 = os.path.join(self.models_dir, transport_fname2)
    transport_fname3 = os.path.join(self.models_dir, transport_fname3)
    self.attention.save(attention_fname1, attention_fname2)
    self.transport.save(transport_fname1,transport_fname2,transport_fname3)
    print('save the snapshot to {}'.format(self.models_dir))
