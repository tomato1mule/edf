import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import numpy as np
import torch
import e2cnn
from e2cnn import gspaces
import torch.nn.functional as F
import e2cnn.nn as enn
import kornia as K
import torchvision
from matplotlib import pyplot as plt
from equ_res_3 import dian_res

class Transport:
    ''''equavariant Transport module'''

    # TODO by Haojie, try Resnet_ns + Resnet_ns
    #                 or Resnet    + Resenet_ns

    def __init__(self, in_shape, n_rotations, crop_size, preprocess, device, lite=False, init=False):
        # TODO BY HAOJIE: add lite model
        self.device = device
        self.preprocess = preprocess
        self.n_rotations = n_rotations
        self.iters = 0
        self.crop_size_2 = crop_size  # crop size must be N*16 (e.g. 96)
        self.crop_size_1 = 96

        # Padding the image to get same size output after the cross-relation
        self.pad_size_2 = int(self.crop_size_2 / 2)
        self.padding_2 = np.zeros((3, 2), dtype=int)
        self.padding_2[:2, :] = self.pad_size_2

        # Padding the image to get 96*96 crop centered at pick location
        self.pad_size_1 = int(self.crop_size_1 / 2)
        self.padding_1 = np.zeros((3, 2), dtype=int)
        self.padding_1[:2, :] = self.pad_size_1

        in_shape = np.array(in_shape)
        in_shape[0:2] += self.pad_size_2 * 2
        in_shape = tuple(in_shape)

        if not hasattr(self, 'output_dim'):
            self.output_dim = 3
        if not hasattr(self, 'kernel_dim'):
            self.kernel_dim = 3

        self.gspace = gspaces.Rot2dOnR2(6)
        self.in_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr] * in_shape[-1])
        # self.model_map = Wide_ResNet(16, 4, 0.2, initial_stride=1, N=6, f=False, r=0, num_classes=3).to(self.device)
        # self.model_kernel = Wide_ResNet(16, 4, 0.2, initial_stride=1, N=6, f=False, r=0, num_classes=3).to(self.device)
        if lite:
            self.model_map = dian_res(in_dim=6, out_dim=3, N=6, middle_dim=(16, 32, 64, 128), init=init).to(self.device)
            self.model_kernel = dian_res(in_dim=6, out_dim=3, N=6, middle_dim=(16, 32, 64, 128), init=init).to(self.device)
        else:
            self.model_map = dian_res(in_dim=6, out_dim=3, N=6, middle_dim=(32, 64, 128, 256),init=init).to(self.device)
            self.model_kernel = dian_res(in_dim=6, out_dim=3, N=6, middle_dim=(32, 64, 128, 256),init=init).to(self.device)


        self.parameter = list(self.model_map.parameters()) + list(self.model_kernel.parameters())
        self.optim = torch.optim.Adam(self.parameter, lr=1e-4)

    def forward(self, in_img, p, softmax=True, train=True):
        # The entire image
        img_unprocessed = np.pad(in_img, self.padding_2, mode='constant')
        input_data = self.preprocess(img_unprocessed)
        in_shape = (1,) + input_data.shape
        input_data = input_data.reshape(in_shape).transpose(0, 3, 1, 2)
        input_tensor = torch.from_numpy(input_data).to(self.device)
        # The crop
        crop = np.pad(in_img, self.padding_1, mode='constant')
        crop = self.preprocess(crop)
        in_shape = (1,) + crop.shape
        crop = crop.reshape(in_shape).transpose(0, 3, 1, 2)
        # pivot = np.array([p[1], p[0]]) + self.pad_size_1 # the pivot in the entrire image with 96/2 padding each side
        crop = crop[:, :, p[0]:(p[0] + self.crop_size_1), p[1]:(p[1] + self.crop_size_1)]
        crop_input = torch.from_numpy(crop).to(self.device)

        if not train:
            self.model_map.eval()
            self.model_kernel.eval()
            with torch.no_grad():
                _, logits = self.model_map(input_tensor)
                _, kernel_raw = self.model_kernel(crop_input)
        else:
            _, logits = self.model_map(input_tensor)
            _, kernel_raw = self.model_kernel(crop_input)
        # get the tensor in the field
        logits = logits.tensor
        kernel_raw = kernel_raw.tensor
        # Rotate the cropped feature conterclockwise and conduct another crop to get 65x65 kernels
        pivot = int(self.crop_size_1 / 2)
        assert pivot == int(kernel_raw.shape[-1] / 2)
        # print('pivot',pivot)
        half_length = self.pad_size_2
        l, r = pivot - half_length, pivot + half_length + 1
        b, u = pivot - half_length, pivot + half_length + 1
        kernel_raw = kernel_raw.repeat(self.n_rotations, 1, 1, 1)
        kernel_raw = K.geometry.rotate(kernel_raw,
                                       torch.from_numpy(np.linspace(0., 360., self.n_rotations,
                                                                    endpoint=False, dtype=np.float32)).to(self.device),
                                       mode='nearest')
        kernel = kernel_raw[:, :, l:r, b:u]
        output = F.conv2d(input=logits, weight=kernel)

        if softmax:
            output_shape = output.shape
            output = output.reshape(-1)
            output = F.softmax(output, dim=-1)
            output = output.reshape(output_shape[1:]).detach().cpu().numpy()
            output = output.transpose(1, 2, 0)
        return output

    def train(self, in_img, p, q, theta, backprop=True):
        """Transport pixel p to pixel q.

        Args:
          in_img: input image.
          p: pixel (y, x)
          q: pixel (y, x)
          theta: rotation label in radians.
          backprop: True if backpropagating gradients.

        Returns:
          loss: training loss.
        """
        # print('hello from equ_transporter')
        self.model_map.train()
        self.model_kernel.train()
        output = self.forward(in_img, p, softmax=False)
        output = output.reshape(1, -1)

        itheta = theta / (2 * np.pi / self.n_rotations)
        itheta = np.int32(np.round(itheta)) % self.n_rotations
        # Get one-hot pixel label map.
        label_size = (self.n_rotations,) + in_img.shape[:2]
        label = torch.zeros(label_size, dtype=torch.long, device=self.device)
        label[itheta, q[0], q[1],] = 1
        label = label.reshape(1, -1)
        label = torch.argmax(label).unsqueeze(dim=0)
        # Get loss
        loss = F.cross_entropy(input=output, target=label)

        if backprop:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        self.iters += 1
        return np.float32(loss.item())

    def load(self, path1, path2):
        # safe operation for e2cnn
        self.model_map.eval()
        self.model_kernel.eval()
        self.model_map.load_state_dict(torch.load(path1, map_location=self.device))
        self.model_kernel.load_state_dict(torch.load(path2, map_location=self.device))

    def save(self, filename1, filename2):
        # safe operation for e2cnn
        self.model_map.eval()
        self.model_kernel.eval()
        torch.save(self.model_map.state_dict(), filename1)
        torch.save(self.model_kernel.state_dict(), filename2)

    def imshow(self, input: torch.Tensor, size: tuple = None, center: bool = False):
        input_ = input[:, 0:3, :, :]
        if center:
            center_x = int(input_.shape[-2] / 2)
            center_y = int(input_.shape[-1] / 2)
            # input_[:,:,center_x,center_y]=[0,1,0]
        out = torchvision.utils.make_grid(input_, nrow=4, padding=5)
        out_np: np.ndarray = K.utils.tensor_to_image(out)
        plt.figure(figsize=size)
        plt.imshow(out_np)
        plt.axis('off')
        plt.show()
