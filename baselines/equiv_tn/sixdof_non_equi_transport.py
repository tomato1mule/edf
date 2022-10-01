import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import numpy as np
import torch
from .pytorch_resnet import ResNet43
from e2cnn import gspaces
import torch.nn.functional as F
import e2cnn.nn as enn
import kornia as K
import torchvision
from matplotlib import pyplot as plt

debug = {}

class Transport:
    ''''equavariant Transport module'''
    #TODO by Haojie, try Resnet_ns + Resnet_ns
    #                 or Resnet    + Resenet_ns

    def __init__(self, in_shape, n_rotations, crop_size, device, preprocess=None, log_std = None, lr=1e-4):
        # TODO BY HAOJIE: add lite model
        self.device = device
        self.preprocess = preprocess
        self.n_rotations = n_rotations
        self.iters = 0
        self.crop_size_2 = crop_size  # crop size must be N*16 (e.g. 96)
        self.crop_size_1 = crop_size#96

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


        # scale
        self.zrp_scale = torch.tensor([(0.4 - 0.0) / 2, np.pi, np.pi/2], dtype = torch.float32, device=device) / 4
        self.zrp_offset = torch.tensor([(0.4 + 0.0) / 2, np.pi, 0.], dtype = torch.float32, device=device)


        log_std = np.log(np.array([1., 1., 1.])) #np.log(np.array([0.3, 3., 1.]))

        
        # get the location
        emb_in = 10
        emb_out = 10
        self.log_std = log_std
        if log_std is None:
            self.zrp_dim = 6
            self.output_dim = emb_in
            self.kernel_dim = emb_in * emb_out
        else:
            self.zrp_dim = 3
            self.log_std = torch.tensor(log_std, dtype = torch.float32, device=device)
            assert self.log_std.dim() == 1 and len(self.log_std) == 3
            self.output_dim = emb_in
            self.kernel_dim = emb_in * emb_out
        self.emb_in = emb_in
        self.emb_out = emb_out

        self.use_pick_zrp = True

        self.in_type = in_shape[-1] # 6
        self.model_map = ResNet43(self.in_type, outdim=self.output_dim).to(self.device)
        self.model_kernel = ResNet43(self.in_type + 1*self.use_pick_zrp, outdim=self.kernel_dim).to(self.device)
        self.prob_linear = torch.nn.Linear(emb_out, 1).to(device)
        self.zrp_linear = torch.nn.Linear(emb_out, self.zrp_dim).to(device)


        self.parameter = list(self.model_map.parameters()) + list(self.model_kernel.parameters()) + list(self.prob_linear.parameters()) + list(self.zrp_linear.parameters())
        self.optim = torch.optim.Adam(self.parameter, lr=lr)

    def forward(self,in_img,p, z_pick, roll_pick, pitch_pick, softmax=True,train=True, return_crop = False):
        # The entire image
        img_unprocessed = np.pad(in_img, self.padding_2, mode='constant')
        if self.preprocess is not None:
            input_data = self.preprocess(img_unprocessed.copy())
        else:
            input_data = img_unprocessed.copy()
        in_shape = (1,) + input_data.shape
        input_data = input_data.reshape(in_shape).transpose(0, 3, 1, 2)
        input_tensor = torch.from_numpy(input_data).to(self.device)
        #print('input map',input_tensor.shape)
        # The crop
        #print('before padding',in_img.shape)
        crop = np.pad(in_img, self.padding_1, mode='constant')
        if self.preprocess is not None:
            raise NotImplementedError
            crop = self.preprocess(crop)
        in_shape = (1,) + crop.shape
        crop = crop.reshape(in_shape).transpose(0, 3, 1, 2)
        crop = torch.from_numpy(crop)
        crop = crop.repeat(self.n_rotations,1,1,1)

        #print('before rotate',crop.shape)
        #self.imshow(crop,size=(36,36))
        pivot = np.array([p[1], p[0]]) + self.pad_size_1
        pivot = torch.from_numpy(pivot).repeat(self.n_rotations,1).to(torch.float32)
        crop = K.geometry.rotate(crop,torch.from_numpy(np.linspace(0., 360., self.n_rotations,
                                                                    endpoint=False,dtype=np.float32)),
                                 mode='nearest',center=pivot).to(self.device)                            # this is nondeterministic on cuda so run on cpu
        #print('after rotate', crop.shape)
        # self.imshow(crop, size=(36, 36))

        crop_input = crop[:,:,p[0]:(p[0] + self.crop_size_1),p[1]:(p[1] + self.crop_size_1)] # (nRot, 6, cropH, cropW)
        #print('after crop', crop_input.shape)
        #self.imshow(crop_input, size=(36, 36))
        #self.imshow(crop_input)
        #print('after crop',crop.shape)
        # pass the entire image and crop to the network

        if self.use_pick_zrp is True:
            pick_zrp = np.array([z_pick, roll_pick, pitch_pick])
            pick_zrp = (pick_zrp - self.zrp_offset.detach().cpu().numpy()) / self.zrp_scale.detach().cpu().numpy()
            pick_zrp = torch.tensor(pick_zrp, device=crop_input.device, dtype=crop_input.dtype)[None, :, None, None].repeat(crop_input.shape[-4], 1, crop_input.shape[-2], crop_input.shape[-1]) # (nRot, 3, cropH, cropW)
            crop_input = torch.cat([crop_input[...,:4, :, :], pick_zrp], dim=-3)

        if not train:
            self.model_map.eval()
            self.model_kernel.eval()
            self.prob_linear.eval()
            self.zrp_linear.eval()
            with torch.no_grad():
                logits = self.model_map(input_tensor) # (1, emb_in, cropH, cropW)
                kernel_raw = self.model_kernel(crop_input) # (nRot, emb_in * emb_out, cropH, cropW)
        else:
            logits = self.model_map(input_tensor) # (1, emb_in, cropH, cropW)
            kernel_raw = self.model_kernel(crop_input) # (nRot, emb_in * emb_out, cropH, cropW)
        #print('after model',kernel_raw.shape)
        pivot = int(self.crop_size_1 / 2)
        assert pivot == int(kernel_raw.shape[-1] / 2)
        # print('pivot',pivot)
        half_length = self.pad_size_2
        l, r = pivot - half_length, pivot + half_length+1
        b, u = pivot - half_length, pivot + half_length+1

        kernel_raw = kernel_raw[:,:,l:r,b:u] # (nRot, emb_in * emb_out, cropH, cropW)

        # kernel_normalized = kernel_raw.detach()
        # kernel_normalized = kernel_normalized - kernel_normalized.view(len(kernel_normalized),-1).min(dim=1).values[:,None,None,None]
        # kernel_normalized = (kernel_normalized) / (kernel_normalized.view(len(kernel_normalized),-1).max(dim=1).values[:,None,None,None] + 1e-8)
        # kernel_normalized = kernel_normalized.cpu()
        # self.imshow(kernel_normalized, size=(36, 36))
        
        #print('crop')
        #np.save('crop.npy',kernel_raw.cpu().detach().numpy())
        #print('after model kernel', kernel_raw.shape)
        #self.imshow(kernel_raw, size=(36, 36))
        #p2d = (0, 1, 0, 1)
        #kernel_raw = F.pad(kernel_raw,p2d)
        #print('after pad',kernel_raw.size())
        #self.imshow(kernel_raw, size=(36, 36))
        # correlation step
        kernel_raw = kernel_raw.reshape(self.n_rotations, self.emb_out, self.emb_in, *(kernel_raw.shape[-2:])).reshape(self.n_rotations*self.emb_out, self.emb_in, *(kernel_raw.shape[-2:])) # (nRot*emb_out, emb_in, cropH, cropW)
        output = F.conv2d(input=logits,weight=kernel_raw) # (1, nRot*emb_out, H, W)
        output = output.reshape(self.n_rotations, -1, *(output.shape[-2:])) # (nRot, emb_out, H, W)
        output = output[...,:-1,:-1]
        assert output.shape[-2:] == in_img.shape[-3:-1]

        output = output / np.sqrt(kernel_raw.shape[-1] * kernel_raw.shape[-2] * self.emb_in * self.emb_out)
        #print(output.std().item())

        logits = self.prob_linear(output.permute(0,2,3,1)).permute(0,3,1,2) # (nRot, 1, H, W)
        zrp = self.zrp_linear(output.permute(0,2,3,1)).permute(0,3,1,2) # (nRot, zrp_dim, H, W)

        if self.log_std is None:
            zrp_log_std = zrp[:, 3:] # (nRot, 3, H, W)
        else:
            zrp_log_std = self.log_std.clone() # (3,)
            zrp_log_std = zrp_log_std[None, :, None, None].repeat(zrp.shape[0], 1, zrp.shape[-2], zrp.shape[-1])
        zrp = zrp[:, 0:3] # (nRot, 3, H, W)


        # if softmax:
        #     output_shape = output.shape
        #     output = output.reshape(-1)
        #     output = F.softmax(output,dim=-1)
        #     output = output.reshape(output_shape[1:]).detach().cpu().numpy()
        #     output = output.transpose(1,2,0)
        # return output

        assert logits.shape[-2:] == in_img.shape[-3:-1]
        prob = logits.reshape(1,-1) # (1, nRot * H * W)
        if softmax:
            prob = F.softmax(prob,dim=-1)
            prob = prob.reshape(logits.shape[0],logits.shape[-2],logits.shape[-1]).cpu().detach().numpy()
            #print('prob',prob.shape)
            prob = prob.transpose(1,2,0) # (H, W, nRot)
        else:
            prob = F.log_softmax(prob, dim=-1)
            prob = prob.reshape(logits.shape[0],logits.shape[-2],logits.shape[-1])
            prob = prob.permute(1,2,0) # (H, W, nRot)

        zrp = zrp.permute(2,3,0,1) # (H,W,nRot,3)
        zrp_log_std = zrp_log_std.permute(2,3,0,1) # (H,W,nRot,3)

        zrp = zrp * self.zrp_scale
        zrp_log_std = zrp_log_std + torch.log(self.zrp_scale)
        zrp = zrp + self.zrp_offset
        r = (zrp[..., 1] + 2*np.pi)%(2*np.pi)
        zrp = torch.stack([zrp[...,0], r, zrp[...,2]], dim=-1)

        if not return_crop:
            return prob, zrp, zrp_log_std # (H,W,nRot), (H,W,nRot,3), (H,W,nRot,3)
        else:
            return prob, zrp, zrp_log_std, crop_input

    def train(self, in_img, p, q, theta, z, roll, pitch, z_pick, roll_pick, pitch_pick, backprop=True):
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
        self.model_map.train()
        self.model_kernel.train()
        self.prob_linear.train()
        self.zrp_linear.train()
        if backprop:
            self.optim.zero_grad(set_to_none=True)
        logits, zrp, zrp_log_std = self.forward(in_img,p, z_pick, roll_pick, pitch_pick, softmax=False) # (H,W,nRot), (H,W,nRot,3), (H,W,nRot,3)
        
        # # Get one-hot pixel label map.
        # itheta = theta / (2 * np.pi / self.n_rotations)
        # itheta = np.int32(np.round(itheta)) % self.n_rotations
        # label_size = (self.n_rotations,) + in_img.shape[:2]
        # label = torch.zeros(label_size, dtype=torch.long,device=self.device)
        # label[itheta, q[0], q[1],] = 1
        # label = label.reshape(-1)
        # label = torch.argmax(label).unsqueeze(dim=0)
        # # Get loss
        # loss = F.cross_entropy(input=output, target=label)

        theta = (theta + 2*np.pi)%(2*np.pi)
        theta_i = np.int32(np.round(theta / (2*np.pi/self.n_rotations))) % self.n_rotations
        loss = -logits[q[0], q[1], theta_i]

        # Get zrp loss
        zrp_target = torch.tensor([z, roll, pitch], device=loss.device, dtype = loss.dtype)
        zrp = zrp[q[0], q[1], theta_i, :] #(3,)
        zrp_log_std = zrp_log_std[q[0], q[1], theta_i, :] #(3,)
        #print((zrp_log_std-self.zrp_scale.log()).detach().cpu().exp())
        zrp_diff = zrp - zrp_target
        r_diff = (zrp_diff[1] + 2*np.pi)%(2*np.pi) # (1,)
        r_diff_rev = (2*np.pi - r_diff)            # (1,)
        r_diff = torch.stack([r_diff, r_diff_rev], dim=0) # (2,1)
        r_diff = torch.min(r_diff, dim=0).values # (1,)
        # if r_diff.item() > np.pi*2*3/4:
        #     r_diff = (2*np.pi - r_diff)            # (1,)
        zrp_diff = torch.stack([zrp_diff[0], r_diff, zrp_diff[2]], dim=-1) # (3,)

        loss_zrp = (zrp_diff).square() * (-2 * zrp_log_std).exp() #(3,)
        if self.log_std is None:
             loss_zrp = loss_zrp + (2*zrp_log_std)
        loss_zrp = loss_zrp.sum(dim=-1) * 0.5 # (1,)
        #loss_zrp = loss_zrp[0] * 0.5 # (1,)

        loss = loss + loss_zrp

        if backprop:
            loss.backward()
            self.optim.step()
        self.iters +=1
        return np.float32(loss.item())

    def load(self,path1,path2):
        # safe operation for e2cnn
        self.model_map.eval()
        self.model_kernel.eval()
        self.prob_linear.eval()
        self.zrp_linear.eval()

        self.model_map.load_state_dict(torch.load(path1,map_location=self.device))
        self.model_kernel.load_state_dict(torch.load(path2,map_location=self.device))
        self.prob_linear.load_state_dict(torch.load(path1[:-3] + '_prob.pt', map_location=self.device))
        self.zrp_linear.load_state_dict(torch.load(path1[:-3] + '_zrp.pt', map_location=self.device))

    def save(self,filename1,filename2):
        # safe operation for e2cnn
        self.model_map.eval()
        self.model_kernel.eval()
        self.prob_linear.eval()
        self.zrp_linear.eval()
        torch.save(self.model_map.state_dict(), filename1)
        torch.save(self.model_kernel.state_dict(), filename2)
        torch.save(self.prob_linear.state_dict(), filename1[:-3] + '_prob.pt')
        torch.save(self.zrp_linear.state_dict(), filename1[:-3] + '_zrp.pt')

    def imshow(self,input: torch.Tensor, size: tuple = None, center: bool= False):
        input_ = input[:,0:3,:,:]
        if center:
            center_x = int(input_.shape[-2]/2)
            center_y = int(input_.shape[-1]/2)
            #input_[:,:,center_x,center_y]=[0,1,0]
        out = torchvision.utils.make_grid(input_, nrow=6, padding=5)
        out_np: np.ndarray = K.utils.tensor_to_image(out)
        plt.figure(figsize=size)
        plt.imshow(out_np)
        plt.axis('off')
        plt.show()

