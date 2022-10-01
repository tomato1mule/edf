import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import numpy as np
import torch
from .pytorch_resnet import ResNet43
import torch.nn.functional as F
import kornia as K
import torchvision
from matplotlib import pyplot as plt


class Attention:
    def __init__(self,in_shape,n_rotations,device,lite=False,preprocess=None, log_std = None, lr=1e-4):
        self.device = device
        self.preprocess = preprocess
        self.n_rotations = n_rotations
        #max_dim = np.max(in_shape[:2])
        #max_dim = 480
        max_dim = int(np.ceil(np.sqrt(in_shape[-2]**2 + in_shape[-3]**2)))
        self.padding = np.zeros((3, 2), dtype=int)
        pad = (max_dim - np.array(in_shape[:2])) / 2
        self.padding[:2] = pad.reshape(2, 1)
        in_shape = np.array(in_shape)
        in_shape += np.sum(self.padding, axis=1)
        in_shape = tuple(in_shape) # (480, 480, 6)
        self.in_type = in_shape[-1]
        # self.in_type = 6, self.outdim=1
        

        # scale
        self.zrp_scale = torch.tensor([(0.4 - 0.0) / 2, np.pi, np.pi/2], dtype = torch.float32, device=device) / 4
        self.zrp_offset = torch.tensor([(0.4 + 0.0) / 2, np.pi, 0.], dtype = torch.float32, device=device)

        # debug
        log_std = np.log(np.array([0.3, 1., 1.]))
        # get the location
        emb_dim = 20
        self.log_std = log_std
        if log_std is None:
            zrp_dim = 6
            #self.log_std_min = torch.log([torch.tensor()])
        else:
            zrp_dim = 3
            self.log_std = torch.tensor(log_std, dtype = torch.float32, device=device)
            assert self.log_std.dim() == 1 and len(self.log_std) == 3

        self.model = ResNet43(self.in_type,outdim=emb_dim,include_batch_normal=False).to(self.device)
        self.prob_linear = torch.nn.Linear(emb_dim, 1).to(device)
        self.zrp_linear = torch.nn.Linear(emb_dim, zrp_dim).to(device)

        # use the location as pivot to rotate the image and get the angle
        #self.angle_model = ResNet43(self.in_type,outdim=1,include_batch_normal=False).to(self.device)
        self.optim = torch.optim.Adam(list(self.model.parameters()) + list(self.prob_linear.parameters()) + list(self.zrp_linear.parameters()), lr=lr)
        
        #self.pad_2 = (80,80,80,80)
        

    def forward(self,in_img,softmax=True,train=True):
        #print('padding',self.padding)
        #print('img',in_img.shape)
        in_data = np.pad(in_img, self.padding, mode='constant')
        #print('indata',in_data.shape)
        if self.preprocess is not None:
            in_data = self.preprocess(in_data)
        in_shape = (1,) + in_data.shape
        in_data = in_data.reshape(in_shape).transpose(0, 3, 1, 2)
        in_data = torch.from_numpy(in_data)
        #print(in_data.size())
        
        # rotate image
        pivot = torch.as_tensor([in_data.shape[-2]/2,in_data.shape[-1]/2])
        #pivot =pivot.to(self.device).repeat(self.n_rotations//2,1).to(torch.float32)
        #in_data = in_data.repeat(self.n_rotations//2,1,1,1)
        pivot =pivot.repeat(self.n_rotations,1).to(torch.float32)
        in_data = in_data.repeat(self.n_rotations,1,1,1)
        #in_data = K.geometry.rotate(in_data,torch.from_numpy(-np.linspace(0., 360., self.n_rotations, endpoint=False, dtype=np.float32))[0:18].to(self.device), mode='nearest',center=pivot)
        in_data = K.geometry.rotate(in_data,torch.from_numpy(-np.linspace(0., 360., self.n_rotations, endpoint=False, dtype=np.float32)), mode='nearest', center=pivot).to(self.device) # (n_rot, 6, H, W)     # this is nondeterministic on cuda so run on cpu
        #print('indata rotate 36/2',in_data.shape)
        #self.imshow(in_data,size=(36,12),name='rotation')

        if not train:
            self.model.eval()
            self.prob_linear.eval()
            self.zrp_linear.eval()
            with torch.no_grad():
                outs = self.model(in_data) # (nRot, emb, H, W)
        else:
            outs = self.model(in_data) # (nRot, emb, H, W)
        
        
        outs = K.geometry.rotate(outs.to('cpu'),torch.from_numpy(np.linspace(0., 360., self.n_rotations,
                                                                    endpoint=False,dtype=np.float32)),
                                 mode='nearest',center=pivot).to(self.device) # (nRot, emb, H, W)                # this is nondeterministic on cuda so run on cpu

        #print('atenion logits1',logits.shape)
        #self.imshow(logits)
        #self.imshow(logits,size=(36,12),name='rotation_back')
        #logits = logits[:,:,80:-80,80:-80]
        #print('first crop',logits.size())
        c0 = self.padding[:2, 0]
        c1 = c0 + in_img.shape[:2]
        #print('crop',c0)
        #print('crop',c1)
        outs = outs[:, :, c0[0]:c1[0], c0[1]:c1[1]] # (nRot, emb, H, W)
        #print('second crop',logits.size())
        #print('attention logits',logits.shape)
        #self.imshow(logits)

        outs = outs * 10. # Normalization
        #print(f"out: {outs.std().detach().cpu().item()}")

        #print(outs.std().item())
        logits = self.prob_linear(outs.permute(0,2,3,1)).permute(0,3,1,2) # (nRot, 1, H, W)
        zrp = self.zrp_linear(outs.permute(0,2,3,1)).permute(0,3,1,2) # (nRot, 6, H, W)

        #print(f"log: {logits.std().detach().cpu().item()}")
        #print(f"zrp: {zrp.std().detach().cpu().item()}")

        if self.log_std is None:
            zrp_log_std = zrp[:, 3:] # (nRot, 3, H, W)
        else:
            zrp_log_std = self.log_std.clone() # (3,)
            zrp_log_std = zrp_log_std[None, :, None, None].repeat(zrp.shape[0], 1, zrp.shape[-2], zrp.shape[-1])
        zrp = zrp[:, 0:3] # (nRot, 3, H, W)



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

        return prob, zrp, zrp_log_std # (H,W,nRot), (H,W,nRot,3), (H,W,nRot,3)

    def train(self,in_img,p,theta, z, roll, pitch, backprop=True):
        self.model.train()
        self.prob_linear.train()
        self.zrp_linear.train()
        if backprop:
            self.optim.zero_grad(set_to_none=True)
        logits, zrp, zrp_log_std = self.forward(in_img,softmax=False) # (H,W,nRot), (H,W,nRot,3), (H,W,nRot,3)
        # Get label
        
        theta = (theta + 2*np.pi)%(2*np.pi)
        # if theta >= np.pi:
        #   theta = theta -np.pi
        # # angle label
        # # dgree interval: 10
        # theta_i = theta / (2 * np.pi / 36)
        # # theta_i is in range [0,17]
        # theta_i = np.int32(np.round(theta_i)) % 18
        theta_i = np.int32(np.round(theta / (2*np.pi/self.n_rotations))) % self.n_rotations
        

        # label_theta = torch.as_tensor(theta_i, dtype=torch.long, device=self.device).unsqueeze(dim=0)
        # #label_size = (self.n_rotations//2,) + in_img.shape[:2]
        # label_size = (self.n_rotations,) + in_img.shape[:2]
        # label = torch.zeros(label_size,dtype=torch.long,device=self.device)
        # label[theta_i, p[0], p[1],] = 1
        # label = label.reshape(-1)
        # label = torch.argmax(label).unsqueeze(dim=0)
        # #print('label size',label.shape)
        # #print('out size', output.shape)
        # # Get loss
        # loss = F.cross_entropy(input=logits, target=label)

        loss = -logits[p[0], p[1], theta_i]

        # Get zrp loss
        zrp_target = torch.tensor([z, roll, pitch], device=loss.device, dtype = loss.dtype)
        zrp = zrp[p[0], p[1], theta_i, :] #(3,)
        zrp_log_std = zrp_log_std[p[0], p[1], theta_i, :] #(3,)
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

        # Backpropagation
        if backprop:
            loss.backward()
            self.optim.step()
        return np.float32(loss.item())

    def load(self,path):
        # safe operation for e2cnn
        self.model.eval()
        self.prob_linear.eval()
        self.zrp_linear.eval()
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.prob_linear.load_state_dict(torch.load(path[:-3] + '_prob.pt', map_location=self.device))
        self.zrp_linear.load_state_dict(torch.load(path[:-3] + '_zrp.pt', map_location=self.device))

    def save(self,filename):
        # safe operation for e2cnn
        self.model.eval()
        self.prob_linear.eval()
        self.zrp_linear.eval()
        torch.save(self.model.state_dict(), filename)
        torch.save(self.prob_linear.state_dict(), filename[:-3] + '_prob.pt')
        torch.save(self.zrp_linear.state_dict(), filename[:-3] + '_zrp.pt')

    def imshow(self,input: torch.Tensor, size: tuple = None, center: bool= False, name: str = 'name'):
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
        plt.savefig(name)