import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import numpy as np
import torch
from pytorch_resnet import ResNet43
import torch.nn.functional as F
import kornia as K
import torchvision
from matplotlib import pyplot as plt


class Attention:
    def __init__(self,in_shape,n_rotations,device,lite=False,preprocess=None):
        # TODO BY HAOJIE: add lite model
        self.device = device
        self.preprocess = preprocess
        self.n_rotations = n_rotations
        max_dim = np.max(in_shape[:2])
        max_dim = 480
        self.padding = np.zeros((3, 2), dtype=int)
        pad = (max_dim - np.array(in_shape[:2])) / 2
        self.padding[:2] = pad.reshape(2, 1)
        in_shape = np.array(in_shape)
        in_shape += np.sum(self.padding, axis=1)
        in_shape = tuple(in_shape)
        self.in_type = in_shape[-1]
        # self.in_type = 6, self.outdim=1

        # get the location
        self.model = ResNet43(self.in_type,outdim=1,include_batch_normal=False).to(self.device)

        # use the location as pivot to rotate the image and get the angle
        #self.angle_model = ResNet43(self.in_type,outdim=1,include_batch_normal=False).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
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
        in_data = torch.from_numpy(in_data).to(self.device)
        #print(in_data.size())
        
        # rotate image
        pivot = torch.as_tensor([in_data.shape[-2]/2,in_data.shape[-1]/2])
        pivot =pivot.to(self.device).repeat(self.n_rotations//2,1).to(torch.float32)
        in_data = in_data.repeat(self.n_rotations//2,1,1,1)
        in_data = K.geometry.rotate(in_data,torch.from_numpy(-np.linspace(0., 360., self.n_rotations, endpoint=False, dtype=np.float32))[0:18].to(self.device), mode='nearest',center=pivot)
        #print('indata rotate 36/2',in_data.shape)
        #self.imshow(in_data,size=(36,12),name='rotation')

        print(in_data.shape)
        if not train:
            self.model.eval()
            with torch.no_grad():
                logits = self.model(in_data)
        else:
            logits = self.model(in_data)
        
        #print('logits',logits.shape)
        # rotate back
        logits = K.geometry.rotate(logits,torch.from_numpy(np.linspace(0., 360., self.n_rotations,
                                                                    endpoint=False,dtype=np.float32))[0:18].to(self.device),
                                 mode='nearest',center=pivot)
                                 

        #print('atenion logits1',logits.shape)
        #self.imshow(logits)
        #self.imshow(logits,size=(36,12),name='rotation_back')
        #logits = logits[:,:,80:-80,80:-80]
        #print('first crop',logits.size())
        c0 = self.padding[:2, 0]
        c1 = c0 + in_img.shape[:2]
        #print('crop',c0)
        #print('crop',c1)
        logits = logits[:, :, c0[0]:c1[0], c0[1]:c1[1]]
        #print('second crop',logits.size())
        #print('attention logits',logits.shape)
        #self.imshow(logits)
        output = logits.reshape(1,-1)
        if softmax:
            output = F.softmax(output,dim=-1)
            output = output.reshape(logits.shape[0],logits.shape[-2],logits.shape[-1]).cpu().detach().numpy()
            #print('output',output.shape)
            output = output.transpose(1,2,0)
        return output

    def train(self,in_img,p,theta,backprop=True):
        self.model.train()
        self.optim.zero_grad()
        output = self.forward(in_img,softmax=False)
        # Get label
        
        theta = (theta + 2*np.pi)%(2*np.pi)
        if theta >= np.pi:
          theta = theta -np.pi
        # angle label
        # dgree interval: 10
        theta_i = theta / (2 * np.pi / 36)
        # theta_i is in range [0,17]
        theta_i = np.int32(np.round(theta_i)) % 18
        label_theta = torch.as_tensor(theta_i, dtype=torch.long, device=self.device).unsqueeze(dim=0)
        label_size = (self.n_rotations//2,) + in_img.shape[:2]
        label = torch.zeros(label_size,dtype=torch.long,device=self.device)
        label[theta_i, p[0], p[1],] = 1
        label = label.reshape(-1)
        label = torch.argmax(label).unsqueeze(dim=0)
        #print('label size',label.shape)
        #print('out size', output.shape)
        # Get loss
        loss = F.cross_entropy(input=output, target=label)

        # Backpropagation
        if backprop:
            loss.backward()
            self.optim.step()
        return np.float32(loss.item())

    def load(self,path):
        # safe operation for e2cnn
        self.model.eval()
        self.model.load_state_dict(torch.load(path,map_location=self.device))

    def save(self,filename):
        # safe operation for e2cnn
        self.model.eval()
        torch.save(self.model.state_dict(), filename)

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