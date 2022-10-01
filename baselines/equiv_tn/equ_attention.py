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
from equ_res_3 import dian_res
from pick_angle_model import EquRes as lite_pick_angle
from pick_angle_model_2 import EquRes as pick_angle


class Attention:
    def __init__(self,in_shape,n_rotations,preprocess,device,lite=True,angle_lite=False,init=False):
        # TODO BY HAOJIE: add lite model
        self.device = device
        self.preprocess = preprocess
        self.n_rotations = n_rotations
        max_dim = np.max(in_shape[:2])
        self.padding = np.zeros((3, 2), dtype=int)
        pad = (max_dim - np.array(in_shape[:2])) / 2
        self.padding[:2] = pad.reshape(2, 1)
        in_shape = np.array(in_shape)
        in_shape += np.sum(self.padding, axis=1)
        in_shape = tuple(in_shape)
        self.gspace = gspaces.Rot2dOnR2(4)
        self.in_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr] * in_shape[-1])
        
        if lite:
          self.model = dian_res(in_dim=6,out_dim=1,N=4,middle_dim=(16, 32, 64, 128),init=init).to(self.device)
        else:
          self.model = dian_res(in_dim=6,out_dim=1,N=4,middle_dim=(32, 64, 128, 256),init=init).to(self.device)
        if angle_lite:
          self.angle_model = lite_pick_angle(init=init).to(self.device)
          self.crop_size = 64
        else:
          self.angle_model = pick_angle(init=init).to(self.device)
          self.crop_size = 96
        
        self.pad_size_2 = int(self.crop_size / 2)
        self.padding_2 = np.zeros((3, 2), dtype=int)
        self.padding_2[:2, :] = self.pad_size_2
        
        #self.parameters = list(self.model.parameters()) + list(self.angle_model.parameters())
        self.optim1 = torch.optim.Adam(self.model.parameters(),lr=1e-4)
        self.optim2 = torch.optim.Adam(self.angle_model.parameters(),lr=1e-4)

    def forward(self,in_img,softmax=True,train=True):
        in_data = np.pad(in_img, self.padding, mode='constant')
        in_data = self.preprocess(in_data)
        in_shape = (1,) + in_data.shape
        in_data = in_data.reshape(in_shape).transpose(0, 3, 1, 2)
        in_data = torch.from_numpy(in_data).to(self.device)
        #pading image for crop
        img_unprocessed = np.pad(in_img, self.padding_2, mode='constant')
        input_data = self.preprocess(img_unprocessed)
        in_shape_2 = (1,) + input_data.shape
        input_data = input_data.reshape(in_shape_2).transpose(0, 3, 1, 2)
        input_tensor = torch.from_numpy(input_data).to(self.device)
        angle_index = None
        if not train:
            self.model.eval()
            with torch.no_grad():
                _,logits = self.model(in_data)
        else:
            _, logits = self.model(in_data)

        c0 = self.padding[:2, 0]
        c1 = c0 + in_img.shape[:2]
        logits = logits.tensor
        logits = logits[:, :, c0[0]:c1[0], c0[1]:c1[1]]
        output = logits.reshape(1,-1)
        
        if softmax:
            output = F.softmax(output,dim=-1)
            output = output.reshape(logits.shape[2:]).cpu().detach().numpy()
            output = output[...,np.newaxis]
        #get the pick_angele if not train
        if not train:
            argmax = np.argmax(output)
            argmax = np.unravel_index(argmax, shape=output.shape)
            p = argmax[:2]
            crop = input_tensor[:,:,p[0]:(p[0] + self.crop_size),p[1]:(p[1] + self.crop_size)]
            #print('crop',crop.size())
            self.angle_model.eval()
            with torch.no_grad():
              angle_index = self.angle_model(crop)
              angle_index = angle_index.tensor.reshape(1,-1)
              angle_index = angle_index.detach().cpu().numpy()
            #print('max angle',angle_index.shape, np.argmax(angle_index.shape))
        
        return output, angle_index, input_tensor

    def train(self,in_img,p,theta,backprop=True):
        self.model.train()
        self.angle_model.train()
        
        output,_,input_tensor = self.forward(in_img,softmax=False)
        crop = input_tensor[:,:,p[0]:(p[0] + self.crop_size),p[1]:(p[1] + self.crop_size)]
        #print('crop',crop.size())
        angle_index = self.angle_model(crop)
        angle_index = angle_index.tensor.reshape(1,-1)
        #print('angle_index',angle_index.shape)
        # Get label
        theta = (theta + 2*np.pi)%(2*np.pi)
        if theta >= np.pi:
          theta = theta -np.pi
        # angle label
        # dgree interval: 10
        theta_i = theta / (2 * np.pi / 36)
        # theta_i is in range [0,17]
        theta_i = np.int32(np.round(theta_i)) % 18
        label_theta = torch.as_tensor(theta_i,dtype=torch.long,device=self.device).unsqueeze(dim=0)
        # location label
        label_size = (self.n_rotations,) + in_img.shape[:2]
        label = torch.zeros(label_size,dtype=torch.long,device=self.device)
        label[0, p[0], p[1],] = 1
        label = label.reshape(-1)
        label = torch.argmax(label).unsqueeze(dim=0)
        #print('label size',label.shape)
        #print('out size', output.shape)
        # Get loss
        loss1 = F.cross_entropy(input=output, target=label)
        loss2 = F.cross_entropy(input=angle_index,target=label_theta)

        # Backpropagation
        if backprop:
            self.optim1.zero_grad()
            loss1.backward()
            self.optim1.step()
            self.optim2.zero_grad()
            loss2.backward()
            self.optim2.step()
        return np.float32(loss1.item()),np.float32(loss2.item())

    def load(self,path1,path2):
        # safe operation for e2cnn
        self.model.eval()
        self.model.load_state_dict(torch.load(path1,map_location=self.device))
        
        self.angle_model.eval()
        self.angle_model.load_state_dict(torch.load(path2,map_location=self.device))


    def save(self,filename1,filename2):
        # safe operation for e2cnn
        self.model.eval()
        torch.save(self.model.state_dict(), filename1)
        self.angle_model.eval()
        torch.save(self.angle_model.state_dict(), filename2)


