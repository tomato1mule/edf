from typing import Tuple
import torch
import torch.nn.functional as F
import math
import e2cnn.nn as enn
from e2cnn.nn import init
from e2cnn import gspaces

from argparse import ArgumentParser

#TODO add init function

def conv5x5(in_channel, out_channel, stride=1, padding=2,dilation=1, bias=True):
    """5x5 convolution with padding"""
    return torch.nn.Conv2d(in_channel,
                           out_channel,
                           kernel_size=(5,5),
                           stride=(stride,stride),
                           padding=padding,
                           dilation=(dilation,dilation),
                           bias=bias,)

def conv3x3(in_channel, out_channel, stride=1, padding=1,dilation=1, bias=True):
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(in_channel,
                           out_channel,
                           kernel_size=(3,3),
                           stride=(stride,stride),
                           padding=padding,
                           dilation=(dilation,dilation),
                           bias=bias,)

def conv1x1(in_channel, out_channel, stride=1, padding=0,dilation=1, bias=True):
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(in_channel,
                           out_channel,
                           kernel_size=(1,1),
                           stride=(stride,stride),
                           padding=padding,
                           dilation=(dilation,dilation),
                           bias=bias,)


class conv_block(torch.nn.Module):

    def __init__(self,
                 in_type,
                 kernel_size,
                 filters,
                 stride,
                 activation=True,
                 include_batch_norm=False,
                 short_cut =True):
        super(conv_block,self).__init__()
        if kernel_size==3:
            conv = conv3x3
        elif kernel_size==5:
            conv = conv5x5
        else:
            print('kernel size be either 3 or 5')

        self.activation = activation
        self.include_batch_norm = include_batch_norm
        self.stride = stride
        self.in_type = in_type #TO CHECK LATER
        self.short_cut = short_cut
        filters1, filters2, filters3 = filters

        self.relu = torch.nn.ReLU(inplace=True)
        self.layer1 = conv1x1(self.in_type,filters1,stride=stride)
        self.batch_norm_1 = torch.nn.BatchNorm2d(filters1)

        self.layer2 = conv(filters1,filters2)
        self.batch_norm_2 = torch.nn.BatchNorm2d(filters2)

        self.layer3 = conv1x1(filters2, filters3)
        self.batch_norm_3 = torch.nn.BatchNorm2d(filters3)

        self.short_cut_layer = conv1x1(self.in_type, filters3,stride=stride)
        self.short_cut_batch_norm = torch.nn.BatchNorm2d(filters3)

    ## forward pass
    def forward(self, input_tensor):

        x = self.layer1(input_tensor)
        if self.include_batch_norm:
            x = self.batch_norm_1(x)
        x = self.relu(x)

        x = self.layer2(x)
        if self.include_batch_norm:
            x = self.batch_norm_2(x)
        x = self.relu(x)

        x = self.layer3(x)
        if self.include_batch_norm:
            x = self.batch_norm_3(x)

        if self.short_cut:
            short_cut = self.short_cut_layer(input_tensor)
            if self.include_batch_norm:
                short_cut = self.short_cut_batch_norm(short_cut)
            x = x+short_cut
        if self.activation:
            x = self.relu(x)
        return x



class identity_block(torch.nn.Module):

    def __init__(self,
                 in_type,
                 kernel_size,
                 filters,
                 activation=True,
                 include_batch_norm=False,):

        super(identity_block,self).__init__()
        self.activation = activation
        self.include_batch_norm = include_batch_norm
        self.in_type = in_type
        filters1, filters2, filters3 = filters

        if kernel_size==3:
            conv = conv3x3
        elif kernel_size==5:
            conv = conv5x5
        else:
            print('kernel size be either 3 or 5')

        self.relu = torch.nn.ReLU(inplace=True)
        self.layer1 = conv1x1(self.in_type, filters1)
        self.batch_norm_1 = torch.nn.BatchNorm2d(filters1)

        self.layer2 = conv(filters1, filters2)
        self.batch_norm_2 = torch.nn.BatchNorm2d(filters2)

        self.layer3 = conv1x1(filters2, filters3)
        self.batch_norm_3 = torch.nn.BatchNorm2d(filters3)

    ## forward pass
    def forward(self, input_tensor):

        x = self.layer1(input_tensor)
        if self.include_batch_norm:
            x = self.batch_norm_1(x)
        x = self.relu(x)

        x = self.layer2(x)
        if self.include_batch_norm:
            x = self.batch_norm_2(x)
        x = self.relu(x)

        x = self.layer3(x)
        if self.include_batch_norm:
            x = self.batch_norm_3(x)

        x = x+input_tensor
        if self.activation:
            x = self.relu(x)
        return x


class ResNet43(torch.nn.Module):
    def __init__(self,in_type,outdim,include_batch_normal=False,cutoff_early=False):
        super(ResNet43,self).__init__()
        self.in_type = in_type
        self.outdim = outdim

        self.conv0 = conv3x3(self.in_type ,64)
        self.batch_norm0 = torch.nn.BatchNorm2d(64)

        self.relu = torch.nn.ReLU(inplace=True)
        self.include_batch_normal = include_batch_normal
        self.cutoff_early = cutoff_early

        if self.cutoff_early:
            self.cut1 = conv_block(in_type=64,kernel_size=5,filters=[64,64,outdim],stride=1,
                                   include_batch_norm=self.include_batch_normal)
            self.cut2 = identity_block(outdim,kernel_size=5,filters=[64,64,outdim],
                                       include_batch_norm=self.include_batch_normal)
        else:
            # block1: chanell = 64 = 8*8     stride ==1
            self.conv_block1 = conv_block(64, kernel_size=3,filters=[64,64,64],stride=1,
                                   include_batch_norm=self.include_batch_normal)

            output_type = 64

            self.identity_block1 = identity_block(output_type, kernel_size=3, filters=[64, 64, 64],
                                                  include_batch_norm=self.include_batch_normal)


            # block2: chanell = 128 = 16*8      stride ==2

            self.conv_block2 = conv_block(output_type,  kernel_size=3, filters=[128, 128, 128], stride=2,
                                          include_batch_norm=self.include_batch_normal)

            output_type = 128


            self.identity_block2 = identity_block(output_type,  kernel_size=3, filters=[128, 128, 128],
                                                  include_batch_norm=self.include_batch_normal)

            output_type = 128

            # block3: chanell = 256 = 32*8      stride ==2

            self.conv_block3 = conv_block(output_type,  kernel_size=3, filters=[256, 256, 256], stride=2,
                                          include_batch_norm=self.include_batch_normal)

            output_type = 256

            self.identity_block3 = identity_block(output_type,  kernel_size=3, filters=[256, 256, 256],
                                                  include_batch_norm=self.include_batch_normal)

            output_type =256

            # block4: chanell = 512 = 64*8      stride ==2

            self.conv_block4 = conv_block(output_type,  kernel_size=3, filters=[512, 512, 512], stride=2,
                                          include_batch_norm=self.include_batch_normal)

            output_type = 512

            self.identity_block4 = identity_block(output_type, kernel_size=3, filters=[512, 512, 512],
                                                  include_batch_norm=self.include_batch_normal)

            output_type = 512

            # block5: chanell = 256 = 32*8      stride ==1

            self.conv_block5 = conv_block(output_type, kernel_size=3, filters=[256, 256, 256], stride=1,
                                          include_batch_norm=self.include_batch_normal)

            output_type = 256

            self.identity_block5 = identity_block(output_type, kernel_size=3, filters=[256, 256, 256],
                                                  include_batch_norm=self.include_batch_normal)

            output_type = 256

            self.upsample1 = torch.nn.Upsample(scale_factor=2,mode='bilinear') # don't use bilinear with cuda, which is nondeterministic

            # block6: chanell = 128 = 16*8      stride ==1

            self.conv_block6 = conv_block(output_type, kernel_size=3, filters=[128, 128, 128], stride=1,
                                          include_batch_norm=self.include_batch_normal)

            output_type = 128

            self.identity_block6 = identity_block(output_type, kernel_size=3, filters=[128, 128, 128],
                                                  include_batch_norm=self.include_batch_normal)

            output_type = 128

            self.upsample2 = torch.nn.Upsample(scale_factor=2,mode='bilinear') # don't use bilinear with cuda, which is nondeterministic

            # block7: chanell = 64 = 8*8      stride ==1

            self.conv_block7 = conv_block(output_type,  kernel_size=3, filters=[64, 64, 64], stride=1,
                                          include_batch_norm=self.include_batch_normal)

            output_type = 64

            self.identity_block7 = identity_block(output_type, kernel_size=3, filters=[64, 64, 64],
                                                  include_batch_norm=self.include_batch_normal)

            output_type = 64

            self.upsample3 = torch.nn.Upsample(scale_factor=2,mode='bilinear') # don't use bilinear with cuda, which is nondeterministic

            # final layers

            self.final_1 = conv_block(output_type, kernel_size=3, filters=[16,16,self.outdim],stride=1,activation=False)

            output_type = self.outdim

            self.final_2 = identity_block(output_type, kernel_size=3, filters=[16,16,self.outdim], activation=False)



    def forward(self, input_tensor):
        x = self.conv0(input_tensor)
        if self.include_batch_normal:
            x = self.batch_norm0(x)
        x = self.relu(x)

        if self.cutoff_early:
            x = self.cut1(x)
            x = self.cut2(x)
            return input_tensor,x
        else:
            #######################
            x = self.conv_block1(x)
            x= self.identity_block1(x)
            #######################
            x = self.conv_block2(x)
            x = self.identity_block2(x)
            ########################
            x = self.conv_block3(x)
            x = self.identity_block3(x)
            ##########################
            x = self.conv_block4(x)
            x = self.identity_block4(x)
            ##########################
            x = self.conv_block5(x)
            x = self.identity_block5(x)
            #x = self.upsample1(x)
            x = self.upsample1(x.to('cpu')).to(x.device) # for reproducibility
            ###########################
            x = self.conv_block6(x)
            x = self.identity_block6(x)
            #x = self.upsample2(x)
            x = self.upsample2(x.to('cpu')).to(x.device) # for reproducibility
            #######################
            x = self.conv_block7(x)
            x = self.identity_block7(x)
            #x = self.upsample3(x)
            x = self.upsample3(x.to('cpu')).to(x.device) # for reproducibility
            ########################
            x = self.final_1(x)
            x = self.final_2(x)
        return x
