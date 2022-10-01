import torch
from e2cnn import gspaces
import e2cnn.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class EquiResBlock(torch.nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size, N, flip=False, quotient=False, initialize=True):
        super(EquiResBlock, self).__init__()

        if flip:
            r2_act = gspaces.FlipRot2dOnR2(N=N)
        else:
            r2_act = gspaces.Rot2dOnR2(N=N)

        if quotient:
            if flip:
                rep = r2_act.quotient_repr((None, 2))
            else:
                rep = r2_act.quotient_repr(2)
        else:
            rep = r2_act.regular_repr

        feat_type_in = nn.FieldType(r2_act, input_channels * [rep])
        feat_type_hid = nn.FieldType(r2_act, hidden_dim * [rep])

        self.layer1 = nn.SequentialModule(
            nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, initialize=initialize),
            nn.ReLU(feat_type_hid, inplace=True)
        )

        self.layer2 = nn.SequentialModule(
            nn.R2Conv(feat_type_hid, feat_type_hid, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, initialize=initialize),

        )
        self.relu = nn.ReLU(feat_type_hid, inplace=True)

        self.upscale = None
        if input_channels != hidden_dim:
            self.upscale = nn.SequentialModule(
                nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, initialize=initialize),
            )

    def forward(self, xx):
        residual = xx
        out = self.layer1(xx)
        out = self.layer2(out)
        if self.upscale:
            out += self.upscale(residual)
        else:
            out += residual
        out = self.relu(out)
        return out

class EquResUNet(torch.nn.Module):
    def __init__(self, n_input_channel=1, n_output_channel=16, n_middle_channels=(8, 16, 32), kernel_size=3, N=6, flip=False, quotient=False, initialize=False):
        super().__init__()
        self.N = N
        self.quotient = quotient
        if flip:
            self.r2_act = gspaces.FlipRot2dOnR2(N=N)
        else:
            self.r2_act = gspaces.Rot2dOnR2(N=N)

        if quotient:
            if flip:
                self.repr = self.r2_act.quotient_repr((None, 2))
            else:
                self.repr = self.r2_act.quotient_repr(2)
        else:
            self.repr = self.r2_act.regular_repr

        assert len(n_middle_channels) == 3
        self.l1_c = n_middle_channels[0]
        self.l2_c = n_middle_channels[1]
        self.l3_c = n_middle_channels[2]

        self.conv_down_1 = torch.nn.Sequential(OrderedDict([
            ('enc-e2conv-0', nn.R2Conv(nn.FieldType(self.r2_act, n_input_channel * [self.r2_act.trivial_repr]),
                                       nn.FieldType(self.r2_act, self.l1_c * [self.repr]),
                                       kernel_size=3, padding=1, initialize=initialize)),
            ('enc-e2relu-0', nn.ReLU(nn.FieldType(self.r2_act, self.l1_c * [self.repr]), inplace=True)),
            ('enc-e2res-1', EquiResBlock(self.l1_c, self.l1_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, initialize=initialize)),
        ]))

        self.conv_down_2 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-2', nn.PointwiseMaxPool(nn.FieldType(self.r2_act, self.l1_c * [self.repr]), 2)),
            ('enc-e2res-2', EquiResBlock(self.l1_c, self.l2_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, initialize=initialize)),
        ]))
        self.conv_down_4 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-3', nn.PointwiseMaxPool(nn.FieldType(self.r2_act, self.l2_c * [self.repr]), 2)),
            ('enc-e2res-3', EquiResBlock(self.l2_c, self.l3_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, initialize=initialize)),
        ]))
        self.conv_down_8 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-4', nn.PointwiseMaxPool(nn.FieldType(self.r2_act, self.l3_c * [self.repr]), 2)),
            ('enc-e2res-4', EquiResBlock(self.l3_c, self.l3_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, initialize=initialize)),
        ]))
        # self.conv_down_16 = torch.nn.Sequential(OrderedDict([
        #     ('enc-pool-5', nn.PointwiseMaxPool(nn.FieldType(self.r2_act, self.l4_c * [self.repr]), 2)),
        #     ('enc-e2res-5', EquiResBlock(self.l4_c, self.l4_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, initialize=initialize)),
        # ]))
        #
        # self.conv_up_8 = torch.nn.Sequential(OrderedDict([
        #     ('dec-e2res-1', EquiResBlock(2*self.l4_c, self.l3_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, initialize=initialize)),
        # ]))
        self.conv_up_4 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-2', EquiResBlock(2*self.l3_c, self.l2_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, initialize=initialize)),
        ]))
        self.conv_up_2 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-3', EquiResBlock(2*self.l2_c, self.l1_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, initialize=initialize)),
        ]))
        self.conv_up_1 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-4', EquiResBlock(2*self.l1_c, n_output_channel, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, initialize=initialize)),
        ]))

        #self.upsample_16_8 = nn.R2Upsampling(nn.FieldType(self.r2_act, self.l4_c * [self.repr]), 2)
        self.upsample_8_4 = nn.R2Upsampling(nn.FieldType(self.r2_act, self.l3_c * [self.repr]), 2)
        self.upsample_4_2 = nn.R2Upsampling(nn.FieldType(self.r2_act, self.l2_c * [self.repr]), 2)
        self.upsample_2_1 = nn.R2Upsampling(nn.FieldType(self.r2_act, self.l1_c * [self.repr]), 2)

    def forwardEncoder(self, obs):
        obs_gt = nn.GeometricTensor(obs, nn.FieldType(self.r2_act, obs.shape[1] * [self.r2_act.trivial_repr]))
        feature_map_1 = self.conv_down_1(obs_gt)
        feature_map_2 = self.conv_down_2(feature_map_1)
        feature_map_4 = self.conv_down_4(feature_map_2)
        feature_map_8 = self.conv_down_8(feature_map_4)
        return feature_map_1, feature_map_2, feature_map_4, feature_map_8

    def forwardDecoder(self, feature_map_1, feature_map_2, feature_map_4, feature_map_8):
        # concat_8 = torch.cat((feature_map_8.tensor, self.upsample_16_8(feature_map_16).tensor), dim=1)
        # concat_8 = nn.GeometricTensor(concat_8, nn.FieldType(self.r2_act, 2*self.l4_c * [self.repr]))
        # feature_map_up_8 = self.conv_up_8(concat_8)

        concat_4 = torch.cat((feature_map_4.tensor, self.upsample_8_4(feature_map_8).tensor), dim=1)
        concat_4 = nn.GeometricTensor(concat_4, nn.FieldType(self.r2_act, 2*self.l3_c * [self.repr]))
        feature_map_up_4 = self.conv_up_4(concat_4)

        concat_2 = torch.cat((feature_map_2.tensor, self.upsample_4_2(feature_map_up_4).tensor), dim=1)
        concat_2 = nn.GeometricTensor(concat_2, nn.FieldType(self.r2_act, 2*self.l2_c * [self.repr]))
        feature_map_up_2 = self.conv_up_2(concat_2)

        concat_1 = torch.cat((feature_map_1.tensor, self.upsample_2_1(feature_map_up_2).tensor), dim=1)
        concat_1 = nn.GeometricTensor(concat_1, nn.FieldType(self.r2_act, 2*self.l1_c * [self.repr]))
        feature_map_up_1 = self.conv_up_1(concat_1)

        return feature_map_up_1

    def forward(self, obs):
        feature_map_1, feature_map_2, feature_map_4, feature_map_8 = self.forwardEncoder(obs)
        return self.forwardDecoder(feature_map_1, feature_map_2, feature_map_4, feature_map_8)


class Tail(torch.nn.Module):
    def __init__(self,in_dim, out_dim, N=6, middle_dim=(8, 16, 32,),init=False):
        super(Tail, self).__init__()
        self.r2_act = gspaces.Rot2dOnR2(N=N)
        self.main_block = EquResUNet(n_input_channel=in_dim,n_output_channel=middle_dim[0],n_middle_channels=middle_dim,N=N)
        self.final = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.r2_act, [self.r2_act.regular_repr]*middle_dim[0]),
                      nn.FieldType(self.r2_act, [self.r2_act.trivial_repr]*out_dim), kernel_size=3,padding=1,initialize=False))
        for name, module in self.named_modules():
            if isinstance(module, nn.R2Conv):
                if init:
                    #nn.init.generalized_he_init(module.weights.data, module.basisexpansion)
                    nn.init.deltaorthonormal_init(module.weights.data, module.basisexpansion)
                else:
                    pass
    def forward(self,x):
        x = x.permute(1,0,2,3)
        out = self.main_block(x)
        #print(out.shape)
        out = self.final(out)
        #print(out.shape)
        out = out.tensor.permute(1,0,2,3)
        return x,out



