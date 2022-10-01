import torch
from e2cnn import gspaces
import e2cnn.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class EquiResBlock(torch.nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size, N, flip=False, quotient=True, initialize=True,padding=True):
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

        if padding:
            self.pading = (kernel_size - 1) // 2
        else:
            self.pading =0

        feat_type_in = nn.FieldType(r2_act, input_channels * [rep])
        feat_type_hid = nn.FieldType(r2_act, hidden_dim * [rep])

        self.layer1 = nn.SequentialModule(
            nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, initialize=initialize),
            nn.ReLU(feat_type_hid, inplace=True)
        )

        self.layer2 = nn.SequentialModule(
            nn.R2Conv(feat_type_hid, feat_type_hid, kernel_size=kernel_size, padding=self.pading, initialize=initialize),

        )
        self.relu = nn.ReLU(feat_type_hid, inplace=True)

        self.upscale = None
        if input_channels != hidden_dim or self.pading==0:
            self.upscale = nn.SequentialModule(
                nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=kernel_size, padding=self.pading, initialize=initialize),
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

class EquRes(torch.nn.Module):

    def __init__(self, n_input_channel=6, n_output_channel=1, n_middle_channels=(36, 72, 36, 18), kernel_size=7, N=36, flip=False, quotient=True, initialize=False, init=False):
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

        #assert len(n_middle_channels) == 4
        self.l1_c = n_middle_channels[0]
        self.l2_c = n_middle_channels[1]
        self.l3_c = n_middle_channels[2]
        self.l4_c = n_middle_channels[3]


        # 1* 6(tri) * 96 * 96  --> 1 * 36 * 96 * 96 --> 1 * 36 *  96 * 96
        self.conv_down_1 = torch.nn.Sequential(OrderedDict([
            ('enc-e2conv-0', nn.R2Conv(nn.FieldType(self.r2_act, n_input_channel * [self.r2_act.trivial_repr]),
                                       nn.FieldType(self.r2_act, self.l1_c * [self.repr]),
                                       kernel_size=7, padding=3, initialize=initialize)),
            ('enc-e2relu-0', nn.ReLU(nn.FieldType(self.r2_act, self.l1_c * [self.repr]), inplace=True)),
            ('enc-e2res-1',
             EquiResBlock(self.l1_c, self.l1_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient,
                          initialize=initialize)),
        ]))
        # maxpool(2,2) ---> 1 * 36 * 48 * 48 --> 1 * 72 * 48 * 48
        self.conv_down_2 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-2', nn.PointwiseMaxPool(nn.FieldType(self.r2_act, self.l1_c * [self.repr]), 2)),
            ('enc-e2res-2',
             EquiResBlock(self.l1_c, self.l2_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient,
                          initialize=initialize)),
        ]))
        # maxpool (2,2) --> 1 * 72 * 24 * 24 --> 1 * 36 * 24 * 24
        self.conv_down_4 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-3', nn.PointwiseMaxPool(nn.FieldType(self.r2_act, self.l2_c * [self.repr]), 2)),
            ('enc-e2res-3',
             EquiResBlock(self.l2_c, self.l3_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient,
                          initialize=initialize)),
        ]))
        #  1* 36 * 24 * 24 --> 1 * 18 * 24 * 24 --> maxpool(2,2) --> 1 * 18 * 12 * 12
        self.conv_down_8 = torch.nn.Sequential(OrderedDict([
            ('enc-e2res-4',
             EquiResBlock(self.l3_c, self.l4_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient,
                          initialize=initialize)),
            ('enc-pool-4', nn.PointwiseMaxPool(nn.FieldType(self.r2_act, self.l4_c * [self.repr]), 2)),
        ]))

        # 1 * 18 * 12 * 12 --> 1 * 9 * 6 * 6
        self.final_0 = torch.nn.Sequential(OrderedDict([
            ('enc-final-0', nn.R2Conv(nn.FieldType(self.r2_act, self.l4_c * [self.repr]),
                                       nn.FieldType(self.r2_act, 9 * [self.repr]),
                                       kernel_size=7, padding=0, initialize=initialize)),
            ('enc-f_relu-0', nn.ReLU(nn.FieldType(self.r2_act, 9 * [self.repr]), inplace=True)),]))
        # 1 * 9 * 6 * 6 --> 1 * 1 * 1 * 1
        self.final_1 = torch.nn.Sequential(OrderedDict([
            ('enc-final-1', nn.R2Conv(nn.FieldType(self.r2_act, 9 * [self.repr]),
                                      nn.FieldType(self.r2_act, n_output_channel * [self.repr]),
                                      kernel_size=6, padding=0, initialize=initialize)),]))

        #feat_type_out = nn.FieldType(self.r2_act, n_output_channel * [self.repr])
        #self.pool = nn.PointwiseAvgPool(feat_type_out, kernel_size=4,stride=1,padding=0)

        for name, module in self.named_modules():
            if isinstance(module, nn.R2Conv):
                if init:
                    #nn.init.generalized_he_init(module.weights.data, module.basisexpansion)
                    nn.init.deltaorthonormal_init(module.weights.data, module.basisexpansion)
                else:
                    pass

    def forward(self,obs):

        obs_gt = nn.GeometricTensor(obs, nn.FieldType(self.r2_act, obs.shape[1] * [self.r2_act.trivial_repr]))
        feature_map = self.conv_down_1(obs_gt)
        #print(feature_map.shape)
        feature_map = self.conv_down_2(feature_map)
        #print(feature_map.shape)
        feature_map = self.conv_down_4(feature_map)
        #print(feature_map.shape)
        feature_map = self.conv_down_8(feature_map)
        #print(feature_map.shape)
        feature_map = self.final_0(feature_map)
        #print(feature_map.shape)
        feature_map = self.final_1(feature_map)
        #print(feature_map.shape)
        #feature_map = self.pool(feature_map)

        return feature_map

# pick = EquRes()
# image = torch.rand(1,6,96,96)
# feature = pick(image)
# print(feature.shape)
# for i in range(36):
#     print(feature.transform(i).tensor[0,...].detach().numpy().squeeze())