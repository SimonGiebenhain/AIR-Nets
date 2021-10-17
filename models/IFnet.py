'''
IF-Nets
Paper: https://virtualhumans.mpi-inf.mpg.de/papers/chibane20ifnet/chibane20ifnet.pdf
Code Author: Julian Chibane
Code from: https://github.com/jchibane/if-net (as much of the rest of this repository)
Code adapted to fit into encoder-decoder schema.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1D convolution is used for the decoder. It acts as a standard FC, but allows to use a batch of point samples features,
# additionally to the batch over the input objects.
# The dimensions are used as follows:
# batch_size (N) = #3D objects , channels = features, signal_lengt (L) (convolution dimension) = #point samples
# kernel_size = 1 i.e. every convolution is done over only all features of one point sample, this makes it a FC.


# ShapeNet Voxel Super-Resolution --------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
class ShapeNet32Vox(nn.Module):

    def __init__(self, hidden_dim=256):
        super(ShapeNet32Vox, self).__init__()

        self.conv_1 = nn.Conv3d(1, 32, 3, padding=1)  # out: 32
        self.conv_1_1 = nn.Conv3d(32, 64, 3, padding=1)  # out: 32
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1)  # out: 16
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1)  # out: 16
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1)  # out: 8
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1)  # out: 8

        feature_size = (1 + 64 + 128 + 128 ) * 7
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim*2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim*2, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)


        displacment = 0.035
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments).cuda()

    def forward(self, p, x):
        x = x.unsqueeze(1)

        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)  # (B,1,7,num_samples,3)
        feature_0 = F.grid_sample(x, p)  # out : (B,C (of x), 1,1,sample_num)

        net = self.actvn(self.conv_1(x))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        feature_1 = F.grid_sample(net, p)  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        feature_2 = F.grid_sample(net, p)
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        feature_3 = F.grid_sample(net, p)

        # here every channel corresponse to one feature.

        features = torch.cat((feature_0, feature_1, feature_2, feature_3),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        #features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.fc_out(net)
        out = net.squeeze(1)

        return out

class ShapeNet128Vox(nn.Module):

    def __init__(self, hidden_dim=256):
        super(ShapeNet128Vox, self).__init__()
        # accepts 128**3 res input
        self.conv_in = nn.Conv3d(1, 16, 3, padding=1)  # out: 128
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1)  # out: 64
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1)  # out: 64
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1)  # out: 32
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1)  # out: 32
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1)  # out: 16
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1)  # out: 16
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1)  # out: 8
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1)  # out: 8

        feature_size = (1 +  16 + 32 + 64 + 128 + 128 ) * 7
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim, 1)
        self.fc_1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)


        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments).cuda()

    def forward(self, p, x):
        x = x.unsqueeze(1)

        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)  # (B,1,7,num_samples,3)
        feature_0 = F.grid_sample(x, p)  # out : (B,C (of x), 1,1,sample_num)

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        feature_1 = F.grid_sample(net, p)  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        feature_2 = F.grid_sample(net, p)  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        feature_3 = F.grid_sample(net, p)  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        feature_4 = F.grid_sample(net, p)
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        feature_5 = F.grid_sample(net, p)

        # here every channel corresponse to one feature.

        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        #features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.fc_out(net)
        out = net.squeeze(1)

        return out




# ShapeNet Pointcloud Completion ---------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

class ShapeNetPoints(nn.Module):

    def __init__(self, hidden_dim=256):
        super(ShapeNetPoints, self).__init__()
        self.encoder = IFPointEncoder()
        self.decoder = IFPointDecoder()

    def forward(self, p, x):
        return self.decoder(p, self.encoder(x))

class ShapeNetPointsORI(nn.Module):

    def __init__(self, hidden_dim=256):
        super(ShapeNetPointsORI, self).__init__()
        # 128**3 res input
        self.conv_in = nn.Conv3d(1, 16, 3, padding=1,)
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1,)
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1,)
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1,)
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1,)
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1,)
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1,)
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1,)
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1,)

        feature_size = (1 +  16 + 32 + 64 + 128 + 128 ) * 7
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim, 1)
        self.fc_1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)


        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments).cuda()

    def forward(self, p, x):
        x = x.unsqueeze(1)

        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)  # (B,1,7,num_samples,3)
        feature_0 = F.grid_sample(x, p, padding_mode='border')  # out : (B,C (of x), 1,1,sample_num)

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        feature_1 = F.grid_sample(net, p, padding_mode='border')  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        feature_2 = F.grid_sample(net, p, padding_mode='border')  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        feature_3 = F.grid_sample(net, p, padding_mode='border')  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        feature_4 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        feature_5 = F.grid_sample(net, p, padding_mode='border')

        # here every channel corresponds to one feature.

        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        #features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.fc_out(net)
        out = net.squeeze(1)

        return out

########
#### Note that splitting the network in an encoder and decoder this way is not as efficient as having a single model.
#### This is only done, such that it fits the repository structure!
########
class IFPointEncoder(nn.Module):
    def __init__(self):
        super(IFPointEncoder, self).__init__()
        # 128**3 res input
        self.conv_in = nn.Conv3d(1, 16, 3, padding=1)
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1)
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1)
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1)
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1)
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1)

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)

        self.actvn = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        feature_tensor = {}
        feature_tensor['input'] = x

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        feature_tensor['level1'] = net.clone()
        net = self.maxpool(net)

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        feature_tensor['level2'] = net.clone()
        net = self.maxpool(net)

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        feature_tensor['level3'] = net.clone()
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        feature_tensor['level4'] = net.clone()
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        feature_tensor['level5'] = net.clone()

        return feature_tensor


class IFPointDecoder(nn.Module):
    def __init__(self, hidden_dim=256):
        super(IFPointDecoder, self).__init__()

        feature_size = (1 +  16 + 32 + 64 + 128 + 128 ) * 7
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim, 1)
        self.fc_1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()

        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments).cuda()

    def forward(self, p, feature_tensor):
        p_features = p.transpose(1, -1)
        # *2 to scale from -1 to 1
        p_clone = p.clone()
        p[..., 0], p[..., 2] = p_clone[..., 2], p_clone[..., 0]

        p = 2 * p.unsqueeze(1).unsqueeze(1)

        p = torch.cat([p + d for d in self.displacments], dim=2)  # (B,1,7,num_samples,3)

        feature_0 = F.grid_sample(feature_tensor['input'],  p, padding_mode='border')
        feature_1 = F.grid_sample(feature_tensor['level1'], p, padding_mode='border')
        feature_2 = F.grid_sample(feature_tensor['level2'], p, padding_mode='border')
        feature_3 = F.grid_sample(feature_tensor['level3'], p, padding_mode='border')
        feature_4 = F.grid_sample(feature_tensor['level4'], p, padding_mode='border')
        feature_5 = F.grid_sample(feature_tensor['level5'], p, padding_mode='border')

        # here every channel corresponds to one feature.
        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.fc_out(net)
        out = net.squeeze(1)

        return out



# 3D Single View Reconsturction (for 256**3 input voxelization) --------------------------------------
# ----------------------------------------------------------------------------------------------------

class SVR(nn.Module):


    def __init__(self, hidden_dim=256):
        super(SVR, self).__init__()

        self.conv_in = nn.Conv3d(1, 16, 3, padding=1, padding_mode='border')  # out: 256 ->m.p. 128
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1, padding_mode='border')  # out: 128
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1, padding_mode='border')  # out: 128 ->m.p. 64
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1, padding_mode='border')  # out: 64
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1, padding_mode='border')  # out: 64 -> mp 32
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1, padding_mode='border')  # out: 32
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')  # out: 32 -> mp 16
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')  # out: 16
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')  # out: 16 -> mp 8
        self.conv_4 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')  # out: 8
        self.conv_4_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')  # out: 8

        feature_size = (1 +  16 + 32 + 64 + 128 + 128 + 128) * 7 + 3
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim * 2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim *2, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim , hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)
        self.conv4_1_bn = nn.BatchNorm3d(128)


        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments).cuda()

    def forward(self, p, x):
        x = x.unsqueeze(1)

        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)
        feature_0 = F.grid_sample(x, p, padding_mode='border')

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        feature_1 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net) #out 128

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        feature_2 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net) #out 64

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        feature_3 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        feature_4 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        feature_5 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net)

        net = self.actvn(self.conv_4(net))
        net = self.actvn(self.conv_4_1(net))
        net = self.conv4_1_bn(net)
        feature_6 = F.grid_sample(net, p, padding_mode='border')

        # here every channel corresponse to one feature.

        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.fc_out(net)
        out = net.squeeze(1)

        return out


#model = ShapeNetPoints()

def get_encoder(cfg):
    #return ShapeNetPointsORI()
    #return model.encoder
    return IFPointEncoder()


def get_decoder(cfg):
    #return ShapeNetPointsORI()
    #return model.decoder
    return IFPointDecoder()
