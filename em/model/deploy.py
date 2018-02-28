# deployed model without much flexibility
# useful for stand-alone test, model translation, quantization
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from .block import mergeAdd, mergeCrop


class unet3D_m1(nn.Module): # deployed Toufiq model
    def __init__(self, in_num=1, out_num=3, filters=[24,72,216,648],relu_slope=0.005, rescale_skip=0):
        super(unet3D_m1, self).__init__()
        self.filters = filters 
        self.io_num = [in_num, out_num]
        self.relu_slope = relu_slope

        filters_in = [in_num] + filters[:-1]
        self.depth = len(filters)-1
        self.seq_num = self.depth*3+2

        self.downC = nn.ModuleList([nn.Sequential(
                nn.Conv3d(filters_in[x], filters_in[x+1], kernel_size=3, stride=1, bias=True),
                nn.LeakyReLU(relu_slope),
                nn.Conv3d(filters_in[x+1], filters_in[x+1], kernel_size=3, stride=1, bias=True),
                nn.LeakyReLU(relu_slope))
            for x in range(self.depth)]) 
        self.downS = nn.ModuleList(
                [nn.MaxPool3d((1,2,2), (1,2,2))
            for x in range(self.depth)]) 
        self.center = nn.Sequential(
                nn.Conv3d(filters[-2], filters[-1], kernel_size=3, stride=1, bias=True),
                nn.LeakyReLU(relu_slope),
                nn.Conv3d(filters[-1], filters[-1], kernel_size=3, stride=1, bias=True),
                nn.LeakyReLU(relu_slope))
        self.upS = nn.ModuleList([nn.Sequential(
                nn.ConvTranspose3d(filters[3-x], filters[3-x], (1,2,2), (1,2,2), groups=filters[3-x], bias=False),
                nn.Conv3d(filters[3-x], filters[2-x], kernel_size=1, stride=1, bias=True))
            for x in range(self.depth)]) 
        # initialize upsample
        for x in range(self.depth):
            self.upS[x]._modules['0'].weight.data.fill_(1.0)

        # double input channels: merge-crop
        self.upC = nn.ModuleList([nn.Sequential(
                nn.Conv3d(2*filters[2-x], filters[2-x], kernel_size=3, stride=1, bias=True),
                nn.LeakyReLU(relu_slope),
                nn.Conv3d(filters[2-x], filters[2-x], kernel_size=3, stride=1, bias=True),
                nn.LeakyReLU(relu_slope))
            for x in range(self.depth)]) 

        self.final = nn.Sequential(nn.Conv3d(filters[0], out_num, kernel_size=1, stride=1, bias=True))


    def getLearnableSeq(self, seq_id): # learnable variable 
        if seq_id < self.depth:
            return self.downC[seq_id]
        elif seq_id == self.depth:
            return self.center
        elif seq_id != self.seq_num-1:
            seq_id = seq_id-self.depth-1
            if seq_id % 2 ==0:
                return self.upS[seq_id/2]
            else:
                return self.upC[seq_id/2]
        else:
            return self.final

    def setLearnableSeq(self, seq_id, seq): # learnable variable 
        if seq_id < self.depth:
            self.downC[seq_id] = seq
        elif seq_id==self.depth:
            self.center = seq
        elif seq_id!=self.seq_num-1:
            seq_id = seq_id-self.depth-1
            if seq_id % 2 ==0:
                self.upS[seq_id/2] = seq
            else:
                self.upC[seq_id/2] = seq
        else:
            self.final = seq

    def forward(self, x):
        down_u = [None]*self.depth
        for i in range(self.depth):
            down_u[i] = self.downC[i](x)
            x = self.downS[i](down_u[i])
        x = self.center(x)
        for i in range(self.depth):
            x = mergeCrop(down_u[self.depth-1-i], self.upS[i](x))
            x = self.upC[i](x)
        return F.sigmoid(self.final(x))

def unet_m2_conv(in_num, out_num, kernel_size, pad_size, stride_size, has_bias, has_BN, has_relu):
    layers = []
    for i in range(len(in_num)):
        layers.append(nn.Conv3d(in_num[i], out_num[i], kernel_size=kernel_size[i], padding=pad_size[i], stride=stride_size[i], bias=has_bias[i])) 
        if has_BN[i]:
            layers.append(nn.BatchNorm3d(out_num[i]))
        if has_relu[i]==0:
            layers.append(nn.ReLU(inplace=True))
        elif has_relu[i]==1:
            layers.append(nn.ELU(inplace=True))
    return nn.Sequential(*layers)

class unet_m2_BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, out_planes, is_3D = True, has_BN = False, has_relu = 0):
        super(unet_m2_BasicBlock, self).__init__()
        self.block1 = unet_m2_conv([in_planes], [out_planes], [(1,3,3)], [(0,1,1)], \
                                   [1], [False], [has_BN], [has_relu])
        # no relu for the second block
        if is_3D:
            self.block2 = unet_m2_conv([out_planes]*2, [out_planes]*2, [(3,3,3)]*2, [(1,1,1)]*2, \
                                       [1]*2, [False]*2, [has_BN]*2, [has_relu,-1])
        else: # a bit different due to bn-2D vs. bn-3D
            self.block2 = unet_m2_conv([out_planes]*2, [out_planes]*2, [(1,3,3)]*2, [(0,1,1)]*2, \
                                       [1]*2, [False]*2, [has_BN]*2, [has_relu,-1])
        if has_relu==0:
            self.block3 = nn.ReLU(inplace=True)
        else:
            self.block3 = nn.ELU(inplace=True)

    def forward(self, x):
        residual = self.block1(x)
        out = residual + self.block2(residual)
        out = self.block3(out)
        return out

class unet3D_m2(nn.Module): # deployed PNI model
    # Superhuman Accuracy on the SNEMI3D Connectomics Challenge. Lee et al.
    # https://arxiv.org/abs/1706.00120
    def __init__(self, in_num=1, out_num=3, filters=[28,36,48,64,80], has_BN=True, has_relu=0):
        super(unet3D_m2, self).__init__()
        self.filters = filters 
        self.io_num = [in_num, out_num]
        self.res_num = len(filters)-2
        self.seq_num = (self.res_num+1)*2+1
        
        self.downC = nn.ModuleList(
                [unet_m2_conv([in_num], [filters[0]], [(1,5,5)], [(0,2,2)], [1], [False], [has_BN], [has_relu])]
                + [unet_m2_BasicBlock(filters[x], filters[x+1], True, has_BN, has_relu)
                      for x in range(self.res_num)]) 
        self.downS = nn.ModuleList(
                [nn.MaxPool3d((1,2,2), (1,2,2))
            for x in range(self.res_num+1)]) 
        self.center = unet_m2_BasicBlock(filters[-2], filters[-1], True, has_BN, has_relu)
        self.upS = nn.ModuleList(
            [nn.Sequential(
                nn.ConvTranspose3d(filters[self.res_num+1-x], filters[self.res_num+1-x], (1,2,2), (1,2,2), groups=filters[self.res_num+1-x], bias=False),
                nn.Conv3d(filters[self.res_num+1-x], filters[self.res_num-x], kernel_size=(1,1,1), stride=1, bias=True))
                for x in range(self.res_num+1)]) 
        # initialize upsample
        for x in range(self.res_num+1):
            self.upS[x]._modules['0'].weight.data.fill_(1.0)

        self.upC = nn.ModuleList(
            [unet_m2_BasicBlock(filters[self.res_num-x], filters[self.res_num-x], True, has_BN, has_relu)
                for x in range(self.res_num)]
            + [nn.Conv3d(filters[0], out_num, kernel_size=(1,5,5), stride=1, padding=(0,2,2), bias=True)]) 

    def forward(self, x):
        down_u = [None]*(self.res_num+1)
        for i in range(self.res_num+1):
            down_u[i] = self.downC[i](x)
            x = self.downS[i](down_u[i])
        x = self.center(x)
        for i in range(self.res_num+1):
            x = down_u[self.res_num-i] + self.upS[i](x)
            x = self.upC[i](x)
        return F.sigmoid(x)

class unet3D_m2_v2(nn.Module):
    # changes from unet3D_m2
    # - add 2D residual module
    # - add 2D residual module
    def __init__(self, in_num=1, out_num=3, filters=[28,36,48,64,80], has_BN=True, has_relu=0):
        super(unet3D_m2_v2, self).__init__()
        self.filters = filters 
        self.io_num = [in_num, out_num]
        self.res_num = len(filters)-2
        self.seq_num = (self.res_num+1)*2+1

        self.downC = nn.ModuleList(
                [unet_m2_conv([in_num], [1], [(1,5,5)], [(0,2,2)], [1], [False], [has_BN], [has_relu])]
                + [unet_m2_BasicBlock(1, filters[0], False, has_BN, has_relu)]
                + [unet_m2_BasicBlock(filters[x], filters[x+1], True, has_BN, has_relu)
                      for x in range(1, self.res_num)]) 
        self.downS = nn.ModuleList(
                [nn.MaxPool3d((1,2,2), (1,2,2))
            for x in range(self.res_num+1)]) 
        self.center = unet_m2_BasicBlock(filters[-2], filters[-1], True, has_BN, has_relu)
        self.upS = nn.ModuleList(
            [nn.Sequential(
                nn.ConvTranspose3d(filters[self.res_num+1-x], filters[self.res_num+1-x], (1,2,2), (1,2,2), groups=filters[self.res_num+1-x], bias=False),
                nn.Conv3d(filters[self.res_num+1-x], filters[self.res_num-x], kernel_size=(1,1,1), stride=1, bias=True))
                for x in range(self.res_num+1)]) 
        # initialize upsample
        for x in range(self.res_num+1):
            self.upS[x]._modules['0'].weight.data.fill_(1.0)

        self.upC = nn.ModuleList( # same number of channels from the left
            [unet_m2_BasicBlock(filters[self.res_num-x], filters[self.res_num-x], True, has_BN, has_relu)
                for x in range(self.res_num-1)]
            + [unet_m2_BasicBlock(filters[0], filters[0], False, has_BN, has_relu)]
            + [nn.Conv3d(filters[0], out_num, kernel_size=(1,5,5), stride=1, padding=(0,2,2), bias=True)]) 

    def forward(self, x):
        down_u = [None]*(self.res_num+1)
        x = self.downC[0](x) # first 1x5x5
        for i in range(self.res_num+1):
            down_u[i] = self.downC[1+i](x)
            x = self.downS[i](down_u[i])
        x = self.center(x)
        for i in range(self.res_num+1):
            x = down_u[self.res_num-i] + self.upS[i](x)
            x = self.upC[i](x)
        x = self.upC[-1](x) # last 1x5x5
        return F.sigmoid(x)

    
class unet3D_m3(nn.Module):
    # 3d unet model for isotropic dataset
    def __init__(self, in_num=1, out_num=3, filters=[28,36,48,64,80], has_BN=True):
        super(unet3D_m3, self).__init__()
        self.filters = filters 
        self.io_num = [in_num, out_num]
        self.res_num = len(filters)-2 #3
        self.seq_num = (self.res_num+1)*2+1

        self.downC = nn.ModuleList(
                  [unet_m3_BasicBlock(1, filters[0], has_BN)]
                + [unet_m3_BasicBlock(filters[x], filters[x+1], has_BN)
                      for x in range(self.res_num)]) 
        self.downS = nn.ModuleList(
                [nn.MaxPool3d((2,2,2), (2,2,2))
            for x in range(self.res_num+1)]) 
        self.center = unet_m3_BasicBlock(filters[-2], filters[-1], has_BN)
        self.upS = nn.ModuleList(
            [nn.Sequential(
                nn.ConvTranspose3d(filters[self.res_num+1-x], filters[self.res_num+1-x], 2, stride=2, padding=0, groups=filters[self.res_num+1-x], bias=False),
                nn.Conv3d(filters[self.res_num+1-x], filters[self.res_num-x], kernel_size=(3,3,3), stride=1, padding=1, bias=True))
                for x in range(self.res_num+1)]) 
        # initialize upsample
        for x in range(self.res_num+1):
            self.upS[x]._modules['0'].weight.data.fill_(1.0)

        self.upC = nn.ModuleList(
            [unet_m3_BasicBlock(filters[self.res_num-x], filters[self.res_num-x], has_BN)
                for x in range(self.res_num)]        
            + [nn.Sequential(
                  unet_m3_BasicBlock(filters[0], filters[0], has_BN),
                  nn.Conv3d(filters[0], out_num, kernel_size=(3,3,3), stride=1, padding=(1,1,1), bias=True))])

    def forward(self, x):
        down_u = [None]*(self.res_num+1)
        for i in range(self.res_num+1):
            down_u[i] = self.downC[i](x)
            #print("downC:",i,down_u[i].size())
            x = self.downS[i](down_u[i])
            #print("downS:",i,x.size())
        x = self.center(x)
        for i in range(self.res_num+1):
            #print("upS:",i,self.upS[i](x).size())
            #print("downU:",self.res_num-i,down_u[self.res_num-i].size())
            x = down_u[self.res_num-i] + self.upS[i](x)
            x = self.upC[i](x)
            #print("upC:",i, x.size())
        return F.sigmoid(x)        


class unet_m3_BasicBlock(nn.Module):
    # Basic module for isotropic dataset
    expansion = 1
    def __init__(self, in_planes, out_planes, has_BN = False):
        super(unet_m3_BasicBlock, self).__init__()
        if has_BN:
            self.block1 = nn.Sequential(
                nn.Conv3d(in_planes,  out_planes, kernel_size=(3,3,3), stride=1, padding=(1,1,1), bias=False),
                nn.BatchNorm3d(out_planes),
                nn.ReLU(inplace=True))
            self.block2 = nn.Sequential(
                nn.Conv3d(out_planes, out_planes, kernel_size=(3,3,3), stride=1, padding=(1,1,1), bias=False),
                nn.BatchNorm3d(out_planes),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_planes, out_planes, kernel_size=(3,3,3), stride=1, padding=(1,1,1), bias=False),
                nn.BatchNorm3d(out_planes))
        else:
            self.block1 = nn.Sequential(
                nn.Conv3d(in_planes,  out_planes, kernel_size=(3,3,3), stride=1, padding=(1,1,1), bias=False),
                nn.ReLU(inplace=True))
            self.block2 = nn.Sequential(
                nn.Conv3d(out_planes, out_planes, kernel_size=(3,3,3), stride=1, padding=(1,1,1), bias=False),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_planes, out_planes, kernel_size=(3,3,3), stride=1, padding=(1,1,1), bias=False))
        self.block3 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual  = self.block1(x)
        out = residual + self.block2(residual)
        out = self.block3(out)
        return out                    

#--------------------
# CNN for VI-approx 
# follow basic res-net architectures, adapted from:
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

def conv3_3D(in_planes, out_planes, stride=1):
    """3x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3,3,3), stride=stride,
                     padding=(1,1,1), bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3_3D(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3_3D(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(6, self.inplanes, kernel_size=5, stride=1, padding=2,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d((2,2,2), (2,2,2))
        self.layer1 = self._make_layer(block, 32, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128,layers[2], stride=2)
        self.avgpool = nn.AvgPool3d(8, stride=1)
        self.fc = nn.Linear(self.inplanes*8 * block.expansion, num_classes)
        self.relu = nn.ReLU(inplace=True)                            

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.relu(x)
                                    
        return x

                                    
def cnn2_v1():
    # Constructs a ResNet model to approximate VI.
    model = ResNet(BasicBlock, [3, 6, 3])
    return model
