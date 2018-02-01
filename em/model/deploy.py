# deployed model without much flexibility
# useful for stand-alone test, model translation, quantization
import torch
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

class unet_m2_BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, out_planes, has_BN = False):
        super(unet_m2_BasicBlock, self).__init__()
        if has_BN:
            self.block1 = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
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
                nn.Conv3d(in_planes, out_planes, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
                nn.ReLU(inplace=True))
            self.block2 = nn.Sequential(
                nn.Conv3d(out_planes, out_planes, kernel_size=(3,3,3), stride=1, padding=(1,1,1), bias=False),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_planes, out_planes, kernel_size=(3,3,3), stride=1, padding=(1,1,1), bias=False))
        self.block3 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.block1(x)
        out = residual + self.block2(residual)
        out = self.block3(out)
        return out

class unet3D_m2(nn.Module): # deployed PNI model
    # Superhuman Accuracy on the SNEMI3D Connectomics Challenge. Lee et al.
    # https://arxiv.org/abs/1706.00120
    def __init__(self, in_num=1, out_num=3, filters=[28,36,48,64,80], has_BN=True):
        super(unet3D_m2, self).__init__()
        self.filters = filters 
        self.io_num = [in_num, out_num]
        self.res_num = len(filters)-2
        self.seq_num = (self.res_num+1)*2+1

        self.downC = nn.ModuleList(
                [nn.Sequential(
                    nn.Conv3d(in_num, filters[0], kernel_size=(1,5,5), stride=1, padding=(0,2,2), bias=False),
                    nn.BatchNorm3d(filters[0]),
                    nn.ReLU(True))]
                + [unet_m2_BasicBlock(filters[x], filters[x+1], has_BN)
                      for x in range(self.res_num)]) 
        self.downS = nn.ModuleList(
                [nn.MaxPool3d((1,2,2), (1,2,2))
            for x in range(self.res_num+1)]) 
        self.center = unet_m2_BasicBlock(filters[-2], filters[-1], has_BN)
        self.upS = nn.ModuleList(
            [nn.Sequential(
                nn.ConvTranspose3d(filters[self.res_num+1-x], filters[self.res_num+1-x], (1,2,2), (1,2,2), groups=filters[self.res_num+1-x], bias=False),
                nn.Conv3d(filters[self.res_num+1-x], filters[self.res_num-x], kernel_size=(1,1,1), stride=1, bias=True))
                for x in range(self.res_num+1)]) 
        # initialize upsample
        for x in range(self.res_num+1):
            self.upS[x]._modules['0'].weight.data.fill_(1.0)

        self.upC = nn.ModuleList(
            [unet_m2_BasicBlock(filters[self.res_num-x], filters[self.res_num-x], has_BN)
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

    def getLearnableSeq(self, seq_id): # learnable variable 
        if seq_id < self.res_num+1:
            return self.downC[seq_id]
        elif seq_id == self.res_num+1:
            return self.center
        elif seq_id != self.seq_num-1:
            seq_id = seq_id-self.res_num
            if seq_id % 2 ==0:
                return self.upS[seq_id/2]
            else:
                return self.upC[seq_id/2]
        else:
            return self.final

    def setLearnableSeq(self, seq_id, seq): # learnable variable 
        if seq_id < self.res_num+1:
            self.downC[seq_id] = seq
        elif seq_id==self.res_num+1:
            self.center = seq
        elif seq_id!=self.seq_num-1:
            seq_id = seq_id-self.res_num
            if seq_id % 2 ==0:
                self.upS[seq_id/2] = seq
            else:
                self.upC[seq_id/2] = seq
        else:
            self.final = seq
