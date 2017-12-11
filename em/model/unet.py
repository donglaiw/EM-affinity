import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from .block import *

class unetDown(nn.Module): # type 
    def __init__(self, opt, in_num, out_num, 
                 cfg_down={'pool_kernel': (1,2,2), 'pool_stride': (1,2,2), 'out_num': 1},
        cfg_conv={'pad_size':0, 'pad_type':'', 'has_bias':True, 'has_BN':False, 'relu_slope':-1, 'has_dropout':0}, block_id=0):
        super(unetDown, self).__init__()
        self.opt = opt
        if opt[0]==0: # max-pool
            self.down = nn.MaxPool3d(cfg_down['pool_kernel'], cfg_down['pool_stride'])
        elif opt[0]==1: # resBasic
            self.down = blockResNet(unitResBasic, 1, in_num, cfg_down['out_num'], stride_size=cfg_down['pool_stride'], cfg=cfg_conv)

        if opt[1]==0: # vgg-3x3 
            self.conv = blockVgg(2, in_num, out_num, cfg=cfg_conv)
        elif opt[1]==1: # res-18
            self.conv = blockResNet(unitResBasic, 2, in_num, out_num, cfg=cfg_conv)
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.down(x1)
        return x1, x2

class unetUp(nn.Module):
    # in1: skip layer, in2: previous layer
    def __init__(self, opt, in_num, outUp_num, inLeft_num, outConv_num,
        cfg_up={'pool_kernel': (1,2,2), 'pool_stride': (1,2,2)},
        cfg_conv={'pad_size':0, 'pad_type':'', 'has_bias':True, 'has_BN':False, 'relu_slope':0.005, 'has_dropout':0}, block_id=0):
        super(unetUp, self).__init__()
        self.opt = opt
        if opt[0]==0: # upsample+conv
            self.up = nn.Sequential(nn.ConvTranspose3d(in_num, in_num, cfg_up['pool_kernel'], cfg_up['pool_stride'], groups=in_num, bias=False),
                unitConv3dRBD(in_num, outUp_num, 1, 1, 0, '', True))
            self.up._modules['0'].weight.data.fill_(1.0)
            # not supported yet: anisotropic upsample
            # self.up = nn.Sequential(nn.Upsample(scale_factor=cfg_up.pool_kernel, mode='nearest'),
            #    unitConv3dRBD(in_num, outUp_num, 1, 1, 0, '', True))
        elif opt[0]==1: # group deconv, remember to initialize with (1,0)
            self.up = nn.Sequential(nn.ConvTranspose3d(in_num, in_num, cfg_up['pool_kernel'], cfg_up['pool_stride'], groups=in_num, bias=True),
                unitConv3dRBD(in_num, outUp_num, 1, 1, 0, '', True))
            self.up._modules['0'].weight.data.fill_(1.0)
        elif opt[0]==2: # residual deconv
            self.up = blockResNet(unitResBasic, 1, in_num, in_num, stride_size=cfg_down['pool_stride'], do_sample=-1, cfg=cfg_conv)
            outUp_num = in_num

        if opt[1]==0: # merge-crop
            self.mc = mergeCrop
            inConv_num = outUp_num+inLeft_num
        elif opt[1]==1: # merge-add
            self.mc = mergeAdd
            inConv_num = min(outUp_num,inLeft_num)

        if opt[2]==0: # deconv
            self.conv = blockVgg(2, inConv_num, outConv_num, cfg=cfg_conv)
        elif opt[2]==1: # residual
            self.conv = blockResNet(unitResBasic, 2, inConv_num, outConv_num, cfg=cfg_conv)

    def forward(self, x1, x2):
        # inputs1 from left-side (bigger)
        x2_up = self.up(x2)
        mc = self.mc(x1, x2_up)
        return self.conv(mc)

class unetCenter(nn.Module):
    def __init__(self, opt, in_num, out_num,
        cfg_conv={'pad_size':0, 'pad_type':'', 'has_bias':True, 'has_BN':False, 'relu_slope':0.005, 'has_dropout':0} ):
        super(unetCenter, self).__init__()
        self.opt = opt
        if opt[0]==0: # vgg
            self.conv = blockVgg(2, in_num, out_num, cfg=cfg_conv)
        elif opt[0]==1: # residual
            self.conv = blockResNet(unitResBasic, 2, in_num, out_num, cfg=cfg_conv)
    def forward(self, x):
        return self.conv(x)

class unetFinal(nn.Module):
    def __init__(self, opt, in_num, out_num):
        super(unetFinal, self).__init__()
        self.opt = opt
        if opt[0]==0: # vgg
            self.conv = unitConv3dRBD(in_num, out_num, 1, 1, 0, '', True)
        elif opt[0]==1: # resnet
            self.conv = blockResNet(unitResBasic, 1, in_num, out_num, 1, cfg={'pad_size':0, 'pad_type':'constant,0', 'has_BN':False, 'relu_slope':-1})

    def forward(self, x):
        return F.sigmoid(self.conv(x))


class unet3D(nn.Module): # symmetric unet
    # default global parameter 
    # opt_arch: change component 
    # component parameter
    def __init__(self, opt_arch=[[0,0],[0],[0,0,0],[0]], opt_param=[[0],[0],[0],[0]], 
                 in_num=1, out_num=3, filters=[24,72,216,648],
                 has_bias=True, has_BN=False,has_dropout=0,pad_size=0,pad_type='',relu_slope=0.005,
                 pool_kernel=(1,2,2), pool_stride=(1,2,2)):
        super(unet3D, self).__init__()
        self.depth = len(filters)-1 
        filters_in = [in_num] + filters[:-1]
        cfg_conv={'has_bias':has_bias,'has_BN':has_BN, 'has_dropout':has_dropout, 'pad_size':pad_size, 'pad_type':pad_type, 'relu_slope':relu_slope}
        cfg_pool={'pool_kernel':pool_kernel, 'pool_stride':pool_stride}

        # --- down arm ---
        self.down = nn.ModuleList([
                    unetDown(opt_arch[0], filters_in[x], filters_in[x+1], cfg_pool, cfg_conv, x) 
                    for x in range(len(filters)-1)]) 

        # --- center arm ---
        cfg_conv_c = copy.deepcopy(cfg_conv)
        if opt_param[1][0]==1: # pad for conv
            cfg_conv_c['pad_size']=1;cfg_conv_c['pad_type']='replicate';
        self.center = unetCenter(opt_arch[1], filters[-2], filters[-1], cfg_conv_c)

        # --- up arm ---
        cfg_conv_u = copy.deepcopy(cfg_conv)
        if opt_param[2][0]==1: # pad for conv
            cfg_conv_u['pad_size']=1;cfg_conv_c['pad_type']='replicate';
        self.up = nn.ModuleList([
                    unetUp(opt_arch[2], filters[x], filters[x-1], filters[x-1], filters[x-1], cfg_pool, cfg_conv_u, x)
                    for x in range(len(filters)-1,0,-1)])

        # --- final arm ---
        self.final = unetFinal(opt_arch[3], filters[0], out_num) 

    def forward(self, x):
        down_u = [None]*self.depth
        for i in range(self.depth):
            down_u[i], x = self.down[i](x)
        x = self.center(x)
        for i in range(self.depth):
             x = self.up[i](down_u[self.depth-1-i],x)
        return self.final(x)

class unet3D_m1(nn.Module): # deployed model-1
    def __init__(self, in_num=1, out_num=3, filters=[24,72,216,648],relu_slope=0.005):
        super(unet3D_m1, self).__init__()
        if len(filters) != 4: raise AssertionError 
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
