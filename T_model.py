import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# --------------------------------
# building blocks for unet construction
# --------------------------------
## level-1: single unit
def mergeCrop(x1, x2):
    # x1 left, x2 right
    offset = [(x1.size()[x]-x2.size()[x])/2 for x in range(2,x1.dim()) for i in range(2)] 
    return torch.cat([x2, x1[:,:,offset[0]:offset[0]+x2.size(2),
            offset[1]:offset[1]+x2.size(3),offset[2]:offset[2]+x2.size(4)]], 1)
def mergeAdd(x1, x2):
    # x1 bigger
    offset = [(x1.size()[x]-x2.size()[x])/2 for x in range(1,x1.dim()) for i in range(2)] 
    #print x1.size(),x2.size(),offset
    return x2 + x1[:,offset[0]:offset[0]+x2.size(1),offset[1]:offset[1]+x2.size(2),
            offset[2]:offset[2]+x2.size(3),offset[3]:offset[3]+x2.size(4)]

class unitConv3dRBD(nn.Module):
    # one conv-bn-relu-dropout neuron with different padding
    def __init__(self, in_num=1, out_num=1, kernel_size=1, stride_size=1, pad_size=0, pad_type='constant,0', has_bias=False, has_BN=False, relu_slope=-1, has_dropout=0):
        super(unitConv3dRBD, self).__init__()
        p_conv = pad_size if pad_type == 'constant,0' else 0 #  0-padding in conv layer 
        layers = [nn.Conv3d(in_num, out_num, kernel_size=kernel_size, padding=p_conv, stride=stride_size, bias=has_bias)] 
        if has_BN:
            layers.append(nn.BatchNorm3d(out_num))
        if relu_slope==0:
            layers.append(nn.ReLU(inplace=True))
        elif relu_slope>0:
            layers.append(nn.LeakyReLU(relu_slope))
        if has_dropout>0:
            layers.append(nn.Dropout3d(has_dropout, True))
        self.cbrd = nn.Sequential(*layers)
        self.pad_type = pad_type
        if pad_size==0:
            self.pad_size = 0
        else:
            self.pad_size = tuple([pad_size]*6)
        if ',' in pad_type:
            self.pad_value = float(pad_type[pad_type.find(',')+1:])

    def forward(self, x):
        if isinstance(self.pad_size,int) or self.pad_type == 'constant,0': # no padding or 0-padding
            return self.cbrd(x)
        else:
            if ',' in self.pad_type:# constant padding
                return self.cbrd(F.pad(x,self.pad_size,'constant',self.pad_value))
            else:
                if self.pad_type!='reflect':# reflect: hack with numpy (not implemented in)...
                    return self.cbrd(F.pad(x,self.pad_size,self.pad_type))

class unitResBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_num, out_num,  stride_size=1, downsample=None, 
                 cfg={'pad_size':1, 'pad_type':'constant,0', 'has_BN':True}):
        super(unitResBottleneck, self).__init__()
        self.conv = nn.Sequential(
            unitConv3dRBD(in_num, out_num, 1, has_BN=has_BN),
            unitConv3dRBD(out_num, out_num, 3, stride, pad_size, pad_type, has_BN=has_BN),
            unitConv3dRBD(out_num, out_num*4, 1, has_BN=has_BN))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.convs(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class unitResBasic(nn.Module):
    expansion = 1
    def __init__(self, in_num, out_num, kernel_size=1, stride_size=1, do_sample=1, 
                 pad_size=1, pad_type='constant,0', has_BN=True, relu_slope=0):
        super(unitResBasic, self).__init__()
        self.sample = None
        if do_sample>=0: # 1=downsample, 0=same size:
            self.conv= nn.Sequential(
                unitConv3dRBD(in_num, out_num, kernel_size, stride_size, pad_size, pad_type, has_BN=has_BN, relu_slope=relu_slope),
                unitConv3dRBD(out_num, out_num, kernel_size, 1, pad_size, pad_type, has_BN=has_BN))
            if (in_num != out_num * self.expansion) or (isinstance(stride_size,int) and stride_size!=1) or (not isinstance(stride_size,int) and  max(stride_size) != 1): # downsample:
                self.sample = unitConv3dRBD(in_num, out_num * self.expansion, 1, stride_size, has_BN=has_BN)
        else: # -1: upsample
            assert in_num == out_num
            self.conv= nn.Sequential(
                nn.ConvTranspose3d(in_num, in_num, stride_size, stride_size, groups=in_num, bias=False),
                unitConv3dRBD(in_num, in_num, kernel_size, 1, pad_size, pad_type, has_BN=has_BN))
            self.sample = nn.ConvTranspose3d(in_num, in_num, stride_size, stride_size, groups=in_num, bias=False)
        self.relu = None
        if relu_slope==0:
            self.relu = nn.ReLU(inplace=True)
        elif relu_slope>0 and relu_slope<1:
            self.relu = nn.LeakyReLU(relu_slope)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        if self.sample is not None:
            residual = self.sample(x)
        out = mergeAdd(residual, out)
        if self.relu is not None:
            out = self.relu(out)
        return out

# level-2: list of level-1 blocks
def blockResNet(unit, unit_num, in_num, out_num, kernel_size=3, stride_size=1, do_sample=0,
                cfg={'pad_size':1, 'pad_type':'constant,0', 'has_BN':True, 'relu_slope':0.005}):
    layers = []
    pre_pad_size=cfg['pad_size']
    pre_pad_type=cfg['pad_type']
    cfg['pad_size'] = (kernel_size-1)/2
    cfg['pad_type'] = 'replicate'
    for i in range(unit_num):
        if i==unit_num-1:
            cfg['pad_size'] = pre_pad_size
            cfg['pad_type'] = pre_pad_type
        layers.append(unit(in_num, out_num, kernel_size, stride_size, do_sample, 
                           cfg['pad_size'], cfg['pad_type'], cfg['has_BN'], cfg['relu_slope']))
        in_num = out_num * unit.expansion
    return nn.Sequential(*layers)

def blockVgg(unit_num, in_num, out_num, kernel_size=3, stride_size=1, 
        cfg={'pad_size':0, 'pad_type':'', 'has_bias':True, 'has_BN':False, 'relu_slope':0.005, 'has_dropout':0}):
        layers= [unitConv3dRBD(in_num, out_num, kernel_size, stride_size, 
                        cfg['pad_size'], cfg['pad_type'], cfg['has_bias'], cfg['has_BN'], cfg['relu_slope'], cfg['has_dropout'])]
        for i in range(unit_num-1):
            if i==unit_num-2: # no dropout in the last layer
                cfg['has_dropout'] = 0
            layers.append(unitConv3dRBD(out_num, out_num, kernel_size, stride_size, cfg['pad_size'], cfg['pad_type'], cfg['has_bias'], cfg['has_BN'], cfg['relu_slope'], cfg['has_dropout']))
        return nn.Sequential(*layers)

# level-3: down-up module
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
    def __init__(self, opt_arch=[[0,0],[0],[0,0,0],[0]], in_num=1, out_num=3, filters=[24,72,216,648],
                 has_bias=True, has_BN=False,has_dropout=0,pad_size=0,pad_type='',relu_slope=0.005,
                 pool_kernel=(1,2,2), pool_stride=(1,2,2)):
        super(unet3D, self).__init__()
        cfg_conv={'has_bias':has_bias,'has_BN':has_BN, 'has_dropout':has_dropout, 'pad_size':pad_size, 'pad_type':pad_type, 'relu_slope':relu_slope}
        cfg_pool={'pool_kernel':pool_kernel, 'pool_stride':pool_stride}

        self.depth = len(filters)-1 
        filters_in = [in_num] + filters[:-1]
        self.down = nn.ModuleList([
                    unetDown(opt_arch[0], filters_in[x], filters_in[x+1], cfg_pool, cfg_conv, x) 
                    for x in range(len(filters)-1)]) 
        self.center = unetCenter(opt_arch[1], filters[-2], filters[-1], cfg_conv)
        self.up = nn.ModuleList([
                    unetUp(opt_arch[2], filters[x], filters[x-1], filters[x-1], filters[x-1], cfg_pool, cfg_conv, x)
                    for x in range(len(filters)-1,0,-1)])
        self.final = unetFinal(opt_arch[3], filters[0], out_num) 

    def forward(self, x):
        down_u = [None]*self.depth
        for i in range(self.depth):
            down_u[i], x = self.down[i](x)
        x = self.center(x)
        for i in range(self.depth):
             x = self.up[i](down_u[self.depth-1-i],x)
        return self.final(x)
