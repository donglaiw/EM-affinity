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

def unet_m2_conv(in_num, out_num, kernel_size, pad_size, stride_size, has_bias, has_BN, relu_opt):
    layers = []
    for i in range(len(in_num)):
        layers.append(nn.Conv3d(in_num[i], out_num[i], kernel_size=kernel_size[i], padding=pad_size[i], stride=stride_size[i], bias=has_bias[i])) 
        if has_BN[i]:
            layers.append(nn.BatchNorm3d(out_num[i]))
        if relu_opt[i]==0:
            layers.append(nn.ReLU(inplace=True))
        elif relu_opt[i]==1:
            layers.append(nn.ELU(inplace=True))
    return nn.Sequential(*layers)

class unet_m2_BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, out_planes, is_3D = True, has_BN = False, relu_opt = 0):
        super(unet_m2_BasicBlock, self).__init__()
        self.block1 = unet_m2_conv([in_planes], [out_planes], [(1,3,3)], [(0,1,1)], \
                                   [1], [False], [has_BN], [relu_opt])
        # no relu for the second block
        if is_3D:
            self.block2 = unet_m2_conv([out_planes]*2, [out_planes]*2, [(3,3,3)]*2, [(1,1,1)]*2, \
                                       [1]*2, [False]*2, [has_BN]*2, [relu_opt,-1])
        else: # a bit different due to bn-2D vs. bn-3D
            self.block2 = unet_m2_conv([out_planes]*2, [out_planes]*2, [(1,3,3)]*2, [(0,1,1)]*2, \
                                       [1]*2, [False]*2, [has_BN]*2, [relu_opt,-1])
        if relu_opt==0:
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
    def __init__(self, in_num=1, out_num=3, filters=[28,36,48,64,80], has_BN=True, relu_opt=0):
        super(unet3D_m2, self).__init__()
        self.filters = filters 
        self.io_num = [in_num, out_num]
        self.res_num = len(filters)-2
        self.seq_num = (self.res_num+1)*2+1
        
        self.downC = nn.ModuleList(
                [unet_m2_conv([in_num], [filters[0]], [(1,5,5)], [(0,2,2)], [1], [False], [has_BN], [relu_opt])]
                + [unet_m2_BasicBlock(filters[x], filters[x+1], True, has_BN, relu_opt)
                      for x in range(self.res_num)]) 
        self.downS = nn.ModuleList(
                [nn.MaxPool3d((1,2,2), (1,2,2))
            for x in range(self.res_num+1)]) 
        self.center = unet_m2_BasicBlock(filters[-2], filters[-1], True, has_BN, relu_opt)
        self.upS = nn.ModuleList(
            [nn.Sequential(
                nn.ConvTranspose3d(filters[self.res_num+1-x], filters[self.res_num+1-x], (1,2,2), (1,2,2), groups=filters[self.res_num+1-x], bias=False),
                nn.Conv3d(filters[self.res_num+1-x], filters[self.res_num-x], kernel_size=(1,1,1), stride=1, bias=True))
                for x in range(self.res_num+1)]) 
        # initialize upsample
        for x in range(self.res_num+1):
            self.upS[x]._modules['0'].weight.data.fill_(1.0)

        self.upC = nn.ModuleList(
            [unet_m2_BasicBlock(filters[self.res_num-x], filters[self.res_num-x], True, has_BN, relu_opt)
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
    def __init__(self, in_num=1, out_num=3, filters=[28,36,48,64,80], has_BN=True, relu_opt=0):
        super(unet3D_m2_v2, self).__init__()
        self.filters = filters 
        self.io_num = [in_num, out_num]
        self.res_num = len(filters)-2
        self.seq_num = (self.res_num+1)*2+1

        self.downC = nn.ModuleList(
                [unet_m2_conv([in_num], [1], [(1,5,5)], [(0,2,2)], [1], [False], [has_BN], [relu_opt])]
                + [unet_m2_BasicBlock(1, filters[0], False, has_BN, relu_opt)]
                + [unet_m2_BasicBlock(filters[x], filters[x+1], True, has_BN, relu_opt)
                      for x in range(1, self.res_num)]) 
        self.downS = nn.ModuleList(
                [nn.MaxPool3d((1,2,2), (1,2,2))
            for x in range(self.res_num+1)]) 
        self.center = unet_m2_BasicBlock(filters[-2], filters[-1], True, has_BN, relu_opt)
        self.upS = nn.ModuleList(
            [nn.Sequential(
                nn.ConvTranspose3d(filters[self.res_num+1-x], filters[self.res_num+1-x], (1,2,2), (1,2,2), groups=filters[self.res_num+1-x], bias=False),
                nn.Conv3d(filters[self.res_num+1-x], filters[self.res_num-x], kernel_size=(1,1,1), stride=1, bias=True))
                for x in range(self.res_num+1)]) 
        # initialize upsample
        for x in range(self.res_num+1):
            self.upS[x]._modules['0'].weight.data.fill_(1.0)

        self.upC = nn.ModuleList( # same number of channels from the left
            [unet_m2_BasicBlock(filters[self.res_num-x], filters[self.res_num-x], True, has_BN, relu_opt)
                for x in range(self.res_num-1)]
            + [unet_m2_BasicBlock(filters[0], filters[0], False, has_BN, relu_opt)]
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

class ffn_BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=(3,3,3), pad_size=1):
        super(ffn_BasicBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad_size, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_planes, out_planes, kernel_size=kernel_size, padding=pad_size, bias=True))

    def forward(self, x):
        return x+self.conv(x)

class ffn(nn.Module): # deployed FFN model
    # https://github.com/google/ffn/blob/master/ffn/training/models/convstack_3d.py
    # https://github.com/google/ffn/blob/master/ffn/training/model.py
    def __init__(self, in_seed=1, in_patch=1, deltas=(4,4,4), 
                 depth=11, filter_num=32, pad_size=1, kernel_size=(3,3,3)):
        super(ffn, self).__init__()
        self.build_rnn(in_seed, in_patch, deltas)
        self.build_conv(depth, filter_num, pad_size, kernel_size)

    def set_uniform_io_size(self, patch_size):
        """Initializes unset input/output sizes to 'patch_size', sets input shapes.
        This assumes that the inputs and outputs are of equal size, and that exactly
        one step is executed in every direction during training.
        Args:
          patch_size: (x, y, z) specifying the input/output patch size
        Returns:
          None
        """
        if self.pred_mask_size is None:
            self.pred_mask_size = patch_size
        if self.input_seed_size is None:
            self.input_seed_size = patch_size
        if self.input_image_size is None:
            self.input_image_size = patch_size
        self.set_input_shapes()

    def set_input_shapes(self):
        """Sets the shape inference for input_seed and input_patches.
        Assumes input_seed_size and input_image_size are already set.
        """
        self.input_seed.set_shape([self.batch_size] +
                              list(self.input_seed_size[::-1]) + [1])
        self.input_patches.set_shape([self.batch_size] +
                              list(self.input_image_size[::-1]) + [1])

    def build_rnn(self, in_seed, in_patch, deltas): 
        # parameters for recurrent
        self.in_seed = in_seed
        self.in_patch = in_patch
        self.deltas = deltas
        self.shifts = []
        for dx in (-self.deltas[0], 0, self.deltas[0]):
            for dy in (-self.deltas[1], 0, self.deltas[1]):
                for dz in (-self.deltas[2], 0, self.deltas[2]):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    self.shifts.append((dx, dy, dz))

    def build_conv(self, depth, filter_num, pad_size, kernel_size):
        self.depth = depth 
        # build convolution model 
        self.conv0 = nn.Sequential(
            nn.Conv3d(2, filter_num, kernel_size=kernel_size, stride=1, padding=pad_size, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(filter_num, filter_num, kernel_size=kernel_size, stride=1, padding=pad_size, bias=True))

        self.conv1 = nn.ModuleList(
                [ffn_BasicBlock(filter_num, filter_num, kernel_size, pad_size)
                      for x in range(self.depth)]) 

        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(filter_num, filter_num, kernel_size=1, bias=True))
        self.out = nn.Sigmoid()

    def predict_object_mask(x):
        out = self.conv0(x)
        for i in range(self.depth):
            out = self.conv1[i](out)
        return self.conv2(out)

    def update_seed(self, seed, update):
        """Updates the initial 'seed' with 'update'."""
        dx = self.input_seed_size[0] - self.pred_mask_size[0]
        dy = self.input_seed_size[1] - self.pred_mask_size[1]
        dz = self.input_seed_size[2] - self.pred_mask_size[2]

        if dx == 0 and dy == 0 and dz == 0:
            seed += update
        else:
            seed += F.pad(update, [[0, 0],
                              [dz // 2, dz - dz // 2],
                              [dy // 2, dy - dy // 2],
                              [dx // 2, dx - dx // 2],
                              [0, 0]])
        return seed

    def forward(self, x):
        # input 
        logit_update = self.predict_object_mask(x)

        logit_seed = self.update_seed(self.input_seed, logit_update)
        
        self.logits = logit_seed
        self.logistic = self.out(logit_seed)

