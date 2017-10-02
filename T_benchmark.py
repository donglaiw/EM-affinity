import numpy as np
import pickle
import os,sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import malis


def load_weights(model, weights):
    # down
    for i in range(3):
        ww = weights['Convolution'+str(2*i+1)]
        model.down[i].conv.conv1._modules['0'].weight.data.copy_(torch.from_numpy(ww['w']))
        model.down[i].conv.conv1._modules['0'].bias.data.copy_(torch.from_numpy(ww['b']))
        ww = weights['Convolution'+str(2*i+2)]
        model.down[i].conv.conv2._modules['0'].weight.data.copy_(torch.from_numpy(ww['w']))
        model.down[i].conv.conv2._modules['0'].bias.data.copy_(torch.from_numpy(ww['b']))
    # center
    ww = weights['Convolution7']
    model.center.conv1._modules['0'].weight.data.copy_(torch.from_numpy(ww['w']))
    model.center.conv1._modules['0'].bias.data.copy_(torch.from_numpy(ww['b']))
    ww = weights['Convolution8']
    model.center.conv2._modules['0'].weight.data.copy_(torch.from_numpy(ww['w']))
    model.center.conv2._modules['0'].bias.data.copy_(torch.from_numpy(ww['b']))
    # up
    for i in range(3):
        ww = weights['Deconvolution'+str(i+1)]
        model.up[i].up.weight.data.copy_(torch.from_numpy(ww['w']))
        ww = weights['Convolution'+str(9+i*3)]
        model.up[i].up_conv.weight.data.copy_(torch.from_numpy(ww['w']))
        model.up[i].up_conv.bias.data.copy_(torch.from_numpy(ww['b']))
        ww = weights['Convolution'+str(10+3*i)]
        model.up[i].conv.conv1._modules['0'].weight.data.copy_(torch.from_numpy(ww['w']))
        model.up[i].conv.conv1._modules['0'].bias.data.copy_(torch.from_numpy(ww['b']))
        ww = weights['Convolution'+str(11+3*i)]
        model.up[i].conv.conv2._modules['0'].weight.data.copy_(torch.from_numpy(ww['w']))
        model.up[i].conv.conv2._modules['0'].bias.data.copy_(torch.from_numpy(ww['b']))
    ww = weights['Convolution18']
    model.final.conv.weight.data.copy_(torch.from_numpy(ww['w']))
    model.final.conv.bias.data.copy_(torch.from_numpy(ww['b']))

class unetVgg3(nn.Module):
    def __init__(self, in_size, out_size, pad_size, relu_slope, is_batchnorm):
        super(unetVgg3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, 3, padding=pad_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.LeakyReLU(relu_slope))
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, 3, padding=pad_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.LeakyReLU(relu_slope))
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, 3, padding=pad_size),
                                       nn.LeakyReLU(relu_slope))
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, 3, padding=pad_size),
                                       nn.LeakyReLU(relu_slope))
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetDown(nn.Module):
    def __init__(self, in_size, out_size, pad_size, relu_slope, pool_kernel, pool_stride, is_batchnorm):
        super(unetDown, self).__init__()
        self.conv = unetVgg3(in_size, out_size, pad_size, relu_slope, is_batchnorm)
        self.down = nn.MaxPool3d(pool_kernel, pool_stride)

    def forward(self, inputs):
        outputs1 = self.conv(inputs)
        outputs2 = self.down(outputs1)
        return outputs1, outputs2

class unetUp(nn.Module):
    # in1: skip layer, in2: previous layer
    def __init__(self, in1_size, in2_size, out_size, pad_size, relu_slope, up_kernel, up_stride, upC_kernel, num_group, has_bias, is_batchnorm):
        super(unetUp, self).__init__()
        self.up = nn.ConvTranspose3d(in2_size, in2_size, up_kernel, up_stride,groups=num_group, bias=has_bias)
        self.up_conv = nn.Conv3d(in2_size, out_size, upC_kernel)
        self.conv = unetVgg3(out_size+in1_size, out_size, pad_size, relu_slope, is_batchnorm)

    def forward(self, inputs1, inputs2):
        # inputs1 is bigger
        outputs2 = self.up_conv(self.up(inputs2))
        offset = [(inputs1.size()[x+2]-outputs2.size()[x+2])/2 for x in range(3)]
        offset2 = [inputs1.size()[x+2]-outputs2.size()[x+2]-offset[x] for x in range(3)]
        # crop merge
        outputs1 = inputs1[:,:, offset[0]:-offset2[0], offset[1]:-offset2[1], offset[2]:-offset2[2]]
        return self.conv(torch.cat([outputs2, outputs1], 1))

class unetFinal(nn.Module):
    def __init__(self, in_size, out_size, kernel_size):
        super(unetFinal, self).__init__()
        self.conv = nn.Conv3d(in_size, out_size, kernel_size)
    def forward(self, inputs):
        outputs = self.conv(inputs)
        return F.sigmoid(outputs)


class unet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, filters=[24,72,216,648], has_bias=False, num_group=[24,72,216,648], vgg_pad=0, relu_slope=0.005, pool_kernel=(1,2,2), pool_stride=(1,2,2), upC_kernel=(1,1,1),is_batchnorm=False):
        super(unet3D, self).__init__()
        self.is_batchnorm = is_batchnorm

        self.down =  nn.ModuleList([unetDown(in_channels, filters[0], vgg_pad, relu_slope, pool_kernel, pool_stride, self.is_batchnorm), 
                     unetDown(filters[0], filters[1], vgg_pad, relu_slope, pool_kernel, pool_stride, self.is_batchnorm),
                     unetDown(filters[1], filters[2], vgg_pad, relu_slope, pool_kernel, pool_stride, self.is_batchnorm)])
        self.center = unetVgg3(filters[2], filters[3], vgg_pad, relu_slope, self.is_batchnorm)
        self.up =  nn.ModuleList([unetUp(filters[2], filters[3], filters[2], vgg_pad, relu_slope, pool_kernel, pool_stride, upC_kernel, num_group[3], has_bias, self.is_batchnorm),
                   unetUp(filters[1], filters[2], filters[1], vgg_pad, relu_slope, pool_kernel, pool_stride, upC_kernel, num_group[2], has_bias, self.is_batchnorm),
                   unetUp(filters[0], filters[1], filters[0], vgg_pad, relu_slope, pool_kernel, pool_stride, upC_kernel, num_group[1], has_bias, self.is_batchnorm)])
        self.final = unetFinal(filters[0], out_channels, (1,1,1))

    def forward(self, inputs):
        down1_u, down1 = self.down[0](inputs)
        down2_u, down2 = self.down[1](down1)
        down3_u, down3 = self.down[2](down2)
        center = self.center(down3)
        up3 = self.up[0](down3_u, center)
        up2 = self.up[1](down2_u, up3)
        up1 = self.up[2](down1_u, up2)
        return self.final(up1)


if __name__ == "__main__":
    # config 
    DATA_DIR = '/n/coxfs01/donglai/malis_trans/malis_test/'
    # select implementation method
    opt_impl = int(sys.argv[1]) # 0: no pre-compute, 1: pre-compute 
    # select data type
    opt_data = int(sys.argv[2]) # 0: only 1-seg (pos edge), 1: multiple-seg (both pos and neg edge)
    data_suf=['_pos','_pair'][opt_data]

    # build model
    res = DATA_DIR+'test_malis_ecs_res'+data_suf+'.pkl'
    if not os.path.exists(res):
        net = unet3D().cuda()
        # load weight
        ww=pickle.load(open(DATA_DIR+'net_weight.pkl','rb'))
        # set weight
        load_weights(net, ww) 
        # predict
        data=pickle.load(open(DATA_DIR+'test_malis_ecs'+data_suf+'.pkl','rb'))
        x = Variable(torch.from_numpy(data.astype(np.float32)).cuda())
        net.eval()
        y = net(x)
        pickle.dump(y.data.cpu().numpy(),open(res,'wb'))
    else:
        y=Variable(torch.from_numpy(pickle.load(open(res,'rb'))),requires_grad=False).cuda()
    seg_cpu = pickle.load(open(DATA_DIR+'test_malis_ecs_seg'+data_suf+'.pkl','rb')).astype(np.uint64)
    # from T_util import bwlabel;print bwlabel(seg_cpu)
    aff_cpu = pickle.load(open(DATA_DIR+'test_malis_ecs_aff'+data_suf+'.pkl','rb')).astype(np.float32)
    conn_dims = aff_cpu.shape
    criterion = malis.MalisLoss(conn_dims, opt_impl).cuda() #
    l = criterion(y,seg_cpu, aff_cpu)
    st=time.time()
    for i in range(1000):
        l = criterion(y,seg_cpu, aff_cpu)
    et=time.time()
    print et-st
    #print l.data[0] #malis:0.00302
