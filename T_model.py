import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --------------------------------
# 1. building block
# level-1: one conv-bn-relu neuron with different padding
class CBR3D(nn.Module):
    def __init__(self, in_size=1, out_size=1, kernel_size=1, stride_size=1, pad_size=0, pad_type='constant,0', has_bias=False, has_BN=False, relu_slope=-1):
        super(CBR3D, self).__init__()
        p_conv = pad_size if pad_type == 'constant,0' else 0 #  0-padding in conv layer 
        layers = [nn.Conv3d(in_size, out_size, kernel_size=kernel_size, padding=p_conv, stride=stride_size, bias=has_bias)] 
        if has_BN:
            layers.append(nn.BatchNorm3d(out_size))
        if relu_slope==0:
            layers.append(nn.ReLU(inplace=True))
        elif relu_slope>0:
            layers.append(nn.LeakyReLU(relu_slope))
        self.cbr = nn.Sequential(*layers)
        self.pad_type = pad_type
        if pad_size==0:
            self.pad_size = 0
        else:
            self.pad_size = tuple([pad_size]*6)
        if ',' in pad_type:
            self.pad_value = float(pad_type[pad_type.find(',')+1:])

    def forward(self, x):
        if isinstance(self.pad_size,int) or self.pad_type == 'constant,0': # no padding or 0-padding
            return self.cbr(x)
        else:
            if ',' in self.pad_type:# constant padding
                return self.cbr(F.pad(x,self.pad_size,'constant',self.pad_value))
            else:
                if self.pad_type=='reflect':# reflect: hack with numpy (not implemented in)...
                    return self.cbr(pad5DReflect(x,self.pad_size))
                else: # other types
                    return self.cbr(F.pad(x,self.pad_size,self.pad_type))

# level-2: list of level-1 blocks
class unetResidualBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_size, out_size, stride=1, pad_size=1, pad_type='constant,0', downsample=None):
        super(residualBottleneck, self).__init__(pad_size, pad_type)
        self.convs = nn.Sequential(
            CBR3D(in_size, out_size, 1, has_BN=True),
            CBR3D(out_size, out_size, 3, stride, pad_size, pad_type, has_BN=True),
            CBR3D(out_size, out_size*4, 1, has_BN=True))
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

class unetResidualBasic(nn.Module):
    expansion = 1
    def __init__(self, in_size, out_size, stride=1, pad_size=1, pad_type='constant,0', downsample=None):
        super(unetResidual, self).__init__()
        self.convs = nn.Sequential(
            CBR3D(in_size, out_size, 3, stride, pad_size, pad_type, has_BN=True, relu_slope=0),
            CBR3D(in_size, out_size, 3, 1,      pad_size, pad_type, has_BN=True))
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.convs(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class unetVgg(nn.Module):# default: basic convolution kernel
    def __init__(self, num_conv, kernel_size, in_size, out_size, stride_size=1, pad_size=0, pad_type='constant,0', has_BN=False, relu_slope=-1):
        super(unetVgg, self).__init__()
        layers= [CBR3D(in_size, out_size, kernel_size, stride_size, pad_size, pad_type, True, has_BN, relu_slope)]
        for i in range(num_conv-1):
            layers.append(CBR3D(out_size, out_size, kernel_size, stride_size, pad_size, pad_type, True, has_BN, relu_slope))
        self.convs = nn.Sequential(*layers)

    def forward(self, inputs):
        # print('vgg:',outputs.size())
        return self.convs(inputs)

# level-3: down-up module
class unetDown(nn.Module):
    def __init__(self, in_size, out_size, pad_size, pad_type, has_BN, relu_slope, pool_kernel, pool_stride):
        super(unetDown, self).__init__()
        self.conv = unetVgg(2, 3, in_size, out_size, 1, pad_size, pad_type, has_BN, relu_slope)
        self.down = nn.MaxPool3d(pool_kernel, pool_stride)

    def forward(self, x):
        outputs1 = self.conv(x)
        outputs2 = self.down(outputs1)
        return outputs1, outputs2

class unetUp(nn.Module):
    # in1: skip layer, in2: previous layer
    def __init__(self, in2_size, up_kernel, up_stride, upC_kernel, num_group, has_bias, 
                 out_size, p_upad_size, p_upad_type, 
                 in1_size, p_out_size, p_out_type, has_BN, relu_slope):
        super(unetUp, self).__init__()
        self.up = nn.ConvTranspose3d(in2_size, in2_size, up_kernel, up_stride, groups=num_group, bias=has_bias)
        self.up_conv = unetVgg(1, upC_kernel, in2_size, out_size, 1, p_upad_size, p_upad_type)
        self.conv = unetVgg(2, 3, out_size+in1_size, out_size, 1, p_out_size, p_out_type, has_BN, relu_slope)

    def forward(self, x1, x2):
        # inputs1 from left-side (bigger)
        outputs2 = self.up_conv(self.up(x2))
        offset = [(x1.size()[x+2]-outputs2.size()[x+2])/2 for x in range(3)]
        offset2 = [x1.size()[x+2]-outputs2.size()[x+2]-offset[x] for x in range(3)]
        # print(x1.size(),outputs2.size())
        # crop merge
        if max(offset2)==0:
            outputs1 = x1[:,:, offset[0]:, offset[1]:, offset[2]:]
        else:
            outputs1 = x1[:,:, offset[0]:-offset2[0], offset[1]:-offset2[1], offset[2]:-offset2[2]]
        # print('up:',outputs1.size(),outputs2.size())
        return self.conv(torch.cat([outputs2, outputs1], 1))

# level-1: one conv-bn-relu neuron
class unet3D(nn.Module):
    def __init__(self, in_size=1, out_size=3, filters=[24,72,216,648], has_bias=False, num_group=None, pad_vgg_size=0, pad_vgg_type='constant,0', pad_up_size=0, pad_up_type='constant,0',relu_slope=0.005, pool_kernel=(1,2,2), pool_stride=(1,2,2), upC_kernel=(1,1,1),has_BN=False):
        super(unet3D, self).__init__()
        if num_group is None:
            num_group = filters
        self.depth = len(filters)-1 
        filters_in = [in_size] + filters[:-1]
        self.down = nn.ModuleList([
                    unetDown(filters_in[x], filters_in[x+1], pad_vgg_size, pad_vgg_type, has_BN, relu_slope, pool_kernel, pool_stride) 
                    for x in range(len(filters)-1)]) 

        self.center = unetVgg(2, 3, filters[-2], filters[-1], 1, pad_vgg_size, pad_vgg_type, has_BN, relu_slope)
        
        self.up = nn.ModuleList([
                    unetUp(filters[x], pool_kernel, pool_stride, upC_kernel, num_group[x], has_bias,
                       filters[x-1], pad_up_size, pad_up_type, 
                       filters[x-1], pad_vgg_size, pad_vgg_type, has_BN, relu_slope)
                    for x in range(len(filters)-1,0,-1)])
        self.final = unetVgg(1, 1, filters[0], out_size)

    def forward(self, x):
        down_u = [None]*self.depth
        for i in range(self.depth):
            down_u[i], x = self.down[i](x)
        x = self.center(x)
        for i in range(self.depth):
             x = self.up[i](down_u[self.depth-1-i],x)
        return F.sigmoid(self.final(x))


# --------------------------------
# 2. utility function
# for L2 training: re-weight the error by label bias (far more 1 than 0)
def error_scale(data, clip_low=0.01, clip_high=0.99, thres=0.5):
    frac_pos = np.clip(data.mean(), clip_low, clip_high) #for binary labels
    # can't be all zero
    w_pos = 1.0/(2.0*frac_pos)
    w_neg = 1.0/(2.0*(1.0-frac_pos))
    scale = np.add((data >= thres) * w_pos, (data < thres) * w_neg)
    return scale

def decay_lr(optimizer, base_lr, iter, policy='inv',gamma=0.0001, power=0.75, step=100): 
    if policy=='fixed':
        return
    elif policy=='inv':
        new_lr = base_lr * ((1+gamma*iter)**(-power)) 
    elif policy=='exp':
        new_lr = base_lr * (gamma**iter)
    elif policy=='step':
        new_lr = base_lr * (gamma**(np.floor(iter/step)))
    for group in optimizer.param_groups:
        group['lr'] = new_lr 

def save_checkpoint(model, filename='checkpoint.pth', optimizer=None, epoch=1):
    if optimizer is None:
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }, filename)
    else:
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()
        }, filename)

def load_checkpoint(snapshot, num_gpu):
    # take care of multi-single gpu conversion
    cp = torch.load(snapshot)
    if num_gpu==1 and cp['state_dict'].keys()[0][:7]=='module.':
        # modify the saved model for single GPU
        for k,v in cp['state_dict'].items():
            cp['state_dict'][k[7:]] = cp['state_dict'].pop(k,None)
    elif num_gpu>1 and (len(cp['state_dict'].keys()[0])<7 or cp['state_dict'].keys()[0][:7]!='module.'):
        # modify the single gpu model for multi-GPU
        for k,v in cp['state_dict'].items():
            cp['state_dict']['module.'+k] = v
            cp['state_dict'].pop(k,None)
    return cp

def weight_filler(ksizes, opt_scale=2.0, opt_norm=2):
    kk=0
    if opt_norm==0:# n_in
        kk = np.prod(ksizes[1:])
    elif opt_norm==1:# n_out
        kk = ksizes[0]*np.prod(ksizes[2:])
    elif opt_norm==2:# (n_in+n_out)/2
        kk = np.mean(ksizes[:2])*np.prod(ksizes[2:])
    # opt_scale: 1.0=xavier, 2.0=kaiming, 3.0=caffe
    ww = np.sqrt(opt_scale/float(kk))
    return ww

def init_weights(model,opt_init=0):
    opt=[[2.0,2],[3.0,0]][opt_init]
    for i in range(3):
        for j in range(2):
            ksz = model.down[i].conv.convs[j].cbr._modules['0'].weight.size()
            model.down[i].conv.conv.convs[j].cbr._modules['0'].weight.data.copy_(torch.from_numpy(np.random.normal(0, weight_filler(ksz,opt[0],opt[1]), ksz)))
            model.down[i].conv.convs[j].cbr._modules['0'].bias.data.fill_(0.0)
    # center
    for j in range(2):
        ksz = model.center.convs[j].cbr._modules['0'].weight.size()
        model.center.convs[j].cbr._modules['0'].weight.data.copy_(torch.from_numpy(np.random.normal(0, weight_filler(ksz,opt[0],opt[1]), ksz)))
        model.center.convs[j].cbr._modules['0'].bias.data.fill_(0.0)
    # up
    for i in range(3):
        model.up[i].up.weight.data.fill_(1.0)
        for j in range(2):
        ksz = model.up[i].up_conv.convs[0].cbr._modules['0'].weight.size()
        model.up[i].up_conv.convs[0].cbr._modules['0'].weight.data.copy_(torch.from_numpy(np.random.normal(0, weight_filler(ksz,opt[0],opt[1]), ksz)))
        model.up[i].up_conv.convs[0].cbr._modules['0'].bias.data.fill_(0.0)
        for j in range(2):
            ksz = model.up[i].conv.convs[j].cbr._modules['0'].weight.size()
            model.up[i].conv.convs[j].cbr._modules['0'].weight.data.copy_(torch.from_numpy(np.random.normal(0, weight_filler(ksz,opt[0],opt[1]), ksz)))
            model.up[i].conv.convs[j].cbr._modules['0'].bias.data.fill_(0.0)
    ksz = model.final.conv.weight.size()
    model.final.convs[0].cbr.weight.data.copy_(torch.from_numpy(np.random.normal(0, weight_filler(ksz,opt[0],opt[1]), ksz)))
    model.final.convs[0].cbr.bias.data.fill_(0.0)

def load_weights_pkl(model, weights):
    # down
    for i in range(3):
        for j in range(2):
            ww = weights['Convolution'+str(2*i+j+1)]
            model.down[i].conv.convs[j].cbr._modules['0'].weight.data.copy_(torch.from_numpy(ww['w']))
            model.down[i].conv.convs[j].cbr._modules['0'].bias.data.copy_(torch.from_numpy(ww['b']))
    # center
    for j in range(2):
        ww = weights['Convolution'+str(7+j)]
        model.center.convs[j].cbr._modules['0'].weight.data.copy_(torch.from_numpy(ww['w']))
        model.center.convs[j].cbr._modules['0'].bias.data.copy_(torch.from_numpy(ww['b']))
    # up
    for i in range(3):
        ww = weights['Deconvolution'+str(i+1)]
        model.up[i].up.weight.data.copy_(torch.from_numpy(ww['w']))
        ww = weights['Convolution'+str(9+i*3)]
        model.up[i].up_conv.convs[0].cbr._modules['0'].weight.data.copy_(torch.from_numpy(ww['w']))
        model.up[i].up_conv.convs[0].cbr._modules['0'].bias.data.copy_(torch.from_numpy(ww['b']))
        for j in range(2):
            ww = weights['Convolution'+str(10+3*i+j)]
            model.up[i].conv.convs[j].cbr._modules['0'].weight.data.copy_(torch.from_numpy(ww['w']))
            model.up[i].conv.convs[j].cbr._modules['0'].bias.data.copy_(torch.from_numpy(ww['b']))
    ww = weights['Convolution18']
    model.final.convs[0].cbr._modules['0'].weight.data.copy_(torch.from_numpy(ww['w']))
    model.final.convs[0].cbr._modules['0'].bias.data.copy_(torch.from_numpy(ww['b']))

def pad5DReflect(data,pad_size,opt=0):
    if opt==0:
        # method 1: transfer to cpu
        # forward is correct, but backward is wrong
        out = np.lib.pad(data.data.cpu().numpy(),pad_size,'reflect')
        return torch.from_numpy(out)
    elif opt==1:
        # method 2: hack with 4D reflect
        pass

