import sys
import numpy as np
import h5py

# 1. model i/o
def save_checkpoint(model, filename='checkpoint.pth', optimizer=None, epoch=1):
    # model or model_state
    out = model if type(model) is dict else model.state_dict()
         
    import torch
    if optimizer is None:
        torch.save({
            'epoch': epoch,
            'state_dict': out,
        }, filename)
    else:
        torch.save({
            'epoch': epoch,
            'state_dict': out,
            'optimizer' : optimizer.state_dict()
        }, filename)

def convert_state_dict(state_dict, num_gpu):
    # multi-single gpu conversion
    if num_gpu==1 and state_dict.keys()[0][:7]=='module.':
        # modify the saved model for single GPU
        for k,v in state_dict.items():
            state_dict[k[7:]] = state_dict.pop(k,None)
    elif num_gpu>1 and (len(state_dict.keys()[0])<7 or state_dict.keys()[0][:7]!='module.'):
        # modify the single gpu model for multi-GPU
        for k,v in state_dict.items():
            state_dict['module.'+k] = v
            state_dict.pop(k,None)

def load_checkpoint(snapshot, num_gpu):
    import torch
    if isinstance(snapshot, basestring):
        cp = torch.load(snapshot)
        if type(cp) is not dict:
            # model -> state_dict
            cp={'epoch':0, 'state_dict': cp.state_dict()}
    else:
        cp={'epoch':0, 'state_dict': snapshot}
    convert_state_dict(cp['state_dict'], num_gpu)
    return cp

# 2. model conversion
def keras2pkl(mn,outN=None):
    pass
def caffe2pkl(mn,wn,outN=None):
    import caffe
    net0 = caffe.Net(mn,caffe.TEST,weights=wn)
    net={}
    for k,v in net0.params.items():
        layer={}
        layer['w']=v[0].data
        if len(v)>1: # deconv layer: no bias term
            layer['b']=v[1].data
        net[k]=layer
    if outN is None:
        return net
    else:
        import pickle
        pickle.dump(net,open(outN,'wb'))

def pth2issac(net):
    import isaac.pytorch
    # now only supports: 
    result = isaac.pytorch.UNet(out_num=net.io_num[1], 
                filters=[net.io_num[0]]+net.filters,
                relu_slope=net.relu_slope).cuda()

    # Reorder indices because new state dict has upsample-upconv interleaved
    depth = net.depth
    ndown = 4*(depth + 1)
    reorder = list(range(ndown))
    for i in range(depth):
        upsamples = list(range(ndown + i*3, ndown + i*3 + 3))
        upconvs = list(range(ndown + depth*3 + i*4, ndown + depth*3 + i*4 + 4))
        reorder +=  upsamples + upconvs
    reorder += [ndown + 7*depth, ndown + 7*depth + 1]

    # Copy in proper order
    net_keys = list(net.state_dict().keys())
    result_keys = list(result.state_dict().keys())
    net_dict = net.state_dict()
    result_dict = result.state_dict()
    for i, j in enumerate(reorder):
        result_dict[result_keys[i]] = net_dict[net_keys[j]].clone()
    result.load_state_dict(result_dict)

    return result


def pth2pkl(mn,outN=None):
    import torch
    net0 = torch.load(mn)
    net={}
    cc=1
    bn=1
    ks= net0['state_dict'].keys()
    for k in ks:
        if 'cbrd.0.weight' in k and 'up.0.weight' not in k: #no deconv
            layer={}
            print cc,k
            layer['w']=net0['state_dict'][k].cpu().numpy()
            kk=k.replace('weight','bias')
            if kk in ks:
                layer['b']=net0['state_dict'][kk].cpu().numpy()
            net['Convolution'+str(cc)]=layer
            cc+=1
        elif 'cbrd.1.running_mean' in k: #batchnorm
            layer={}
            print bn,k
            layer['w0']=net0['state_dict'][k.replace('running_mean','weight')].cpu().numpy()
            layer['w1']=net0['state_dict'][k.replace('running_mean','bias')].cpu().numpy()
            layer['w2']=net0['state_dict'][k].cpu().numpy()
            layer['w3']=net0['state_dict'][k.replace('running_mean','running_var')].cpu().numpy()
            net['BatchNorm'+str(bn)]=layer
            bn+=1
    if outN is None:
        return net
    else:
        import pickle
        pickle.dump(net,open(outN,'wb'))

# 3. initialization
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
    import torch
    opt=[[2.0,2],[3.0,0]][opt_init]
    for i in range(3):
        for j in range(2):
            ksz = model.down[i].conv.convs[j].cbrd._modules['0'].weight.size()
            model.down[i].conv.conv.convs[j].cbrd._modules['0'].weight.data.copy_(torch.from_numpy(np.random.normal(0, weight_filler(ksz,opt[0],opt[1]), ksz)))
            model.down[i].conv.convs[j].cbrd._modules['0'].bias.data.fill_(0.0)
    # center
    for j in range(2):
        ksz = model.center.convs[j].cbrd._modules['0'].weight.size()
        model.center.convs[j].cbrd._modules['0'].weight.data.copy_(torch.from_numpy(np.random.normal(0, weight_filler(ksz,opt[0],opt[1]), ksz)))
        model.center.convs[j].cbrd._modules['0'].bias.data.fill_(0.0)
    # up
    for i in range(3):
        model.up[i].up.weight.data.fill_(1.0)
        ksz = model.up[i].up_conv.convs[0].cbrd._modules['0'].weight.size()
        model.up[i].up_conv.convs[0].cbrd._modules['0'].weight.data.copy_(torch.from_numpy(np.random.normal(0, weight_filler(ksz,opt[0],opt[1]), ksz)))
        model.up[i].up_conv.convs[0].cbrd._modules['0'].bias.data.fill_(0.0)
        for j in range(2):
            ksz = model.up[i].conv.convs[j].cbrd._modules['0'].weight.size()
            model.up[i].conv.convs[j].cbrd._modules['0'].weight.data.copy_(torch.from_numpy(np.random.normal(0, weight_filler(ksz,opt[0],opt[1]), ksz)))
            model.up[i].conv.convs[j].cbrd._modules['0'].bias.data.fill_(0.0)
    ksz = model.final.conv.weight.size()
    model.final.convs[0].cbrd.weight.data.copy_(torch.from_numpy(np.random.normal(0, weight_filler(ksz,opt[0],opt[1]), ksz)))
    model.final.convs[0].cbrd.bias.data.fill_(0.0)

def load_weights_pkl(model, weights):
    import torch
    # todo: rewrite for more flexibility
    # todo: add BN param: running mean
    # down
    for i in range(3):
        for j in range(2):
            ww = weights['Convolution'+str(2*i+j+1)]
            model.down[i].conv[j].cbrd._modules['0'].weight.data.copy_(torch.from_numpy(ww['w']))
            model.down[i].conv[j].cbrd._modules['0'].bias.data.copy_(torch.from_numpy(ww['b']))
    # center
    for j in range(2):
        ww = weights['Convolution'+str(7+j)]
        model.center.conv[j].cbrd._modules['0'].weight.data.copy_(torch.from_numpy(ww['w']))
        model.center.conv[j].cbrd._modules['0'].bias.data.copy_(torch.from_numpy(ww['b']))
    # up
    for i in range(3):
        ww = weights['Deconvolution'+str(i+1)]
        if model.up[i].opt[0]==0:# upsample+conv
            model.up[i].up._modules['0'].weight.data.copy_(torch.from_numpy(ww['w']))
            ww = weights['Convolution'+str(9+i*3)]
            model.up[i].up._modules['1'].cbrd._modules['0'].weight.data.copy_(torch.from_numpy(ww['w']))
            model.up[i].up._modules['1'].cbrd._modules['0'].bias.data.copy_(torch.from_numpy(ww['b']))
        elif model.up[i].opt[0]==1:# deconv
            model.up[i].up.weight.data.copy_(torch.from_numpy(ww['w']))
        for j in range(2):
            ww = weights['Convolution'+str(10+3*i+j)]
            model.up[i].conv[j].cbrd._modules['0'].weight.data.copy_(torch.from_numpy(ww['w']))
            model.up[i].conv[j].cbrd._modules['0'].bias.data.copy_(torch.from_numpy(ww['b']))
    ww = weights['Convolution18']
    model.final.conv.cbrd._modules['0'].weight.data.copy_(torch.from_numpy(ww['w']))
    model.final.conv.cbrd._modules['0'].bias.data.copy_(torch.from_numpy(ww['b']))

def load_weights_pkl_m1(model, weights):
    import torch
    # down
    for i in range(3):
        for j in range(2):
            ww = weights['Convolution'+str(2*i+j+1)]
            model.downC[i]._modules[str(j*2)].weight.data.copy_(torch.from_numpy(ww['w']))
            model.downC[i]._modules[str(j*2)].bias.data.copy_(torch.from_numpy(ww['b']))
    # center
    for j in range(2):
        ww = weights['Convolution'+str(7+j)]
        model.center._modules[str(j*2)].weight.data.copy_(torch.from_numpy(ww['w']))
        model.center._modules[str(j*2)].bias.data.copy_(torch.from_numpy(ww['b']))
    # up
    for i in range(3):
        ww = weights['Deconvolution'+str(i+1)]
        model.upS[i]._modules['0'].weight.data.copy_(torch.from_numpy(ww['w']))
        ww = weights['Convolution'+str(9+i*3)]
        model.upS[i]._modules['1'].weight.data.copy_(torch.from_numpy(ww['w']))
        model.upS[i]._modules['1'].bias.data.copy_(torch.from_numpy(ww['b']))
        for j in range(2):
            ww = weights['Convolution'+str(10+3*i+j)]
            model.upC[i]._modules[str(2*j)].weight.data.copy_(torch.from_numpy(ww['w']))
            model.upC[i]._modules[str(2*j)].bias.data.copy_(torch.from_numpy(ww['b']))
    ww = weights['Convolution18']
    model.final._modules['0'].weight.data.copy_(torch.from_numpy(ww['w']))
    model.final._modules['0'].bias.data.copy_(torch.from_numpy(ww['b']))

