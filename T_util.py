import sys
import numpy as np
import malis_core
import torch
import h5py

# ---------------------
# 1. utility layers
class malisWeight():
    def __init__(self, conn_dims, opt_weight=0.5, opt_nb=1):
        # pre-compute 
        self.opt_weight=opt_weight
        if opt_nb==1:
            self.nhood_data = malis_core.mknhood3d(1).astype(np.int32).flatten()
        else:
            self.nhood_data = malis_core.mknhood3d(1,1.8).astype(np.uint64).flatten()
        self.nhood_dims = np.array((3,3),dtype=np.uint64)
        self.conn_dims = np.array(conn_dims[1:]).astype(np.uint64) # dim=4
        self.pre_ve, self.pre_prodDims, self.pre_nHood = malis_core.malis_init(self.conn_dims, self.nhood_data, self.nhood_dims)
        self.weight = np.zeros(conn_dims,dtype=np.float32)#pre-allocate

    def getWeight(self, x_cpu, aff_cpu, seg_cpu):
        for i in range(x_cpu.shape[0]):
            self.weight[i] = malis_core.malis_loss_weights_both(seg_cpu[i].flatten(), self.conn_dims, self.nhood_data, self.nhood_dims, self.pre_ve, self.pre_prodDims, self.pre_nHood, x_cpu[i].flatten(), aff_cpu[i].flatten(), self.opt_weight).reshape(self.conn_dims)
        return self.weight


# for L2 training: re-weight the error by label bias (far more 1 than 0)
class labelWeight():
    def __init__(self, conn_dims, opt_weight=2, clip_low=0.01, clip_high=0.99, thres=0.5):
        self.opt_weight=opt_weight
        self.clip_low=clip_low
        self.clip_high=clip_high
        self.thres = thres
        self.num_elem = np.prod(conn_dims[1:]).astype(float)
        self.weight = np.zeros(conn_dims,dtype=np.float32) #pre-allocate

    def getWeight(self, data):
        w_pos = self.opt_weight
        w_neg = 1.0-self.opt_weight
        for i in range(data.shape[0]):
            if self.opt_weight==2:
                frac_pos = np.clip(data[i].mean(), self.clip_low, self.clip_high) #for binary labels
                # can't be all zero
                w_pos = 1.0/(2.0*frac_pos)
                w_neg = 1.0/(2.0*(1.0-frac_pos))
            self.weight[i] = np.add((data[i] >= self.thres) * w_pos, (data[i] < self.thres) * w_neg)
        return self.weight/self.num_elem

def weightedMSE_np(input, target, weight=None, normalize_weight=False):
    # normalize by batchsize
    if weight is None:
        return np.sum((input - target) ** 2)/input.shape[0]
    else:
        if not normalize_weight: # malis loss: weight already normalized
            return np.sum(weight * (input - target) ** 2)/input.shape[0]
        else: # standard avg error
            return np.mean(weight * (input - target) ** 2)

def weightedMSE(input, target, weight=None, normalize_weight=False):
    # normalize by batchsize
    if weight is None:
        return torch.sum((input - target) ** 2)/input.size(0)
    else:
        if not normalize_weight:
            return torch.sum(weight * (input - target) ** 2)/input.size(0)
        else:
            return torch.mean(weight * (input - target) ** 2)

# ---------------------
# 2. learning/model utility
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
        if group['lr'] != 0.0: # not frozen
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

def writeh5(filename, datasetname, dtarray):
    fid=h5py.File(filename,'w')
    fid.create_dataset(datasetname,data=dtarray)
    fid.close()

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
    # todo: rewrite for more flexibility
    # todo: add BN param: running mean
    # down
    for i in range(3):
        for j in range(2):
            ww = weights['Convolution'+str(2*i+j+1)]
            model.down[i].conv.convs[j].cbrd._modules['0'].weight.data.copy_(torch.from_numpy(ww['w']))
            model.down[i].conv.convs[j].cbrd._modules['0'].bias.data.copy_(torch.from_numpy(ww['b']))
    # center
    for j in range(2):
        ww = weights['Convolution'+str(7+j)]
        model.center.conv.convs[j].cbrd._modules['0'].weight.data.copy_(torch.from_numpy(ww['w']))
        model.center.conv.convs[j].cbrd._modules['0'].bias.data.copy_(torch.from_numpy(ww['b']))
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
            model.up[i].conv.convs[j].cbrd._modules['0'].weight.data.copy_(torch.from_numpy(ww['w']))
            model.up[i].conv.convs[j].cbrd._modules['0'].bias.data.copy_(torch.from_numpy(ww['b']))
    ww = weights['Convolution18']
    model.final.conv.cbrd._modules['0'].weight.data.copy_(torch.from_numpy(ww['w']))
    model.final.conv.cbrd._modules['0'].bias.data.copy_(torch.from_numpy(ww['b']))

# ---------------------
# 3. for visualization
def getLoss(fn):
    a=open(fn,'r')
    line = a.readline()
    loss = np.zeros((300000))
    lid=0
    while line:
        if 'loss=' in line:
            loss[lid] = float(line[line.find('=')+1:line.find('=')+5])
            lid+=1
        line = a.readline()
    a.close()
    return loss[:lid]

def getVI(fn):
    return [float(x) for x in open(fn,'r').readlines()[5].split(',')]

def getAvg(rr, avg_step):
    num_avg = int(np.floor(len(rr)/float(avg_step)))
    rr_avg = rr[:num_avg*avg_step].reshape((num_avg,avg_step)).mean(axis=1)
    return rr_avg
