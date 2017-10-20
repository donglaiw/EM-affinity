import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import malis_core
def hook_a(m,go,gi):
    import pdb; pdb.set_trace()
    pass
#class MalisLoss(torch.autograd.Function):
class MalisLoss(nn.Module):
    def __init__(self, conn_dims, opt_weight=0, opt_impl=1, opt_nb=1):
        super(MalisLoss, self).__init__()
        # pre-compute 
        self.opt_weight=opt_weight
        #self.register_backward_hook(hook_a)
        if opt_nb==1:
            self.nhood_data = malis_core.mknhood3d(1).astype(np.int32).flatten()
        else:
            self.nhood_data = malis_core.mknhood3d(1,1.8).astype(np.uint64).flatten()
        self.nhood_dims = np.array((3,3),dtype=np.uint64)
        self.num_vol = conn_dims[0]
        self.conn_dims = np.array(conn_dims[1:]).astype(np.uint64) # dim=4
        self.opt_impl = opt_impl
        if opt_impl==1:
            self.pre_ve, self.pre_prodDims, self.pre_nHood = malis_core.malis_init(self.conn_dims, self.nhood_data, self.nhood_dims)

    def mse_loss(self, src, dst):
        # already normalized in malis code
        return torch.sum(((src - dst)**2))
        #return torch.sum(((src - dst)**2)) / src.data.nelement()

    #def backward(self):
    #    loss.backward()
    def backward(self, x):
        print "do back"
        import pdb; pdb.set_trace()
        return x

    def forward(self,x, seg_cpu, aff_cpu):
        print "do forward"
        # pre-computed affinity
        x_cpu = x.data.cpu().numpy()
        loss = 0
        #ll=range(self.num_vol)
        for i in range(self.num_vol):
            xMask = Variable(torch.from_numpy(aff_cpu[i]), requires_grad=False).cuda()
            if self.opt_impl == 0:
                tmp_w = malis_core.malis_loss_weights(seg_cpu[i].flatten(), self.conn_dims, self.nhood_data, self.nhood_dims, np.minimum(x_cpu[i], aff_cpu[i]).flatten(), 1)
            elif self.opt_impl == 1:
                tmp_w = malis_core.malis_loss_weights_pos(seg_cpu[i].flatten(), self.conn_dims, self.nhood_data, self.nhood_dims, self.pre_ve, self.pre_prodDims, self.pre_nHood, np.minimum(x_cpu[i], aff_cpu[i]).flatten(), 1)
            outP = Variable(torch.from_numpy(tmp_w.reshape(self.conn_dims)), requires_grad=False).cuda()
            numP = tmp_w.mean()
            numN = 0
            #ll[i] = 0.5*torch.sum(((torch.min(x[i],xMask)-1.0)**2)*outP).data[0]
            if np.count_nonzero(np.unique(seg_cpu[i])) > 1: # both pos and neg
                if self.opt_impl == 0:
                    tmp_w = malis_core.malis_loss_weights(seg_cpu[i].flatten(), self.conn_dims, self.nhood_data, self.nhood_dims, np.maximum(x_cpu[i], aff_cpu[i]).flatten(), 0)
                elif self.opt_impl == 1: # pre-compute dim
                    tmp_w = malis_core.malis_loss_weights_pos(seg_cpu[i].flatten(), self.conn_dims, self.nhood_data, self.nhood_dims, self.pre_ve, self.pre_prodDims, self.pre_nHood, np.maximum(x_cpu[i], aff_cpu[i]).flatten(), 0)
                outN = Variable(torch.from_numpy(tmp_w.reshape(self.conn_dims)), requires_grad=False).cuda()
                numN = tmp_w.mean()
                #ll[i] += 0.5*torch.sum(((torch.max(x[i],xMask))**2)*outN).data[0]
            if self.opt_weight==0: # equal weight
                loss += 0.5*torch.sum(((torch.min(x[i],xMask)-1.0)**2)*outP)
                if numN>0:
                    loss += 0.5*torch.sum((torch.max(x[i],xMask)**2)*outN)
            elif self.opt_weight<=1 and self.opt_weight>0: # constant weight
                loss += self.opt_weight*torch.sum(((torch.min(x[i],xMask)-1.0)**2)*outP)
                if numN>0:
                    loss += (1-self.opt_weight)*torch.sum((torch.max(x[i],xMask)**2)*outN)
            elif self.opt_weight==2: # adaptive weight
                loss += torch.sum(((torch.min(x[i],xMask)-1.0)**2)*outP)
                if numN>0:
                    loss += torch.sum((torch.max(x[i],xMask)**2)*outN)*numP/numN
        return loss/self.num_vol

    def forward_diff_eval(self,x_cpu, seg_cpu, aff_cpu):
        # no need for grad, for cpu version of x
        # [len(np.unique(seg_cpu[x])) for x in range(self.num_vol)]
        # [len(np.unique(aff_cpu[x])) for x in range(self.num_vol)]
        self.dloss = 0
        for i in range(self.num_vol):
            if self.opt_impl == 0:
                tmp_w = malis_core.malis_loss_weights(seg_cpu[i].flatten(), self.conn_dims, self.nhood_data, self.nhood_dims, np.minimum(x_cpu[i], aff_cpu[i]).flatten(), 1)
            elif self.opt_impl == 1:
                tmp_w = malis_core.malis_loss_weights_pos(seg_cpu[i].flatten(), self.conn_dims, self.nhood_data, self.nhood_dims, self.pre_ve, self.pre_prodDims, self.pre_nHood, np.minimum(x_cpu[i], aff_cpu[i]).flatten(), 1)
            outP = tmp_w.reshape(self.conn_dims).copy()
            numP = tmp_w.mean()
            numN = 0
            if np.count_nonzero(np.unique(seg_cpu[i])) > 1: # both pos and neg
                if self.opt_impl == 0:
                    tmp_w = malis_core.malis_loss_weights(seg_cpu[i].flatten(), self.conn_dims, self.nhood_data, self.nhood_dims, np.maximum(x_cpu[i], aff_cpu[i]).flatten(), 0)
                elif self.opt_impl == 1:
                    tmp_w = malis_core.malis_loss_weights_pos(seg_cpu[i].flatten(), self.conn_dims, self.nhood_data, self.nhood_dims, self.pre_ve, self.pre_prodDims, self.pre_nHood, np.maximum(x_cpu[i], aff_cpu[i]).flatten(), 0)
                outN = tmp_w.reshape(self.conn_dims).copy()
                numN = tmp_w.mean()
            self.dloss += ((np.minimum(x_cpu[i],aff_cpu[i])-1.0)*outP)
            if numN>0:
                self.dloss += ((np.maximum(x_cpu[i],aff_cpu[i]))*outN)
        return self.dloss/self.num_vol
   
    def forward_eval(self,x_cpu, seg_cpu, aff_cpu):
        # no need for grad, for cpu version of x
        # [len(np.unique(seg_cpu[x])) for x in range(self.num_vol)]
        # [len(np.unique(aff_cpu[x])) for x in range(self.num_vol)]
        loss = 0
        for i in range(self.num_vol):
            if self.opt_impl == 0:
                tmp_w = malis_core.malis_loss_weights(seg_cpu[i].flatten(), self.conn_dims, self.nhood_data, self.nhood_dims, np.minimum(x_cpu[i], aff_cpu[i]).flatten(), 1)
            elif self.opt_impl == 1:
                tmp_w = malis_core.malis_loss_weights_pos(seg_cpu[i].flatten(), self.conn_dims, self.nhood_data, self.nhood_dims, self.pre_ve, self.pre_prodDims, self.pre_nHood, np.minimum(x_cpu[i], aff_cpu[i]).flatten(), 1)
            outP = tmp_w.reshape(self.conn_dims).copy()
            numP = tmp_w.mean()
            numN = 0
            if np.count_nonzero(np.unique(seg_cpu[i])) > 1: # both pos and neg
                if self.opt_impl == 0:
                    tmp_w = malis_core.malis_loss_weights(seg_cpu[i].flatten(), self.conn_dims, self.nhood_data, self.nhood_dims, np.maximum(x_cpu[i], aff_cpu[i]).flatten(), 0)
                elif self.opt_impl == 1:
                    tmp_w = malis_core.malis_loss_weights_pos(seg_cpu[i].flatten(), self.conn_dims, self.nhood_data, self.nhood_dims, self.pre_ve, self.pre_prodDims, self.pre_nHood, np.maximum(x_cpu[i], aff_cpu[i]).flatten(), 0)
                outN = tmp_w.reshape(self.conn_dims).copy()
                numN = tmp_w.mean()
            if self.opt_weight==0: # equal weight
                loss += 0.5*np.sum(((np.minimum(x_cpu[i],aff_cpu[i])-1.0)**2)*outP)
                if numN>0:
                    loss += 0.5*np.sum((np.maximum(x_cpu[i],aff_cpu[i])**2)*outN)
            elif self.opt_weight<=1 and self.opt_weight>0: # constant weight
                loss += self.opt_weight*np.sum(((np.minimum(x[i],aff_cpu[i])-1.0)**2)*outP)
                if numN>0:
                    loss += (1-self.opt_weight)*np.sum((np.maximum(x_cpu[i],aff_cpu[i])**2)*outN)
            elif self.opt_weight==2: # adaptive weight
                loss += np.sum(((np.minimum(x[i],aff_cpu[i])-1.0)**2)*outP)
                if numN>0:
                    loss += np.sum((np.maximum(x[i],aff_cpu[i])**2)*outN)*numP/numN
        return loss/self.num_vol


