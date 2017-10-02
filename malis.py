import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import malis_core

class MalisLoss(nn.Module):
    def __init__(self, conn_dims, opt_impl=0, opt_nb=1):
        super(MalisLoss, self).__init__()
        # pre-compute 
        if opt_nb==1:
            self.nhood_data = malis_core.mknhood3d(1).astype(np.int32).flatten()
        else:
            self.nhood_data = malis_core.mknhood3d(1,1.8).astype(np.uint64).flatten()
        self.nhood_dims = np.array((3,3),dtype=np.uint64)
        self.loss = 0
        self.num_vol = conn_dims[0]
        self.conn_dims = np.array(conn_dims[1:]).astype(np.uint64) # dim=4
        self.opt_impl = opt_impl
        if opt_impl==1:
            self.pre_ve, self.pre_prodDims, self.pre_nHood = malis_core.malis_init(self.conn_dims, self.nhood_data, self.nhood_dims)

    def mse_loss(self, src, dst):
        # already normalized in malis code
        return torch.sum(((src - dst)**2))
        #return torch.sum(((src - dst)**2)) / src.data.nelement()

    def forward(self,x, seg_cpu, aff_cpu):
        # pre-computed affinity
        x_cpu = x.data.cpu().numpy()
        # batch size 1 for now
        self.loss = 0
        for i in range(self.num_vol):
            if self.opt_impl == 0:
                tmp_w = malis_core.malis_loss_weights(seg_cpu[i].flatten(), self.conn_dims, self.nhood_data, self.nhood_dims, np.minimum(x_cpu[i], aff_cpu[i]).flatten(), 1)
            elif self.opt_impl == 1:
                tmp_w = malis_core.malis_loss_weights_pos(seg_cpu[i].flatten(), self.conn_dims, self.nhood_data, self.nhood_dims, self.pre_ve, self.pre_prodDims, self.pre_nHood, np.minimum(x_cpu[i], aff_cpu[i]).flatten(), 1)
            outP = Variable(torch.from_numpy(tmp_w.reshape(self.conn_dims)), requires_grad=False).cuda()
            self.loss += 0.5*torch.sum(((x[i]-1.0)**2)*outP)
            if np.count_nonzero(np.unique(seg_cpu[i])) > 1: # both pos and neg
                print np.unique(seg_cpu[i])
                if self.opt_impl == 0:
                    tmp_w = malis_core.malis_loss_weights(seg_cpu[i].flatten(), self.conn_dims, self.nhood_data, self.nhood_dims, np.maximum(x_cpu[i], aff_cpu[i]).flatten(), 0)
                elif self.opt_impl == 1:
                    tmp_w = malis_core.malis_loss_weights_pos(seg_cpu[i].flatten(), self.conn_dims, self.nhood_data, self.nhood_dims, self.pre_ve, self.pre_prodDims, self.pre_nHood, np.maximum(x_cpu[i], aff_cpu[i]).flatten(), 0)
                outN = Variable(torch.from_numpy(tmp_w.reshape(self.conn_dims)), requires_grad=False).cuda()
                self.loss += 0.5*torch.sum((x[i]**2)*outN)
        return self.loss/self.num_vol

    def backward(self):
        self.loss.backward()


