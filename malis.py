import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MalisLoss(nn.Module):
    def __init__(self, opt_nb=1):
        super(MalisLoss, self).__init__()
        if opt_nb==1:
            self.nhood=malis.mknhood3d(1)
        else:
            self.nhood=malis.mknhood3d(1,1.8)
        self.loss_pos = nn.MSELoss()
        self.loss_neg = nn.MSELoss()
        self.loss = None

    def forward(self,x, seg_cpu, aff_cpu, aff_gpu):
        # pre-computed affinity
        conn_shape = x.size()
        nhood_shape = self.nhood.shape
        if conn_shape[0] == 1:
            outP = Variable(torch.from_numpy(malis.malis_loss_weights(seg_cpu[0], conn_shape, self.nhood, nhood_shape, np.minimum(x_cpu[0], aff_cpu[0]),1)), requires_grad=False).cuda()
            outN = Variable(torch.from_numpy(malis.malis_loss_weights(seg_cpu[0], conn_shape, self.nhood, nhood_shape, np.maximum(x_cpu[0], aff_cpu[0]),0)), requires_grad=False).cuda()
            self.loss = 0.5*self.loss_pos(outP,x*aff_gpu*outP) + 0.5*self.loss_neg(0,x*(1.-aff_gpu)*outN)
        else:
            outP = Variable(torch.cuda.FloatTensor(conn_shape), requires_grad=False)
            outN = Variable(torch.cuda.FloatTensor(conn_shape), requires_grad=False)
            tmp_c = torch.cuda.FloatTensor(1,conn_shape[1],conn_shape[2],conn_shape[3],conn_shape[4])
            x_cpu = x.data.cpu().numpy()
            for i in range(conn_shape[0]):
                tmp_c.data.copy_()
                outP[i] = torch.from_numpy(malis.malis_loss_weights(seg_cpu[i], conn_shape, self.nhood, nhood_shape, np.minimum(x_cpu[i], aff_cpu[i]),1)).cuda()
                outN[i] = torch.from_numpy(malis.malis_loss_weights(seg_cpu[i], conn_shape, self.nhood, nhood_shape, np.maximum(x_cpu[i], aff_cpu[i]),0)).cuda()
            self.loss = 0.5*self.loss_pos(outP,x*aff_gpu*outP) + 0.5*self.loss_neg(0,x*(1.-aff_gpu)*outN)
        return self.loss

    def forward_bk(self,x, gt_seg, gt_aff):
        # pre-computed affinity
        conn_shape = x.size()
        nhood_shape = self.nhood.shape
        outP = np.zeros(conn_shape).astype(np.uint64);
        outN = np.zeros(conn_shape).astype(np.uint64);
        #gt_aff = np.zeros(conn_shape).astype(np.uint64);
        # with np, no gradient is calculated
        for i in range(conn_shape[0]):
            # gt_aff[i] = malis.seg_to_affgraph(y[i], self.nhood)
            outP[i] = malis.malis_loss_weights(gt_seg[i].data.cpu().numpy(), conn_shape, self.nhood, nhood_shape, np.minimum(x[i], gt_aff[i]), 1)
            outN[i] = malis.malis_loss_weights(gt_seg[i], conn_shape, self.nhood, nhood_shape, np.maximum(x[i],gt_aff[i]), 0)
        # compute l2-loss
        # gt_aff are binary mask 
        self.loss = 0.5*self.loss_pos(outP,x*gt_aff*outP) + 0.5*self.loss_neg(0,x*(1.-gt_aff)*outN)
        return self.loss

    def backward(self):
        self.loss.backward()


