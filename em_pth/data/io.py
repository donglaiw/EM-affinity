# utility function to load data
# train-L2: img+label
# train-malis: img+label+seg
# test-pred: img
# test-err-L2: img+label
# test-err-malis: img+label+seg
import numpy as np
import h5py
import os
from ..util.misc import writeh5, segToAffinity

def setPred(pred, y_pred, pred_sz, pos, lt=None, st=0, pred_ww=None, ww=None):
    if lt is None:
        lt = pos.shape[0]
    if pred_ww is None: # simply assign
        for j in range(st,lt):
            pp = pos[j] 
            pred[:,pp[1]:pp[1]+pred_sz[0],
                  pp[2]:pp[2]+pred_sz[1],
                  pp[3]:pp[3]+pred_sz[2]] = y_pred[j].copy()
    else:
        for j in range(st,lt):
            pp = pos[j] 
            pred[:,pp[1]:pp[1]+pred_sz[0],
                  pp[2]:pp[2]+pred_sz[1],
                  pp[3]:pp[3]+pred_sz[2]] += y_pred[j]*ww
            pred_ww[:,pp[1]:pp[1]+pred_sz[0],
                  pp[2]:pp[2]+pred_sz[1],
                  pp[3]:pp[3]+pred_sz[2]] += ww


def getVar(batch_size, model_io_size, do_input=[True, False, False]):
    import torch
    from torch.autograd import Variable
    input_vars = [None] *3  
    # img
    input_vars[0] = Variable(torch.zeros(batch_size, 1, model_io_size[0][0], model_io_size[0][1], model_io_size[0][2]).cuda(), requires_grad=False)
    # label
    if do_input[1]:
        input_vars[1] = Variable(torch.zeros(batch_size, 3, model_io_size[1][0], model_io_size[1][1], model_io_size[1][2]).cuda(), requires_grad=False)
    if do_input[2]:
        input_vars[2] = Variable(torch.zeros(batch_size, 3, model_io_size[1][0], model_io_size[1][1], model_io_size[1][2]).cuda(), requires_grad=False)
    return input_vars
