import numpy as np
import pickle
import os,sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import malis
from T_model import unet3D, load_weights


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
    # benchmark speed
    st=time.time()
    for i in range(1000):
        l = criterion(y,seg_cpu, aff_cpu)
    et=time.time()
    print et-st
    #print l.data[0] #malis:0.00302
