import numpy as np
import pickle, h5py, time, argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os, sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from em.model.io import load_weights_pkl, load_weights_pkl_m1
from em.model.unet import unet3D_m1, unet3D

def get_args():
    parser = argparse.ArgumentParser(description='Training Model')
    parser.add_argument('-m','--model',  type=int, default=0,
                        help='model id')
    parser.add_argument('-t','--test-data',  default='/n/coxfs01/donglai/malis_trans/malis_test/test_malis_ecs_pair.pkl',
                        help='test data')
    parser.add_argument('-tn','--test-name',  default='main',
                        help='test data name')
    parser.add_argument('-w','--weight',  default='/n/coxfs01/donglai/micron100_1217/model/Toufiq/net_iter_100000.pkl',
                        help='model weight')
    parser.add_argument('-o','--output',  default='',
                        help='output name')
    parser.add_argument('-n','--num',  type=int, default=0,
                        help='number of iteration for benchmark')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # config 
    args = get_args()

    # load data
    if args.test_data[-3:]=='pkl':
        data = pickle.load(open(args.test_data,'rb'))
    elif args.test_data[-3:]=='h5':
        data = np.array(h5py.File(args.test_data,args)[test_name])
    x = Variable(torch.from_numpy(data.astype(np.float32)).cuda())

    # load model
    ww=pickle.load(open(args.weight,'rb'))
    if args.model==0:
        net = unet3D(filters=[24,72,216,648])
        load_weights_pkl(net, ww) 
    elif args.model==1:
        net = unet3D_m1(filters=[24,72,216,648])
        load_weights_pkl_m1(net, ww) 
    net.cuda()
    net.eval()
    y = net(x)
    if args.output != '': 
        pickle.dump(y.data.cpu().numpy(),open(args.output,'wb'))
    # benchmark speed
    tt = np.zeros((args.num))
    for i in range(args.num):
        st = time.time()
        y = net(x)
        tt[i] = time.time()-st
    print tt.mean()*1000,tt.std()*1000
