import numpy as np
import pickle
import time
import os,sys
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data

import malis_core
from T_model import unet3D,error_scale,save_checkpoint,init_weights 
from T_data import VolumeDatasetTrain, np_collate
import argparse

# dc=0;dr=0;CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python E_train.py -m 1 -dc ${dc} -dr ${dr} -l 1 -b 16 --iter-total 2000 --iter-save 1000 -lr 0.001 -g 6 -c 16 -o result/16_8_1e-3_bn_dc${dc}_dr${dr}/

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python E_train.py -m 1 -d 2 -l 1 -b 16 --iter-total 625 --iter-save 625 -lr 0.001 -g 8 -c 16 -o result/16_8_1e-3_bn_d2/ 

# CUDA_VISIBLE_DEVICES=2,5,6,7,8,9 python E_train.py -m 1 -d 2 -l 1 -b 16 --iter-total 625 --iter-save 625 -lr 0.001 -g 6 -c 16 -o result/16_8_1e-3_bn_dc2_dr1/ -s result/16_8_1e-3_bn_dc2_dr1/iter_16_625_0.001.pth 

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,6 python E_train.py -m 1 -dc 2 -dr 1 -l 0 -b 16 --iter-total 1250 --iter-save 625 -lr 0.001 -g 6 -c 16 -o result/16_8_1e-3_bn_dc2_dr1_nw/ 

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python E_train.py -it 0 -m 1 -dc 0 -dr 0 -l 1 -b 14 --iter-total 2500 --iter-save 625 -lr 0.001 -g 7 -c 16 -o result/14_8_1e-3_bn_dc0_dr0_init0/ 

def get_args():
    parser = argparse.ArgumentParser(description='Training Model')
    # I/O
    parser.add_argument('-i','--input',  default='/n/coxfs01/donglai/malis_trans/data/ecs-3d/ecs-gt-3x6x6/',
                        help='input folder')
    parser.add_argument('-dn','--data-name',  default='im_uint8.h5',
                        help='image data')
    parser.add_argument('-ln','--label-name',  default='seg-groundtruth2-malis.h5',
                        help='segmentation label')
    parser.add_argument('-o','--output', default='result/train/',
                        help='output path')
    parser.add_argument('-s','--snapshot',  default='',
                        help='pre-train snapshot path')
    # training option
    parser.add_argument('-it','--init', type=int,  default=0,
                        help='model initialization type')
    parser.add_argument('-m','--model-opt', type=int,  default=0,
                        help='model type')
    parser.add_argument('-l','--loss-opt', type=int, default=2,
                        help='loss type')
    parser.add_argument('-dc','--data-color-opt', type=int,  default=2,
                        help='data color aug type')
    parser.add_argument('-dr','--data-rotation-opt', type=int,  default=0,
                        help='data rotation aug type')
    parser.add_argument('-lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('-betas', default='0.9,0.99',
                        help='beta for adam')
    parser.add_argument('-wd', type=float, default=5e-6,
                        help='weight decay')
    parser.add_argument('-b','--batch-size', type=int,  default=16,
                        help='batch size')
    parser.add_argument('--iter-total', type=int, default=1000,
                        help='total number of iteration')
    parser.add_argument('--iter-save', type=int, default=100,
                        help='number of iteration to save')
    parser.add_argument('-g','--num-gpu', type=int,  default=1,
                        help='number of gpu')
    parser.add_argument('-c','--num-cpu', type=int,  default=1,
                        help='number of cpu')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    sn = args.output+'/'
    if not os.path.isdir(sn):
        os.makedirs(sn)

    print '1. setup data'
    # data/seg: initially same size
    train_size = np.array(((31,204,204), (3,116,116)));
    train_offset = (train_size[0]-train_size[1])/2
    if args.data_name[-3:] == '.h5':
        train_data =  np.array(h5py.File(args.input+args.data_name,'r')['main'],dtype=np.float32)[None,:]/(2.**8)
    elif args.data_name[-4:] == '.pkl':
        train_data =  np.array(pickle.load(args.input+args.data_name,'rb'),dtype=np.float32)[None,:]/(2.**8)

    train_nhood = malis_core.mknhood3d()
    if os.path.exists(args.input+args.label_name+'_aff.h5'):
        train_label = np.array(h5py.File(args.input+args.label_name[:-3]+'_aff.h5','r')['main'])
    else: # pre-compute for faster i/o
        train_seg = np.array(h5py.File(args.input+args.label_name,'r')['main'])[train_offset[0]-2:-train_offset[0]+2,train_offset[1]-2:-train_offset[1]+2,train_offset[2]-2:-train_offset[2]+2]
        train_label = malis_core.seg_to_affgraph(train_seg,train_nhood)[:,1:-1,1:-1,1:-1]
        from T_util import writeh5
        writeh5(args.input+args.label_name[:-3]+'_aff.h5', 'main', train_label)
    # add sampler
    nhood = None if args.loss_opt in [0,1] else train_nhood

    color_scale = [(0.8,1.2), (0.9,1.1), None][args.data_color_opt]
    color_shift = [(0.2,0.2), (-0.1,0.1), None][args.data_color_opt]
    color_clip = [(0.05,0.95), (0.05,0.95), None][args.data_color_opt]
    rot = [[(1,1,1),True],[(0,0,0),False]][args.data_rotation_opt]
    train_dataset = VolumeDatasetTrain(train_data, train_label, nhood, data_size=train_data.shape[1:], 
                                       reflect=rot[0], swapxy=rot[1],
                                       color_scale=color_scale,color_shift=color_shift,clip=color_clip,
                                       out_data_size=train_size[0],out_label_size=train_size[1])

    train_loader =  torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn = np_collate,
            num_workers=args.num_cpu, pin_memory=True)

    print '2. setup model'
    if args.model_opt == 0: #regular
        model = unet3D()
    elif args.model_opt == 1: #batchnorm
        model = unet3D(is_batchnorm=True)
    # initialize model
    if args.init>=0:
        init_weights(model,args.init)

    if args.num_gpu>1: model = nn.DataParallel(model, range(args.num_gpu)) 
    model.cuda()
    if args.lr==0: #for eval
        model.eval()
    else:
        model.train()
    betas = [float(x) for x in args.betas.split(',')]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=betas, weight_decay=args.wd)
    pre_epoch = 0
    
    if args.loss_opt in [0,1]: # weighted L2 training
        loss_fn = torch.nn.MSELoss()
        loss_suf = '_L2_'+str(args.lr)
    elif args.loss_opt == 2: # malis training
        import malisLoss
        loss_fn = malisLoss.MalisLoss([args.batch_size,3]+list(train_size[1]),1).cuda()
        loss_suf = '_malis_'+str(args.lr)
    # load previous model
    if len(args.snapshot)>0:
        loss_suf += '_'+args.snapshot[:-4] if '/' not in args.snapshot else args.snapshot[args.snapshot.rfind('/')+1:-4]
        cp = torch.load(args.snapshot)
        model.load_state_dict(cp['state_dict'])
        pre_epoch = cp['epoch']

    print '3. start training'
    log = open(args.output+'log'+loss_suf+'.txt','w',0) # unbuffered, write instantly
    # pre-allocate torch cuda tensor
    x = Variable(torch.zeros(args.batch_size, 1, 31, 204, 204).cuda(), requires_grad=False)
    if args.loss_opt in [0,1]:
        y = Variable(torch.randn(args.batch_size, 3, 3, 116, 116).cuda(), requires_grad=False)
        if args.loss_opt in [1]:
            ww = Variable(torch.zeros(args.batch_size, 3, 3, 116, 116).cuda(), requires_grad=False)
    for iter_id, data in enumerate(train_loader):
        optimizer.zero_grad()
        # copy data
        t1 = time.time()
        x.data.copy_(torch.from_numpy(data[0]))

        # forward-backward
        t2 = time.time()
        y_pred = model(x)
        if args.loss_opt in [0,1]: # L2
            y.data.copy_(torch.from_numpy(data[1]))
            if args.loss_opt == 0: # L2
                loss = loss_fn(y_pred, y)
            else: # weighted L2
                ww.data.copy_(torch.from_numpy(error_scale(data[1],0.01,0.99)))
                loss = loss_fn(y_pred*ww, y*ww)
        elif args.loss_opt == 2: # malis loss
            loss = loss_fn(y_pred, data[2], data[1])
        loss.backward()
        optimizer.step()
        
        # print log
        t3 = time.time()
        log.write("[Iter %d] loss=%0.2f ModelTime=%.2f TotalTime=%.2f\n" % (iter_id,loss.data[0],t3-t2,t3-t1))
        if (iter_id+1) % args.iter_save == 0: 
            print ('saving: [%d/%d]') % (pre_epoch + iter_id, args.iter_save)
            save_checkpoint(model, optimizer, iter_id, sn+('iter_%d_%d_%s.pth' % (args.batch_size,pre_epoch+iter_id+1,str(args.lr))))
        iter_id+=1
        if iter_id == args.iter_total:
            break
    log.close()

if __name__ == "__main__":
    main()
