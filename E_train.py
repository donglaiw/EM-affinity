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
from T_model import unet3D
from T_data import VolumeDatasetTrain, np_collate

# for L2 training: re-weight the error by label bias (far more 1 than 0)
def error_scale(data, clip_low, clip_high):
    frac_pos = np.clip(data.mean(), clip_low, clip_high) #for binary labels
    # can't be all zero
    w_pos = 1.0/(2.0*frac_pos)
    w_neg = 1.0/(2.0*(1.0-frac_pos))
    scale = np.add((data >= 0.5) * w_pos, (data < 0.5) * w_neg)
    return scale

   
def save_checkpoint(model, optimizer, epoch=1, filename='checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict()
    }, filename)

# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 python T_trainPytorch.py 1 16 18125 3625 s0921/16_8_1e-3 im_uint8 seg-groundtruth2-malis 8 0.001 16 iter_16_625.pth

if __name__ == "__main__":
    # constant to change
    D0 = '/n/coxfs01/donglai/malis_trans/data/ecs-3d/ecs-gt-3x6x6/'
    # arg parameters
    opt = sys.argv[1] # 0: unet3d-L2, 1: unet3d-malis, 0.1: unet3dBN-L2, 1.1: unet3dBN-malis
    batch_size = int(sys.argv[2])
    num_epoch = int(sys.argv[3])
    num_epoch_save = int(sys.argv[4])
    save_dir = sys.argv[5]+'/'
    data_name = sys.argv[6] # data name
    label_name = sys.argv[7] # groundtruth label name
    num_gpu = int(sys.argv[8]) # number of gpu
    lr = float(sys.argv[9]) # learning rate
    nDonkey = int(sys.argv[10]) # number of thread to load data
    preM = None # previous model

    if len(sys.argv)>11:
        preM = sys.argv[11]

    sn = 'pytorch_train/'+save_dir
    if not os.path.isdir(sn):
        os.mkdir(sn)

    if float(opt) < 2: # train L2 net
        print '1. setup data'
        print '\t loading data'
        # data/seg: initially same size
        train_size = np.array(((31,204,204), (3,116,116)));
        train_offset = (train_size[0]-train_size[1])/2
        train_data =  np.array(h5py.File(D0+data_name+'.h5','r')['main'],dtype=np.float32)[None,:]/(2.**8)
        train_nhood = malis_core.mknhood3d()
        if os.path.exists(D0+label_name+'_aff.h5'):
            train_label = np.array(h5py.File(D0+label_name+'_aff.h5','r')['main'])
        else:
            train_seg = np.array(h5py.File(D0+label_name+'.h5','r')['main'])[train_offset[0]-2:-train_offset[0]+2,train_offset[1]-2:-train_offset[1]+2,train_offset[2]-2:-train_offset[2]+2]
            train_label = malis_core.seg_to_affgraph(train_seg,train_nhood)[:,1:-1,1:-1,1:-1]
            from T_util import writeh5
            writeh5(D0+label_name+'_aff.h5', 'main', train_label)
        # naive implementation

        print '\t creating dataloader'
        # add sampler
        nhood = None if opt[0]=='0' else train_nhood
        train_dataset = VolumeDatasetTrain(train_data, train_label, nhood, batch_size, data_size=train_data.shape[1:], reflect=(1,1,1),swapxy=True,color_scale=(0.8,1.2),color_shift=(-0.2,0.2),clip=(0.05,0.95),out_data_size=train_size[0],out_label_size=train_size[1])
        if '_df' in data_name: # mean already normalized
            train_dataset.color_scale = (0.9,1.1)
            train_dataset.color_shift = (-0.1,0.1)

        train_loader =  torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, collate_fn = np_collate,
                num_workers=nDonkey, pin_memory=True)

        print '2. setup model'
        # initialization matters ?
        if len(opt)==1: #regular
            model = unet3D()
        else:
            if opt[2]=='1': #batchnorm
                model = unet3D(is_batchnorm=True)
        if num_gpu>1: model = nn.DataParallel(model, range(num_gpu)) 
        model.cuda()
        if lr==0: #for eval
            model.eval()
        else:
            model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.99, 0.999), weight_decay=5e-6)
        pre_epoch = 0
        
        if opt[0]=='0': # weighted L2 training
            loss_fn = torch.nn.MSELoss()
            loss_suf = '_L2_'+str(lr)
        elif opt[0]=='1': # malis training
            import malisLoss
            loss_fn = malisLoss.MalisLoss([batch_size,3]+list(train_size[1]),1).cuda()
            loss_suf = '_malis_'+str(lr)
            # load previous model
            if preM is not None:
                loss_suf += '_'+preM[:-4] if '/' not in preM else preM[preM.rfind('/')+1:-4]
                cp = torch.load(sn+preM)
                model.load_state_dict(cp['state_dict'])
                pre_epoch = cp['epoch']
            

        print '3. start training'
        log = open(sn+'log'+loss_suf+'.txt','w',0) # unbuffered, write instantly
        # pre-allocate torch cuda tensor
        x = Variable(torch.zeros(batch_size, 1, 31, 204, 204).cuda(), requires_grad=False)
        if opt[0]=='0':
            y = Variable(torch.randn(batch_size, 3, 3, 116, 116).cuda(), requires_grad=False)
            ww = Variable(torch.zeros(batch_size, 3, 3, 116, 116).cuda(), requires_grad=False)
        for iter_id, data in enumerate(train_loader):
            optimizer.zero_grad()
            # copy data
            t1 = time.time()
            x.data.copy_(torch.from_numpy(data[0]))

            # forward-backward
            t2 = time.time()
            y_pred = model(x)
            if opt[0]=='0': # weighted L2
                y.data.copy_(torch.from_numpy(data[1]))
                ww.data.copy_(torch.from_numpy(error_scale(data[1],0.01,0.99)))
                loss = loss_fn(y_pred*ww, y*ww)
            elif opt[0]=='1':
                loss = loss_fn(y_pred, data[2], data[1])
            loss.backward()
            optimizer.step()

            # print log
            t3 = time.time()
            log.write("[Iter %d] loss=%0.2f ModelTime=%.2f TotalTime=%.2f\n" % (iter_id,loss.data[0],t3-t2,t3-t1))
            if (iter_id+1) % num_epoch_save == 0: 
                print ('saving: [%d/%d]') % (iter_id, num_epoch)
                save_checkpoint(model, optimizer, iter_id, sn+('iter_%d_%d_%s.pth' % (batch_size,pre_epoch+iter_id+1,str(lr))))
            iter_id+=1
            if iter_id == num_epoch:
                break
           
    log.close()
