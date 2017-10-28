import numpy as np
import pickle, h5py, time, os, sys, argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data

from T_model import unet3D
from T_util import save_checkpoint,load_checkpoint,decay_lr,weightedMSE,malisWeight,labelWeight
from T_data import VolumeDatasetTrain, VolumeDatasetTest, np_collate
import malis_core

def get_args():
    parser = argparse.ArgumentParser(description='Training Model')
    # I/O
    parser.add_argument('-i','--input',  default='/n/coxfs01/donglai/malis_trans/data/ecs-3d/ecs-gt-3x6x6/',
                        help='input folder (train)')
    parser.add_argument('-t','--test',  default='/n/coxfs01/donglai/malis_trans/data/ecs-3d/ecs-gt-4x6x6/',
                        help='input folder (test)')
    parser.add_argument('-dn','--data-name',  default='im_uint8.h5',
                        help='image data')
    parser.add_argument('-ln','--label-name',  default='seg-groundtruth2-malis.h5',
                        help='segmentation label')
    parser.add_argument('-lnd','--label-dataset-name',  default='main',
                        help='dataset name in label')
    parser.add_argument('-dnd','--data-dataset-name',  default='main',
                        help='dataset name in data')
    parser.add_argument('-o','--output', default='result/train/',
                        help='output path')
    parser.add_argument('-s','--snapshot',  default='',
                        help='pre-train snapshot path')

    # model option
    parser.add_argument('-a','--opt-arch', type=str,  default='0-0@0@0-0-0@0',
                        help='model type')
    parser.add_argument('-f', '--num-filter', default='24,72,216,648',
                        help='number of filters per layer')
    parser.add_argument('-ps', '--pad-size', type=int, default=0,
                        help='pad size')
    parser.add_argument('-pt', '--pad-type', default='constant,0',
                        help='pad type')
    parser.add_argument('-bn', '--has-BN', type=int, default=0,
                        help='use BatchNorm')
    parser.add_argument('-rs', '--relu-slope', type=float, default=0.005,
                        help='relu type')
    parser.add_argument('-do', '--has-dropout', type=float, default=0,
                        help='use dropout')
    parser.add_argument('-it','--init', type=int,  default=-1,
                        help='model initialization type')
    # data option
    parser.add_argument('-dc','--data-color-opt', type=int,  default=2,
                        help='data color aug type')
    parser.add_argument('-dr','--data-rotation-opt', type=int,  default=0,
                        help='data rotation aug type')
    parser.add_argument('-l','--loss-opt', type=int, default=0,
                        help='loss type')
    parser.add_argument('-lw','--loss-weight-opt', type=float, default=2.0,
                        help='weighted loss type')
    # optimization option
    parser.add_argument('-lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('-lr_decay', default='inv,0.0001,0.75',
                        help='learning rate decay')
    parser.add_argument('-betas', default='0.99,0.999',
                        help='beta for adam')
    parser.add_argument('-wd', type=float, default=5e-6,
                        help='weight decay')
    parser.add_argument('--volume-total', type=int, default=1000,
                        help='total number of iteration')
    parser.add_argument('--volume-save', type=int, default=100,
                        help='number of iteration to save')
    parser.add_argument('-e', '--pre-epoch', type=int, default=0,
                        help='previous number of epoch')
    parser.add_argument('-g','--num-gpu', type=int,  default=1,
                        help='number of gpu')
    parser.add_argument('-c','--num-cpu', type=int,  default=1,
                        help='number of cpu')
    parser.add_argument('-b','--batch-size', type=int,  default=1,
                        help='batch size')
    args = parser.parse_args()
    return args

def get_data(args):
    # data/seg: initially same size
    if args.pad_size==0:
        model_io_size = np.array(((31,204,204), (3,116,116)));
        aff_suf='';
    elif args.pad_size==1:
        model_io_size = np.array(((18,224,224), (18,224,224)));
        aff_suf='2';
    train_offset = (model_io_size[0]-model_io_size[1])/2
    if args.data_name[-3:] == '.h5':
        train_data =  np.array(h5py.File(args.input+args.data_name,'r')[args.data_dataset_name],dtype=np.float32)[None,:]/(2.**8)
    elif args.data_name[-4:] == '.pkl':
        train_data =  np.array(pickle.load(args.input+args.data_name,'rb'),dtype=np.float32)[None,:]/(2.**8)

    train_nhood = malis_core.mknhood3d()
    # load whole aff -> remove offset for label, pad a bit for rotation augmentation
    if os.path.exists(args.input+args.label_name[:-3]+'_aff'+aff_suf+'.h5'):
        train_label = np.array(h5py.File(args.input+args.label_name[:-3]+'_aff'+aff_suf+'.h5','r')['main'])
    else: # pre-compute for faster i/o
        train_seg = np.array(h5py.File(args.input+args.label_name,'r')[args.label_dataset_name])
        train_label = malis_core.seg_to_affgraph(train_seg,train_nhood)
        if aff_suf=='2':
            train_label = np.lib.pad(train_label,((0,0),(1,1),(1,1),(1,1)),mode='reflect')
        from T_util import writeh5
        writeh5(args.input+args.label_name[:-3]+'_aff'+aff_suf+'.h5', 'main', train_label)
    # either crop or pad whole
    if train_offset[0]!=0:
        train_label = train_label[:,train_offset[0]-1:-train_offset[0]+1,train_offset[1]-1:-train_offset[1]+1,train_offset[2]-1:-train_offset[2]+1]
    # add sampler
    nhood = None if args.loss_opt in [0] else train_nhood
    color_scale = [(0.8,1.2), (0.9,1.1), None][args.data_color_opt]
    color_shift = [(-0.2,0.2), (-0.1,0.1), None][args.data_color_opt]
    color_clip = [(0.05,0.95), (0.05,0.95), None][args.data_color_opt]
    rot = [[(1,1,1),True],[(0,0,0),False]][args.data_rotation_opt]
    train_dataset = VolumeDatasetTrain(train_data, train_label, nhood, data_size=train_data.shape[1:],
                           reflect=rot[0], swapxy=rot[1],
                           color_scale=color_scale,color_shift=color_shift,clip=color_clip,
                           out_data_size=model_io_size[0],out_label_size=model_io_size[1])
    train_loader =  torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn = np_collate,
            num_workers=args.num_cpu, pin_memory=True)

    # pre-allocate torch cuda tensor
    train_vars = [None]*3
    # input data
    train_vars[0] = Variable(torch.zeros(args.batch_size, 1, model_io_size[0][0], model_io_size[0][1], model_io_size[0][2]).cuda(), requires_grad=False)
    # gt label
    train_vars[1] = Variable(torch.zeros(args.batch_size, 3, model_io_size[1][0], model_io_size[1][1], model_io_size[1][2]).cuda(), requires_grad=False)
    if not (args.loss_opt == 0 and args.loss_weight_opt == 0):
        # weight
        train_vars[2] = Variable(torch.zeros(args.batch_size, 3, model_io_size[1][0], model_io_size[1][1], model_io_size[1][2]).cuda(), requires_grad=False)

    return train_loader, train_vars,model_io_size


def get_test_data(args):
    # Handle padding
    if args.pad_size==0:
        model_io_size = np.array(((31,204,204), (3,116,116)));
        aff_suf='';
    elif args.pad_size==1:
        model_io_size = np.array(((18,224,224), (18,224,224)));
        aff_suf='2';

    # Features
    test_data =  np.array(h5py.File(args.test + args.data_name,'r')['main'],dtype=np.float32)[None,:]/(2.**8)
    out_data_size = model_io_size[0]

    # Labels
    test_label = np.array(h5py.File(args.test + args.label_name[:-3]+'_crop_aff'+aff_suf+'.h5','r')['main'])
    nhood = malis_core.mknhood3d() if args.loss_opt==1 else None

    # Loaders
    test_dataset = VolumeDatasetTest(test_data, test_label, nhood, data_size=test_data.shape[1:],sample_stride=model_io_size[1],
                out_data_size=out_data_size,out_label_size=model_io_size[1])
    test_loader =  torch.utils.data.DataLoader(
            test_dataset, batch_size= args.batch_size, shuffle=False, collate_fn = np_collate,
            num_workers= args.num_cpu, pin_memory=True)

    return test_loader

def get_model(args, model_io_size):
    num_filter = [int(x) for x in args.num_filter.split(',')]
    opt_arch = [[int(x) for x in y.split('-')] for y in  args.opt_arch.split('@')]
    model = unet3D(filters=num_filter,opt_arch = opt_arch,
                   has_BN = args.has_BN==1, has_dropout = args.has_dropout, relu_slope = args.relu_slope,
                   pad_size = args.pad_size, pad_type= args.pad_type)
    if args.num_gpu>1: model = nn.DataParallel(model, range(args.num_gpu))
    model.cuda()
    conn_dims = [args.batch_size,3]+list(model_io_size[1])
    if args.loss_opt == 0: # L2 training
        loss_w = labelWeight(conn_dims, args.loss_weight_opt)
        save_suf = '_L2_'+str(args.lr)
    elif args.loss_opt == 1: # malis training
        loss_w = malisWeight(conn_dims, args.loss_weight_opt)
        save_suf = '_malis_'+str(args.lr)
    save_suf += '_'+str(args.loss_weight_opt)

    # load previous model
    pre_epoch = args.pre_epoch
    if len(args.snapshot)>0:
        save_suf += '_'+args.snapshot[:args.snapshot.rfind('.')] if '/' not in args.snapshot else args.snapshot[args.snapshot.rfind('/')+1:args.snapshot.rfind('.')]
        cp = load_checkpoint(args.snapshot, args.num_gpu)
        model.load_state_dict(cp['state_dict'])
        if pre_epoch == 0:
            pre_epoch = cp['epoch']

    logger = open(args.output+'log'+save_suf+'.txt','w',0) # unbuffered, write instantly
    return model, loss_w, pre_epoch, logger

def get_optimizer(args, model, pre_epoch=0):
    betas = [float(x) for x in args.betas.split(',')]
    frozen_id = []
    if args.num_gpu==1 and model.up[0].opt[0]==0: # hacked upsampling layer
        for i in range(len(model.up)):
            frozen_id +=  list(map(id,model.up[i].up._modules['0'].parameters()))
    elif model.module.up[0].opt[0]==0: # hacked upsampling layer
        for i in range(len(model.module.up)):
            frozen_id +=  list(map(id,model.module.up[i].up._modules['0'].parameters()))
    frozen_params = filter(lambda p: id(p) in frozen_id, model.parameters())
    rest_params = filter(lambda p: id(p) not in frozen_id, model.parameters())
    optimizer = torch.optim.Adam([
        {'params': rest_params},
        {'params': frozen_params, 'lr':0.0, 'weight_decay':0.0, 'betas':[0.0, 0.0]}],
        lr=args.lr, betas=betas, weight_decay=args.wd)

    lr_decay = args.lr_decay.split(',')
    for i in range(1,len(lr_decay)):
        lr_decay[i] =  float(lr_decay[i])
    if pre_epoch != 0:
        decay_lr(optimizer, args.lr, pre_epoch-1, lr_decay[0], lr_decay[1], lr_decay[2])
    return optimizer, lr_decay

def forward(model, data, vars, loss_w, args):
    y_pred = model(vars[0])
    vars[1].data.copy_(torch.from_numpy(data[1]))
    # Weighted (L2)
    if args.loss_opt == 0 and args.loss_weight_opt != 0:
        vars[2].data.copy_(torch.from_numpy(loss_w.getWeight(data[1])))
    # Weighted (MALIS)
    elif args.loss_opt == 1 and args.loss_weight_opt != 0:
        vars[2].data.copy_(torch.from_numpy(loss_w.getWeight(y_pred.data.cpu().numpy(), data[1], data[2])))
    return weightedMSE(y_pred, vars[1], vars[2])

def main():
    args = get_args()
    sn = args.output+'/'
    if not os.path.isdir(sn):
        os.makedirs(sn)

    print '1. setup data'
    train_loader, train_vars, model_io_size = get_data(args)
    test_loader = get_test_data(args)

    print '2. setup model'
    model, loss_w, pre_epoch, logger = get_model(args, model_io_size)

    print '3. setup optimizer'
    optimizer, lr_decay = get_optimizer(args, model, pre_epoch)

    print '4. start training'
    if args.lr == 0:
        model.eval()
    else:
        model.train()

    # Normalize learning rate
    args.lr = args.lr * args.batch_size / 2
    train_iter, test_iter = iter(train_loader), iter(test_loader)
    for iter_id, data in enumerate(train_iter):
        optimizer.zero_grad()
        volume_id = (iter_id + 1) * args.batch_size
        pre_volume = pre_epoch * args.batch_size

        # copy data
        t1 = time.time()

        # Forward
        t2 = time.time()
        # Validation error
        #if iter_id % 10 == 0:
        #    test_data = next(test_iter)
        #    train_vars[0].data.copy_(torch.from_numpy(test_data[0]))
        #    test_loss = forward(model, test_data, train_vars, loss_w, args)
        # Training error
        train_vars[0].data.copy_(torch.from_numpy(data[0]))
        train_loss = forward(model, data, train_vars, loss_w, args)

        # Backward
        if args.lr > 0:
            train_loss.backward()
            optimizer.step()

        # Print log
        t3 = time.time()
        logger.write("[Volume %d] train_loss=%0.3f test_loss=%0.3f lr=%.5f ModelTime=%.2f TotalTime=%.2f\n" % (volume_id,train_loss.data[0],0.25,optimizer.param_groups[0]['lr'],t3-t2,t3-t1))

        # Save progress
        if volume_id % args.volume_save == 0 or volume_id >= args.volume_total:
            save_checkpoint(model, sn+('volume_%d.pth' % (pre_volume + volume_id)), optimizer, volume_id)

        # Terminate
        if volume_id >= args.volume_total:
            break

        # LR update
        if args.lr > 0:
            decay_lr(optimizer, args.lr, pre_epoch+iter_id, lr_decay[0], lr_decay[1], lr_decay[2])

    logger.close()

if __name__ == "__main__":
    main()
