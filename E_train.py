import numpy as np
import pickle, h5py, time, os, sys, argparse, itertools

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data

from T_model import unet3D
from T_util import save_checkpoint,load_checkpoint,decay_lr,weightedMSE,malisWeight,labelWeight
from T_data import VolumeDatasetTrain, VolumeDatasetTest, np_collate
import malis_core
#from T_vis import visSliceSeg, visSlice

def get_args():
    parser = argparse.ArgumentParser(description='Training Model')
    # I/O
    parser.add_argument('-t','--train',  default='/n/coxfs01/donglai/malis_trans/data/ecs-3d/ecs-gt-3x6x6/',
                        help='input folder (train)')
    parser.add_argument('-v','--val',  default='',
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
    parser.add_argument('-ma','--opt-arch', type=str,  default='0-0@0@0-0-0@0',
                        help='model type')
    parser.add_argument('-mp','--opt-param', type=str,  default='0@0@0@0',
                        help='model param')
    parser.add_argument('-mi','--model-input', type=str,  default='31,204,204',
                        help='model input size')
    parser.add_argument('-mo','--model-output', type=str,  default='3,116,116',
                        help='model input size')
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

def init(args):
    sn = args.output+'/'
    if not os.path.isdir(sn):
        os.makedirs(sn)
    model_io_size = np.array([[int(x) for x in args.model_input.split(',')],
                              [int(x) for x in args.model_output.split(',')]])

    # pre-allocate torch cuda tensor for malis loss
    # input data
    pre_vars = [None] *3  
    pre_vars[0] = Variable(torch.zeros(args.batch_size, 1, model_io_size[0][0], model_io_size[0][1], model_io_size[0][2]).cuda(), requires_grad=False)
    # gt label
    pre_vars[1] = Variable(torch.zeros(args.batch_size, 3, model_io_size[1][0], model_io_size[1][1], model_io_size[1][2]).cuda(), requires_grad=False)
    if not (args.loss_opt == 0 and args.loss_weight_opt == 0): # weight
        pre_vars[2] = Variable(torch.zeros(args.batch_size, 3, model_io_size[1][0], model_io_size[1][1], model_io_size[1][2]).cuda(), requires_grad=False)
    return model_io_size, pre_vars

def get_data(args, model_io_size, opt='train'):
    from T_util import cropCentral
    if opt=='train':
        dirName = args.train.split('@')
        numWorker = args.num_cpu
    else:
        dirName = args.val.split('@')
        numWorker = 1
    if len(dirName[0])==0:
        return None

    # 1. load data
    d_data = [None]*len(dirName)
    d_label = [None]*len(dirName)
    suf_aff = '_aff'+args.opt_param
    model_io_offset = (model_io_size[0]-model_io_size[1])/2
    d_nhood = malis_core.mknhood3d()
    nhood = None if args.loss_opt in [0] else d_nhood
    for i in range(len(dirName)):
        if args.data_name[-3:] == '.h5':
            d_data[i] =  np.array(h5py.File(dirName[i]+args.data_name,'r')[args.data_dataset_name],dtype=np.float32)[None,:]/(2.**8)
        elif args.data_name[-4:] == '.pkl':
            d_data[i] =  np.array(pickle.load(dirName[i]+args.data_name,'rb'),dtype=np.float32)[None,:]/(2.**8)
        else: # folder of images
            import glob
            from scipy import misc
            imN=sorted(glob.glob(dirName[i]+'*'+args.data_name))
            im0 =  misc.imread(imN[0])
            d_data[i] =  np.zeros((len(imN),im0.shape[1],im0.shape[0]),dtype=np.float32)
            for j in range(len(imN)):
                d_data[i][j] = misc.imread(imN[j]).astype(np.float32)/(2.**8)  

        # load whole aff -> remove offset for label, pad a bit for rotation augmentation
        if os.path.exists(dirName[i]+args.label_name[:-3]+suf_aff+'.h5'):
            d_label[i] = np.array(h5py.File(dirName[i]+args.label_name[:-3]+suf_aff+'.h5','r')['main'])
        else: # pre-compute for faster i/o
            d_seg = np.array(h5py.File(dirName[i]+args.label_name,'r')[args.label_dataset_name])
            d_label[i] = malis_core.seg_to_affgraph(d_seg,d_nhood)
            from T_util import writeh5
            writeh5(dirName[i]+args.label_name[:-3]+suf_aff+'.h5', 'main', d_label[i])
        d_data[i], d_label[i] = cropCentral(d_data[i], d_label[i], model_io_offset)

    # 2. add sampler
    if opt=='train':
        color_scale = [(0.8,1.2), (0.9,1.1), None][args.data_color_opt]
        color_shift = [(-0.2,0.2), (-0.1,0.1), None][args.data_color_opt]
        color_clip = [(0.05,0.95), (0.05,0.95), None][args.data_color_opt]
        rot = [[(1,1,1),True],[(0,0,0),False]][args.data_rotation_opt]
    else:
        color_scale=None; color_shift=None; color_clip=None; rot=[(0,0,0),False]

    dataset = VolumeDatasetTrain(d_data, d_label, d_nhood,
                           reflect=rot[0], swapxy=rot[1],
                           color_scale=color_scale,color_shift=color_shift,clip=color_clip,
                           out_data_size=model_io_size[0],out_label_size=model_io_size[1])
    # to have evaluation during training (two dataloader), has to set num_worker=0
    data_loader =  torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, collate_fn = np_collate,
            num_workers=numWorker, pin_memory=True)
    return data_loader

def get_model(args, model_io_size):
    num_filter = [int(x) for x in args.num_filter.split(',')]
    opt_arch = [[int(x) for x in y.split('-')] for y in  args.opt_arch.split('@')]
    opt_param = [[int(x) for x in y.split('-')] for y in  args.opt_param.split('@')]
    model = unet3D(filters=num_filter, opt_arch = opt_arch, opt_param = opt_param,
                   has_BN = args.has_BN==1, has_dropout = args.has_dropout, relu_slope = args.relu_slope,
                   pad_size = args.pad_size, pad_type= args.pad_type)
    if args.num_gpu>1: model = nn.DataParallel(model, range(args.num_gpu))
    model.cuda()
    conn_dims = [args.batch_size,3]+list(model_io_size[1])
    if args.loss_opt == 0: # L2 training
        loss_w = labelWeight(conn_dims, args.loss_weight_opt)
        log_suf = '_L2_'+str(args.lr)
    elif args.loss_opt == 1: # malis training
        loss_w = malisWeight(conn_dims, args.loss_weight_opt)
        log_suf = '_malis_'+str(args.lr)
    log_suf += '_'+str(args.loss_weight_opt)

    # load previous model
    pre_epoch = args.pre_epoch
    if len(args.snapshot)>0:
        log_suf += '_'+args.snapshot[:args.snapshot.rfind('.')] if '/' not in args.snapshot else args.snapshot[args.snapshot.rfind('/')+1:args.snapshot.rfind('.')]
        cp = load_checkpoint(args.snapshot, args.num_gpu)
        model.load_state_dict(cp['state_dict'])
        if pre_epoch == 0:
            pre_epoch = cp['epoch']

    logger = open(args.output+'log'+log_suf+'.txt','w',0) # unbuffered, write instantly
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

    print '0. initial setup'
    model_io_size, pre_vars = init(args) 

    print '1. setup data'
    train_loader = get_data(args, model_io_size, 'train')
    test_loader = get_data(args, model_io_size, 'val')

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
    # args.lr = args.lr * args.batch_size / 2
    test_iter = test_loader.__iter__() if test_loader is not None else None
    test_loss = 0
    volume_id = pre_epoch
    for iter_id, data in enumerate(train_loader):
        optimizer.zero_grad()
        volume_id += args.batch_size

        # copy data
        t1 = time.time()

        # Training error
        #visSliceSeg(data[0], data[2], offset=[14,44,44],outN='result/db/train_'+str(iter_id)+'_'+str(data[3][0][0])+'.png', frame_id=0)
        #visSliceSeg(data[0], data[2], offset=[6,14,14],outN='result/db/train_'+str(iter_id)+'_'+str(data[3][0][0])+'.png', frame_id=0)
        #visSlice(data[0][0,0],outN='result/db/train_im.png',frame_id=6)
        pre_vars[0].data.copy_(torch.from_numpy(data[0]))
        train_loss = forward(model, data, pre_vars, loss_w, args)

        # Forward
        t2 = time.time()
        # Backward
        if args.lr > 0:
            train_loss.backward()
            optimizer.step()

        t3 = time.time()
        # Validation error
        if test_iter is not None and iter_id % 5 == 0:
            test_data = next(test_iter)
            #visSliceSeg(test_data[0], test_data[2], offset=[14,44,44],outN='result/db/test_'+str(iter_id)+'_'+str(test_data[3][0][0])+'.png', frame_id=0)
            pre_vars[0].data.copy_(torch.from_numpy(test_data[0]))
            test_loss = forward(model, test_data, pre_vars, loss_w, args).data[0]

        # Print log
        logger.write("[Volume %d] train_loss=%0.3f test_loss=%0.3f lr=%.5f ModelTime=%.2f TotalTime=%.2f\n" % (volume_id,train_loss.data[0],test_loss,optimizer.param_groups[0]['lr'],t3-t2,t3-t1))

        # Save progress
        if volume_id % args.volume_save <args.batch_size or volume_id >= args.volume_total:
            save_checkpoint(model, args.output+('/volume_%d.pth' % (volume_id)), optimizer, volume_id)
        # Terminate
        if volume_id >= args.volume_total:
            break

        # LR update
        if args.lr > 0:
            decay_lr(optimizer, args.lr, volume_id, lr_decay[0], lr_decay[1], lr_decay[2])
    logger.close()

if __name__ == "__main__":
    main()
