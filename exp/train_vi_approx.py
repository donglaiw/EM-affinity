import numpy as np
import pickle, h5py, time, argparse, itertools

import torch
import torch.nn as nn
import torch.utils.data

import os, sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from em.model.io import load_checkpoint,save_checkpoint
from em.model.unet import unet3D
from em.model.deploy import cnn_vi_v1
from em.model.optim import decay_lr
from em.model.loss import viWeight
from em.data.volumeData import VolumeDatasetTrain, VolumeDatasetTest, np_collate
from em.data.io import getVar, getImg, getLabel, cropCentralN
from em.data.augmentation import DataAugment 

def get_args():
    parser = argparse.ArgumentParser(description='Training Model')
    # I/O
    parser.add_argument('-m','--model-id',  type=float, default=0,
                        help='model id')
    parser.add_argument('-t','--train',  default='/n/coxfs01/donglai/malis_trans/data/ecs-3d/ecs-gt-3x6x6/',
                        help='input folder (train)')
    parser.add_argument('-v','--val',  default='',
                        help='input folder (test)')
    parser.add_argument('-dn','--img-name',  default='im_uint8.h5',
                        help='image data')
    parser.add_argument('-ln','--seg-name',  default='seg-groundtruth2-malis.h5',
                        help='segmentation label')
    parser.add_argument('-dnv','--img-name-val',  default='im_uint8.h5',
                        help='image data for val')
    parser.add_argument('-lnv','--seg-name-val',  default='seg-groundtruth2-malis.h5',
                        help='segmentation label for val')
    parser.add_argument('-dnd','--img-dataset-name',  default='main',
                        help='dataset name in data')
    parser.add_argument('-lnd','--seg-dataset-name',  default='main',
                        help='dataset name in label')
    parser.add_argument('-o','--output', default='result/train/',
                        help='output path')
    parser.add_argument('-s','--snapshot',  default='',
                        help='pre-train snapshot path')

    # model option
    parser.add_argument('-ma','--opt-arch', type=str,  default='0,0@0@0,0,0@0',
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
    parser.add_argument('-ao','--aug-opt', type=str,  default='1@-1@0@5',
                        help='data aug type')
    parser.add_argument('-apw','--aug-param-warp', type=str,  default='15@3@1.1@0.1',
                        help='data warp aug parameter')
    parser.add_argument('-apc','--aug-param-color', type=str,  default='0.95,1.05@-0.15,0.15@0.5,2@0,1',
                        help='data color aug parameter')

    # optimization option
    parser.add_argument('-l','--loss-opt', type=int, default=0,
                        help='loss type')
    parser.add_argument('-lw','--loss-weight-opt', type=float, default=2.0,
                        help='weighted loss type')
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
    train_vars = getVar(args.batch_size, model_io_size, [True, True, not (args.loss_opt == 0 and args.loss_weight_opt == 0)])
    return model_io_size, train_vars

def get_img(args, model_io_size, opt='train'):
    # two dataLoader, can't be both multiple-cpu (pytorch issue)
    if opt=='train':
        dir_name = args.train.split('@')
        num_worker = args.num_cpu
        img_name = args.img_name.split('@')
        seg_name = args.seg_name.split('@')
    else:
        dir_name = args.val.split('@')
        num_worker = 1
        img_name = args.img_name_val.split('@')
        seg_name = args.seg_name_val.split('@')
    img_dataset_name = args.img_dataset_name.split('@')
    seg_dataset_name = args.seg_dataset_name.split('@')

    # should be either one or the same as dir_name
    seg_name = [dir_name[x]+seg_name[0] for x in range(len(dir_name))] \
            if len(seg_name) == 1 else [dir_name[x]+seg_name[x] for x in range(len(dir_name))]
    img_name = [dir_name[x]+img_name[0] for x in range(len(dir_name))] \
            if len(img_name) == 1 else [dir_name[x]+img_name[x] for x in range(len(dir_name))]
    seg_dataset_name = seg_dataset_name*len(dir_name) if len(seg_dataset_name) == 1 else seg_dataset_name
    img_dataset_name = img_dataset_name*len(dir_name) if len(img_dataset_name) == 1 else img_dataset_name

    if len(dir_name[0])==0: # don't load data
        return None

    # 1. load data
    # make sure img and label have the same size
    # assume img and label have the same center
    suf_aff = '_aff'+args.opt_param
    train_img = getImg(img_name, img_dataset_name)
    train_label = getLabel(seg_name, seg_dataset_name, suf_aff)
    #train_img, train_label = cropCentralN(train_img, train_label)

    # 2. get dataAug
    aug_opt = [int(x) for x in args.aug_opt.split('@')]
    aug_param_warp = [float(x) for x in args.aug_param_warp.split('@')]
    aug_param_color = [[float(y) for y in x.split(',')] for x in args.aug_param_color.split('@')]
    data_aug = DataAugment(aug_opt, aug_param_warp, aug_param_color)

    # if malis, then need seg
    do_seg = args.loss_opt==1 # need seg if do malis loss
    import pdb; pdb.set_trace()
    dataset = VolumeDatasetTrain(train_img, train_label, do_seg, np.inf, \
                                 model_io_size[0], model_io_size[1], data_aug=data_aug)
    # to have evaluation during training (two dataloader), has to set num_worker=0
    img_loader =  torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, collate_fn = np_collate,
            num_workers=num_worker, pin_memory=True)
    return img_loader

def get_model(args, model_io_size):
    # 1. get model
    num_filter = [int(x) for x in args.num_filter.split(',')]
    if args.model_id==0: # flexible framework
        opt_arch = [[int(x) for x in y.split(',')] for y in  args.opt_arch.split('@')]
        opt_param = [[int(x) for x in y.split(',')] for y in  args.opt_param.split('@')]
        model = unet3D(filters=num_filter, opt_arch = opt_arch, opt_param = opt_param,
                       has_BN = args.has_BN==1, has_dropout = args.has_dropout, relu_slope = args.relu_slope,
                       pad_size = args.pad_size, pad_type= args.pad_type)
    elif args.model_id==1: # cnn_vi_v1
        model = cnn_vi_v1(filters=num_filter, has_BN = args.has_BN==1)  

    # 2. load previous model weight
    pre_epoch = args.pre_epoch
    if len(args.snapshot)>0:
        cp = load_checkpoint(args.snapshot)
        model.load_state_dict(cp['state_dict'])
        if pre_epoch == 0:
            pre_epoch = cp['epoch']
        print '\t continue to train from epoch '+str(pre_epoch)

    # 3. get loss weight
    conn_dims = [args.batch_size,3]+list(model_io_size[1])
    if args.loss_opt == 0: # VI training
        loss_w = viWeight(conn_dims, args.loss_weight_opt)

    return model, loss_w, pre_epoch

def get_logger(args):
    log_name = args.output+'/log'
    log_name += ['_L2_','_malis_'][args.loss_opt]+str(args.lr)+ '_'+str(args.loss_weight_opt)
    if len(args.snapshot)>0:
        log_name += '_'+args.snapshot[:args.snapshot.rfind('.')] if '/' not in args.snapshot else args.snapshot[args.snapshot.rfind('/')+1:args.snapshot.rfind('.')]
    logger = open(log_name+'.txt','w',0) # unbuffered, write instantly
    return logger

def get_optimizer(args, model, pre_epoch=0):
    betas = [float(x) for x in args.betas.split(',')]
    frozen_id = []
    if args.model_id==0 and model.up[0].opt[0]==0: # hacked upsampling layer
        for i in range(len(model.up)):
            frozen_id +=  list(map(id,model.up[i].up._modules['0'].parameters()))
    elif args.model_id==2: # hacked upsampling layer
        for i in range(len(model.upS)):
            frozen_id +=  list(map(id,model.upS[i]._modules['0'].parameters()))
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
    model_io_size, train_vars = init(args) 

    print '1. setup data'
    #train_loader = get_img(args, model_io_size, 'train')
    #test_loader = get_img(args, model_io_size, 'val') if args.val != '' else None

    print '2. setup model'
    model, loss_w, pre_epoch = get_model(args, model_io_size)
    logger = get_logger(args)

    print '3. setup optimizer'
    optimizer, lr_decay = get_optimizer(args, model, pre_epoch)

    print '4. start training'
    if args.num_gpu>1: model = nn.DataParallel(model, range(args.num_gpu))
    model.cuda()
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
        #print data[0].shape,data[1].shape,data[2].shape
        #visSliceSeg(data[0], data[2], offset=[14,44,44],outN='tmp/train_seg'+str(iter_id)+'_'+str(data[3][0][0])+'.png', frame_id=0)
        #visSliceSeg(data[0], data[1][0][1], offset=[14,44,44],outN='tmp/train_affy'+str(iter_id)+'_'+str(data[3][0][0])+'.png', frame_id=0)
        #visSliceSeg(data[0], data[2], offset=[6,14,14],outN='result/db/train_'+str(iter_id)+'_'+str(data[3][0][0])+'.png', frame_id=0)
        #visSlice(data[0][0,0],outN='result/db/train_im.png',frame_id=6)
        train_vars[0].data.copy_(torch.from_numpy(data[0]))
        train_loss = forward(model, data, train_vars, loss_w, args)

        # Forward
        t2 = time.time()
        # Backward
        if args.lr > 0:
            train_loss.backward()
            optimizer.step()

        t3 = time.time()
        # Validation error
        if test_iter is not None and iter_id % 5 == 0:
            test_img = next(test_iter)
            #visSliceSeg(test_img[0], test_img[2], offset=[14,44,44],outN='result/db/test_'+str(iter_id)+'_'+str(test_img[3][0][0])+'.png', frame_id=0)
            train_vars[0].data.copy_(torch.from_numpy(test_img[0]))
            test_loss = forward(model, test_img, train_vars, loss_w, args).data[0]

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
