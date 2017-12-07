import numpy as np
import os, sys, time, argparse, h5py

import torch
import torch.nn as nn
from torch.autograd import Variable

from T_model import unet3D
from T_util import load_checkpoint,weightedMSE_np,malisWeight,labelWeight,writeh5
from T_data import VolumeDatasetTest, np_collate
import malis_core

def get_args():
    parser = argparse.ArgumentParser(description='Testing Model')

    parser.add_argument('-t','--task-opt', type=int,  default=0,
                        help='task: 0=prediction, 1=evaluation, 2=evaluation-heatmap')
    parser.add_argument('-l','--loss-opt', type=int, default=0,
                        help='loss type') 
    parser.add_argument('-lw','--loss-weight-opt', type=float, default=2.0,                         
                        help='weighted loss type') 
# I/O
    parser.add_argument('-dc','--data-color-opt', type=int,  default=2,
                        help='data color aug type')
    parser.add_argument('-i','--input',  default='/n/coxfs01/donglai/malis_trans/data/ecs-3d/ecs-gt-4x6x6/',
                        help='input path')
    parser.add_argument('-s','--snapshot',  default='/n/coxfs01/donglai/malis_trans/pytorch_train/w0921/16_8_1e-3_bnL2/iter_16_11499_0.001.pth',
                        help='snapshot path')
    parser.add_argument('-dn','--data-name',  default='im_uint8.h5',
                        help='image data name')
    parser.add_argument('-dnd','--data-dataset-name',  default='main',
                        help='dataset name in data')
    parser.add_argument('-o','--output', default='result/my-pred.h5',
                        help='output path')
    parser.add_argument('-ln','--label-name',  default='seg-groundtruth2-malis_crop.h5',
                        help='segmentation label')
    parser.add_argument('-lnd','--label-dataset-name',  default='main',
                        help='dataset name in label')

    # model option
    parser.add_argument('-a','--opt-arch', type=str,  default='0-0@0@0-0-0@0',
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
    parser.add_argument('-do', '--has-dropout', type=float, default=0,
                        help='use dropout')
    # data option
    parser.add_argument('-b','--batch-size', type=int,  default=16,
                        help='batch size')
    parser.add_argument('-g','--num-gpu', type=int,  default=8,
                        help='number of gpu')
    parser.add_argument('-c','--num-cpu', type=int,  default=16,
                        help='number of cpu')
    parser.add_argument('-e', '--batch-end', type=int,  default=-1,
                        help='last batch to test')
    args = parser.parse_args()
    return args

def get_data(args):
    from T_util import cropCentral
    suf_aff = '_aff'+args.opt_param
    model_io_size = np.array([[int(x) for x in args.model_input.split(',')],
                              [int(x) for x in args.model_output.split(',')]])
    if args.task_opt==0:
        test_data =  np.array(h5py.File(args.input+args.data_name,'r')[args.data_dataset_name],dtype=np.float32)[None,:]
        if test_data.max()>10:
            test_data=test_data/(2.**8)
        out_data_size = model_io_size[0]
    elif args.task_opt in [1,2]: # load test prediction 
        test_data =  np.array(h5py.File(args.output,'r')[args.data_dataset_name],dtype=np.float32)
        out_data_size = model_io_size[1]
    color_clip = [(0.05,0.95), (0.05,0.95), None][args.data_color_opt]

    nhood = None
    test_label = None
    extra_pad = 1
    if args.task_opt in [1,2]: # do evaluation
        extra_pad = 0
        if os.path.exists(args.input+args.label_name[:-3]+suf_aff+'.h5'):
            test_label = np.array(h5py.File(args.input+args.label_name[:-3]+suf_aff+'.h5','r')['main'])
        else: # pre-compute for faster i/o
            d_seg = np.array(h5py.File(args.input+args.label_name,'r')[args.label_dataset_name])
            d_nhood = malis_core.mknhood3d()
            test_label = malis_core.seg_to_affgraph(d_seg,d_nhood)
            from T_util import writeh5
            writeh5(args.input+args.label_name[:-3]+suf_aff+'.h5', 'main', test_label)
        test_data, test_label = cropCentral(test_data, test_label, [0,0,0], False)
        if args.loss_opt==1: # evaluate malis: need segmentation
            nhood = malis_core.mknhood3d()
    test_dataset = VolumeDatasetTest([test_data], [test_label], nhood,
            clip=color_clip, sample_stride=model_io_size[1],
            extra_pad=extra_pad, out_data_size=out_data_size, out_label_size=model_io_size[1])

    test_loader =  torch.utils.data.DataLoader(
            test_dataset, batch_size= args.batch_size, shuffle=False, collate_fn = np_collate,
            num_workers= args.num_cpu, pin_memory=True)
    
    # pre-allocate torch cuda tensor
    test_var = Variable(torch.zeros(args.batch_size, 1, model_io_size[0][0], model_io_size[0][1], model_io_size[0][2]).cuda(), requires_grad=False)
    output_size = [3]+list(test_data.shape[1:]-(model_io_size[0]-model_io_size[1]))
    return test_loader, test_var, output_size, model_io_size, test_dataset.sample_size[0]

def get_model(args):
    # create model
    num_filter = [int(x) for x in args.num_filter.split(',')]

    opt_arch = [[int(x) for x in y.split('-')] for y in  args.opt_arch.split('@')]
    opt_param = [[int(x) for x in y.split('-')] for y in  args.opt_param.split('@')]
    model = unet3D(filters=num_filter,opt_arch = opt_arch, opt_param = opt_param,
                   has_BN = args.has_BN==1, has_dropout = args.has_dropout,
                   pad_size = args.pad_size, pad_type= args.pad_type)
    model.cuda()
    if args.num_gpu>1: model = nn.DataParallel(model, range(args.num_gpu)) 
    # load parameter
    cp = load_checkpoint(args.snapshot, args.num_gpu)
    model.load_state_dict(cp['state_dict'])
    return model

def main():
    args = get_args()
    if not os.path.exists(args.output[:args.output.rfind('/')]):
        os.makedirs(args.output[:args.output.rfind('/')])

    print '1. setup data'
    test_loader, test_var, output_size, model_io_size, sample_size = get_data(args)
    
    if not os.path.exists(args.output):
        if not os.path.exists(args.snapshot):
            raise ValueError(args.snapshot+" doesn't exist for prediction")
        else:
            print '-- start prediction --'
            print '2. load model'
            model = get_model(args)

            print '3. start testing'
            model.eval()
            st=time.time()
            num_batch = test_loader.__len__()
            pred = np.zeros(output_size,dtype=np.float32)
            for batch_id, data in enumerate(test_loader):
                # print batch_id,data[0].min(),data[0].max()
                test_var.data[:data[0].shape[0]].copy_(torch.from_numpy(data[0]))
                # forward-backward
                y_pred = model(test_var).data.cpu().numpy()
                for j in range(data[0].shape[0]):
                    pp = data[3][j] 
                    pred[:,pp[1]:pp[1]+model_io_size[1][0],
                          pp[2]:pp[2]+model_io_size[1][1],
                          pp[3]:pp[3]+model_io_size[1][2]] = y_pred[j].copy()
                print "finish batch: [%d/%d] " % (batch_id, num_batch)
                sys.stdout.flush()
                if batch_id == args.batch_end:
                    # early stop for debug
                    pred=pred[:,:pp[0]+model_io_size[1][0]]
                    break

            et=time.time()
            print 'testing time: '+str(et-st)+' sec'
            
            print '4. start saving'
            outhdf5 = h5py.File(args.output, 'w')
            outdset = outhdf5.create_dataset('main', pred.shape, np.float32, data=pred)
            outhdf5.close()
    else:
        print '-- start evaluate prediction loss'
        print '2. load loss'
        conn_dims = [args.batch_size, 3]+list(model_io_size[1])                                                     
        if args.loss_opt==0: # l2 
            loss_w = labelWeight(conn_dims, args.loss_weight_opt)
        elif args.loss_opt==1: # malis
            loss_w = malisWeight(conn_dims, args.loss_weight_opt) 
        print '3. start evaluation'
        loss=0;
        num_batch = test_loader.__len__()
        if args.task_opt==1: # avg test error
            for batch_id, data in enumerate(test_loader):
                if args.loss_opt == 0: # L2
                    ww = loss_w.getWeight(data[1])
                elif args.loss_opt == 1: # malis
                    ww = loss_w.getWeight(data[0], data[1], data[2])
                loss += weightedMSE_np(data[0], data[1], ww)
                print '%d/%d: avg loss = %.5f' % (batch_id,num_batch,loss/(1+batch_id))
                if batch_id == num_batch-2: # pass the last one (not enough batch)
                    break 
        elif args.task_opt==2: # heatmap test error
            pred = np.zeros(sample_size, dtype=np.float32)
            for batch_id, data in enumerate(test_loader):
                if args.loss_opt == 0: # L2
                    ww = loss_w.getWeight(data[1])
                elif args.loss_opt == 1: # malis
                    ww = loss_w.getWeight(data[0], data[1], data[2])
                for j in range(data[0].shape[0]):
                    pp2 = [int(np.ceil(float(data[3][j][1+x])/model_io_size[1][x])) for x in range(3)]
                    print pp2
                    pred[pp2[0], pp2[1], pp2[2]] = weightedMSE_np(data[0][j], data[1][j], ww[j])
            writeh5(args.output.replace('.h5','_err.h5'), 'main', pred)

if __name__ == "__main__":
    main()
