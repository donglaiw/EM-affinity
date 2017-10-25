import numpy as np
import os, sys, time, argparse, h5py

import torch
import torch.nn as nn
from torch.autograd import Variable

from T_model import unet3D
from T_util import load_checkpoint,weightedMSE_np,malisWeight,labelWeight
from T_data import VolumeDatasetTest, np_collate
import malis_core

def get_args():
    parser = argparse.ArgumentParser(description='Testing Model')

    parser.add_argument('-t','--task-opt', type=int,  default=0,
                        help='task: 0=prediction, 1=evaluation')
    parser.add_argument('-l','--loss-opt', type=int, default=0,
                        help='loss type') 
    parser.add_argument('-lw','--loss-weight-opt', type=float, default=2.0,                         
                        help='weighted loss type') 
# I/O
    parser.add_argument('-i','--input',  default='/n/coxfs01/donglai/malis_trans/data/ecs-3d/ecs-gt-4x6x6/',
                        help='input path')
    parser.add_argument('-s','--snapshot',  default='/n/coxfs01/donglai/malis_trans/pytorch_train/w0921/16_8_1e-3_bnL2/iter_16_11499_0.001.pth',
                        help='snapshot path')
    parser.add_argument('-dn','--data-name',  default='im_uint8.h5',
                        help='image data name')
    parser.add_argument('-o','--output', default='result/my-pred.h5',
                        help='output path')
    parser.add_argument('-ln','--label-name',  default='seg-groundtruth2-malis_crop.h5',
                        help='segmentation label')
    parser.add_argument('-lnd','--label-dataset-name',  default='main',
                        help='dataset name in label')

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
    if args.pad_size==0:                                                                            
        model_io_size = np.array(((31,204,204), (3,116,116)));                                      
        aff_suf='';                                                                                 
    elif args.pad_size==1:                                                                          
        model_io_size = np.array(((18,224,224), (18,224,224)));                                     
        aff_suf='2';

    if args.task_opt==0:
        test_data =  np.array(h5py.File(args.input+args.data_name,'rb')['main'],dtype=np.float32)[None,:]/(2.**8)
        out_data_size = model_io_size[0]
    elif args.task_opt==1: # load test prediction 
        test_data =  np.array(h5py.File(args.output,'r')['main'],dtype=np.float32)
        out_data_size = model_io_size[1]
    
    nhood = None
    test_label = None
    if args.task_opt == 1: # do evaluation
        test_label = np.array(h5py.File(args.input+args.label_name[:-3]+'_aff'+aff_suf+'.h5','r')['main'])
        if args.loss_opt==1: # evaluate malis: need segmentation
            nhood = malis_core.mknhood3d()
    test_dataset = VolumeDatasetTest(test_data, test_label, nhood, data_size=test_data.shape[1:],sample_stride=model_io_size[1],
                    out_data_size=out_data_size,out_label_size=model_io_size[1])

    test_loader =  torch.utils.data.DataLoader(
            test_dataset, batch_size= args.batch_size, shuffle=False, collate_fn = np_collate,
            num_workers= args.num_cpu, pin_memory=True)
    
    # pre-allocate torch cuda tensor
    test_var = Variable(torch.zeros(args.batch_size, 1, 31, 204, 204).cuda(), requires_grad=False)
    output_size = [3]+list(test_data.shape[1:]-(model_io_size[0]-model_io_size[1]))
    return test_loader, test_var, output_size, model_io_size

def get_model(args):
    # create model
    num_filter = [int(x) for x in args.num_filter.split(',')]

    opt_arch = [[int(x) for x in y.split('-')] for y in  args.opt_arch.split('@')]
    model = unet3D(filters=num_filter,opt_arch = opt_arch,
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
    test_loader, test_var, output_size, model_io_size = get_data(args)
    
    if not os.path.exists(args.output):
        if not os.path.exists(args.snapshot):
            raise IOException(args.snapshot+" doesn't exist for prediction")
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
                test_var.data[:data[0].shape[0]].copy_(torch.from_numpy(data[0]))
                # forward-backward
                y_pred = model(test_var).data.cpu().numpy()
                for j in range(data[0].shape[0]):
                    pp = data[3][j] 
                    pred[:,pp[0]:pp[0]+model_io_size[1][0],
                          pp[1]:pp[1]+model_io_size[1][1],
                          pp[2]:pp[2]+model_io_size[1][2]] = y_pred[j].copy()
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
        for batch_id, data in enumerate(test_loader):
            if args.loss_opt == 0: # L2
                ww = loss_w.getWeight(data[1])
            elif args.loss_opt == 1: # malis
                ww = loss_w.getWeight(data[0], data[1], data[2])
            loss += weightedMSE_np(data[0], data[1], ww)
            print '%d/%d: avg loss = %.5f' % (batch_id,num_batch,loss/(1+batch_id))
            if batch_id == num_batch-2: # pass the last one                                             
                break 

if __name__ == "__main__":
    main()
