import numpy as np
import h5py
import os,sys
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import malis_core
from T_model import unet3D,load_checkpoint
from T_data import VolumeDatasetTest, np_collate
import argparse

# ii=7874;CUDA_VISIBLE_DEVICES=0,1,2,3 python E_test.py -m 1 -s /n/coxfs01/donglai/malis_trans/pytorch_train/w0921/16_8_1e-3_bnL2/iter_16_${ii}_0.001.pth -dn im_uint8.h5 -b 16 -g 4 -c 16 -o result/16_8_1e-3_bnL2/ecs-gt-4x6x6-${ii}-pred.h5

def get_args():
    parser = argparse.ArgumentParser(description='Testing Model')
    # I/O
    parser.add_argument('-i','--input',  default='/n/coxfs01/donglai/malis_trans/data/ecs-3d/ecs-gt-4x6x6/',
                        help='input path')
    parser.add_argument('-s','--snapshot',  default='/n/coxfs01/donglai/malis_trans/pytorch_train/w0921/16_8_1e-3_bnL2/iter_16_11499_0.001.pth',
                        help='snapshot path')
    parser.add_argument('-dn','--data-name',  default='im_uint8.h5',
                        help='image data name')
    parser.add_argument('-o','--output', default='result/my-pred.h5',
                        help='output path')
    # training option
    parser.add_argument('-m','--model', type=int, default=0,
                        help='model type') 
    parser.add_argument('-ps', '--pad-size', type=int, default=0,                                                                   
                        help='pad size')                                                                                            
    parser.add_argument('-pt', '--pad-type', default='constant,0',                                                                  
                        help='pad type')                                                                                            
    parser.add_argument('-bn', '--has-BN', type=int, default=0,                                                                     
                        help='use BatchNorm')
    parser.add_argument('-b','--batch-size', type=int,  default=16,
                        help='batch size')
    parser.add_argument('-g','--num-gpu', type=int,  default=8,
                        help='number of gpu')
    parser.add_argument('-c','--num-cpu', type=int,  default=16,
                        help='number of cpu')
    parser.add_argument('-e', '--batch-end', type=int,  default=-1,
                        help='last batch to test')
    parser.add_argument('-f', '--num-filter', default='24,72,216,648',
                        help='number of filters per layer')
    args = parser.parse_args()
    return args
def get_data(args):
    model_io_size = np.array(((31,204,204), (3,116,116)));
    if args.data_name[-3:] == '.h5':
        test_data =  np.array(h5py.File(args.input+args.data_name,'r')['main'],dtype=np.float32)[None,:]/(2.**8)
    elif args.data_name[-4:] == '.pkl':
        test_data =  np.array(pickle.load(args.input+args.data_name,'rb'),dtype=np.float32)[None,:]/(2.**8)

    test_dataset = VolumeDatasetTest(test_data, data_size=test_data.shape[1:],sample_stride=model_io_size[1],
                                     out_data_size=model_io_size[0],out_label_size=model_io_size[1])

    test_loader =  torch.utils.data.DataLoader(
            test_dataset, batch_size= args.batch_size, shuffle=False, collate_fn = np_collate,
            num_workers= args.num_cpu, pin_memory=True)

    test_var = Variable(torch.zeros(args.batch_size, 1, 31, 204, 204).cuda(), requires_grad=False)
    output_size = [3]+list(test_data.shape[1:]-(model_io_size[0]-model_io_size[1]))
    return test_loader, test_var, output_size, model_io_size

def get_model(args):
    # create model
    num_filter = [int(x) for x in args.num_filter.split(',')]
    model = unet3D(has_BN=True,filters=num_filter,
                  pad_vgg_size = args.pad_size, pad_vgg_type = args.pad_type)
    model.cuda()
    # load parameter
    if args.num_gpu>1: model = nn.DataParallel(model, range(args.num_gpu)) 
    cp = load_checkpoint(args.snapshot, args.num_gpu)
    model.load_state_dict(cp['state_dict'])
    return model

def main():
    args = get_args()
    if not os.path.exists(args.output[:args.output.rfind('/')]):
        os.makedirs(args.output[:args.output.rfind('/')])

    if os.path.exists(args.output):
        print "already done:",args.output
        return
    print '1. setup data'
    test_loader, test_var, output_size, model_io_size = get_data(args)

    print '2. load model'
    model = get_model(args)

    print '3. start testing'
    model.eval()
    st=time.time()
    # pre-allocate torch cuda tensor
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

if __name__ == "__main__":
    main()
