# quantize model: float32 -> int8

import argparse
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import os, sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from em.model.unet import unet3D_m1
from em.model.io import load_checkpoint,save_checkpoint
from em.quant.quant_core import quantize_weight, quantize_feat

from test_affinity import get_data

def get_args():
    parser = argparse.ArgumentParser(description='Training Model')
    # get model
    parser.add_argument('-m','--model-id',  type=int, default=1,
                        help='model id')
    parser.add_argument('-s','--snapshot',  default='/n/coxfs01/donglai/malis_trans/pytorch_train/w0921/16_8_1e-3_bnL2/iter_16_11499_0.001.pth',
                        help='snapshot path')
    parser.add_argument('-f','--num-filter', type=str,  default='24,72,216,648',
                        help='filter numbers')
    parser.add_argument('-pb','--param-bits', type=int, default=8, 
                        help='bit-width for parameters')
    parser.add_argument('-qb','--quant-bias', type=int, default=0, 
                        help='quantize bias')
    parser.add_argument('-qm','--quant-method', default='linear', 
                        help='linear|minmax|log|tanh')
    parser.add_argument('-o','--output',  default='',
                        help='output name')
    parser.add_argument('--overflow_rate', type=float, default=0.0, 
                        help='overflow rate')
    # train data 
    parser.add_argument('-i','--input',  default='/n/coxfs01/donglai/malis_trans/data/ecs-3d/ecs-gt-4x6x6/',
                        help='input path')
    parser.add_argument('-dn','--data-name',  default='im_uint8.h5',
                        help='image data name')
    parser.add_argument('-dnd','--data-dataset-name',  default='main',
                        help='dataset name in data')
    parser.add_argument('-dc','--data-color-opt', type=int,  default=2,
                        help='data color aug type')
    parser.add_argument('-b','--batch-size', type=int, default=10,
                        help='batch size')
    parser.add_argument('-g','--num-gpu', type=int,  default=8,
                        help='number of gpu')
    parser.add_argument('-c','--num-cpu', type=int,  default=16,
                        help='number of cpu')
    parser.add_argument('-nb','--batch-num', type=int, default=1000, 
                        help='number of batches to infer the scaling factor')
    args = parser.parse_args()

    # for test_data loader
    args.opt_param = '0@0@0@0'
    args.sample_stride = '1@1@1'
    args.task_opt = 0.1 # random shuffle
    args.data_color_opt = 2 # no aug 
    if len(args.output)<4 or args.output[-4:] != '.pth':
        if args.output == '':
            args.output = args.snapshot[:-4]
        args.output += '_p'+str(args.param_bits)+'_'+args.quant_method+'_v'+str(args.batch_num*args.batch_size)
    return args

def get_model(args, state_dict):
    if args.model_id==1:
        model = unet3D_m1(filters=[int(x) for x in args.num_filter.split(',')])
        model_io_size = np.array([[31,204,204],[3,116,116]])
    model.load_state_dict(state_dict)
    model.cuda()
    return model, model_io_size

def main():
    args = get_args()

    print '1. quantize weight'
    state_dict = torch.load(args.snapshot)['state_dict']
    state_dict_quant = quantize_weight(state_dict, bits=args.param_bits, do_bias=args.quant_bias==1, overflow_rate=args.overflow_rate, quant_method=args.quant_method)

    print '2. quantize feature' 
    # modify model: add extra layers to quantize feature
    model, model_io_size = get_model(args, state_dict_quant)
    quantize_feat(model, bits=args.param_bits, overflow_rate=args.overflow_rate, quant_method=args.quant_method, counter=args.batch_num)
    if args.num_gpu>1: model = nn.DataParallel(model, range(args.num_gpu)) 
    
    print '3. infer quantization param'
    test_loader, output_size, sample_num = get_data(args, model_io_size)
    test_var = Variable(torch.zeros(args.batch_size, 1, model_io_size[0][0], model_io_size[0][1], model_io_size[0][2]).cuda(), requires_grad=False)

    for batch_id, data in enumerate(test_loader):
        if batch_id % 10 ==0:
            print 'process batch [%d/%d]' % (batch_id, args.batch_num)
        test_var.data.copy_(torch.from_numpy(data[0]))
        y_pred = model(test_var)
        if batch_id == args.batch_num-1:
            break

    # 5. save model
    torch.save(model, args.output+'.pth')

if __name__ == "__main__":
    main()
