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
from em.quant.quant_model import unet3D_m1_quant

from test_affinity import get_data

def get_args():
    parser = argparse.ArgumentParser(description='Training Model')
    parser.add_argument('-opt','--opt',  type=int, default=0,
                        help='script options')
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
    parser.add_argument('-rs','--rescale-skip', type=int,  default=0,
                        help='if rescale skip connection')
    # train data 
    parser.add_argument('-i','--input',  default='',
                        help='input path')
    parser.add_argument('-dn','--data-name',  default='im_uint8.h5',
                        help='image data name')
    parser.add_argument('-dnd','--data-dataset-name',  default='main',
                        help='dataset name in data')
    parser.add_argument('-dc','--data-color-opt', type=int,  default=2,
                        help='data color aug type')
    parser.add_argument('-b','--batch-size', type=int, default=1,
                        help='batch size')
    parser.add_argument('-c','--num-cpu', type=int,  default=16,
                        help='number of cpu')
    parser.add_argument('-nb','--batch-num', type=int, default=1000, 
                        help='number of batches to infer the scaling factor')
    parser.add_argument('-di','--do-issac', type=int,  default=0,
                        help='using issac int8 quantization')
    parser.add_argument('-ds','--do-save', type=int,  default=1,
                        help='whether to save the model')

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

    if args.opt == 0: 
        print 'quantize model: '+args.snapshot
        print '1. quantize weight'
        unet_ref = torch.load(args.snapshot)
        if args.do_issac==1: # issac quant
            pass
        elif args.do_issac==0: #  simulation
            state_dict = unet_ref['state_dict']
            state_dict_quant = quantize_weight(state_dict, bits=args.param_bits, do_bias=args.quant_bias==1, overflow_rate=args.overflow_rate, quant_method=args.quant_method)

            print '2. quantize feature' 
            # modify model: add extra layers to quantize feature
            model0, model_io_size = get_model(args, state_dict_quant)
            quantize_feat(model0, bits=args.param_bits, overflow_rate=args.overflow_rate, quant_method=args.quant_method, counter=args.batch_num)
            
            print '3. infer quantization param'
            model = model0
            test_var = Variable(torch.zeros(args.batch_size, 1, model_io_size[0][0], model_io_size[0][1], model_io_size[0][2]).cuda(), requires_grad=False)
            if args.input=='': # use random data
                for batch_id in range(args.batch_num):
                    test_var.data.copy_(torch.rand(test_var.data.size()))
                    y_pred = model(test_var)
            else: # use provided data
                test_loader, output_size, sample_num = get_data(args, model_io_size)
                for batch_id, data in enumerate(test_loader):
                    if batch_id % 10 ==0:
                        print 'process batch [%d/%d]' % (batch_id, args.batch_num)
                    test_var.data.copy_(torch.from_numpy(data[0]))
                    y_pred = model(test_var)
                    if batch_id == args.batch_num-1:
                        break
            if args.do_save == 1:
                torch.save(model, args.output+'.pth')
    elif args.opt == 1 and os.path.exists(args.output+'.pth'): 
        print 'benchmark quantization result'
        if args.rescale_skip != 0:
            model = unet3D_m1_quant(model0, args.param_bits, args.quant_method, args.overflow_rate, -1) 
            model.set_rescale_skip_scale() # consolidate the scale
        torch.manual_seed(0)
        test_var.data.copy_(torch.rand(test_var.data.size()))
        y_sc = model(test_var)
        model_ref, model_io_size = get_model(args, state_dict)
        y_ref = model_ref(test_var)
        error = torch.norm(y_ref - y_sc)/torch.norm(y_ref)
        print 'Error:',args.quant_method, error.data[0]
    elif args.opt == 2:
        if not os.path.exists(args.output+'.pth'): 
            raise IOError('Not found: '+args.output+'.pth')
        print 'add rescale-skip to the quantized model'
        model0 = torch.load(args.output+'.pth')
        model = unet3D_m1_quant(model0, args.param_bits, args.quant_method, args.overflow_rate, -1) 
        model.set_rescale_skip_scale() # consolidate the scale
        import pdb; pdb.set_trace()
        torch.save(model, args.output+'_rs.pth')

if __name__ == "__main__":
    main()
