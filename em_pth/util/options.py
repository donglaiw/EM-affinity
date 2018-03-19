import argparse
from em_data.options import *

def optResource(parser):
    parser.add_argument('-g','--num-gpu', type=int,  default=1,
                        help='number of gpu')
    parser.add_argument('-c','--num-cpu', type=int,  default=1,
                        help='number of cpu')
    parser.add_argument('-b','--batch-size', type=int,  default=1,
                        help='batch size')

def optTrain(parser):
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
                        help='pre-train number of epoch')
    parser.add_argument('-es','--snapshot',  default='',
                        help='pre-train snapshot path')


def optModel(parser):
    parser.add_argument('-m','--model-id',  type=float, default=0,
                        help='model id')
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

def optIO(parser, mode='train'):
    if mode == 'train':
        parser.add_argument('-dti', '--train-img', default='',
                            help='input train image')
        parser.add_argument('-dts', '--train-seg', default='',
                            help='input train segmentation')
        parser.add_argument('-dvi', '--val-img', default='',
                            help='input validataion image')
        parser.add_argument('-dvs', '--val-seg', default='',
                            help='input validation segmentation')
        # if h5
        parser.add_argument('-dtid','--train-img-name',  default='main',
                            help='dataset name in train image')
        parser.add_argument('-dtsd','--train-seg-name',  default='main',
                            help='dataset name in train segmentation')
        parser.add_argument('-dvid','--val-img-name',  default='main',
                            help='dataset name in validation image')
        parser.add_argument('-dvsd','--val-seg-name',  default='main',
                            help='dataset name in validation segmentation')

    elif mode=='test':
        parser.add_argument('-dei', '--test-img', default='',
                            help='input test image')
        parser.add_argument('-des', '--test-seg', default='',
                            help='input test segmentation')
        # if h5
        parser.add_argument('-deid','--test-img-name',  default='main',
                            help='dataset name in test image')
        parser.add_argument('-desd','--test-seg-name',  default='main',
                            help='dataset name in test segmentation')
    parser.add_argument('-o','--output', default='result/train/',
                        help='output path')

def optDataAug(parser):
    # reduce the number of input arguments by stacking into one string
    parser.add_argument('-ao','--aug-opt', type=str,  default='1@-1@0@5',
                        help='data aug type')
    parser.add_argument('-apw','--aug-param-warp', type=str,  default='15@3@1.1@0.1',
                        help='data warp aug parameter')
    parser.add_argument('-apc','--aug-param-color', type=str,  default='0.95,1.05@-0.15,0.15@0.5,2@0,1',
                        help='data color aug parameter')

