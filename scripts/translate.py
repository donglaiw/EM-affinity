import argparse
import pickle

from em.model.io import caffe2pkl, keras2pkl, pth2pkl, load_weights_pkl,load_weights_pkl_m1, save_checkpoint
from em.model.unet import unet3D,unet3D_m1

# translate model from other packages
def get_args():
    parser = argparse.ArgumentParser(description='Training Model')
    parser.add_argument('-op','--opt', type=float,  default=0,
                        help='caffe->pkl')
    parser.add_argument('-w','--weight', type=str,  default='',
                        help='weight pkl file')

    parser.add_argument('-cp','--caffe-prototxt', type=str,  default='../../malis_trans/unet3d/net_deploy_big20.prototxt',
                        help='caffe model prototxt')
    parser.add_argument('-cm','--caffe-model', type=str,  default='/n/coxfs01/fgonda/experiments/3d/ecs-3d/affinity_20_3/net_iter_10000.caffemodel',
                        help='caffe model')
    parser.add_argument('-bn', '--has-BN', type=int, default=0,
                        help='use BatchNorm')
    parser.add_argument('-km','--keras-model', type=str,  default='/n/coxfs01/fgonda/experiments/3d/ecs-3d/affinity_20_3/net_iter_10000.caffemodel',
                        help='caffe model')
    parser.add_argument('-pm','--pth-model', type=str,  default='',
                        help='caffe model')
    parser.add_argument('-o','--output', type=str,  default='',
                        help='output')

    parser.add_argument('-f','--num-filter', type=str,  default='24,72,216,648',
                        help='caffe output')
    parser.add_argument('-pok','--pool-kernel', type=str,  default='1,2,2',
                        help='pool kernel')
    parser.add_argument('-pos','--pool-stride', type=str,  default='1,2,2',
                        help='pool stride')
    args = parser.parse_args()
    if args.output=='':
        args.output = args.weight
    return args

def main():
    args = get_args()
    if args.opt==0: 
        caffe2pkl(args.caffe_prototxt, args.caffe_model, args.output+'.pkl')
    elif args.opt==0.1: 
        keras2pkl(args.keras_model, args.output+'.pkl')
    elif args.opt==0.2: 
        pth2pkl(args.pth_model, args.output+'.pkl')
    elif args.opt==1: 
        model = unet3D(filters=[int(x) for x in args.num_filter.split(',')],
                      pool_kernel=[int(x) for x in args.pool_kernel.split(',')],
                      pool_stride=[int(x) for x in args.pool_stride.split(',')],
                      has_BN=args.has_BN==1)
        ww=pickle.load(open(args.weight+'.pkl','rb'))
        load_weights_pkl(model,ww)
        sn =''
        if args.has_BN==1:
            sn+='_bn'
        save_checkpoint(model, args.output+sn+'.pth')
    elif args.opt==1.1: 
        model = unet3D_m1(filters=[int(x) for x in args.num_filter.split(',')])
        ww=pickle.load(open(args.weight+'.pkl','rb'))
        load_weights_pkl_m1(model,ww)
        save_checkpoint(model, args.output+'.pth')


if __name__ == "__main__":
    main()
