import argparse
# translate model from other packages

# python E_translate.py -cp ../../malis_trans/unet3d/net_deploy_big24.prototxt -cm ../../malis_trans/unet3d/net_iter_150000.caffemodel -o ../../malis_trans/unet3d/net_weight_24_150k -f 24,72,216,648
# python E_translate.py -op 1 -o ../../malis_trans/unet3d/net_weight_24_150k -f 24,72,216,648
def get_args():
    parser = argparse.ArgumentParser(description='Training Model')
    parser.add_argument('-op','--opt', type=float,  default=0,
                        help='caffe->pkl')

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
    parser.add_argument('-o','--output', type=str,  default='../../malis_trans/unet3d/net_weight_10k',
                        help='caffe output')

    parser.add_argument('-f','--num-filter', type=str,  default='20,60,180,540',
                        help='caffe output')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    if args.opt==0: 
        from T_util import caffe2pkl
        caffe2pkl(args.caffe_prototxt, args.caffe_model, args.output+'.pkl')
    elif args.opt==0.1: 
        from T_util import keras2pkl
        keras2pkl(args.keras_model, args.output+'.pkl')
    elif args.opt==0.2: 
        from T_util import pth2pkl
        pth2pkl(args.pth_model, args.output+'.pkl')
    elif args.opt==1: 
        import pickle
        from T_util import load_weights_pkl,save_checkpoint
        from T_model import unet3D
        model = unet3D(filters=[int(x) for x in args.num_filter.split(',')],
                      has_BN=args.has_BN==1)
        ww=pickle.load(open(args.output+'.pkl','rb'))
        load_weights_pkl(model,ww)
        sn =''
        if args.has_BN==1:
            sn+='_bn'
        save_checkpoint(model, args.output+sn+'.pth')

if __name__ == "__main__":
    main()
