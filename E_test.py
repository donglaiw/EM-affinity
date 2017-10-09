import numpy as np
import h5py
import os,sys
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import malis_core
from T_model import unet3D
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
    parser.add_argument('-b','--batch-size', type=int,  default=16,
                        help='batch size')
    parser.add_argument('-g','--num-gpu', type=int,  default=8,
                        help='number of gpu')
    parser.add_argument('-c','--num-cpu', type=int,  default=16,
                        help='number of cpu')

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    if not os.path.exists(args.output[:args.output.rfind('/')]):
        os.makedirs(args.output[:args.output.rfind('/')])

    print '1. setup data'
    train_size = np.array(((31,204,204), (3,116,116)));
    if args.data_name[-3:] == '.h5':
        train_data =  np.array(h5py.File(args.input+args.data_name,'r')['main'],dtype=np.float32)[None,:]/(2.**8)
    elif args.data_name[-4:] == '.pkl':
        train_data =  np.array(pickle.load(args.input+args.data_name,'rb'),dtype=np.float32)[None,:]/(2.**8)

    test_dataset = VolumeDatasetTest(train_data, data_size=train_data.shape[1:],sample_stride=train_size[1],
                                     out_data_size=train_size[0],out_label_size=train_size[1])

    test_loader =  torch.utils.data.DataLoader(
            test_dataset, batch_size= args.batch_size, shuffle=False, collate_fn = np_collate,
            num_workers= args.num_cpu, pin_memory=True)

    print '2. load model'
    # create model
    if args.model == 0: 
        model = unet3D()
    elif args.model == 1: 
        model = unet3D(is_batchnorm=True)
    model.cuda()
    model.eval()
    # load parameter
    if args.num_gpu>1: model = nn.DataParallel(model, range(args.num_gpu)) 
    cp = torch.load(args.snapshot)
    if args.num_gpu==1 and cp['state_dict'].keys()[0][:7]=='module.':
        # modify the saved model for single GPU
        for k,v in cp['state_dict'].items():
            cp['state_dict'][k[7:]] = v
            cp['state_dict'].pop(k,None)
    model.load_state_dict(cp['state_dict'])

    print '3. start testing'
    st=time.time()
    # pre-allocate torch cuda tensor
    x = Variable(torch.zeros(args.batch_size, 1, 31, 204, 204).cuda(), requires_grad=False)
    num_batch = test_loader.__len__()
    output_size = [3]+list(train_data.shape[1:]-(train_size[0]-train_size[1]))
    pred = np.zeros(output_size,dtype=np.float32)
    for batch_id, data in enumerate(test_loader):
        x.data[:data[0].shape[0]].copy_(torch.from_numpy(data[0]))
        # forward-backward
        y_pred = model(x).data.cpu().numpy()
        for j in range(data[0].shape[0]):
            pp = data[3][j] 
            pred[:,pp[0]:pp[0]+train_size[1][0],
                  pp[1]:pp[1]+train_size[1][1],
                  pp[2]:pp[2]+train_size[1][2]] = y_pred[j][0].copy()
        print "finish batch: [%d/%d] " % (batch_id, num_batch)
        sys.stdout.flush()

    et=time.time()
    print 'testing time: '+str(et-st)+' sec'
    if args.output is not None:
        print '4. start saving'
        outhdf5 = h5py.File(args.output, 'w')
        outdset = outhdf5.create_dataset('main', pred.shape, np.float32, data=pred)
        outhdf5.close()

if __name__ == "__main__":
    main()
