import numpy as np
import os, sys, time, argparse, h5py

import torch
import torch.nn as nn
from torch.autograd import Variable

import os, sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from em.model.io import load_checkpoint, pth2issac
from em.model.unet import unet3D, unet3D_m1
from em.model.loss import weightedMSE_np, malisWeight, labelWeight
from em.data.volumeData import VolumeDatasetTest, np_collate
from em.data.io import getVar, getData, getLabel, getDataAug, cropCentralN, setPred
from em.util.misc import writeh5, writetxt

def get_args():
    parser = argparse.ArgumentParser(description='Testing Model')

    parser.add_argument('-m','--model-id',  type=int, default=0,
                        help='model id')
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
    parser.add_argument('-sn','--seg-name',  default='seg-groundtruth2-malis_crop.h5',
                        help='segmentation label')
    parser.add_argument('-snd','--seg-dataset-name',  default='main',
                        help='dataset name in label')
    parser.add_argument('-ss','--sample-stride', default='', #1@1@1
                        help='sample stride')

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
    parser.add_argument('-pok','--pool-kernel', type=str,  default='1,2,2',
                        help='pool kernel')
    parser.add_argument('-pos','--pool-stride', type=str,  default='1,2,2',
                        help='pool stride')
    parser.add_argument('-di','--do-issac', type=int,  default=0,
                        help='using issac int8 quantization')
    # computation option
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

def init(args):
    model_io_size = np.array([[int(x) for x in args.model_input.split(',')],
                              [int(x) for x in args.model_output.split(',')]])

    # pre-allocate torch cuda tensor
    test_var = Variable(torch.zeros(args.batch_size, 1, model_io_size[0][0], model_io_size[0][1], model_io_size[0][2]).cuda(), requires_grad=False)
    return model_io_size, test_var


def get_data(args, model_io_size, test_var):
    # task_opt=0: prediction
    # task_opt=1: error per slice
    # task_opt=2: error per input volume

    suf_aff = '_aff'+args.opt_param
    color_clip = [(0.05,0.95), (0.05,0.95), None][args.data_color_opt]
    
    do_shuffle = False # test in serial
    sample_stride = model_io_size[1] if args.sample_stride=='' else [int(x) for x in args.sample_stride.split('@')] # no overlap
    extra_pad = 0 # no need to pad
    output_size = None
    if args.task_opt in [0,0.1]: 
        # load test data
        dirName = args.input.split('@')
        test_data = getData(dirName, args.data_name, args.data_dataset_name)
        test_label = None

        out_data_size = model_io_size[0]
        do_seg = False
        # prediction output size
        output_size = [[3]+list(test_data[x].shape[1:]-(model_io_size[0]-model_io_size[1])) for x in range(len(test_data))]
        if args.task_opt in [0.1]:
            do_shuffle = True
    elif args.task_opt in [1,2]: 
        # load test affinity prediction 
        dirName = args.output.split('@')
        test_data =  getLabel(dirName, '','', args.data_dataset_name)
        # load gt affinity
        dirName_l = args.input.split('@')
        test_label =  getLabel(dirName_l, args.seg_name, suf_aff, args.seg_dataset_name)
        test_data, test_label = cropCentralN(test_data, test_label, [0,0,0], False)
        
        out_data_size = model_io_size[1]
        do_seg = args.loss_opt==1 # evaluate malis: need segmentation
            
    color_scale, color_shift, color_clip, rot = getDataAug('test', args.data_color_opt)

    test_dataset = VolumeDatasetTest(test_data, test_label, do_seg,
            clip=color_clip, sample_stride=sample_stride,
            extra_pad=extra_pad, out_data_size=out_data_size, out_label_size=model_io_size[1])
    test_loader =  torch.utils.data.DataLoader(
            test_dataset, batch_size= args.batch_size, shuffle=do_shuffle, collate_fn = np_collate,
            num_workers= args.num_cpu, pin_memory=True)
    
    batch_num = [np.ceil(x/float(args.batch_size)) for x in test_dataset.sample_num]

    if output_size is None:
        output_size = test_dataset.sample_size

    if args.do_issac==1:
        # need to initiate the test_var with real data
        tmp_data = np.zeros([args.batch_size,1]+list(model_io_size[0]))
        tmp_ind = np.floor(np.random.random(args.batch_size)*test_dataset.__len__()).astype(int)
        for i in range(args.batch_size):
            tmp_data[i,:] = test_dataset.__getitem__(tmp_ind[i])[0]
        test_var.data.copy_(torch.from_numpy(tmp_data))
    return test_loader, output_size, batch_num

def get_model(args, test_var):
    # create model
    if args.model_id == 0:
        num_filter = [int(x) for x in args.num_filter.split(',')]
        opt_arch = [[int(x) for x in y.split('-')] for y in  args.opt_arch.split('@')]
        opt_param = [[int(x) for x in y.split('-')] for y in  args.opt_param.split('@')]
        pool_kernel=tuple([int(x) for x in args.pool_kernel.split(',')])
        pool_stride=tuple([int(x) for x in args.pool_stride.split(',')])
        if args.do_issac==1:
            model = unet3D_m1(filters=num_filter)
        else:
            model = unet3D(filters=num_filter,opt_arch = opt_arch, opt_param = opt_param,
                           has_BN = args.has_BN==1, has_dropout = args.has_dropout,
                           pool_kernel = pool_kernel, pool_stride = pool_stride,
                           pad_size = args.pad_size, pad_type= args.pad_type)
        # load parameter
        cp = load_checkpoint(args.snapshot, 1)
        model.load_state_dict(cp['state_dict'])
    elif args.model_id == 1:#load model directly
        model = torch.load(args.snapshot)

    if args.num_gpu>0:
        model.cuda()
        if args.do_issac==1:
            print 'do issac optimization'
            # test_var.data.copy_(torch.rand(test_var.data.size()))
            model = pth2issac(model).fuse().quantize(test_var)

    if args.num_gpu>1: model = nn.DataParallel(model, range(args.num_gpu)) 

    return model

def main():
    args = get_args()
    if not os.path.exists(args.output[:args.output.rfind('/')]):
        os.makedirs(args.output[:args.output.rfind('/')])

    print '1. setup data'
    model_io_size, test_var = init(args)
    test_loader, output_size, batch_num = get_data(args, model_io_size, test_var)
    
    if not os.path.exists(args.output):
        if not os.path.exists(args.snapshot):
            raise ValueError(args.snapshot+" doesn't exist for prediction")
        else:
            print '-- start prediction --'
            print '2. load model'
            model = get_model(args, test_var)

            print '3. start testing'
            model.eval()
            st0 = time.time()
            st = st0
            # multiple dataset
            print 'start dataset: 1/'+str(len(batch_num))
            did = 0
            num_pre=0
            num_total = test_loader.__len__() 
            pred = np.zeros(output_size[0], dtype=np.float32)
            for batch_id, data in enumerate(test_loader):
                # prediction 
                num_b = data[3].shape[0]
                test_var.data[:num_b].copy_(torch.from_numpy(data[0][:num_b]))
                y_pred = model(test_var).data.cpu().numpy()

                # put into pred
                num_bd = np.count_nonzero(data[3][:,0]==did) 
                setPred(pred, y_pred, model_io_size[1], data[3], num_bd)
                print "finish batch: [%d/%d/%d] " % (did, batch_id-num_pre, batch_num[did])
                if num_bd == 0 or num_b != num_bd or batch_id==num_total-1: 
                    # need to save previous prediction
                    print 'save dataset: '+str(did+1)+'/'+str(len(batch_num))
                    writeh5(args.output, 'main', pred)
                    et=time.time()
                    print 'time: '+str(et-st)+' sec'
                    st = time.time()
                    if batch_id < num_total-1: #start new dataset
                        did+=1
                        num_pre=batch_id
                        pred = np.zeros(output_size[did], dtype=np.float32)
                        setPred(pred, y_pred, model_io_size[1], data[3], num_b, num_bd)
                        print "finish batch: [%d/%d/%d] " % (did, batch_id-num_pre, batch_num[did])
                        sys.stdout.flush()

                if batch_id == args.batch_end:
                    # early stop for debug
                    pp = data[3][-1]
                    pred=pred[:,:pp[1]+model_io_size[1][0]]
                    writeh5(args.output, 'main', pred)
                    break
            et=time.time()
            print 'total time: '+str(et-st0)+' sec'
    else:
        print '-- start evaluate prediction loss'
        print '2. load loss'
        conn_dims = [args.batch_size, 3]+list(model_io_size[1])                                                     
        if args.loss_opt==0: # l2 
            loss_w = labelWeight(conn_dims, args.loss_weight_opt)
        elif args.loss_opt==1: # malis
            loss_w = malisWeight(conn_dims, args.loss_weight_opt) 
        print '3. start evaluation'
        for did in range(len(batch_num)):
            print 'test dataset: '+str(did)+'/'+str(len(batch_num))
            num_batch = batch_num[did]
            loss=0;
            if args.task_opt==1: # avg test error
                out = ''
                for batch_id, data in enumerate(test_loader):
                    if args.loss_opt == 0: # L2
                        ww = loss_w.getWeight(data[1])
                    elif args.loss_opt == 1: # malis
                        ww = loss_w.getWeight(data[0], data[1], data[2])
                    loss += weightedMSE_np(data[0], data[1], ww)
                    print '%d/%d: avg loss = %.5f' % (batch_id,num_batch,loss/(1+batch_id))
                    out+='%d/%d: avg loss = %.5f' % (batch_id,num_batch,loss/(1+batch_id)) + '\n'
                writetxt(args.output.replace('.h5','_err.txt'), out)
            elif args.task_opt==2: # heatmap test error
                pred = np.zeros(output_size[did], dtype=np.float32)
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
