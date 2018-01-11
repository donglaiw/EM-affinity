import os, sys, time, argparse, h5py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from em.util.misc import resizeh5, writeh5, writetxt, runBash, pred_reorder
import numpy as np

# call external library
from subprocess import check_output, call

def get_args():
    parser = argparse.ArgumentParser(description='Testing Model')
    # .h5 files
    parser.add_argument('-p', '--pred-pref',  default='',
                        help='prediction file')
    parser.add_argument('-s', '--seg-gt',  default='',
                        help='prediction file')
    parser.add_argument('-sn', '--seg-gt-dn',  default='stack',
                        help='prediction dataset name')

    # env
    parser.add_argument('-ae', '--ag-env', default='nproof',
                        help='env for agglomeration')

    parser.add_argument('-ze', '--zw-env', default='/n/home04/donglai/.conda/envs/zwatershed/bin/',
                        help='env for agglomeration')

    parser.add_argument('-tt', '--ag-do-train', type=int, default=0,
                        help='do agglomeration train')
    parser.add_argument('-ss', '--do-seg', type=int, default=1,
                        help='do seg')
    # lib path
    parser.add_argument('-e', '--eval', default='/n/coxfs01/donglai/malis_trans/vi_test/',
                        help='library folder')
    parser.add_argument('-dz', '--do-z', default='zwatershed_script/generate_watershed_affinity_nomask.py',
                        help='zwatershed code')
    parser.add_argument('-dv', '--do-v', default='comparision/PixelPred2Seg/comparestacks.py',
                        help='VI code')
    parser.add_argument('-dar', '--ag-train', default='agglomeration/Neuroproof_minimal/build/NeuroProof_stack_learn',
                        help='code to train agglomeration')
    parser.add_argument('-dae', '--ag-test', default='agglomeration/Neuroproof_minimal/build/NeuroProof_stack',
                        help='code to test agglomeration')

    # param
    parser.add_argument('-r', '--resize', type=int,  default=1,
                        help='resize prediction')
    parser.add_argument('-zn', '--zw-name', default='stack',
                        help='dataset name')
    parser.add_argument('-zt', '--zw-thres', type=str, default=10000,
                        help='zwatershed thres')
    parser.add_argument('-zr', '--zw-rm', type=str, default='0',
                        help='zwatershed remove small segments')
    parser.add_argument('-dzE', '--do-zwaterEval', type=int, default=0,
                        help='if evaluate zwatershed')
    parser.add_argument('-ap', '--ag-param', default='',
                        help='env for agglomeration') # 2,3,1,0.15,0
    parser.add_argument('-atr', '--ag-train-pref',  default='',
                        help='name of the train data')
    parser.add_argument('-ate', '--ag-test-pref',  default='',
                        help='name of the test data')
    parser.add_argument('-cs', '--crop-suf',  default='',
                        help='suffix for crop')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    root_out = args.pred_pref[:args.pred_pref.rfind('/')+1]
    root_pref = args.pred_pref[args.pred_pref.rfind('/')+1:]

    # step 1: source activate zwatershed
    print '1. prediction'
    predUp = args.pred_pref+'x'+str(args.resize)+'.h5'
    predReorder = predUp[:-3] + '-zyxc.h5'
    print '\t 1.1 upsample'
    if args.resize != 1: 
        if not os.path.exists(predUp):
            resizeh5(args.pred_pref+'.h5', predUp, 'main', ratio=(args.resize, args.resize), interp=1)
    else:
        predUp = args.pred_pref+'.h5'
        predReorder = predUp[:-3] + '-zyxc.h5'
    print '\t 1.2 re-order'
    if not os.path.exists(predReorder):
        pred_reorder(predUp, 'main', root_out)
    
    if args.do_seg == 0:
        sys.exit(0)

    print '2. initial zwatershed'
    zw_seg = args.pred_pref + '-waterz_'+args.zw_thres+'.h5'
    zw_segEval = zw_seg[:-3]+'_eval.txt'
    zw_segRm = zw_seg[:-3]+'_rm'+args.zw_rm+'.h5'

    print '\t 2.1 segmentation'
    if not os.path.exists(zw_seg):
        call([args.zw_env+'python', args.eval+args.do_z, predUp, 'main', zw_seg, args.zw_thres])
    print '\t 2.2 remove small seg'
    if int(args.zw_rm)>0 and not os.path.exists(zw_segRm):
        z0 = np.array(h5py.File(zw_seg)[args.zw_name])
        z0F = z0.reshape((-1))
        z0_id = np.unique(z0)
        num = np.histogram(z0F,np.append(z0_id,z0_id[-1]+1))
        bad_id = z0_id[np.where(num[0] < int(args.zw_rm))[0]]
        z0F[np.in1d(z0F, bad_id)] = 0
        print 'initial #seg:',len(z0_id),'left #seg:',len(z0_id)-len(bad_id)
        writeh5(zw_segRm, args.zw_name, z0)
    print '\t 2.3 evaluation'
    if args.do_zwaterEval==1 and not os.path.exists(zw_segEval):
        vi = check_output([args.zw_env+'python', args.eval+args.do_v, '--stack1', zw_seg, '--stackbase', args.seg_gt, '--dilate1', '1', '--dilatebase', '1', '--relabel1', '--relabelbase', '--filtersize', '100', '--anisotropic'])
        writetxt(zw_segEval, vi)

    if args.ag_param!='':
        print '3. agglomeration'
        ag_str, ag_itr, ag_alg, ag_thr, ag_reg = args.ag_param.split(',')
        ag_pref = args.ag_train_pref 
        ag_suf_train = ag_pref+'_st'+ag_str+'_itr'+ag_itr+'_'+args.zw_thres+'_rm'+args.zw_rm+args.crop_suf
        ag_suf_test = ag_suf_train+'_'+ag_alg+'_'+ag_thr+'_'+ag_reg+'_'+args.ag_test_pref+args.crop_suf
        
        ag_train = root_out+'ag_train_'+ag_suf_train+'.h5'
        ag_test = root_out+'ag_test_'+ag_suf_test+'.h5'
        ag_testEval = ag_test[:-3]+'_eval.txt'
        c0 = '#!/bin/bash \n'
        c1 = 'source activate '+args.ag_env+'\n'
        c2 = 'export LD_LIBRARY_PATH="/n/home04/donglai/.conda/envs/nproof/lib/:$LD_LIBRARY_PATH"\n'

        print '\t 3.1 train RF'
        # need change env: require different versions of some libraries ...
        # can't change env in python directly ..
        # in case of crop
        zw_segRmC = zw_segRm[:-3]+args.crop_suf+'.h5'
        predReorderC = predReorder[:-3]+args.crop_suf+'.h5'
        if args.ag_do_train==1 and not os.path.exists(ag_train):
            c3 = args.eval+args.ag_train+' -watershed '+zw_segRmC+' stack -prediction '+predReorderC+' stack -classifier '+ag_train+' -groundtruth '+args.seg_gt+' '+args.seg_gt_dn+' -strategy '+ag_str+' -iteration '+ag_itr+' -nomito\n'   
            runBash(c0+c1+c2+c3)
        print ag_train, args.ag_test_pref
        if os.path.exists(ag_train) and args.ag_test_pref!='' :
            print '\t 3.2 test RF'
            if not os.path.exists(ag_test):
                # pad extra character for ag_test, due to the agglomeration code..
                c3 = args.eval+args.ag_test+' -watershed '+zw_segRmC+' stack -prediction '+predReorderC+' stack -classifier '+ag_train+' -output '+ag_test[:-2]+'.h5 stack -algorithm '+ag_alg+' -threshold '+ag_thr+' -min_region_sz '+ag_reg+' -nomito\n'
                runBash(c0+c1+c2+c3)

            print '\t 3.3 evaluation'
            if not os.path.exists(ag_testEval):
                # print ' '.join([args.zw_env+'python', args.eval+args.do_v, '--stack1', ag_test, '--stackbase', args.seg_gt, '--dilate1', '1', '--dilatebase', '1', '--relabel1', '--relabelbase', '--filtersize', '100', '--anisotropic'])
                vi = check_output([args.zw_env+'python', args.eval+args.do_v, '--stack1', ag_test, '--stackbase', args.seg_gt, '--dilate1', '1', '--dilatebase', '1', '--relabel1', '--relabelbase', '--filtersize', '100', '--anisotropic'])
                writetxt(ag_testEval, vi)
