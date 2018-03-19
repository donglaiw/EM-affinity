# utility function to load data
# train-L2: img+label
# train-malis: img+label+seg
# test-pred: img
# test-err-L2: img+label
# test-err-malis: img+label+seg
import numpy as np
import h5py
import os
from ..util.misc import writeh5, segToAffinity

def setPred(pred, y_pred, pred_sz, pos, lt=None, st=0, pred_ww=None, ww=None):
    if lt is None:
        lt = pos.shape[0]
    if pred_ww is None: # simply assign
        for j in range(st,lt):
            pp = pos[j] 
            pred[:,pp[1]:pp[1]+pred_sz[0],
                  pp[2]:pp[2]+pred_sz[1],
                  pp[3]:pp[3]+pred_sz[2]] = y_pred[j].copy()
    else:
        for j in range(st,lt):
            pp = pos[j] 
            pred[:,pp[1]:pp[1]+pred_sz[0],
                  pp[2]:pp[2]+pred_sz[1],
                  pp[3]:pp[3]+pred_sz[2]] += y_pred[j]*ww
            pred_ww[:,pp[1]:pp[1]+pred_sz[0],
                  pp[2]:pp[2]+pred_sz[1],
                  pp[3]:pp[3]+pred_sz[2]] += ww


def getVar(batch_size, model_io_size, do_input=[True, False, False]):
    import torch
    from torch.autograd import Variable
    input_vars = [None] *3  
    # img
    input_vars[0] = Variable(torch.zeros(batch_size, 1, model_io_size[0][0], model_io_size[0][1], model_io_size[0][2]).cuda(), requires_grad=False)
    # label
    if do_input[1]:
        input_vars[1] = Variable(torch.zeros(batch_size, 3, model_io_size[1][0], model_io_size[1][1], model_io_size[1][2]).cuda(), requires_grad=False)
    if do_input[2]:
        input_vars[2] = Variable(torch.zeros(batch_size, 3, model_io_size[1][0], model_io_size[1][1], model_io_size[1][2]).cuda(), requires_grad=False)
    return input_vars


def getImg(img_name, img_dataset_name):
    d_img = [None]*len(img_name)
    for i in range(len(img_name)):
        if img_name[i][-3:] == '.h5' or img_name[i][-3:] == 'hdf':
            if '/' in img_dataset_name[i]:
                tmp = img_dataset_name[i].split('/')
                d_img[i] =  np.array(h5py.File(img_name[i], 'r')[tmp[0]][tmp[1]],dtype=np.float32)[None,:]
            else:
                d_img[i] =  np.array(h5py.File(img_name[i], 'r')[img_dataset_name[i]],dtype=np.float32)[None,:]
        elif img_name[i][-4:] == '.pkl':
            d_img[i] =  np.array(pickle.load(img_name[i], 'rb'),dtype=np.float32)[None,:]
        else: # folder of images
            import glob
            from scipy import misc
            imN=sorted(glob.glob(img_name[i]))
            im0 =  misc.imread(imN[0])
            d_img[i] =  np.zeros((len(imN),im0.shape[1],im0.shape[0]),dtype=np.float32)
            for j in range(len(imN)):
                d_img[i][j] = misc.imread(imN[j]).astype(np.float32)
        if d_img[i].max()>5: # normalize uint8 to 0-1
            d_img[i]=d_img[i]/(2.**8)
    return d_img

def getLabel(seg_name, seg_dataset_name, suf_aff):
    # seg_name: segmentation
    # label_name: affinity
    if len(seg_name) == 0:
        d_label = None
    else:
        d_label = [None]*len(seg_name)
        for i in range(len(seg_name)):
            label_name = seg_name[i][:-3]+suf_aff+'.h5' if len(seg_name[i])>3 else ''
            # load whole aff -> remove offset for label, pad a bit for rotation augmentation
            if os.path.exists(label_name):
                d_label[i] = np.array(h5py.File(label_name,'r')['main'])
            else: # pre-compute for faster i/o
                d_seg = np.array(h5py.File(seg_name[i], 'r')[seg_dataset_name[i]])
                d_label[i] = segToAffinity(d_seg)
                writeh5(label_name, 'main', d_label[i])
    return d_label

def countVolume(data_sz, vol_sz, stride):
    return 1+np.ceil((data_sz - vol_sz) / stride.astype(np.float32)).astype(int)
