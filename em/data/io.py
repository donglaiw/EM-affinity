# utility function to load data
# train-L2: data+label
# train-malis: data+label+seg
# test-pred: data
# test-err-L2: data+label
# test-err-malis: data+label+seg
import numpy as np
import h5py
import os
from ..util.misc import writeh5, segToAffinity

def setPred(pred, y_pred, pred_sz, pos, lt=None, st=0):
    if lt is None:
        lt = pos.shape[0]
    for j in range(st,lt):
        pp = pos[j] 
        pred[:,pp[1]:pp[1]+pred_sz[0],
              pp[2]:pp[2]+pred_sz[1],
              pp[3]:pp[3]+pred_sz[2]] = y_pred[j].copy()


def getVar(batch_size, model_io_size, do_input=[True, False, False]):
    import torch
    from torch.autograd import Variable
    input_vars = [None] *3  
    # data
    input_vars[0] = Variable(torch.zeros(batch_size, 1, model_io_size[0][0], model_io_size[0][1], model_io_size[0][2]).cuda(), requires_grad=False)
    # label
    if do_input[1]:
        input_vars[1] = Variable(torch.zeros(batch_size, 3, model_io_size[1][0], model_io_size[1][1], model_io_size[1][2]).cuda(), requires_grad=False)
    if do_input[2]:
        input_vars[2] = Variable(torch.zeros(batch_size, 3, model_io_size[1][0], model_io_size[1][1], model_io_size[1][2]).cuda(), requires_grad=False)
    return input_vars


def getData(dirName, data_name, data_dataset_name):
    d_data = [None]*len(dirName)
    for i in range(len(dirName)):
        if data_name[-3:] == '.h5' or data_name[-3:] == 'hdf':
            if '/' in data_dataset_name:
                tmp = data_dataset_name.split('/')
                d_data[i] =  np.array(h5py.File(dirName[i]+data_name,'r')[tmp[0]][tmp[1]],dtype=np.float32)[None,:]
            else:
                d_data[i] =  np.array(h5py.File(dirName[i]+data_name,'r')[data_dataset_name],dtype=np.float32)[None,:]
        elif data_name[-4:] == '.pkl':
            d_data[i] =  np.array(pickle.load(dirName[i]+data_name,'rb'),dtype=np.float32)[None,:]
        else: # folder of images
            import glob
            from scipy import misc
            imN=sorted(glob.glob(dirName[i]+'*'+data_name))
            im0 =  misc.imread(imN[0])
            d_data[i] =  np.zeros((len(imN),im0.shape[1],im0.shape[0]),dtype=np.float32)
            for j in range(len(imN)):
                d_data[i][j] = misc.imread(imN[j]).astype(np.float32)
        if d_data[i].max()>5: # normalize uint8 to 0-1
            d_data[i]=d_data[i]/(2.**8)
    return d_data

def getLabel(dirName, seg_name, suf_aff, seg_dataset_name):
    label_name = seg_name[:-3]+suf_aff+'.h5' if len(seg_name)>3 else ''
    d_label = [None]*len(dirName)
    for i in range(len(dirName)):
        # load whole aff -> remove offset for label, pad a bit for rotation augmentation
        if os.path.exists(dirName[i] + label_name):
            d_label[i] = np.array(h5py.File(dirName[i] + label_name,'r')[seg_dataset_name])
        else: # pre-compute for faster i/o
            d_seg = np.array(h5py.File(dirName[i] + seg_name, 'r')[seg_dataset_name])
            d_label[i] = segToAffinity(d_seg)
            writeh5(dirName[i] + label_name, 'main', d_label[i])
    return d_label

def getDataAug(opt, data_color_opt=0, data_rotation_opt=0):
    # default: no aug
    color_scale=None; color_shift=None; color_clip=None; rot=[(0,0,0),False]
    if opt=='train':
        color_scale = [(0.8,1.2), (0.9,1.1), None][data_color_opt]
        color_shift = [(-0.2,0.2), (-0.1,0.1), None][data_color_opt]
        color_clip = [(0.05,0.95), (0.05,0.95), None][data_color_opt]
        rot = [[(1,1,1),True],[(0,0,0),False]][data_rotation_opt]
    elif opt=='test':
        color_clip = [(0.05,0.95), (0.05,0.95), None][data_color_opt]
    return color_scale, color_shift, color_clip, rot

# crop data/label correctly
def cropCentralN(data,label,offset,extraPad=True):
    for i in range(len(data)):
        data[i], label[i] = cropCentral(data[i], label[i], offset, extraPad)
    # need return, as label can have new address
    return data,label

def cropCentral(data,label,offset,extraPad=True):
    if len(offset)==2:
        offset = np.abs(offset[0]-offset[1])/2
    # CxDxWxH
    if extraPad: # as z axis is precious, we pad data by 1 (used for affinity flip, not eval)
        label = np.lib.pad(label,((0,0),(1,1),(1,1),(1,1)),mode='reflect')

    sz_diff = np.array(data.shape)-np.array(label.shape)
    sz_offset = sz_diff[1:]/2 # floor
    sz_offset2 = sz_diff[1:]-sz_diff[1:]/2 #ceil
    if extraPad: # extra padding for data augmentation affinity
        sz_offset+=1
        sz_offset2+=1
    if any(sz_offset-offset) or any(sz_offset-offset):
        # z axis
        if offset[0] > sz_offset2[0]: # label is bigger
            label=label[:,offset[0]-sz_offset[0]:label.shape[1]-(offset[0]-sz_offset2[0])]
        else: # data is bigger
            data=data[:,sz_offset[0]-offset[0]:data.shape[1]-(sz_offset2[0]-offset[0])]
        # y axis
        if offset[1] > sz_offset2[1]:
            label=label[:,:,(offset[1]-sz_offset[1]):(label.shape[2]-(offset[1]-sz_offset2[1]))]
        else:
            data=data[:,:,sz_offset[1]-offset[1]:data.shape[2]-(sz_offset2[1]-offset[1])]
        if offset[2] > sz_offset2[2]:
            label=label[:,:,:,offset[2]-sz_offset[2]:label.shape[3]-(offset[2]-sz_offset2[2])]
        else:
            data=data[:,:,:,sz_offset[2]-offset[2]:data.shape[3]-(sz_offset2[2]-offset[2])]
    # need return, as label can have new address
    return data,label
