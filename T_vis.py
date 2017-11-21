import numpy as np
import pickle, h5py, time, os, sys, argparse
from scipy.ndimage.interpolation import zoom

from PIL import Image
from skimage.color import label2rgb


def visSlice(imN,ratio=(1,1), outN=None, frame_id=0, order=0):
    ds = h5py.File(imN)
    data = np.array(ds[ds.keys()[0]])
    ds.close()
    if len(data.shape)==4:
        data = data[1]
    if data.max()<=1.0001:
        data = (data*255)
    data = data[frame_id]
    data = zoom( data, ratio, order=order) # nearest neighbor: no artifact
    out = np.uint8(np.transpose(np.tile(data,(3,1,1)),(2,1,0)))
    if outN is None:
        return out
    im = Image.fromarray(out)
    im.save(outN)

def visSliceSeg(imN, segN, ratio=[1,1,1], offset=[14,44,44],outN=None, frame_id=0):
    if isinstance(segN, basestring): 
        if segN[-3:]=='.h5':
            ds = h5py.File(segN);seg = np.squeeze(np.array(ds[ds.keys()[0]]));ds.close()
    else:
        seg = np.squeeze(segN.copy())
    if len(seg.shape)==4:
        seg = seg[0][frame_id]
    else:
        seg = seg[frame_id]
    if len(ratio) == seg.ndim:
        seg = zoom( seg, ratio, order=0) # nearest neighbor: no artifact
    else:
        seg = zoom( seg, ratio[len(ratio)-seg.ndim:], order=0) # nearest neighbor: no artifact
    if imN is not None:
        if isinstance(imN, basestring): 
            if imN[-3:]=='.h5':
                ds = h5py.File(imN);im = np.array(ds[ds.keys()[0]]);ds.close()
        else:
            im = imN.copy()
        if len(im.shape)==5:# BxCxDxWxH
            im = im[0,:,frame_id+offset[0]]
        elif len(im.shape)==4:# affinity/batch
            im = im[0][frame_id+offset[0]]
        else:
            im = im[frame_id+offset[0]:frame_id+offset[0]+1]
        if offset[1]!=0:
            im = im[:,offset[1]:-offset[1], offset[2]:-offset[2]]
        if im.max()<=1.0001:
            im = (im*255)
        im = zoom( im, ratio, order=1) # nearest neighbor: no artifact
        if im.shape[0]==1:
            im = np.tile(im,(3,1,1))
        im0 = np.uint8(np.transpose(im,(2,1,0)))
        out = label2rgb(np.transpose(seg,(1,0)),image=im0)
    else:
        out = label2rgb(np.transpose(seg,(1,0)))
    if outN is None:
        return out
    im = Image.fromarray(np.uint8(255*out))
    im.save(outN)

def visMontage(imN,imSize=(64,64),outN=None,numCol=5,order=0):
    ds = h5py.File(imN)
    data = np.array(ds[ds.keys()[0]])
    numIm = data.shape[0]
    if numCol is None:
        numCol = int(np.ceil(np.sqrt(numIm)))
    numRow = int(np.ceil(float(numIm)/float(numCol)))
    out = np.zeros((imSize[0]*numRow,imSize[1]*numCol),dtype=np.uint8)
    ratio = (imSize[0]/float(data.shape[1]),imSize[1]/float(data.shape[2]))
    for i in range(numIm):
        ind = [ i/numCol , i%numCol]
        out[ind[0]*imSize[0]:(ind[0]+1)*imSize[0],ind[1]*imSize[1]:(ind[1]+1)*imSize[1]] = zoom(data[i],ratio,order=order)
    if outN is None:
        return out
    im = Image.fromarray(out)
    im.save(outN)
