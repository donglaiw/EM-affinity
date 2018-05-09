import sys
import numpy as np
import h5py
import random
import os

from subprocess import check_output

# 1. h5 i/o
def readh5(filename, datasetname):
    data=np.array(h5py.File(filename,'r')[datasetname])
    return data
 
def writeh5(filename, datasetname, dtarray):
    # reduce redundant
    fid=h5py.File(filename,'w')
    ds = fid.create_dataset(datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype)
    ds[:] = dtarray
    fid.close()

def readh5k(filename, datasetname):
    fid=h5py.File(filename)
    data={}
    for kk in datasetname:
        data[kk]=array(fid[kk])
    fid.close()
    return data

def writeh5k(filename, datasetname, dtarray):
    fid=h5py.File(filename,'w')
    for kk in datasetname:
        ds = fid.create_dataset(kk, dtarray[kk].shape, compression="gzip", dtype=dtarray[kk].dtype)
        ds[:] = dtarray[kk]
    fid.close()

def resizeh5(path_in, path_out, dataset, ratio=(0.5,0.5), interp=2, offset=[0,0,0]):
    from scipy.ndimage.interpolation import zoom
    # for half-res
    im = h5py.File( path_in, 'r')[ dataset ][:]
    shape = im.shape
    if len(shape)==3:
        im_out = np.zeros((shape[0]-2*offset[0], int(np.ceil(shape[1]*ratio[0])), int(np.ceil(shape[2]*ratio[1]))), dtype=im.dtype)
        for i in xrange(shape[0]-2*offset[0]):
            im_out[i,...] = zoom( im[i+offset[0],...], zoom=ratio,  order=interp)
        if offset[1]!=0:
            im_out=im_out[:,offset[1]:-offset[1],offset[2]:-offset[2]]
    elif len(shape)==4:
        im_out = np.zeros((shape[0]-2*offset[0], shape[1], int(shape[2]*ratio[0]), int(shape[3]*ratio[1])), dtype=im.dtype)
        for i in xrange(shape[0]-2*offset[0]):
            for j in xrange(shape[1]):
                im_out[i,j,...] = zoom( im[i+offset[0],j,...], ratio, order=interp)
        if offset[1]!=0:
            im_out=im_out[:,offset[1]:-offset[1],offset[2]:-offset[2],offset[3]:-offset[3]]
    if path_out is None:
        return im_out
    writeh5(path_out, dataset, im_out)


def writetxt(filename, dtarray):
    a = open(filename,'w')
    a.write(dtarray)
    a.close()
# 2. segmentation wrapper
def segToAffinity(seg):
    from ..lib import malis_core as malisL
    nhood = malisL.mknhood3d()
    return malisL.seg_to_affgraph(seg,nhood)

def bwlabel(mat):
    ran = [int(mat.min()),int(mat.max())];
    out = np.zeros(ran[1]-ran[0]+1);
    for i in range(ran[0],ran[1]+1):
        out[i] = np.count_nonzero(mat==i)
    return out

def genSegMalis(gg3,iter_num): # given input seg map, widen the seg border    
    from scipy.ndimage import morphology as skmorph
    #from skimage import morphology as skmorph
    gg3_dz = np.zeros(gg3.shape).astype(np.uint32)
    gg3_dz[1:,:,:] = (np.diff(gg3,axis=0))
    gg3_dy = np.zeros(gg3.shape).astype(np.uint32)
    gg3_dy[:,1:,:] = (np.diff(gg3,axis=1))
    gg3_dx = np.zeros(gg3.shape).astype(np.uint32)
    gg3_dx[:,:,1:] = (np.diff(gg3,axis=2))

    gg3g = ((gg3_dx+gg3_dy)>0)
    #stel=np.array([[1, 1],[1,1]]).astype(bool)
    #stel=np.array([[0, 1, 0],[1,1,1], [0,1,0]]).astype(bool)
    stel=np.array([[1, 1, 1],[1,1,1], [1,1,1]]).astype(bool)
    #stel=np.array([[1,1,1,1],[1, 1, 1, 1],[1,1,1,1],[1,1,1,1]]).astype(bool)
    gg3gd=np.zeros(gg3g.shape)
    for i in range(gg3g.shape[0]):
        gg3gd[i,:,:]=skmorph.binary_dilation(gg3g[i,:,:],structure=stel,iterations=iter_num)
    out = gg3.copy()
    out[gg3gd==1]=0
    #out[0,:,:]=0 # for malis
    return out

# 3. evaluation
def runBash(cmd):
    fn = '/tmp/tmp_'+str(random.random())[2:]+'.sh'
    print 'tmp bash file:',fn
    writetxt(fn, cmd)
    os.chmod(fn,0755)
    out = check_output([fn])
    os.remove(fn)
    print out

def pred_reorder(input_file, input_dataset, output_folder=None):
    if output_folder is None:
        output_folder = input_file[:input_file.rfind('/')]
    filename = input_file[input_file.rfind('/')+1:]
    pred = readh5(input_file,input_dataset)
    predt = np.transpose(pred, (1,2,3,0))
    output_file = output_folder + filename[:-3]+'-zyxc.h5'
    #output_file = input_file[:-3]+'-zyxc.h5'
    writeh5(output_file,'stack',predt)

