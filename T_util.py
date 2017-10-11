import h5py
import numpy as np
import sys

# no pytorch env needed

# keras-to-pkl
def keras2pkl(kerasmodel, output=None):
    import keras
    sys.setrecursionlimit(1000000) # o/w won't pickle keras model
    # bypass model.save(), so that it's easier to save new layers
    net = pickle.load(open(kerasmodel,'rb'))
    out={}
    for layer in net.layers:
        if layer.trainable_weights:
            layer={}
            tmp = layer.get_weights()
            if len(tmp)==1:
                layer['w']=tmp[0]
            elif len(tmp)>1:
                layer['b']=tmp[1]
            out[layer.name]=layer
    if output is None:
        return out
    pickle.dump(out,open(output,'wb'))
# caffe-to-pkl
def caffe2pkl(prototxt, caffemodel, output=None):
    import pickle
    import caffe
    caffe.set_mode_cpu()
    net0 = caffe.Net(prototxt,caffe.TEST,weights=caffemodel)
    out={}
    for k,v in net0.params.items():
        layer={}
        layer['w']=v[0].data
        if len(v)>1: # deconv layer: no bias term
            layer['b']=v[1].data
        out[k]=layer
    if output is None:
        return out
    pickle.dump(out,open(output,'wb'))

def readh5(filename, datasetname,tt=np.float32):
    data=np.array(h5py.File(filename,'r')[datasetname],dtype=tt)
    return data
 
def writeh5(filename, datasetname, dtarray):
    fid=h5py.File(filename,'w')
    fid.create_dataset(datasetname,data=dtarray)
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
        fid.create_dataset(kk,data=dtarray[kk])
    fid.close()

# for half-res
def resizeh5(path_in, path_out, dataset, ratio=(0.5,0.5), interp=1, offset=None):
    # order=1: bilinear
    from scipy.ndimage.interpolation import zoom
    # don't use scipy.misc.imresize, output is uint8...
    im = h5py.File( path_in, 'r')[ dataset ][:]
    shape = im.shape
    if len(shape)==3:
        im_out = np.zeros((shape[0], int(shape[1]*ratio[0]), int(shape[2]*ratio[1])), dtype=im.dtype)
        for i in xrange(shape[0]):
            im_out[i,...] = zoom( im[i,...], zoom=ratio,  order=interp)
        if offset is not None:
            im_out=im_out[:,offset[1]:-offset[1],offset[2]:-offset[2]]
    elif len(shape)==4:
        im_out = np.zeros((shape[0], shape[1], int(shape[2]*ratio[0]), int(shape[3]*ratio[1])), dtype=im.dtype)
        for i in xrange(shape[0]):
            for j in xrange(shape[1]):
                im_out[i,j,...] = zoom( im[i,j,...], ratio, order=interp)
        if offset is not None:
            im_out=im_out[:,offset[1]:-offset[1],offset[2]:-offset[2],offset[3]:-offset[3]]
    if path_out is None:
        return im_out
    h5py.File( path_out, 'w').create_dataset( dataset, data=im_out )

def bwlabel(mat):
    ran = [int(mat.min()),int(mat.max())];
    out = np.zeros(ran[1]-ran[0]+1);
    for i in range(ran[0],ran[1]+1):
        out[i] = np.count_nonzero(mat==i)
    return out

def genSegMalis(gt): # given input seg map, widen the seg border    
    gg3=gt
    gg3_dz = np.zeros(gg3.shape).astype(np.uint32)
    gg3_dz[1:,:,:] = (np.diff(gg3,axis=0))
    gg3_dy = np.zeros(gg3.shape).astype(np.uint32)
    gg3_dy[:,1:,:] = (np.diff(gg3,axis=1))
    gg3_dx = np.zeros(gg3.shape).astype(np.uint32)
    gg3_dx[:,:,1:] = (np.diff(gg3,axis=2))

    gg3g = ((gg3_dx+gg3_dy)>0)
    #stel=np.array([[1, 1],[1,1]]).astype(bool)
    stel=np.array([[0, 1, 0],[1,1,1], [0,1,0]]).astype(bool)
    #stel=np.array([[1,1,1,1],[1, 1, 1, 1],[1,1,1,1],[1,1,1,1]]).astype(bool)
    gg3gd=np.zeros(gg3g.shape)
    for i in range(gg3g.shape[0]):
        gg3gd[i,:,:]=skmorph.binary_dilation(gg3g[i,:,:],structure=stel,iterations=2)
