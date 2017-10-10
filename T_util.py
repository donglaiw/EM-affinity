import h5py
import numpy as np
import scipy.misc

# for L2 training: re-weight the error by label bias (far more 1 than 0)
def error_scale(data, clip_low, clip_high):
    frac_pos = np.clip(data.mean(), clip_low, clip_high) #for binary labels
    # can't be all zero
    w_pos = 1.0/(2.0*frac_pos)
    w_neg = 1.0/(2.0*(1.0-frac_pos))
    scale = np.add((data >= 0.5) * w_pos, (data < 0.5) * w_neg)
    return scale

   
def save_checkpoint(model, optimizer, epoch=1, filename='checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict()
    }, filename)


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

def resizeh5(path_in, path_out, dataset, ratio=(0.5,0.5), interp='bicubic', offset=None):
    # for half-res
    im = h5py.File( path_in, 'r')[ dataset ][:]
    shape = im.shape
    im_out = np.zeros((shape[0], int(shape[1]*ratio[0]), int(shape[2]*ratio[1])), dtype=np.uint8)

    for i in xrange(shape[0]):
        im_out[i,...] = scipy.misc.imresize( im[i,...], size=(int(shape[1]*ratio[0]), int(shape[2]*ratio[1])),  interp=interp)
    if offset is not None:
        im_out=im_out[offset[0]:-offset[0],offset[1]:-offset[1],offset[2]:-offset[2],offset[3]:-offset[3]]
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
