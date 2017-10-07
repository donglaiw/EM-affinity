import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data
import malis_core

# for dataloader
def np_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    return [np.stack([b[x] for b in batch], 0) for x in range(len(batch[0]))]
    
class VolumeDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data, label=None, nhood = None,
                 data_size = (0,0,0),
                 label_size = (0,0,0),
                 zoom_range = None, #((0.8,1.2),(0.8,1.2),(0.8,1.2))
                 shift_range = None, #(0,0,0)
                 reflect = None, #(0,0,0),
                 swapxy = False,
                 out_data_size = (31,204,204),
                 out_label_size = (3,116,116),
                 sample_stride = (1,1,1), # grid sample
                 color_scale = None, #(0.8,1.2)
                 color_shift = None, #(-0.2,0.2)
                 clip = None): # (0.05,0.95)
        # data format: consistent with caffe
        self.data = data
        self.label = label
        self.nhood = nhood
        # samples, channels, depths, rows, cols
        self.data_size = np.array(data_size) # volume size
        self.out_data_size = np.array(out_data_size) # volume size
        self.out_label_size = np.array(out_label_size) # volume size
        self.shift_range = shift_range # shift noise
        self.zoom_range = zoom_range # zoom range
        self.reflect = reflect # reflection
        self.swapxy = swapxy # swap xy
        self.clip = clip # clip value 
        self.color_scale = color_scale # amplify x-mean(x)
        self.color_shift = color_shift # add bias
        # pre allocation [bad for multi-thread]
        # compute sample size
        self.sample_stride = sample_stride
        self.sample_size = 1+np.ceil((self.data_size-self.out_data_size+1)/np.array(sample_stride,dtype=np.float32)).astype(int)
        self.sample_size_prod = np.array([np.prod(self.sample_size),np.prod(self.sample_size[1:3]),self.sample_size[2]])

    def __len__(self): # number of possible position
        return self.sample_size_prod[0]

    def index2zyx(self, index):
        raise NotImplementedError("Need to implement index2zyx !")

    def __getitem__(self, index):
        # todo: add random zoom
        pos = self.index2zyx(index)
        out_data = self.data[:,pos[0]:pos[0]+self.out_data_size[0],
                             pos[1]:pos[1]+self.out_data_size[1],
                             pos[2]:pos[2]+self.out_data_size[2]].copy()
        out_label = False
        out_seg = False;
        # post
        do_reflect=None
        do_swapxy=False
        if self.reflect is not None:
            # random rotation/relection
            do_reflect = [self.reflect[x]>0 and np.random.random()>0.5 for x in range(3)]
            if any(do_reflect):
                if do_reflect[0]:
                    out_data  = out_data[:,::-1,:,:]
                if do_reflect[1]:
                    out_data  = out_data[:,:,::-1,:]
                if do_reflect[2]:
                    out_data  = out_data[:,:,:,::-1]
        if self.swapxy and np.random.random()>0.5:
            do_swapxy = True
            out_data = out_data.transpose((0,1,3,2))
        # color scale/shift
        if self.color_scale is not None:
            out_data_mean = out_data.mean()
            out_data = out_data_mean + (out_data-out_data_mean)*np.random.uniform(low=self.color_scale[0],high=self.color_scale[1])
        if self.color_shift is not None:
            out_data = out_data + np.random.uniform(low=self.color_shift[0],high=self.color_shift[1])
        # clip
        if self.clip is not None:
            out_data = np.clip(out_data,self.clip[0],self.clip[1])

        # do label
        if self.label is not None:
            # pad one on the left, for possible reflection
            # assume label is the same size as data
            out_label = self.label[:,1+pos[0]:1+pos[0]+self.out_label_size[0],
                                   1+pos[1]:1+pos[1]+self.out_label_size[1],
                                   1+pos[2]:1+pos[2]+self.out_label_size[2]].copy().astype(np.float32)
            if do_reflect is not None:
                st = np.ones((3,3),dtype=int)
                tmp_label = self.label[:,pz:2+pz+self.out_label_size[0],py:2+py+self.out_label_size[1],px:2+px+self.out_label_size[2]]
                if do_reflect[0]:
                    out_data  = out_data[:,::-1,:,:]
                    tmp_label = tmp_label[:,::-1,:,:]
                    st[0,0] -= 1
                if do_reflect[1]:
                    out_data  = out_data[:,:,::-1,:]
                    tmp_label = tmp_label[:,:,::-1,:]
                    st[1,1] -= 1
                if do_reflect[2]:
                    out_data  = out_data[:,:,:,::-1]
                    tmp_label = tmp_label[:,:,:,::-1]
                    st[2,2] -= 1
                for i in range(3):
                    out_label[i] = tmp_label[i,st[i,0]:st[i,0]+self.out_label_size[0],st[i,1]:st[i,1]+self.out_label_size[1],st[i,2]:st[i,2]+self.out_label_size[2]]
            if do_swapxy:
                out_label = out_label.transpose((0,1,3,2))
                out_label[[1,2]] = out_label[[2,1]]
            # do local segmentation from affinity
            if self.nhood is not None: # for malis loss, need local segmentation
                out_seg = malis_core.connected_components_affgraph(out_label.astype(np.int32), self.nhood)[0].astype(np.uint64)
        return out_data, out_label, out_seg, pos



class VolumeDatasetTrain(VolumeDataset):
    def index2zyx(self, index):
        # int division = int(floor(.))
        pos = [0,0,0]
        pos[0] = index/self.sample_size_prod[1]
        pz_r = index % self.sample_size_prod[1]
        pos[1] = pz_r/self.sample_size_prod[2]
        pos[0] = pz_r % self.sample_size_prod[2]
        return pos

class VolumeDatasetTest(VolumeDataset):
    def index2zyx(self, index):
        pos = [0,0,0]
        pos[0] = index/self.sample_size_prod[1]
        pz_r = index % self.sample_size_prod[1]
        pos[1] = pz_r/self.sample_size_prod[2]
        pos[2] = pz_r % self.sample_size_prod[2]
        # take care of the boundary case
        for i in range(3):
            if pos[i] != self.sample_size[i]-1:
                pos[i] = pos[i] * self.sample_stride[i]
            else:
                pos[i] = self.data_size[i]-self.out_data_size[i]
        return pos
