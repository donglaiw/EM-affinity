import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data

# for dataloader
def np_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    return [np.stack([b[x] for b in batch], 0) for x in range(len(batch[0]))]
    
class VolumeDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data, label, nhood = None, batch_size = 1,
                 data_size = (0,0,0),
                 label_size = (0,0,0),
                 zoom_range = None, #((0.8,1.2),(0.8,1.2),(0.8,1.2))
                 shift_range = None, #(0,0,0)
                 reflect = None, #(0,0,0),
                 swapxy = False,
                 out_data_size = (31,204,204),
                 out_label_size = (3,116,116),
                 color_scale = None, #(0.8,1.2)
                 color_shift = None, #(-0.2,0.2)
                 clip = None): # (0.05,0.95)
        # data format: consistent with caffe
        self.data = data
        self.label = label
        self.nhood = nhood
        self.batch_size = batch_size
        # samples, channels, depths, rows, cols
        self.data_size = np.array(data_size) # volume size
        self.out_data_size = np.array(out_data_size) # volume size
        self.out_label_size = np.array(out_label_size) # volume size
        self.crop_size = data_size-out_data_size+1 # same for data and label
        self.crop_size_prod = np.array([self.crop_size[1]*self.crop_size[2],self.crop_size[2]],dtype=np.float32)
        self.shift_range = shift_range # shift noise
        self.zoom_range = zoom_range # zoom range
        self.reflect = reflect # reflection
        self.swapxy = swapxy # swap xy
        self.clip = clip # clip value 
        self.color_scale = color_scale # amplify x-mean(x)
        self.color_shift = color_shift # add bias
        # pre allocation [bad for multi-thread]

    def index2zyx(self, index):
        raise NotImplementedError("Need to implement index2zyx ! "):

    def __len__(self): # number of possible position
        raise NotImplementedError("Need to implement __len__ ! "):

    def __getitem__(self, index):
        # todo: add random zoom
        pos = index2zyx(self, index)
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
        pz = int(np.floor(index/self.crop_size_prod[0]))
        pz_r = index % self.crop_size_prod[0]
        py = int(np.floor(pz_r/self.crop_size_prod[1]))
        px = int(pz_r % self.crop_size_prod[1])
        return [pz,py,px]
    def __len__(self): # all possible position
        return np.prod(self.crop_size)

class VolumeDatasetTest(VolumeDataset):
    def index2zyx(self, index):
        pz = int(np.floor(index/self.crop_size_prod[0]))
        pz_r = index % self.crop_size_prod[0]
        py = int(np.floor(pz_r/self.crop_size_prod[1]))
        px = int(pz_r % self.crop_size_prod[1])
        return [pz,py,px]

    def __len__(self): # non-overlapping position
        return np.prod(self.crop_size)
