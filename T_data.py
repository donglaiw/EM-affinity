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

    def __getitem__(self, index):
        # todo: add random zoom
        p0 = int(np.floor(index/self.crop_size_prod[0]))
        p0_r = index % self.crop_size_prod[0]
        p1 = int(np.floor(p0_r/self.crop_size_prod[1]))
        p2 = int(p0_r % self.crop_size_prod[1])
        out_data = self.data[:,p0:p0+self.out_data_size[0],p1:p1+self.out_data_size[1],p2:p2+self.out_data_size[2]].copy()
        # label data: pad one on the left, for possible reflection
        out_label = self.label[:,1+p0:1+p0+self.out_label_size[0],1+p1:1+p1+self.out_label_size[1],1+p2:1+p2+self.out_label_size[2]].copy()
        # post
        if self.reflect is not None:
            # random rotation/relection
            do_flip = [self.reflect[x]>0 and np.random.random()>0.5 for x in range(3)]
            if any(do_flip):
                st = np.ones((3,3),dtype=int)
                tmp_label = self.label[:,p0:2+p0+self.out_label_size[0],p1:2+p1+self.out_label_size[1],p2:2+p2+self.out_label_size[2]]
                if do_flip[0]:
                    out_data  = out_data[:,::-1,:,:]
                    tmp_label = tmp_label[:,::-1,:,:]
                    st[0,0] -= 1
                if do_flip[1]:
                    out_data  = out_data[:,:,::-1,:]
                    tmp_label = tmp_label[:,:,::-1,:]
                    st[1,1] -= 1
                if do_flip[2]:
                    out_data  = out_data[:,:,:,::-1]
                    tmp_label = tmp_label[:,:,:,::-1]
                    st[2,2] -= 1
                for i in range(3):
                    out_label[i] = tmp_label[i,st[i,0]:st[i,0]+self.out_label_size[0],st[i,1]:st[i,1]+self.out_label_size[1],st[i,2]:st[i,2]+self.out_label_size[2]]

        if self.swapxy and np.random.random()>0.5:
            out_data = out_data.transpose((0,1,3,2))
            out_label = out_label.transpose((0,1,3,2))
            out_label[[1,2]] = out_label[[2,1]]
        # color scale/shift
        if self.color_scale is not None:
            out_data_mean = out_data.mean()
            out_data = out_data_mean + (out_data-out_data_mean)*np.random.uniform(low=self.color_scale[0],high=self.color_scale[1])
        if self.color_shift is not None:
            out_data = out_data + np.random.uniform(low=self.color_shift[0],high=self.color_shift[1])
        # clip
        if self.clip is not None:
            out_data = np.clip(out_data,self.clip[0],self.clip[1])

        if self.nhood is not None: # for malis loss, need local segmentation
            return out_data, out_label.astype(np.float32), malis_core.connected_components_affgraph(out_label.astype(np.int32), self.nhood)[0].astype(np.uint64)
        else:
            return out_data, out_label.astype(np.float32)

    def __len__(self): # number of possible position
        return np.prod(self.crop_size)


