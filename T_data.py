import numpy as np
import torch.utils.data
import malis_core

# for dataloader
def np_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    #for b in batch:
    #    print b[2].shape,b[-1]
    return [np.stack([b[x] for b in batch], 0) for x in range(len(batch[0]))]
    
class VolumeDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data, label=None, nhood = None,
                 data_size = [(0,0,0)],
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
        self.data_size = [np.array(x) for x in data_size] # volume size
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
        self.sample_size = [1+np.ceil((x-self.out_data_size)/np.array(sample_stride,dtype=np.float32)).astype(int) for x in self.data_size]
        self.sample_size_dataset = [np.prod(x) for x in self.sample_size]
        self.sample_size_dataset_c = np.cumsum([0]+self.sample_size_dataset)
        self.sample_size_patch = [np.array([np.prod(x[1:3]),x[2]]) for x in self.sample_size]

    def __len__(self): # number of possible position
        return sum(self.sample_size_dataset)

    def index2zyx(self, index):
        # int division = int(floor(.))
        pos = [0,0,0,0]
        did = np.argmax(index<self.sample_size_dataset_c)-1 # which dataset
        pos[0] = did
        index2 = index - self.sample_size_dataset_c[did]
        pos[1] = index2/self.sample_size_patch[did][0]
        pz_r = index2 % self.sample_size_patch[did][0]
        pos[2] = pz_r / self.sample_size_patch[did][1]
        pos[3] = pz_r % self.sample_size_patch[did][1]
        return pos
    
    def getPos(self, index):
        raise NotImplementedError("Need to implement getPos() !")

    def __getitem__(self, index):
        # todo: add random zoom
        pos = self.getPos(index)
        out_data = self.data[pos[0]][:,pos[1]:pos[1]+self.out_data_size[0],
                             pos[2]:pos[2]+self.out_data_size[1],
                             pos[3]:pos[3]+self.out_data_size[2]].copy()
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
        if self.label is not None and self.label[0] is not None:
            # pad one on the left, for possible reflection
            # assume label is the same size as data
            out_label = self.label[pos[0]][:,1+pos[1]:1+pos[1]+self.out_label_size[0],
                                   1+pos[2]:1+pos[2]+self.out_label_size[1],
                                   1+pos[3]:1+pos[3]+self.out_label_size[2]].copy().astype(np.float32)
            if do_reflect is not None and any(do_reflect):
                st = np.ones((3,3),dtype=int)
                tmp_label = self.label[pos[0]][:,pos[1]:2+pos[1]+self.out_label_size[0],
                                       pos[2]:2+pos[2]+self.out_label_size[1],
                                       pos[3]:2+pos[3]+self.out_label_size[2]].copy().astype(np.float32)
                if do_reflect[0]:
                    tmp_label = tmp_label[:,::-1,:,:]
                    st[0,0] -= 1
                if do_reflect[1]:
                    tmp_label = tmp_label[:,:,::-1,:]
                    st[1,1] -= 1
                if do_reflect[2]:
                    tmp_label = tmp_label[:,:,:,::-1]
                    st[2,2] -= 1
                for i in range(3):
                    out_label[i] = tmp_label[i,st[i,0]:st[i,0]+self.out_label_size[0],st[i,1]:st[i,1]+self.out_label_size[1],st[i,2]:st[i,2]+self.out_label_size[2]]
            if do_swapxy:
                out_label = out_label.transpose((0,1,3,2))
                out_label[[1,2]] = out_label[[2,1]] # swap x-, y-affinity
            # do local segmentation from affinity
            if self.nhood is not None: # for malis loss, need local segmentation
                out_seg = malis_core.connected_components_affgraph(out_label.astype(np.int32), self.nhood)[0].astype(np.uint64)
        # print(out_data.shape, out_label.shape, pos)
        return out_data, out_label, out_seg, pos



class VolumeDatasetTrain(VolumeDataset):
    def getPos(self, index):
        return self.index2zyx(index)


class VolumeDatasetTest(VolumeDataset):
    # assume load one dataset at a time
    def getPos(self, index):
        pos = self.index2zyx(index)
        # take care of the boundary case
        for i in range(1,4):
            if pos[i] != self.sample_size[0][i-1]-1:
                pos[i] = pos[i] * self.sample_stride[i-1]
            else:
                pos[i] = self.data_size[0][i-1]-self.out_data_size[i-1]
        return pos
