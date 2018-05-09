import numpy as np
import json
import torch.utils.data
from segLib.seg_core import connected_components_affgraph
from em.data.io import countVolume, cropVolume
from em.data.augmentor import buildAugmentor
from em.data.sampler import buildSampler
from em.data.data_loader import buildLoader

# based on: https://github.com/ELEKTRONN/ELEKTRONN/blob/master/elektronn/training/CNNData.py
class VolumeDataset(torch.utils.data.Dataset):
    # assume for test, no warping [hassle to warp it back..]
    def __init__(self,
                 opt_data,
                 opt_sample,
                 opt_aug=None): 
        self.data = buildLoader(opt_data) 
        self.aug = buildSampler(opt_aug) 
        self.sample = buildAugmentor(self.data, self.aug, opt_sample) 

        self.data_aug = data_aug # data augmentation
        self.data_aug.setSize(vol_img_size, vol_label_size)
        
        # samples, channels, depths, rows, cols
        self.img_size = [np.array(x.shape[1:], dtype=int) for x in img] # volume size
        self.vol_img_size = np.array(vol_img_size, dtype=int) # model input size
        self.vol_label_size = np.array(vol_label_size, dtype=int) # model output size

        # compute number of samples for each dataset
        self.sample_stride = np.array(sample_stride, dtype=np.float32)
        self.sample_size = [ countVolume(x, self.vol_img_size, self.sample_stride) \
                            for x in self.img_size]
        self.sample_num = np.array([np.prod(x) for x in self.sample_size], dtype=int)
        self.sample_num_a = np.sum(self.sample_num)
        self.sample_num_c = np.cumsum([0]+self.sample_num)

        self.num_vol = num_vol
        if self.num_vol == -1: # during test: need to compute it 
            self.num_vol = self.sample_num_a
            self.sample_size_vol = [np.array([np.prod(x[1:3]),x[2]], dtype=int) for x in self.sample_size]

    def __getitem__(self, index):
        # 1. get volume size
        vol_size = self.vol_img_size
        if self.data_aug is not None: # augmentation
            self.data_aug.getParam() # get augmentation parameter
            vol_size = self.data_aug.aug_warp[0]
        # train: random sample based on vol_size
        # test: sample based on index
        pos = self.getPos(index, vol_size) 

        # 2. get initial volume
        out_img = cropVolume(self.img[pos[0]], vol_size, pos[1:])
        out_label = None
        out_seg = None
        if self.label is not None:
            out_label = cropVolume(self.label[pos[0]], vol_size, pos[1:])

        # 3. augmentation
        if self.data_aug is not None: # augmentation
            out_img, out_label = self.data_aug.augment(out_img, out_label)

        # 4. from gt affinity -> gt segmentation
        if self.nhood is not None: # for malis loss, need local segmentation
            out_seg = connected_components_affgraph(out_label.astype(np.int32))[0].astype(np.uint64)

        # print(out_img.shape, out_label.shape, pos)
        return out_img, out_label, out_seg, pos

    def __len__(self): # number of possible position
        return self.num_vol 

    def index2zyx(self, index): # for test
        # int division = int(floor(.))
        pos = [0,0,0,0]
        did = self.getPosDataset(index)
        pos[0] = did
        index2 = index - self.sample_num_c[did]
        pos[1:] = self.getPosLocation(index2, self.sample_size_vol[did])
        return pos
    
    def getPosDataset(self, index):
        return np.argmax(index<self.sample_num_c)-1 # which dataset

    def getPosLocation(self, index, sz):
        # sz: [y*x, x]
        pos= [0,0,0]
        pos[0] = index / sz[0]
        pz_r = index % sz[0]
        pos[1] = pz_r / sz[1]
        pos[2] = pz_r % sz[1]
        return pos

    def getPos(self, index):
        raise NotImplementedError("Need to implement getPos() !")
 
class VolumeDatasetTrain(VolumeDataset):
    def getPos(self, index, vol_size):
        # index: not used
        pos = [0,0,0,0]
        did = np.random.randint(len(self.img_size)) if len(self.img_size)>1 else 0
        pos[0] = did
        tmp_size = countVolume(self.img_size[did], vol_size, self.sample_stride)
        # print('num_vol',tmp_size)
        index = np.random.randint(np.prod(tmp_size))
        tmp_size_vol = [np.prod(tmp_size[1:3]),tmp_size[2]]
        pos[1:] = self.getPosLocation(index, tmp_size_vol)
        return pos

class VolumeDatasetTest(VolumeDataset):
    def getPos(self, index, vol_size):
        # vol_size: not used
        pos = self.index2zyx(index)
        # take care of the boundary case
        for i in range(1,4):
            if pos[i] != self.sample_size[pos[0]][i-1]-1:
                pos[i] = pos[i] * self.sample_stride[i-1]
            else:
                pos[i] = self.img_size[pos[0]][i-1]-self.vol_img_size[i-1]
        return pos

class VolumeDatasetTestJson(VolumeDataset):
    # for image tiles 
    # data: json file
    def __init__(self,
                 data, res=[30,4,4], slice_size=[1,1024,1024],
                 label=None, do_seg = False,
                 extra_pad = 1,
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
        super(VolumeDatasetTestJson,self).__init__(data,label,do_seg,
                extra_pad,zoom_range,shift_range,reflect,swapxy,out_data_size,
                out_label_size,sample_stride,color_scale,color_shift,clip)
        # read one slice at a time
        data_stat = json.load(data)
        self.data_tile_name = data_stat['sections']
        self.data_tile_size = [1,data_stat['dimensions']['width'],data_stat['dimensions']['height']]
        self.data_tile_num = [data_stat['dimensions']['depth'],data_stat['dimensions']['n_columns'],data_stat['dimensions']['n_rows']]

    def getPos(self, index):
        pos = self.index2zyx(index)
        # take care of the boundary case
        for i in range(1,4):
            if pos[i] != self.sample_size[pos[0]][i-1]-1:
                pos[i] = pos[i] * self.sample_stride[i-1]
            else:
                pos[i] = self.img_size[pos[0]][i-1]-self.vol_img_size[i-1]
        return pos


# -- 2. misc --
# for dataloader
def np_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    #for b in batch:
    #    print b[2].shape,b[-1]
    return [np.stack([b[x] for b in batch], 0) for x in range(len(batch[0]))]
