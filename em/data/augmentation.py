import numpy as np
from em.lib.elektronn.warping import getWarpParams, warp3dFast, _warp3dFastLab
from em.data.io import cropVolume

class DataAugment(object):
    def __init__(self,
                 do_opt = [0,0,0,0], # warping, rotation, color, miss Slice
                 param_warp = [15,3,1.1,0.1], # rot_max, shear_max, scale_max, stretch_max
                 param_color = [[0.95,1.05],[-0.15,0.15],[0.5,2],[0,1]] # scale, shift, power, clip
                 ): 
        # data format: C*D*W*H
        # model indepent parameters
        self.pad_offset = np.zeros(3, dtype=int)
        
        # augmentation parameter
        self.do_warp = do_opt[0] 
        self.do_rot = do_opt[1] # 0-15 for 16 choices or -1 for random
        self.do_color = do_opt[2]
        self.do_missD = do_opt[3]
        self.param_warp = param_warp
        self.param_color = param_color


    def setSize(self, vol_img_size, vol_label_size):
        # size parameter
        self.vol_img_size = vol_img_size
        self.vol_label_size = vol_label_size
        self.crop_offset = (vol_img_size-vol_label_size) // 2
        self.initParam()

    def augment(self, vol_img, vol_label): # same size data
        vol_img = self.augmentImg(vol_img)
        if vol_label is not None:
            vol_label = self.augmentLabel(vol_label)
        return vol_img, vol_label

    def augmentImg(self, data):
        # data already cropped, but reference
        if self.do_warp != 0:
            data = self.doWarp(data)
            data = cropVolume(data, self.vol_img_size, self.pad_offset) # crop bottom right
        else: # direct crop, do copy o/w overwrite the crop from the whole volume
            data = data.copy()

        if self.do_rot != 0:
            data = self.doRotation(data) 
        if self.do_color != 0:
            data = self.doColor(data) 
        if self.do_missD != 0:
            data = self.doMissData(data)
        return data

    def augmentLabel(self, data):
        if self.do_warp:
            data = self.doWarp(data)
        else: # direct crop
            data = data.copy()

        if self.do_rot != 0:
            data = self.doRotation(data, is_label = True) 
        if self.do_color != 0:
            data = self.doColor(data) 
        if self.do_missD != 0:
            data = self.doMissData(data)
        return data

    def initParam(self): # initialization
        self.aug_rot = '{0:04b}'.format(self.do_rot)
        self.aug_rot_st = np.ones((3,3), dtype=int) # for affinity label: need a larger data to compute augmented label
        self.aug_warp = []
        if self.do_rot != 0: # need to pad 1 for affinity
            self.pad_offset = np.ones(3, dtype=int) # pad it for all cases (otherwise messy to debug)
        self.aug_warp_size = self.vol_img_size+self.pad_offset # target size
        self.aug_color = []
        self.aug_missD = []
    
    def getParam(self): # initialization
        if self.do_warp >0 : # random
            self.aug_warp = getWarpParams(self.aug_warp_size, self.param_warp[0],\
                                          self.param_warp[1], self.param_warp[2], self.param_warp[3])
        if self.do_rot == -1: # random
            self.aug_rot = ''.join([str(int(np.random.random()>0.5)) for x in range(4)])
            self.aug_rot_st = np.ones((3,3), dtype=int)
            for i in range(3):
                self.aug_rot_st[i,i] = 0 if self.aug_rot[i]=='1' else 1
        if self.do_color >0 : # random
            self.aug_color = [np.random.uniform(low=self.param_color[x][0],high=self.param_color[x][1])\
                              for x in range(3)]+self.param_color[3]
        if self.do_missD >0 : # random sample slices to be unif[0,1]
            miss_num = np.random.randint(self.do_missD)
            self.aug_missD = np.random.permutation(range(self.vol_img_size[0]))[:miss_num]
        print "aug-param:",self.aug_rot,self.aug_color,self.aug_missD
        #print "aug-param:",self.aug_warp,self.aug_rot,self.aug_color,self.aug_missD
 
    def doWarp(self, data, is_seg=False):
        # warp segmentation 
        # img, label: have the same size
        print('aa',data.shape,self.aug_warp_size)
        out = warp3dFast(data, self.aug_warp_size, self.aug_warp[1],\
                         self.aug_warp[2], self.aug_warp[3], self.aug_warp[4], self.aug_warp[5])
        print('db 2:',self.aug_warp_size, out.shape, self.pad_offset)
        return out

    def doRotation(self, data, is_label=False): # augmentation with rotation/relection
        # data = img/label: C*D*W*H 
        if self.aug_rot[0] == '1':
            data  = data[:,::-1,:,:]
        if self.aug_rot[1] == '1':
            data  = data[:,:,::-1,:]
        if self.aug_rot[2] == '1':
            data  = data[:,:,:,::-1]
        # label: need to crop the right region before swap xy
        if is_label:
            print("do-label:", data.shape, self.vol_label_size)
            data = cropVolume(data, self.vol_label_size, self.aug_rot_st)
        if self.aug_rot[3] == '1':
            data = data.transpose((0,1,3,2))
            if is_label: # swap data x-y -> swap x/y affinity
                data[[1,2]] = data[[2,1]]
        return data
    
    def doMissData(self, data):# only apply to data 
        # uniform 0-1
        if len(self.aug_missD)>0:
            data[:,self.aug_missD] = np.random.rand((len(self.aug_missD), \
                                    self.vol_img_size[1], self.vol_img_size[2]))
        return data

    def doColor(self, data): # only apply to data in [0,1]
        # scale+shift
        data_mean = data.mean()
        data = data_mean + (data-data_mean)*self.aug_color[0]+self.aug_color[1]
        # clip + exponential (stay within 0-1)
        data = np.clip(data,self.aug_color[3],self.aug_color[4])**self.aug_color[2]
        return data
