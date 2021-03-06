# Pytorch Package for Unet3D 
## Installation
```
python setup.py install
```

##
1. Basic module 
    - data/: dataLoader for volume data
    - model/: unet3D (block: vgg, residual)
    - quant/: int8 quantization
    - prune/: channel pruning (in progress)
    - lib/: external libraries
    - util/: utility functions
2. Example
    - Affinity prediction from voxel:
        * train unet3D
        ```
        CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 python exp/train_affinity.py -dc 2 -dr 0 -l 0 -lw 2 -b 6 --volume-total 20000 --volume-save 5000 -lr 0.0001 -g 6 -c 10 -o tmp/ -betas 0.99,0.999 -lr_decay inv,0.0001,0.75 -t ../../data/JWR/vol3/ -dn im_uint8_half.h5 -ln seg_groundtruth_myelin_malis_iter3_half.h5 -v ../../data/JWR/vol4/ -bn 1  
        ```
        * test unet3D
        ```
        n1=50000;n2=150k;DD=Dec17_vol1-3_vol4_malis_iter3_half_bn_b6/;i=1;CUDA_VISIBLE_DEVICES=0 python exp/test_affinity.py -s result/${DD}/volume_${n1}.pth -b 3 -g 1 -c 1 -o result/${DD}/vol${i}-${n2}-pred.h5 -i ../../data/JWR/vol${i}/ -dn im_uint8_half.h5 -bn 1;
        ```
    - Model quantization
        * quantization (single-GPU, nn.DataParallel won't update correctly)
        ```
        DD=/n/coxfs01/donglai/micron100_1217/model/Toufiq/;Do=result/quant_m1/;n1=net_iter_100000_m1; python exp/quant.py -s ${DD}${n1}.pth -i ../../data/JWR/vol1/@../../data/JWR/vol3/ -dn im_uint8_half.h5 -b 3 -g 1 -nb 1000 -o ${Do}${n1} -qm linear
        ```
        * prediction
        ```
        n1=net_iter_100000_m1_p8;n2=linear_v1;DD=result/quant_m1/${n1}_${n2};i=2;CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 python exp/test_affinity.py -s ${DD}.pth -b 10 -g 10 -c 16 -o result/quant_m1/vol${i}-${n2}-pred.h5 -i ../../data/JWR/vol${i}/ -dn im_uint8_half.h5 -bn 1 -m 1
        ```
    - Model channel-pruning
    - Utility
        * benchmark model inference speed
        ```
         python exp/benchmark.py -m 0 -o tmp/unet3d.pkl # 228.0\pm 17.0 (gpu05), 191.5\pm 2.8 (gpu06)
        ```
        * model conversion
        ```
        # caffe to pkl
        # pth to pkl
        python exp/translate.py -pm  result/Dec17_vol1-3_vol4_malis_iter3_half_b6_ac4/volume_50000.pth -o bk/tmp -op 0.2   
        # pkl to keras
        # pkl to pytorch
        python exp/translate.py -w ${DD}net_iter_100000 -o ${DD}net_iter_100000_m1 -op 1.1
        ```

## Reference:
1. Malis loss: [[cython code]](https://github.com/TuragaLab/malis), [[caffe code]](https://github.com/naibaf7/caffe/blob/master/src/caffe/layers/malis_loss_layer.cpp)
2. Unet-3D: [[pytorch code]](https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/unet.py)
    - [[residual
      module]](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py), [[inception module]](https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py)
3. Channel pruning: [[pytorch code]](https://github.com/jacobgil/pytorch-pruning/prune.py)
4. Model quantization: [[pytorch code]](https://github.com/aaron-xichen/pytorch-playground/blob/master/utee/quant.py)
