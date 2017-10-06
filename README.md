# malis-pytorch: Malis layer implemented in pytorch
## MalisLoss Intallation
1. Install library dependencies:
```
    conda install cython boost
```
2. modify `setup.py`
```
    CONDA_ENV_INCLUDE=YOUR_PATH
```
3. compile
```
    ./make.sh
```
## Pytorch Package for Unet3D 
1. Basic module 
    - T_data.py: dataLoader for volume data
    - T_model.py: unet3D (with and without BatchNorm)
    - T_pruner.py: channel pruning (in progress)
2. Example
    - E_test.py: test model and benchmark speed
    - E_train.py: train model
    - E_prune.py: channel prune model

## Reference:
1. [cython implementation](https://github.com/TuragaLab/malis)
2. [naibaf7's caffe implementation](https://github.com/naibaf7/caffe/blob/master/src/caffe/layers/malis_loss_layer.cpp)
