#!/bin/bash

# example script to train models

model_opt=${1}
loss_opt=${2}
gpu_id=${3}

case ${model_opt} in 
# 2. unet3d-vgg-pad model

1) # 1. reference unet3d-vgg model
    case ${loss_opt} in
        0) # weighted l2 loss
            CUDA_VISIBLE_DEVICES=${gpu_id} python E_train.py -dc 2 -dr 0 -l 0 -lw 2 --iter-total 2000 --iter-save 500 -lr 0.0005 -o result/10_5_5e-4_dc2_dr0_bn_l2/ -betas 0.99,0.999 -lr_decay inv,0.001,0.75 -bn 1 -g 5 -b 10 -c 16
            ;;
        1) # malis loss
            CUDA_VISIBLE_DEVICES=${gpu_id} python E_train.py -dc 2 -dr 0 -l 1 -lw 0.5 --iter-total 20000 --iter-save 5000 -lr 0.0005 -o result/10_5_1e-3_dc2_dr0/ -betas 0.99,0.999 -lr_decay inv,0.001,0.75 -s result/10_5_5e-4_dc2_dr0_bn_l2/iter_10_2000_0.0005.pth -e 2000 -g 5 -b 10 -c 12 

            ;;
    esac
;;
esac
