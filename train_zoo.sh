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
            CUDA_VISIBLE_DEVICES=${gpu_id} python E_train.py -dc 2 -dr 0 -l 0 -lw 2 --iter-total 2000 --iter-save 500 -lr 0.0005 -o ${4} -betas 0.99,0.999 -lr_decay inv,0.001,0.75 -bn 1 -g 5 -b 10 -c 16
            ;;
        1) # malis loss
            CUDA_VISIBLE_DEVICES=${gpu_id} python E_train.py -dc 2 -dr 0 -l 1 -lw 0.5 --iter-total 20000 --iter-save 5000 -lr 0.0005 -o ${4}/ -betas 0.99,0.999 -lr_decay inv,0.001,0.75 -s ${4}/${5} -e 2000 -g 5 -b 10 -c 12 

            ;;
        2) # pred test volume
            CUDA_VISIBLE_DEVICES=${gpu_id} python E_test.py -s ${4}/${5} -b 20 -g 5 -c 16 -o ${4}/ecs-gt-4x6x6-24-150k-pred.h5
            ;;
        3) # eval pred
            python E_test.py -t 1 -o ${4}/${5} -b 20 -c 16 -l 1 -lw 0.5
            ;;
    esac
;;
esac
