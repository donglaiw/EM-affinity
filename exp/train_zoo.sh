#!/bin/bash
# utility function

countGPU(){
    grep -o "," <<< "${1}" | wc -l
}

# example script to train models

model_opt=${1}
step=${2}
out_dir=${3}
snapshot=${4}
out_name=${5}
cpu_num=${6}
batch_per_gpu=${7}
gpu_id=${8}

loss_opt=${8}
loss_weight_opt=${9}
test_file=${10}

gpu_num=$(( $( countGPU ${gpu_id} ) +1 ))
batch_size=$((batch_per_gpu * gpu_num))

case ${model_opt} in 
# 2. unet3d-vgg-pad model

1) # 1. reference unet3d-vgg model
    case ${step} in
        0) # weighted l2 loss
            CUDA_VISIBLE_DEVICES=${gpu_id} python E_train.py -dc 2 -dr 0 -l 0 -lw 2 --iter-total 2000 --iter-save 500 -lr 0.0005 -o ${4} -betas 0.99,0.999 -lr_decay inv,0.001,0.75 -bn 1 -g ${gpu_num} -b ${batch_size} -c ${cpu_num}
            ;;
        1) # malis loss
            # ./train_zoo.sh 1 1 result/10_5_1e-3_dc2_dr0 iter_10_10000_0.0005.pth -1 16 2 0,1
            CUDA_VISIBLE_DEVICES=${gpu_id} python E_train.py -dc 2 -dr 0 -l 1 -lw 0.5 --iter-total 20000 --iter-save 5000 -lr 0.0005 -o ${4}/ -betas 0.99,0.999 -lr_decay inv,0.001,0.75 -s ${out_dir}/${snapshot} -bn 1 -e 2000 -g ${gpu_num} -b ${batch_size} -c ${cpu_num} 

            ;;
        2) # pred test volume
            # ./train_zoo.sh 1 2 /n/home02/ptillet/Development/malis-pytorch/result/malis/ iter_10_17000_0.0005.pth ecs-gt-4x6x6-150k-pred.h5 16 2 0,1 -1 ecs-gt-4x6x6
            # pred train volume
            # ./train_zoo.sh 1 2 /n/home02/ptillet/Development/malis-pytorch/result/malis/ iter_10_17000_0.0005.pth out_train.h5 16 4 0,1,2,3,4 -1 ecs-gt-3x6x6
            CUDA_VISIBLE_DEVICES=${gpu_id} python E_test.py -s ${out_dir}/${snapshot} -b ${batch_size} -g ${gpu_num} -c ${cpu_num} -o ${out_dir}/${out_name} -bn 1 -i /n/coxfs01/donglai/malis_trans/data/ecs-3d/${out_file}
            ;;
        3) # eval pred
            # train: ./train_zoo.sh 1 3 /n/home02/ptillet/Development/malis-pytorch/result/malis/ -1 out.h5 10 10 1 0.5 ecs-gt-3x6x6
            # test: ./train_zoo.sh 1 3 /n/home02/ptillet/Development/malis-pytorch/result/malis/ -1 out.h5 10 10 1 0.5 ecs-gt-4x6x6
            # malis: 1 0.5
            # weighted-L2: 0 2
            python E_test.py -t 1 -o ${out_dir}/${out_name} -b ${batch_size} -c ${cpu_num} -l ${loss_opt} -lw ${loss_weight_opt} -i /n/coxfs01/donglai/malis_trans/data/ecs-3d/${out_file}
            ;;
    esac
;;
esac
