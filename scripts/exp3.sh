#!/bin/bash

masterK=64
numGPU=$4
chmod 755 /home/donggeonlee/repo/DoubleDescent/train_cifar10.py

cd /home/donggeonlee/repo/DoubleDescent
echo Current Path :
pwd

for ((k=$masterK;k>=1;k--))
do
    touch resample001-k_$masterk.log
    cat /dev/null > resample001-k_$masterk.log

    echo K: $k - Start Code

    CUDA_VISIBLE_DEVICES=$(($k % numGPU)) python -u train_cifar10_imbalance.py \
        --data_name cifar10 \
        --model_name resnet \
        --train_batch_size 128 \
        --start_k $k \
        --end_k $k \
        --imb_type exp \
        --imb_factor 0.01 \
        --train_rule Resample \
        --rand_number 0 \
        --extra_name imbalance_resample_imb001 > resample001-k_$masterk.log 2>&1 &
    # echo End Code

done

tail -f resample001-k_$masterk.log
