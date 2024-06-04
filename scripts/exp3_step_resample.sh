#!/bin/bash
masterK=64
numGPU=4

chmod 755 /home/dhkim0317/DoubleDescent
chmod 755 /home/dhkim0317/DoubleDescent/train_cifar10_imbalance.py

cd /home/dhkim0317/DoubleDescent
echo Current Path :
pwd

for ((k=$masterK;k>=1;k--))
do
    touch ./logtemp/step_resample001-k_$k.log
    cat /dev/null > ./logtemp/step_resample001-k_$k.log

    echo K: $k - Start Code on device $(($k % $numGPU))

    CUDA_VISIBLE_DEVICES=$(($k % $numGPU)) python -u train_cifar10_imbalance.py \
        --data_name cifar10 \
        --model_name resnet \
        --train_batch_size 128 \
        --start_k $k \
        --end_k $k \
        --imb_type step \
        --imb_factor 0.01 \
        --rand_number 0 \
        --train_rule Resample \
	--extra_name step_resample_imb001 > ./logtemp/step_resample001-k_$k.log 2>&1 &
    # echo End Code

done

tail -f ./logtemp/step_resample001-k_$masterK.log
