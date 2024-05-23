#!/bin/bash
masterK=64
numGPU=3

chmod 755 /home/dhkim0317/DoubleDescent
chmod 755 /home/dhkim0317/DoubleDescent/train_cifar10_imbalance.py

cd /home/dhkim0317/DoubleDescent
echo Current Path :
pwd

for ((k=$masterK;k>=1;k--))
do
    touch none001-k_$k.log
    cat /dev/null > none001-k_$k.log

    echo K: $k - Start Code on device $(($k % $numGPU))

    CUDA_VISIBLE_DEVICES=$(($k % $numGPU)) python -u train_cifar10_imbalance.py \
        --data_name cifar10 \
        --model_name resnet \
        --train_batch_size 128 \
        --start_k $k \
        --end_k $k \
        --imb_type exp \
        --imb_factor 0.01 \
        --rand_number 0 \
        --train_rule None \
	--extra_name none_resample_imb001 > none001-k_$k.log 2>&1 &
    # echo End Code

done

tail -f none001-k_$masterK.log
