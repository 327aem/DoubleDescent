#!/bin/bash
echo 1
masterK=64
numGPU=$1

chmod 755 /home/donggeonlee/repo/ImbalanceDDD
chmod 755 /home/donggeonlee/repo/ImbalanceDDD/train_cifar10_imbalance.py

echo 2
cd /home/donggeonlee/repo/ImbalanceDDD
echo Current Path :
pwd

for ((k=$masterK;k>=1;k--))
do
    touch resample001-k_$masterk.log

    echo K: $k - Start Code

    CUDA_VISIBLE_DEVICES=$(($k % numGPU)) python -u train_cifar10_imbalance.py \
        --data_name cifar10 \
        --model_name resnet \
        --train_batch_size 128 \
        --start_k $k \
        --end_k $k \
        --imb_type exp \
        --imb_factor 0.01 \
        --rand_number 0 \
        --train_rule Resample \
	--extra_name imbalance_resample_imb001 > resample001-k_$masterk.log 2>&1 &
    # echo End Code

done

tail -f resample001-k_$masterk.log
